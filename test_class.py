import argparse
import baselineUtils
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from transformer.batch import subsequent_mask
from torch.optim import Adam,SGD,RMSprop,Adagrad
from transformer.noam_opt import NoamOpt
import numpy as np
import scipy.io
import json
import pickle

from torch.utils.tensorboard import SummaryWriter
import quantized_TF





def main():
    parser=argparse.ArgumentParser(description='Train the individual Transformer model')
    parser.add_argument('--dataset_folder',type=str,default='datasets')
    parser.add_argument('--dataset_name',type=str,default='zara1')
    parser.add_argument('--obs',type=int,default=8)
    parser.add_argument('--preds',type=int,default=12)
    parser.add_argument('--emb_size',type=int,default=512)
    parser.add_argument('--heads',type=int, default=8)
    parser.add_argument('--layers',type=int,default=6)
    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('--batch_size',type=int,default=256)
    parser.add_argument('--delim',type=str,default='\t')
    parser.add_argument('--name', type=str, default="zara1")
    parser.add_argument('--epoch',type=str,default="00001")
    parser.add_argument('--num_samples', type=int, default="20")




    args=parser.parse_args()
    model_name=args.name

    try:
        os.mkdir('models')
    except:
        pass
    try:
        os.mkdir('output')
    except:
        pass
    try:
        os.mkdir('output/QuantizedTF')
    except:
        pass
    try:
        os.mkdir(f'models/QuantizedTF')
    except:
        pass

    try:
        os.mkdir(f'output/QuantizedTF/{args.name}')
    except:
        pass

    try:
        os.mkdir(f'models/QuantizedTF/{args.name}')
    except:
        pass

    #log=SummaryWriter('logs/%s'%model_name)

    # log.add_scalar('eval/mad', 0, 0)
    # log.add_scalar('eval/fad', 0, 0)
    device=torch.device("cuda")

    if args.cpu or not torch.cuda.is_available():
        device=torch.device("cpu")

    args.verbose=True


    ## creation of the dataloaders for train and validation

    test_dataset,_ =  baselineUtils.create_dataset(args.dataset_folder,args.dataset_name,0,args.obs,args.preds,delim=args.delim,train=False,eval=True,verbose=args.verbose)

    mat = scipy.io.loadmat(os.path.join(args.dataset_folder, args.dataset_name, "clusters.mat"))

    clusters=mat['centroids']

    model=quantized_TF.QuantizedTF(clusters.shape[0], clusters.shape[0]+1, clusters.shape[0], N=args.layers,
                   d_model=args.emb_size, d_ff=1024, h=args.heads).to(device)

    model.load_state_dict(torch.load(f'models/QuantizedTF/{args.name}/{args.epoch}.pth'))
    model.to(device)


    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    #optim = SGD(list(a.parameters())+list(model.parameters())+list(generator.parameters()),lr=0.01)
    #sched=torch.optim.lr_scheduler.StepLR(optim,0.0005)




    # DETERMINISTIC MODE
    with torch.no_grad():
        model.eval()
        gt=[]
        pr=[]
        inp_=[]
        peds=[]
        frames=[]
        dt=[]
        for id_b,batch in enumerate(test_dl):
            print(f"batch {id_b:03d}/{len(test_dl)}")
            peds.append(batch['peds'])
            frames.append(batch['frames'])
            dt.append(batch['dataset'])
            scale = np.random.uniform(0.5, 2)
            # rot_mat = np.array([[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]])
            n_in_batch = batch['src'].shape[0]
            speeds_inp = batch['src'][:, 1:, 2:4]
            gt_b = batch['trg'][:, :, 0:2]
            inp = torch.tensor(
                scipy.spatial.distance.cdist(speeds_inp.reshape(-1, 2), clusters).argmin(axis=1).reshape(n_in_batch,
                                                                                                         -1)).to(
                device)
            src_att = torch.ones((inp.shape[0], 1,inp.shape[1])).to(device)
            start_of_seq = torch.tensor([clusters.shape[0]]).repeat(n_in_batch).unsqueeze(1).to(device)
            dec_inp = start_of_seq

            for i in range(args.preds):
                trg_att = subsequent_mask(dec_inp.shape[1]).repeat(n_in_batch, 1, 1).to(device)
                out = model(inp, dec_inp, src_att, trg_att)
                dec_inp=torch.cat((dec_inp,out[:,-1:].argmax(dim=2)),1)


            preds_tr_b=clusters[dec_inp[:,1:].cpu().numpy()].cumsum(1)+batch['src'][:,-1:,0:2].cpu().numpy()
            gt.append(gt_b)
            pr.append(preds_tr_b)

        peds=np.concatenate(peds,0)
        frames=np.concatenate(frames,0)
        dt=np.concatenate(dt,0)
        gt=np.concatenate(gt,0)
        dt_names=test_dataset.data['dataset_name']
        pr=np.concatenate(pr,0)
        mad,fad,errs=baselineUtils.distance_metrics(gt,pr)

        #log.add_scalar('eval/DET_mad', mad, epoch)
        #log.add_scalar('eval/DET_fad', fad, epoch)

        scipy.io.savemat(f"output/QuantizedTF/{args.name}/MM_deterministic.mat",{'input':inp,'gt':gt,'pr':pr,'peds':peds,'frames':frames,'dt':dt,'dt_names':dt_names})

        print("Determinitic:")
        print("mad: %6.3f"%mad)
        print("fad: %6.3f" % fad)


        # MULTI MODALITY
        num_samples=args.num_samples

        model.eval()
        gt=[]
        pr_all={}
        inp_=[]
        peds=[]
        frames=[]
        dt=[]
        for sam in range(num_samples):
            pr_all[sam]=[]
        for id_b,batch in enumerate(test_dl):
            print(f"batch {id_b:03d}/{len(test_dl)}")
            peds.append(batch['peds'])
            frames.append(batch['frames'])
            dt.append(batch['dataset'])
            scale = np.random.uniform(0.5, 2)
            # rot_mat = np.array([[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]])
            n_in_batch = batch['src'].shape[0]
            speeds_inp = batch['src'][:, 1:, 2:4]
            gt_b = batch['trg'][:, :, 0:2]
            gt.append(gt_b)
            inp__=batch['src'][:,:,0:2]
            inp_.append(inp__)
            inp = torch.tensor(
                scipy.spatial.distance.cdist(speeds_inp.reshape(-1, 2), clusters).argmin(axis=1).reshape(n_in_batch,
                                                                                                         -1)).to(
                device)
            src_att = torch.ones((inp.shape[0], 1,inp.shape[1])).to(device)
            start_of_seq = torch.tensor([clusters.shape[0]]).repeat(n_in_batch).unsqueeze(1).to(device)

            for sam in range(num_samples):
                dec_inp = start_of_seq

                for i in range(args.preds):
                    trg_att = subsequent_mask(dec_inp.shape[1]).repeat(n_in_batch, 1, 1).to(device)
                    out = model.predict(inp, dec_inp, src_att, trg_att)
                    h=out[:,-1]
                    dec_inp=torch.cat((dec_inp,torch.multinomial(h,1)),1)


                preds_tr_b=clusters[dec_inp[:,1:].cpu().numpy()].cumsum(1)+batch['src'][:,-1:,0:2].cpu().numpy()

                pr_all[sam].append(preds_tr_b)
        peds=np.concatenate(peds,0)
        frames=np.concatenate(frames,0)
        dt=np.concatenate(dt,0)
        gt=np.concatenate(gt,0)
        dt_names=test_dataset.data['dataset_name']
        #pr=np.concatenate(pr,0)
        inp=np.concatenate(inp_,0)
        samp = {}
        for k in pr_all.keys():
            samp[k] = {}
            samp[k]['pr'] = np.concatenate(pr_all[k], 0)
            samp[k]['mad'], samp[k]['fad'], samp[k]['err'] = baselineUtils.distance_metrics(gt, samp[k]['pr'])

        ev = [samp[i]['err'] for i in range(num_samples)]
        e20 = np.stack(ev, -1)
        mad_samp=e20.mean(1).min(-1).mean()
        fad_samp=e20[:,-1].min(-1).mean()
        #mad,fad,errs=baselineUtils.distance_metrics(gt,pr)

        #log.add_scalar('eval/MM_mad', mad_samp, epoch)
        #log.add_scalar('eval/MM_fad', fad_samp, epoch)
        preds_all_fin=np.stack(list([samp[i]['pr'] for i in range(num_samples)]),-1)
        scipy.io.savemat(f"output/QuantizedTF/{args.name}/MM_{num_samples}.mat",{'input':inp,'gt':gt,'pr':preds_all_fin,'peds':peds,'frames':frames,'dt':dt,'dt_names':dt_names})

        print("Determinitic:")
        print("mad: %6.3f"%mad)
        print("fad: %6.3f" % fad)

        print("Multimodality:")
        print("mad: %6.3f"%mad_samp)
        print("fad: %6.3f" % fad_samp)
































if __name__=='__main__':
    main()
