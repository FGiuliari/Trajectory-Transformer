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


def transform_batch(src,trg):
    trg_y = trg.clone()
    trg   = torch.cat((trg,torch.zeros((trg.shape[0],trg.shape[1],1))),2)
    start_seq = torch.zeros((trg.shape[0],1,trg.shape[-1]))
    start_seq[:,:,-1]=1
    trg=torch.cat((start_seq,trg[:,:-1,:]),1)
    src_mask=torch.ones((src.shape[0],1,src.shape[1]))
    trg_mask=subsequent_mask(trg.shape[1]).repeat((trg.shape[0],1,1))

    return src,src_mask,trg,trg_mask,trg_y


def train_epoch(model,optimizer,dataloader,device):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    mean = torch.Tensor(model.mean)
    std  = torch.Tensor(model.std)
    model.train()

    for i,batch in enumerate(dataloader):

        #

        inp = (batch['src']-mean)/std
        trg = (batch['trg']-mean)/std
        src, src_mask, trg, trg_mask, trg_y = transform_batch(inp, trg)
        src, src_mask, trg, trg_mask, trg_y = src.to(device), src_mask.to(device), trg.to(device), trg_mask.to(
            device), trg_y.to(device)
        n_tokens = trg.shape[0] * trg.shape[1]

        # calculate loss

        optimizer.optimizer.zero_grad()
        train_pred = model(src, trg, src_mask, trg_mask)
        loss = F.pairwise_distance(train_pred[:, :].view(-1, 2), trg_y[:, :].view(-1, 2)).mean()
        loss.backward()
        optimizer.step()

        loss=loss*n_tokens

        total_loss += loss
        total_tokens += n_tokens
        tokens += n_tokens
        if i % 10 == 1:
            elapsed = time.time() - start
            print('Epoch step: %d Loss %f Tokens per Sec: %f' % (i, loss / n_tokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens




















def main():
    parser=argparse.ArgumentParser(description='Train the individual Transformer model')
    parser.add_argument('--dataset_folder',type=str,default='datasets')
    parser.add_argument('--dataset_name',type=str,default='zara1')
    parser.add_argument('--obs',type=int,default=8)
    parser.add_argument('--preds',type=int,default=12)
    parser.add_argument('--emb_size',type=int,default=1024)
    parser.add_argument('--heads',type=int, default=8)
    parser.add_argument('--layers',type=int,default=6)
    parser.add_argument('--dropout',type=float,default=0.1)
    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--output_folder',type=str,default='Output')
    parser.add_argument('--val_size',type=int, default=50)
    parser.add_argument('--gpu_device',type=str, default="0")
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('--max_epoch',type=int, default=100)
    parser.add_argument('--batch_size',type=int,default=256)
    parser.add_argument('--validation_epoch_start', type=int, default=30)
    parser.add_argument('--resume_train',action='store_true')
    parser.add_argument('--delim',type=str,default='\t')
    parser.add_argument('--name', type=str, default="zara1")



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
        os.mkdir('output/BERT')
    except:
        pass
    try:
        os.mkdir(f'models/BERT')
    except:
        pass

    try:
        os.mkdir(f'output/BERT/{args.name}')
    except:
        pass

    try:
        os.mkdir(f'models/BERT/{args.name}')
    except:
        pass

    log = SummaryWriter('logs/BERT_%s' % model_name)

    log.add_scalar('eval/mad', 0, 0)
    log.add_scalar('eval/fad', 0, 0)

    try:
        os.mkdir(args.name)
    except:
        pass

    device=torch.device("cuda")
    if args.cpu or not torch.cuda.is_available():
        device=torch.device("cpu")

    args.verbose=True


    ## creation of the dataloaders for train and validation
    train_dataset,_ = baselineUtils.create_dataset(args.dataset_folder,args.dataset_name,0,args.obs,args.preds,delim=args.delim,train=True,verbose=args.verbose)
    val_dataset, _ = baselineUtils.create_dataset(args.dataset_folder, args.dataset_name, 0, args.obs,
                                                                    args.preds, delim=args.delim, train=False,
                                                                    verbose=args.verbose)
    test_dataset,_ =  baselineUtils.create_dataset(args.dataset_folder,args.dataset_name,0,args.obs,args.preds,delim=args.delim,train=False,eval=True,verbose=args.verbose)










    from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig, AdamW






    config= BertConfig(vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='relu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12)
    model = BertModel(config).to(device)


    from individual_TF import LinearEmbedding as NewEmbed,Generator as GeneratorTS
    a=NewEmbed(3, 768).to(device)
    model.set_input_embeddings(a)
    generator=GeneratorTS(768,2).to(device)
    #model.set_output_embeddings(GeneratorTS(1024,2))

    tr_dl=torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    #optim = SGD(list(a.parameters())+list(model.parameters())+list(generator.parameters()),lr=0.01)
    #sched=torch.optim.lr_scheduler.StepLR(optim,0.0005)
    optim = NoamOpt(768, 0.1, len(tr_dl),
                        torch.optim.Adam(list(a.parameters())+list(model.parameters())+list(generator.parameters()), lr=0, betas=(0.9, 0.98), eps=1e-9))
    #optim=Adagrad(list(a.parameters())+list(model.parameters())+list(generator.parameters()),lr=0.01,lr_decay=0.001)
    epoch=0


    mean=train_dataset[:]['src'][:,:,2:4].mean((0,1))*0
    std=train_dataset[:]['src'][:,:,2:4].std((0,1))*0+1

    while epoch<args.max_epoch:
        epoch_loss=0
        model.train()

        for id_b,batch in enumerate(tr_dl):

            optim.optimizer.zero_grad()
            r=0
            rot_mat = np.array([[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]])

            inp=((batch['src'][:,:,2:4]-mean)/std).to(device)
            inp=torch.matmul(inp, torch.from_numpy(rot_mat).float().to(device))
            trg_masked=torch.zeros((inp.shape[0],args.preds,2)).to(device)
            inp_cls=torch.ones(inp.shape[0],inp.shape[1],1).to(device)
            trg_cls= torch.zeros(trg_masked.shape[0], trg_masked.shape[1], 1).to(device)
            inp_cat=torch.cat((inp,trg_masked),1)
            cls_cat=torch.cat((inp_cls,trg_cls),1)
            net_input=torch.cat((inp_cat,cls_cat),2)

            position = torch.arange(0, net_input.shape[1]).repeat(inp.shape[0],1).long().to(device)
            token = torch.zeros((inp.shape[0],net_input.shape[1])).long().to(device)
            attention_mask = torch.ones((inp.shape[0], net_input.shape[1])).long().to(device)

            out=model(input_ids=net_input,position_ids=position,token_type_ids=token,attention_mask=attention_mask)

            pred=generator(out[0])




            loss = F.pairwise_distance(pred[:, :].contiguous().view(-1, 2), torch.matmul(torch.cat((batch['src'][:,:,2:4],batch['trg'][:, :,2:4]),1).contiguous().view(-1, 2).to(device), torch.from_numpy(rot_mat).float().to(device))).mean()
            loss.backward()
            optim.step()
            print("epoch %03i/%03i  frame %04i / %04i loss: %7.4f" % (epoch, args.max_epoch, id_b, len(tr_dl), loss.item()))
            epoch_loss += loss.item()
        #sched.step()
        log.add_scalar('Loss/train', epoch_loss / len(tr_dl), epoch)
        with torch.no_grad():
            model.eval()

            gt=[]
            pr=[]
            val_loss=0
            for batch in val_dl:
                inp = ((batch['src'][:,:,2:4]-mean)/std).to(device)
                trg_masked = torch.zeros((inp.shape[0], args.preds, 2)).to(device)
                inp_cls = torch.ones(inp.shape[0], inp.shape[1], 1).to(device)
                trg_cls = torch.zeros(trg_masked.shape[0], trg_masked.shape[1], 1).to(device)
                inp_cat = torch.cat((inp, trg_masked), 1)
                cls_cat = torch.cat((inp_cls, trg_cls), 1)
                net_input = torch.cat((inp_cat, cls_cat), 2)

                position = torch.arange(0, net_input.shape[1]).repeat(inp.shape[0], 1).long().to(device)
                token = torch.zeros((inp.shape[0], net_input.shape[1])).long().to(device)
                attention_mask = torch.zeros((inp.shape[0], net_input.shape[1])).long().to(device)

                out = model(input_ids=net_input, position_ids=position, token_type_ids=token, attention_mask=attention_mask)

                pred = generator(out[0])

                loss = F.pairwise_distance(pred[:, :].contiguous().view(-1, 2),
                                           torch.cat((batch['src'][:, :, 2:4], batch['trg'][:, :, 2:4]),
                                                     1).contiguous().view(-1, 2).to(device)).mean()
                val_loss += loss.item()

                gt_b=batch['trg'][:,:,0:2]
                preds_tr_b=pred[:,args.obs:].cumsum(1).to('cpu').detach()+batch['src'][:,-1:,0:2]
                gt.append(gt_b)
                pr.append(preds_tr_b)

            gt=np.concatenate(gt,0)
            pr=np.concatenate(pr,0)
            mad,fad,errs=baselineUtils.distance_metrics(gt,pr)
            log.add_scalar('validation/loss', val_loss / len(val_dl), epoch)
            log.add_scalar('validation/mad', mad, epoch)
            log.add_scalar('validation/fad', fad, epoch)


            model.eval()

            gt=[]
            pr=[]
            for batch in test_dl:
                inp = ((batch['src'][:,:,2:4]-mean)/std).to(device)
                trg_masked = torch.zeros((inp.shape[0], args.preds, 2)).to(device)
                inp_cls = torch.ones(inp.shape[0], inp.shape[1], 1).to(device)
                trg_cls = torch.zeros(trg_masked.shape[0], trg_masked.shape[1], 1).to(device)
                inp_cat = torch.cat((inp, trg_masked), 1)
                cls_cat = torch.cat((inp_cls, trg_cls), 1)
                net_input = torch.cat((inp_cat, cls_cat), 2)

                position = torch.arange(0, net_input.shape[1]).repeat(inp.shape[0], 1).long().to(device)
                token = torch.zeros((inp.shape[0], net_input.shape[1])).long().to(device)
                attention_mask = torch.zeros((inp.shape[0], net_input.shape[1])).long().to(device)

                out = model(input_ids=net_input, position_ids=position, token_type_ids=token, attention_mask=attention_mask)

                pred = generator(out[0])

                gt_b=batch['trg'][:,:,0:2]
                preds_tr_b=pred[:,args.obs:].cumsum(1).to('cpu').detach()+batch['src'][:,-1:,0:2]
                gt.append(gt_b)
                pr.append(preds_tr_b)

            gt=np.concatenate(gt,0)
            pr=np.concatenate(pr,0)
            mad,fad,errs=baselineUtils.distance_metrics(gt,pr)

            torch.save(model.state_dict(),"models/BERT/%s/ep_%03i.pth"%(args.name,epoch))
            torch.save(generator.state_dict(),"models/BERT/%s/gen_%03i.pth"%(args.name,epoch))
            torch.save(a.state_dict(),"models/BERT/%s/emb_%03i.pth"%(args.name,epoch))

            log.add_scalar('eval/mad', mad, epoch)
            log.add_scalar('eval/fad', fad, epoch)


        epoch+=1

    ab=1































if __name__=='__main__':
    main()
