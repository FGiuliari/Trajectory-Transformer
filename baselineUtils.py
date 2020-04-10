from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import torch
import random
import scipy.spatial
import scipy.io

def create_dataset(dataset_folder,dataset_name,val_size,gt,horizon,delim="\t",train=True,eval=False,verbose=False):

        if train==True:
            datasets_list = os.listdir(os.path.join(dataset_folder,dataset_name, "train"))
            full_dt_folder=os.path.join(dataset_folder,dataset_name, "train")
        if train==False and eval==False:
            datasets_list = os.listdir(os.path.join(dataset_folder, dataset_name, "val"))
            full_dt_folder = os.path.join(dataset_folder, dataset_name, "val")
        if train==False and eval==True:
            datasets_list = os.listdir(os.path.join(dataset_folder, dataset_name, "test"))
            full_dt_folder = os.path.join(dataset_folder, dataset_name, "test")


        datasets_list=datasets_list
        data={}
        data_src=[]
        data_trg=[]
        data_seq_start=[]
        data_frames=[]
        data_dt=[]
        data_peds=[]

        val_src = []
        val_trg = []
        val_seq_start = []
        val_frames = []
        val_dt = []
        val_peds=[]

        if verbose:
            print("start loading dataset")
            print("validation set size -> %i"%(val_size))


        for i_dt, dt in enumerate(datasets_list):
            if verbose:
                print("%03i / %03i - loading %s"%(i_dt+1,len(datasets_list),dt))
            raw_data = pd.read_csv(os.path.join(full_dt_folder, dt), delimiter=delim,
                                            names=["frame", "ped", "x", "y"],usecols=[0,1,2,3],na_values="?")

            raw_data.sort_values(by=['frame','ped'], inplace=True)

            inp,out,info=get_strided_data_clust(raw_data,gt,horizon,1)

            dt_frames=info['frames']
            dt_seq_start=info['seq_start']
            dt_dataset=np.array([i_dt]).repeat(inp.shape[0])
            dt_peds=info['peds']



            if val_size>0 and inp.shape[0]>val_size*2.5:
                if verbose:
                    print("created validation from %s" % (dt))
                k = random.sample(np.arange(inp.shape[0]).tolist(), val_size)
                val_src.append(inp[k, :, :])
                val_trg.append(out[k, :, :])
                val_seq_start.append(dt_seq_start[k, :, :])
                val_frames.append(dt_frames[k, :])
                val_dt.append(dt_dataset[k])
                val_peds.append(dt_peds[k])
                inp = np.delete(inp, k, 0)
                out = np.delete(out, k, 0)
                dt_frames = np.delete(dt_frames, k, 0)
                dt_seq_start = np.delete(dt_seq_start, k, 0)
                dt_dataset = np.delete(dt_dataset, k, 0)
                dt_peds = np.delete(dt_peds,k,0)
            elif val_size>0:
                if verbose:
                    print("could not create validation from %s, size -> %i" % (dt,inp.shape[0]))

            data_src.append(inp)
            data_trg.append(out)
            data_seq_start.append(dt_seq_start)
            data_frames.append(dt_frames)
            data_dt.append(dt_dataset)
            data_peds.append(dt_peds)





        data['src'] = np.concatenate(data_src, 0)
        data['trg'] = np.concatenate(data_trg, 0)
        data['seq_start'] = np.concatenate(data_seq_start, 0)
        data['frames'] = np.concatenate(data_frames, 0)
        data['dataset'] = np.concatenate(data_dt, 0)
        data['peds'] = np.concatenate(data_peds, 0)
        data['dataset_name'] = datasets_list

        mean= data['src'].mean((0,1))
        std= data['src'].std((0,1))

        if val_size>0:
            data_val={}
            data_val['src']=np.concatenate(val_src,0)
            data_val['trg'] = np.concatenate(val_trg, 0)
            data_val['seq_start'] = np.concatenate(val_seq_start, 0)
            data_val['frames'] = np.concatenate(val_frames, 0)
            data_val['dataset'] = np.concatenate(val_dt, 0)
            data_val['peds'] = np.concatenate(val_peds, 0)

            return IndividualTfDataset(data, "train", mean, std), IndividualTfDataset(data_val, "validation", mean, std)

        return IndividualTfDataset(data, "train", mean, std), None




        return IndividualTfDataset(data,"train",mean,std), IndividualTfDataset(data_val,"validation",mean,std)



class IndividualTfDataset(Dataset):
    def __init__(self,data,name,mean,std):
        super(IndividualTfDataset,self).__init__()

        self.data=data
        self.name=name

        self.mean= mean
        self.std = std

    def __len__(self):
        return self.data['src'].shape[0]


    def __getitem__(self,index):
        return {'src':torch.Tensor(self.data['src'][index]),
                'trg':torch.Tensor(self.data['trg'][index]),
                'frames':self.data['frames'][index],
                'seq_start':self.data['seq_start'][index],
                'dataset':self.data['dataset'][index],
                'peds': self.data['peds'][index],
                }







def create_folders(baseFolder,datasetName):
    try:
        os.mkdir(baseFolder)
    except:
        pass

    try:
        os.mkdir(os.path.join(baseFolder,datasetName))
    except:
        pass



def get_strided_data(dt, gt_size, horizon, step):
    inp_te = []
    dtt = dt.astype(np.float32)
    raw_data = dtt

    ped = raw_data.ped.unique()
    frame=[]
    ped_ids=[]
    for p in ped:
        for i in range(1+(raw_data[raw_data.ped == p].shape[0] - gt_size - horizon) // step):
            frame.append(dt[dt.ped == p].iloc[i * step:i * step + gt_size + horizon, [0]].values.squeeze())
            # print("%i,%i,%i" % (i * 4, i * 4 + gt_size, i * 4 + gt_size + horizon))
            inp_te.append(raw_data[raw_data.ped == p].iloc[i * step:i * step + gt_size + horizon, 2:4].values)
            ped_ids.append(p)

    frames=np.stack(frame)
    inp_te_np = np.stack(inp_te)
    ped_ids=np.stack(ped_ids)

    inp_no_start = inp_te_np[:,1:,0:2] - inp_te_np[:, :-1, 0:2]
    inp_std = inp_no_start.std(axis=(0, 1))
    inp_mean = inp_no_start.mean(axis=(0, 1))
    inp_norm=inp_no_start
    #inp_norm = (inp_no_start - inp_mean) / inp_std

    #vis=inp_te_np[:,1:,2:4]/np.linalg.norm(inp_te_np[:,1:,2:4],2,axis=2)[:,:,np.newaxis]
    #inp_norm=np.concatenate((inp_norm,vis),2)

    return inp_norm[:,:gt_size-1],inp_norm[:,gt_size-1:],{'mean': inp_mean, 'std': inp_std, 'seq_start': inp_te_np[:, 0:1, :].copy(),'frames':frames,'peds':ped_ids}


def get_strided_data_2(dt, gt_size, horizon, step):
    inp_te = []
    dtt = dt.astype(np.float32)
    raw_data = dtt

    ped = raw_data.ped.unique()
    frame=[]
    ped_ids=[]
    for p in ped:
        for i in range(1+(raw_data[raw_data.ped == p].shape[0] - gt_size - horizon) // step):
            frame.append(dt[dt.ped == p].iloc[i * step:i * step + gt_size + horizon, [0]].values.squeeze())
            # print("%i,%i,%i" % (i * 4, i * 4 + gt_size, i * 4 + gt_size + horizon))
            inp_te.append(raw_data[raw_data.ped == p].iloc[i * step:i * step + gt_size + horizon, 2:4].values)
            ped_ids.append(p)

    frames=np.stack(frame)
    inp_te_np = np.stack(inp_te)
    ped_ids=np.stack(ped_ids)

    inp_relative_pos= inp_te_np-inp_te_np[:,:1,:]
    inp_speed = np.concatenate((np.zeros((inp_te_np.shape[0],1,2)),inp_te_np[:,1:,0:2] - inp_te_np[:, :-1, 0:2]),1)
    inp_accel = np.concatenate((np.zeros((inp_te_np.shape[0],1,2)),inp_speed[:,1:,0:2] - inp_speed[:, :-1, 0:2]),1)
    #inp_std = inp_no_start.std(axis=(0, 1))
    #inp_mean = inp_no_start.mean(axis=(0, 1))
    #inp_norm= inp_no_start
    #inp_norm = (inp_no_start - inp_mean) / inp_std

    #vis=inp_te_np[:,1:,2:4]/np.linalg.norm(inp_te_np[:,1:,2:4],2,axis=2)[:,:,np.newaxis]
    #inp_norm=np.concatenate((inp_norm,vis),2)
    inp_norm=np.concatenate((inp_te_np,inp_relative_pos,inp_speed,inp_accel),2)
    inp_mean=np.zeros(8)
    inp_std=np.ones(8)

    return inp_norm[:,:gt_size],inp_norm[:,gt_size:],{'mean': inp_mean, 'std': inp_std, 'seq_start': inp_te_np[:, 0:1, :].copy(),'frames':frames,'peds':ped_ids}

def get_strided_data_clust(dt, gt_size, horizon, step):
    inp_te = []
    dtt = dt.astype(np.float32)
    raw_data = dtt

    ped = raw_data.ped.unique()
    frame=[]
    ped_ids=[]
    for p in ped:
        for i in range(1+(raw_data[raw_data.ped == p].shape[0] - gt_size - horizon) // step):
            frame.append(dt[dt.ped == p].iloc[i * step:i * step + gt_size + horizon, [0]].values.squeeze())
            # print("%i,%i,%i" % (i * 4, i * 4 + gt_size, i * 4 + gt_size + horizon))
            inp_te.append(raw_data[raw_data.ped == p].iloc[i * step:i * step + gt_size + horizon, 2:4].values)
            ped_ids.append(p)

    frames=np.stack(frame)
    inp_te_np = np.stack(inp_te)
    ped_ids=np.stack(ped_ids)

    #inp_relative_pos= inp_te_np-inp_te_np[:,:1,:]
    inp_speed = np.concatenate((np.zeros((inp_te_np.shape[0],1,2)),inp_te_np[:,1:,0:2] - inp_te_np[:, :-1, 0:2]),1)
    #inp_accel = np.concatenate((np.zeros((inp_te_np.shape[0],1,2)),inp_speed[:,1:,0:2] - inp_speed[:, :-1, 0:2]),1)
    #inp_std = inp_no_start.std(axis=(0, 1))
    #inp_mean = inp_no_start.mean(axis=(0, 1))
    #inp_norm= inp_no_start
    #inp_norm = (inp_no_start - inp_mean) / inp_std

    #vis=inp_te_np[:,1:,2:4]/np.linalg.norm(inp_te_np[:,1:,2:4],2,axis=2)[:,:,np.newaxis]
    #inp_norm=np.concatenate((inp_norm,vis),2)
    inp_norm=np.concatenate((inp_te_np,inp_speed),2)
    inp_mean=np.zeros(4)
    inp_std=np.ones(4)

    return inp_norm[:,:gt_size],inp_norm[:,gt_size:],{'mean': inp_mean, 'std': inp_std, 'seq_start': inp_te_np[:, 0:1, :].copy(),'frames':frames,'peds':ped_ids}


def distance_metrics(gt,preds):
    errors = np.zeros(preds.shape[:-1])
    for i in range(errors.shape[0]):
        for j in range(errors.shape[1]):
            errors[i, j] = scipy.spatial.distance.euclidean(gt[i, j], preds[i, j])
    return errors.mean(),errors[:,-1].mean(),errors