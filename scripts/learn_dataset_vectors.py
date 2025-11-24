import os
import sys
from   pathlib import Path
import h5py
import torch
from   torch.utils.data import random_split, DataLoader

sys.path.append(os.path.normcase(Path(__file__).resolve().parents[1]))
from ml_exploration.config.root import ROOT_DIR
from ml_exploration.config.root import ROOT_DIR
from ml_exploration.dataloading.collate import my_collate_fn
from ml_exploration.dataloading.dataset import MyDataset
from ml_exploration.dataloading.sampler import MyScheduledBatchSampler


if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    
    # load data
    with h5py.File(ROOT_DIR/"ml_exploration/dataloading/datasets/vector_seq_simple.h5", "r") as file:
        raw_data = file["data"][()] # grab all the data
    
    # setup ------------------------------------------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # dataset and split
    ds_full            = MyDataset(raw_data)
    ds_len             = len(ds_full)
    split              = 0.80
    ds_train, ds_valid = random_split(ds_full, [round(split * ds_len), ds_len - round(split * ds_len)]) 
    
    # dataloader and batch schedule
    bs_schedule = 10*[32] + 10*[64]
    sampler_train = MyScheduledBatchSampler(ds_train, bs_schedule, shuffle=True, drop_last=False)
    loader_train = DataLoader(ds_train, batch_sampler=sampler_train, collate_fn=my_collate_fn)
    lodaer_valid = DataLoader(ds_valid, batch_size=1_000)
    n_epochs = sampler_train.get_total_epochs()
    # n_samples = sampler_train.get_total_samples()
    
    onebatch = next(iter(loader_train))
    print(onebatch["sample"].shape)
    
      
    
    
    
    
    
