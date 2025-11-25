import os
import sys
from   pathlib import Path

import torch
from   torch.utils.data import DataLoader
import h5py

sys.path.insert(0, os.path.normcase(Path(__file__).resolve().parents[1]))
from ml_exploration.config.root import ROOT_DIR
from ml_exploration.dataloading.collate import my_collate_fn
from ml_exploration.dataloading.dataset import MyDataset
from ml_exploration.dataloading.sampler import MyScheduledBatchSampler


if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    
    # load data from my own dataset
    print(f"Cuda Check: {torch.cuda.is_available()}")
    with h5py.File(ROOT_DIR/"ml_exploration/dataloading/datasets/MNIST.h5", "r") as file:
        data = file["data"][()]
    
    myset     = MyDataset(data)
    bs_sched  = 1*[4] + 1*[8] # needs just a list of batch sizes, and this is a convenient way to construct
    mysampler = MyScheduledBatchSampler(myset, bs_sched, shuffle=True)
    myloader  = DataLoader(
        dataset=myset, 
        batch_sampler=mysampler, # watch out, my sampler is a batch sampler
        collate_fn=my_collate_fn, 
        num_workers=0, 
        pin_memory=True
    )

    onebatch = next(iter(myloader))
    onebatch.to("cuda:0")
    print(onebatch["image"].shape)
    