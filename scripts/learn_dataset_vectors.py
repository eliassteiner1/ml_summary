import os
import sys
from   pathlib import Path
import h5py
import numpy as np
import torch
from   torch.utils.data import random_split, DataLoader
import torch.nn as nn
from   typing import Callable

sys.path.append(os.path.normcase(Path(__file__).resolve().parents[1]))
from ml_exploration.config.root import ROOT_DIR
from ml_exploration.config.root import ROOT_DIR
from ml_exploration.dataloading.collate import my_collate_fn
from ml_exploration.dataloading.dataset import MyDataset
from ml_exploration.dataloading.sampler import MyScheduledBatchSampler
from ml_exploration.layers.self_attention import AttentionBlock


class LearnedPooling(nn.Module):
    def __init__(self, input_feats_dim: int, hidden_layers: list[int]) -> None:
        """ Implements the attention pooling mechanism. A simple mlp learns one weight per input sequence. The output is then a linear combination of all the weights and inputs. """
        super().__init__()
        
        self.activation = nn.GELU()
        self.softmax    = nn.Softmax(dim=1)
        
        # set up the sequential mlp to learn the per-input weights [BS | input length | 1]
        layers        = []
        current_feats = input_feats_dim
        for el in hidden_layers:
            layers.append(nn.Linear(current_feats, el))
            layers.append(self.activation)
            current_feats = el
        layers.append(nn.Linear(current_feats, 1))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x, msk=None):
        """ input x:   [BS (var, broadcastable) | input length (var) | n feats (known, architecture defining)]. """
        
        # calculate the weights by learning them with an mlp    
        weights = self.mlp(x)
        weights = self.softmax(weights).squeeze()

        # recombine the outputs from a linear combination of weights and inputs
        x = torch.einsum("Bi, Bij -> Bj ", weights, x)
        
        return x

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.lin_in = nn.Linear(10, 32)
        
        self.att_1 = AttentionBlock(32, 32, att_drop=0.05, mlp_drop=0.05, res_drop=0.05)
        self.att_2 = AttentionBlock(32, 32, att_drop=0.05, mlp_drop=0.05, res_drop=0.05)
        self.att_3 = AttentionBlock(32, 32, att_drop=0.05, mlp_drop=0.05, res_drop=0.05)
        
        self.lin_out = nn.Linear(32, 10)
        self.pool = LearnedPooling(10, [10, 10])
        
        self.activation = nn.GELU()
        
    def forward(self, x):
        
        x = self.activation(self.lin_in(x))
        x = self.att_1(x)
        x = self.att_2(x)
        x = self.att_3(x)
        x = self.activation(self.lin_out(x))
        x = self.pool(x)

        return x
    
def criterion(pred, label):
    return torch.mean(torch.sum((pred - label)**2, dim=-1))
    
def validation(criterion: Callable, network: nn.Module, validation_loader: DataLoader, DEVICE: str):
    with torch.no_grad():
        losses = []
        for batch in loader_valid:
            batch.to(DEVICE)
            pred = network(batch["sample"])
            loss = criterion(pred, batch["label"])
            losses.append(loss.item())
            
    return np.mean(losses)


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
    bs_schedule   = 5*[32] + 5*[64] + 5*[128] + 5*[256]
    sampler_train = MyScheduledBatchSampler(ds_train, bs_schedule, shuffle=True, drop_last=False)
    loader_train  = DataLoader(ds_train, batch_sampler=sampler_train, collate_fn=my_collate_fn)
    loader_valid  = DataLoader(ds_valid, batch_size=1_000, collate_fn=my_collate_fn)
    n_epochs      = sampler_train.get_total_epochs()
    n_steps       = sampler_train.get_total_steps()
    n_samples     = sampler_train.get_total_samples()
    
    # network
    model         = Network().to(DEVICE)
    optim         = torch.optim.AdamW(model.parameters(), lr=10**-3, betas=(0.9, 0.999), weight_decay=0)
    
    # plotter
    ...
    
    
    for E in range(n_epochs):
        model.train()
        
        losses_train = []
        
        for batch in loader_train:
            
            batch.to(DEVICE)
            optim.zero_grad()
            pred = model(batch["sample"])
            loss = criterion(pred, batch["label"])
            loss.backward()
            optim.step()
            losses_train.append(loss.item())
        
        
        loss_valid = validation(criterion, model, loader_valid, DEVICE)
        loss_train = np.mean(losses_train)
        
        print(f"epoch {E:02} - valid.Loss: {loss_valid:03.4f} (train.Loss = {loss_train:03.4f})")   
        sampler_train.step() # advance batch schedule
        
    
