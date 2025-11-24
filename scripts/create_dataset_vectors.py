import os
import sys
from   pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import h5py
import einops as eo

sys.path.append(os.path.normcase(Path(__file__).resolve().parents[1]))
from ml_exploration.config.root import ROOT_DIR


if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    
    N = 10_000
    
    sample = np.random.rand(N, 20, 10)
    negatives = np.ones((20, 10))
    negatives[:, 1::2] = -1
    
    label_idx = np.sum(sample*negatives[None, :, :], axis=-1)
    label = sample[np.arange(N), np.argmax(label_idx, axis=-1), :]

    dataset_arr = np.empty(N, dtype=[("sample", np.float32, (20, 10)), ("label", np.float32, (10, ))])
    dataset_arr["sample"] = sample
    dataset_arr["label"] = label

    with h5py.File(ROOT_DIR/"ml_exploration/dataloading/datasets/vector_seq_simple.h5", "w") as file: 
        dset = file.create_dataset("data", data=dataset_arr, compression="gzip")
    
    
    # sanity check   
    with h5py.File(ROOT_DIR/"ml_exploration/dataloading/datasets/vector_seq_simple.h5", "r") as file:
        dset = file["data"][()]
        
    print(dset["sample"][:50].shape)
