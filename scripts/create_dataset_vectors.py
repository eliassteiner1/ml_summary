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
    
    # an then reading the file too:
    print("reading file for testing:")
    with h5py.File(ROOT_DIR/"ml_exploration/dataloading/datasets/MNIST.h5", "r") as file:
        dset = file["data"][0:10] # only read the first 10 lines of the array
        dset = file["data"][()] # get a dump of the entire h5py data as ndarray

    test_img = eo.rearrange(dset["image"][:10], "B h w -> h (B w)")
    plt.imshow(test_img, cmap="grey")
    plt.show()