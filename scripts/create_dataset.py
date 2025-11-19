import os
import sys
from   pathlib import Path

import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from   tqdm import tqdm
import h5py

sys.path.insert(0, os.path.normcase(Path(__file__).resolve().parents[1]))
from ml_exploration.config.root import ROOT_DIR


if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")

    # # just load the standard datasets which are provided by pytorch
    mnist_trainset = datasets.MNIST(root=ROOT_DIR/"ml_exploration/dataloading/datasets", train=True, download=True)
    mnist_testset  = datasets.MNIST(root=ROOT_DIR/"ml_exploration/dataloading/datasets", train=False, download=True)
    dataset        = mnist_testset + mnist_trainset
    print(f"length of dataset from torchvision: {len(dataset)}")

    # # extract images to np array + labels, structured array is nice!
    images = []
    labels = []
    for el in tqdm(dataset, desc="extracting MNIST data", unit_scale=True):
        img = np.array(el[0])
        lbl = el[1]
        images.append(img)
        labels.append(lbl)
        
    img_arr              = np.array(images)
    lbl_arr              = np.array(labels)
    dataset_arr          = np.empty(len(dataset), dtype=[("label", np.int32), ("image", np.uint8, (28, 28))])
    dataset_arr["image"] = img_arr
    dataset_arr["label"] = lbl_arr
        
    # STORE: either stacked array (efficient?) or each sample as dict (convenient but inefficient?)

    # 1) h5py is perfect for dumping numpy-esque types directly into files!
    with h5py.File(ROOT_DIR/"ml_exploration/dataloading/datasets/MNIST.h5", "w") as file: 
        dset = file.create_dataset("mydataset", (10, ), dtype="i") # initialized as empty
        dset[()] = list(range(10)) # use () go get a ref to the actual data (ndarray) and not just the h5pydataset obj
        dset[()] = np.arange(10) # overwrite, () -> same here
        
    # 2) for complicated samples it might be nice to have one dict per sample with "label", "datax", "datay", "mask", ...
    #    but dicts fail for h5yp! 

    # 3) so mabye structured array? works! so store the mnist files:
    with h5py.File(ROOT_DIR/"ml_exploration/dataloading/datasets/MNIST.h5", "w") as file:
        dset = file.create_dataset("data", data=dataset_arr, compression="gzip") # directly initialize with data

    # an then reading the file too:
    print("reading file for testing:")
    with h5py.File(ROOT_DIR/"ml_exploration/dataloading/datasets/MNIST.h5", "r") as file:
        dset = file["data"][0:10] # only read the first 10 lines of the array
        dset = file["data"][()] # get a dump of the entire h5py data as ndarray

    print(dset["label"][0:10])
    plt.imshow(dset["image"][0:10].transpose(1, 0, 2).reshape(28, -1), cmap="gray")
    plt.show()
   
    
# NOTES ================================================================================================================

# ideally samples are normally just stored as lists of tuples (e.g. [(x1, y1), (x2, y2), ...]) or for more complicated samples as dicts / lists of dicts. Then, the standard collate_fn from pytorch works out of the box. If using structured array (easier to safe to h5py) the loaded samples either have to be modified, or a custom collate function has to be used to batch them, as one sample (row) from such a structured array is of type np.void and cannot be collated with other samples by default. 