import torch
import numpy as np
from   typing import List

from   .batch import MyBatch


def my_collate_fn(batch: List):
    """ function responsible to stack the samples into batches: stacking the data into torch tensors. The input is the current batch as a LIST of samples [dataset[b1], dataset[b2], ...] where each sample is retrieved by using the __getitem__ of the MyDataset class. Then, the function returns the batch (stacked samples). Ideally should be torch.tensors, but can also be dicts, ndarrays, etc. """

    batch_arr    = np.array(batch)
    batch_fields = batch_arr.dtype.names
    
    return MyBatch({fd: torch.tensor(batch_arr[fd]) for fd in batch_fields})