import torch
from   typing import List
from   .batch import MyBatch

def my_collate_fn(batch: List):
    """ function responsible to stack the samples into batches: stacking the data into torch tensors. The input is the current batch as a LIST of samples [dataset[b1], dataset[b2], ...] where each sample is retrieved by using the __getitem__ of the MyDataset class. Then, the function returns the batch (stacked samples). Ideally should be torch.tensors, but can also be dicts, ndarrays, etc. """
    
    # note: when using structured arrays to load samples, the batch seems to be a list of just one ndarray of the batch
    batch  = batch[0] # remove the this list
    fields = batch.dtype.names

    return MyBatch({f: torch.tensor(batch[f]) for f in fields})