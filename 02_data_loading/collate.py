import torch
from   typing import List

class MyBatch(dict):
    """ this implements a convenience extension for dicts to be used to hold a batch of samples. Mainly the .to(some device) and clone method are added to standard dicts. """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        for key, value in self.items():
            if not torch.is_tensor(value):
                raise TypeError(f"batch field \"{key}\" must be a torch.tensor, got {type(value)} instead!")

    def to(self, device: str, non_blocking=False):
        for key, value in self.items():
            self[key] = value.to(device, non_blocking=non_blocking)
        return self # also return itself to be able to chain calls
        
    def clone(self):
        return MyBatch({key: value.clone() for key, value in self.items()})
    
def my_collate_fn(batch: List):
    """ function responsible to stack the samples into batches: stacking the data into torch tensors. The input is the current batch as a LIST of samples [dataset[b1], dataset[b2], ...] where each sample is retrieved by using the __getitem__ of the MyDataset class. Then, the function returns the batch (stacked samples). Ideally should be torch.tensors, but can also be dicts, ndarrays, etc. """
    
    # note: when using structured arrays to load samples, the batch seems to be a list of just one ndarray of the batch
    batch  = batch[0] # remove the this list
    fields = batch.dtype.names

    return MyBatch({f: torch.tensor(batch[f]) for f in fields})