import random
import math
import torch
from   typing import List
from   torch.utils.data import Sampler, Dataset

class MySampler(Sampler):
    """ responsible for drawing samples (just indices) from the dataset, also shuffling, repetition, etc. Minimal implementation needs to contain a __iter__ function that yields dataset indices and ideally a __len__ method (number of batches yielded!). This is a simple batched sampler"""
    
    def __init__(self, data: torch.utils.data.Dataset, batch_size: int, shuffle=True, drop_last=False):
        """
        Args:
            data_source (torch.utils.data.Dataset): the input dataset
            batch_size (int):                       positive integer constant batch size
            shuffle (bool, optional):               whether to randomize the samples every epoch
            drop_last (bool, optional):             exclude potential last and non-full batches
        """
        
        self.batch_size  = batch_size
        self.shuffle     = shuffle
        self.drop_last   = drop_last
        self.num_samples = len(data)
        self.indices     = list(range(self.num_samples)) # create a base list of dataset sample indices
    
    def __iter__(self):
        """ this is called for every epoch, every time an iterator is "requested" and returned. the iterator then returns one batch worth of (randomized) sample indices at a time. """
        
        if self.shuffle:
            random.shuffle(self.indices)
            
        for start_idx in range(0, self.num_samples, self.batch_size):
            batch = self.indices[start_idx:start_idx+self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                continue # optionally skip the last partial batch
            yield batch
                
    def __len__(self):
        if self.drop_last:
            return math.floor(self.num_samples / self.batch_size)
        else:
            return math.ceil(self.num_samples / self.batch_size)

class MyScheduledBatchSampler(Sampler):
    """ this class implements a simple batch sampler with batch size scheduling (custom input) """
    
    def __init__(self, data: torch.utils.data.Dataset, bs_schedule: List, shuffle=False, drop_last=False, seed=None):
        """
        Args:
            data (torch.utils.data.Dataset): the input dataset
            bs_schedule (List):              batch size schedule as an explicit list of batch sizes
            shuffle (bool, optional):        whether to randomize the samples every epoch
            drop_last (bool, optional):      exclude potential last and non-full batches
        """
        
        assert isinstance(bs_schedule, list), "bs_schedule must be a list!"
        assert all(isinstance(b, int) and b > 0 for b in bs_schedule), "batch sizes must be positive integers!"

        # set up instance variables
        self.bs_schedule   = bs_schedule
        self.num_samples   = len(data) 
        self.indices       = list(range(self.num_samples)) # create a base list of dataset sample indices
        self.current_epoch = 0 
        self.shuffle       = shuffle
        self.drop_last     = drop_last
        if seed is None:
            self.seed = random.randint(0, 2**64)
        else:
            self.seed = seed
                
    def step(self):
        """ analog to optimizer step, this increments the epoch by one, for usage in training loop """
        self.current_epoch += 1
        
        if self.current_epoch > self.get_total_epochs():
            print(f"BS SCHEDULER WARNING: current epoch now outside of range covered by schedule! repeating last bs...")
        
    def set_epoch(self, epoch: int):
        """ update the current epoch explicitly when called in the training loop, just for debugging / customization """
        
        self.current_epoch = epoch
        
    def get_total_epochs(self) -> int:
        """ convenience method for knowing for how many epochs to loop the main training procedure """
        
        return len(self.bs_schedule)

    def _get_current_batch_size(self) -> int:
        """ convenience method for getting the batch size for the current epoch. also implements safeguarding against usage outside of the defined schedule range """

        if self.current_epoch < self.get_total_epochs():
            return self.bs_schedule[self.current_epoch]
        else:
            return self.bs_schedule[-1]

    def __iter__(self):
        """ this function is responsible for returning an iterator. The yield statement is kind of like a return, but instead it just advances the loop further every time the next() method is called, or in a for loop. it kind of remebers the last time it was called """

        if self.shuffle:
            rng = random.Random(self.seed + self.current_epoch)
            rng.shuffle(self.indices) 
            
        batch_size = self._get_current_batch_size()
        
        # note this is only the starting "meta-index" to find the sequence of indices in self.indices for one batch
        for start_idx in range(0, self.num_samples, batch_size):
            batch = self.indices[start_idx:start_idx + batch_size]
            if len(batch) < batch_size and self.drop_last:
                continue # optionally skip the last partial batch
            yield batch

    def __len__(self):
        """ just returns the number of batches in the current epoch. changes by 1 depending on whether the last semi-full one is dropped or not. """
        
        batch_size = self._get_current_batch_size()
        if self.drop_last:
            return math.floor(self.num_samples / batch_size)
        else:
            return math.ceil(self.num_samples / batch_size)
        
        
# SOURCES ==============================================================================================================

# [https://arxiv.org/abs/1711.00489]

# ======================================================================================================================