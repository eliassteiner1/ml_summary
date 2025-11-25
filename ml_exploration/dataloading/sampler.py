import random
import math
import torch
import numpy as np
from   typing import List
from   torch.utils.data import Sampler

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
        assert all(isinstance(bs, int) and bs > 0 for bs in bs_schedule), "batch sizes must be positive integers!"

        # set up instance variables
        self._bs_schedule   = bs_schedule
        self._num_samples   = len(data) 
        self._indices       = list(range(self._num_samples)) # create a base list of dataset sample indices
        self._current_epoch = 0 
        self._shuffle       = shuffle
        self._drop_last     = drop_last
        if seed is None:
            self._seed      = random.randint(0, 2**64)
        else:
            self._seed      = seed
                
    def __iter__(self):
        """ this function is responsible for returning an iterator. The yield statement is kind of like a return, but instead it just advances the loop further every time the next() method is called, or in a for loop. it kind of remebers the last time it was called """

        if self._shuffle:
            rng = random.Random(self._seed + self._current_epoch)
            rng.shuffle(self._indices) 
            
        batch_size = self._get_current_batch_size()
        
        # note this is only the starting "meta-index" to find the sequence of indices in self.indices for one batch
        for start_idx in range(0, self._num_samples, batch_size):
            batch = self._indices[start_idx:start_idx + batch_size]
            if len(batch) < batch_size and self._drop_last:
                continue # optionally skip the last partial batch
            yield batch

    def __len__(self):
        """ just returns the number of batches in the current epoch. changes by 1 depending on whether the last semi-full one is dropped or not. """
        
        batch_size = self._get_current_batch_size()
        if self._drop_last:
            return math.floor(self._num_samples / batch_size)
        else:
            return math.ceil(self._num_samples / batch_size)
    
    def _get_current_batch_size(self) -> int:
        """ convenience method for getting the batch size for the current epoch. also implements safeguarding against usage outside of the defined schedule range """

        if self._current_epoch < self.get_total_epochs():
            return self._bs_schedule[self._current_epoch]
        else:
            return self._bs_schedule[-1]
     
    def step(self):
        """ analog to optimizer step, this increments the epoch by one, for usage in training loop """
        self._current_epoch += 1
        
        if self._current_epoch > self.get_total_epochs():
            print(f"BS SCHEDULER WARNING: current epoch now outside of range covered by schedule! repeating last bs...")
   
    def set_epoch(self, epoch: int):
        """ update the current epoch explicitly when called in the training loop, just for debugging / customization """
        
        self._current_epoch = epoch
        
    def get_total_epochs(self) -> int:
        """ convenience method for knowing for how many epochs to loop the main training procedure """
        
        return len(self._bs_schedule)
    
    def get_total_steps(self):
        """ returns the total number of batches that are yielded by sampler. takes drop_last into account. """

        bs_schedule_arr = np.array(self._bs_schedule)
        
        if self._drop_last is True:
            total_steps = np.sum(np.floor(self._num_samples / bs_schedule_arr))
        else:
            total_steps = np.sum(np.ceil(self._num_samples / bs_schedule_arr))

        return int(total_steps)
    
    def get_total_samples(self) -> int:
        """ returns total number of samples that are returned by the sampler. mainly takes drop_last into account. """
        
        bs_schedule_arr = np.array(self._bs_schedule)
        
        if self._drop_last is True:
            total_samples = np.sum(np.floor(self._num_samples / bs_schedule_arr) * bs_schedule_arr)
        else:
            total_samples = self.get_total_epochs() * self._num_samples
            
        return int(total_samples)
    
        
        
# SOURCES ==============================================================================================================

# [https://arxiv.org/abs/1711.00489]

# ======================================================================================================================