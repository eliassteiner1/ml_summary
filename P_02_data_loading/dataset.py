import torch
from   torch.utils.data import Dataset

class MyDataset(Dataset):
    """ a minimal dataset class needs at least the three functions __init__, __len__ and __getitem__. This is, in it's simplest form, basically just a lean wrapper for your data, that ensures the one-by-one retrieval of samples. """
    
    def __init__(self, data):
        """ just the constructor for the dataset. Initialize the directories to data / data itself, transforms, etc """
        self.data = data

    def __len__(self):
        """ very standard, just needs to return the total length of the dataset """
        return len(self.data)

    def __getitem__(self, idx):
        """ core method: loads and returns one sample given and index, can grab it directly from data if it's already loaded in memory, or load it from the disk if path is specified. Sample can be manipulated and transformed """
        
        return self.data[idx] 
    
    # TODO: Explore Transforms