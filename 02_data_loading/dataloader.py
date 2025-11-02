from torch.utils.data import DataLoader

""" combines dataset and sampler and actually returns an iterable dataset with shuffling, batching and memory handling. Lot of low-level memory handling, so I guess it's usually not necessary to touch this. watch out, num_workers > 0 has issues with jupyter notebooks! put this line into a .py file and import it """

def my_jupyter_dataloader(*args, **kwargs):
    return DataLoader(*args, **kwargs)