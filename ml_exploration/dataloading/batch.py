import torch

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