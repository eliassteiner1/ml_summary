import torch
import torch.nn as nn

class TestNet(nn.Module):
    def __init__(self, init_mode: str="none"):
        super().__init__()
        
        SZ = 25
        self.linear_1x1 = nn.Linear( 1, SZ)
        self.linear_1x2 = nn.Linear(SZ, SZ)
        self.linear_1x3 = nn.Linear(SZ, SZ)
        self.linear_1x4 = nn.Linear(SZ,  1)
        
        self.activation = nn.ReLU()
        
        
        self._initialize_weights(mode=init_mode)
        return
        
    def _initialize_weights(self, mode: str):
        if mode == "none":
            return
        if mode not in ["xavier", "kaiming", "orthogonal"]:
            raise ValueError(f"please choose a valid method from [xavier, kaiming, orthogonal]! (got {mode})")
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if mode == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
                if mode == "kaiming":
                    nn.init.kaiming_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
                if mode == "orthogonal":
                    nn.init.orthogonal_(m.weight)
                    nn.init.zeros_(m.bias)
        return
    
    def forward(self, sample: dict):
        
        x = sample["input"][:, None].float()
        
        x = self.activation(self.linear_1x1(x))
        x = self.activation(self.linear_1x2(x))
        x = self.activation(self.linear_1x3(x))
        x = self.linear_1x4(x)
        
        return x
