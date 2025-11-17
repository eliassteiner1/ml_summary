import torch
import torch.nn as nn


class MyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.linear1 = nn.Linear(10, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, 10)
        
        self.activation = nn.ReLU()
        
    def forward(self, x):
        
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        
        return x

class MyVariableMLP(nn.Module):
    def __init__(self, layer_sizes, activation=nn.ReLU, final_activation=None):
        super().__init__()

        layers = []
        for i in range(len(layer_sizes) - 1):
            in_dim  = layer_sizes[i]
            out_dim = layer_sizes[i+1]
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(layer_sizes) - 2:
                layers.append(activation())
            elif final_activation is not None:
                layers.append(final_activation())
                
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)
            

if __name__ == "__main__":        
    # dummy inference on the normal mlp
    BS = 1
    N  = 7        
    sample1 = torch.rand(10, )  
    sample2 = torch.rand(1, 10) 
    sample3 = torch.rand(BS, 10)  
    sample4 = torch.rand(N, BS, 10)
    net = MyMLP()
    out = net(sample4)
    
    # dummy inference on the variable layer net
    layer_sizes = [10, 32, 32, 32, 10]
    net = MyVariableMLP(layer_sizes, activation=nn.ReLU, final_activation=None)
    BS = 1
    N  = 7        
    sample1 = torch.rand(10, )
    sample2 = torch.rand(1, 10) 
    sample3 = torch.rand(BS, 10)  
    sample4 = torch.rand(N, BS, 10)