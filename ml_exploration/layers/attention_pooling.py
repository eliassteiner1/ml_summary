import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    def __init__(self, input_feats_dim, hidden_layers=[128]) -> None:
        """
        Implements the attention pooling mechanism. A simple mlp learns one weight per input sequence. The output is then a linear combination of all the weights and inputs.
        """
        super().__init__()
        
        self.activation = nn.ReLU()
        self.softmax    = nn.Softmax(dim=1)
        
        # set up the sequential mlp to learn the per-input weights [BS | input length | 1]
        layers        = []
        current_feats = input_feats_dim
        for el in hidden_layers:
            layers.append(nn.Linear(current_feats, el))
            layers.append(self.activation)
            current_feats = el
        layers.append(nn.Linear(current_feats, 1))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x, msk=None):
        """
        input x:   [BS (var, broadcastable) | input length (var) | n feats (known, architecture defining)]  
        input msk: [BS (var, broadcastable) | input length (var) ]
        """
        
        # fallback for when no mask is provided -> [BS (var, broadcastable) | 1 | feats (known, architecture def.)]
        # (so the maskvector is a lying row vector times batch size)
        if msk is None:
            msk = torch.zeros(x.shape[0], x.shape[1]).to(DEVICE)
        
        # calculate the weights by learning them with an mlp    
        d = self.mlp(x)
        d = d.masked_fill(msk.unsqueeze(2) == 1, float("-inf"))
        d = self.softmax(d)
        
        # recombine the outputs from a linear combination of weights and inputs
        x = torch.bmm(d.permute(0, 2, 1), x)
        
        return x