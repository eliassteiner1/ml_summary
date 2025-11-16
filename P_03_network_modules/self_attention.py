import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops as eo

    
class SDPAttentionKernel(nn.Module):
    # TODO: modify naming a bit for crossatttention
    """ just the dot product fc graph attention operation in isolation. can be used for all other attention layers """
    
    def __init__(self, softmax_temp: float = 1.0, att_drop: float = 0.1):
        
        """
        Args:
            softmax_temp (float, optional): scaling the attention weights before softmax
            att_dropout (float, optional): dropout after the softmax
        """
        super().__init__()
        
        self.softmax_temp = softmax_temp
        self.dropout      = nn.Dropout(att_drop)
        self.softmax      = nn.Softmax(dim = 2) # softmax is over rows of attention weights!
        
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, MSK: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            Q   (`torch.Tensor`): query matrix; `dim[Batch, seqLen, d_k]`
            K   (`torch.Tensor`): key   matrix; `dim[Batch, seqLen, d_k]`
            V   (`torch.Tensor`): value matrix; `dim[Batch, seqLen, d_v]`
            MSK (`torch.Tensor`): bool mask for the attention weights; broadcastable to `dim[Batch, seqLen, seqLen]`

        Returns:
            torch.Tensor: context vectors: `dim[Batch, seqLen, d_v]`
        """
        
        ATTW = torch.einsum("Bij, Bkj -> Bik", [Q, K]) # matmul is faster! but syntax is worse
        ATTW = ATTW / self.softmax_temp
        if MSK is not None:
            ATTW = ATTW.masked_fill(MSK.bool(), float("-inf"))
        ATTW = self.softmax(ATTW)
        ATTW = self.dropout(ATTW)
        CTXT = torch.einsum("Bij, Bjk -> Bik", [ATTW, V]) # matmul is faster! but syntax is worse  

        return CTXT

class MLP(nn.Module):
    " the fully connected layer that follows dot product attention"
    
    def __init__(self, input_features: int, hidden_features: int, mlp_drop: float = 0.1):
        
        super().__init__()
        
        self.fc_up = nn.Linear(input_features, hidden_features)
        self.fc_dw = nn.Linear(hidden_features, input_features)
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(mlp_drop)
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        
        X = self.fc_up(X)
        X = self.activation(X)
        X = self.dropout(X)
        X = self.fc_dw(X)
        
        return X

class LayerNorm(nn.Module):
    """ layer norm acts independent on each element of the sequence (of each batch) """
    
    def __init__(self, input_features: int):
        super().__init__()
        
        self.eps   = 1e-8
        
        self.gamma = nn.Parameter(torch.ones([input_features]))
        self.beta  = nn.Parameter(torch.zeros([input_features]))
    

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): residual stream input; `dim[Batch, seqLen, embeddFeats]`

        Returns:
            torch.Tensor: normalized residual stream; `dim[Batch, seqLen, embeddFeats]`
        """
        
        X = X - eo.reduce(X, "B s e -> B s 1", "mean") # shift by mean
        X = X / (eo.reduce(X**2, "B s e -> B s 1", "mean") + self.eps)**0.5 # scale by variance
        X = X * self.gamma[None, None, :] + self.beta[None, None, :]
        
        return X
        
class SelfAttention(nn.Module):
    def __init__(self, input_features, d_k, d_v = None, att_drop = 0.1, mlp_drop = 0.1, res_drop = 0.1) -> None:
        super().__init__()
        
        if d_v is None:
            d_v = d_k # this is the standard, but technically not constrained
            
        self.input_features = input_features
        self.d_k = d_k
        self.d_v = d_v
        
        self.w_q = nn.Linear(input_features, d_k)
        self.w_k = nn.Linear(input_features, d_k)
        self.w_v = nn.Linear(input_features, d_v)
        self.w_z = nn.Linear(d_v, input_features)
        
        self.sdp_attention  = SDPAttentionKernel(softmax_temp = d_k**0.5, att_drop = att_drop)
        self.layer_norm_att = LayerNorm(input_features)
        self.layer_norm_mlp = LayerNorm(input_features)
        self.ff_mlp         = MLP(input_features, 4*input_features, mlp_drop = mlp_drop)
        self.dropout        = nn.Dropout(res_drop)
    
    def forward(self, RES: torch.Tensor, MSK: torch.Tensor = None) -> torch.Tensor:
        
        X = self.layer_norm_att(RES)
        
        Q = self.w_q(X)
        K = self.w_k(X)
        V = self.w_v(X)
        CTXT = self.sdp_attention(Q, K, V, MSK = MSK)
        X = self.w_z(CTXT)
        
        X = self.dropout(X)
        RES = RES + X
        
        X = self.layer_norm_mlp(RES)
        X = self.ff_mlp(X)
        X = self.dropout(X)
        RES = RES + X
        
        return RES






class MultiHeadSelfAttention(nn.Module):
    pass



    
    

if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    
    q = torch.rand(2, 8, 6)
    k = torch.rand(2, 8, 6)
    v = torch.rand(2, 8, 4)
    
    mask = torch.eye(8)
    
    net = SDPAttentionKernel()
    out = net(q, k, v, mask)
    
    norm = LayerNorm(4)    


    


