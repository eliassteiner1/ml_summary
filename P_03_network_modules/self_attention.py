import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops as eo

    
class SDPAttentionKernel(nn.Module):
    """ just the dot product fc graph attention operation in isolation. can be used for all other attention layers """
    
    def __init__(self, d_residual: int, d_k: int, d_v: int=None, att_drop: float=0.1):
        """
        Args:
            d_residual (int): residual stream embedding dimension
            d_k (int):        query and key embedding dimension
            d_v (int):        value embedding dimension (typically = d_k)
            att_drop (float): dropout rate for attention weights
        """
        super().__init__()
        
        if d_v is None:
            d_v = d_k # this is the standard, but technically not constrained
        self.softmax_temp = d_k**0.5
        
        self.w_q = nn.Linear(d_residual, d_k)
        self.w_k = nn.Linear(d_residual, d_k)
        self.w_v = nn.Linear(d_residual, d_v)
        self.w_o = nn.Linear(d_v, d_residual)
         
        self.dropout = nn.Dropout(att_drop)
        self.softmax = nn.Softmax(dim=2) # softmax is over rows of attention weights!
        
    def forward(self, X: torch.Tensor, MSK: torch.Tensor=None) -> torch.Tensor:
        """
        Args:
            X (torch.Tensor):   input residual stream, `dims[batch, seq_len, d_residual]`
            MSK (torch.Tensor): optionsal bool mask, broadcastable to `dims[batch, seq_len, seq_len]`
        Returns:
            torch.Tensor: output residual stream, `dims[batch, seq_len, d_residual]`
        """

        Q = self.w_q(X)
        K = self.w_k(X)
        V = self.w_v(X)
        
        ATTW = torch.einsum("Bij, Bkj -> Bik", [Q, K]) # matmul is faster! but syntax is worse
        ATTW = ATTW / self.softmax_temp
        if MSK is not None:
            ATTW = ATTW.masked_fill(MSK.bool(), float("-inf"))
        ATTW = self.softmax(ATTW)
        ATTW = self.dropout(ATTW)
        CTXT = torch.einsum("Bij, Bjk -> Bik", [ATTW, V]) # matmul is faster! but syntax is worse  
        
        X = self.w_o(CTXT)

        return X

class MLP(nn.Module):
    " the fully connected layer that follows dot product attention. MLP is applied elementwise"
    
    def __init__(self, d_residual: int, d_hidden: int, mlp_drop: float=0.1):
        """
        Args:
            d_residual (int):           residual stream embedding dimension
            d_hidden (int):             dimension of the intermediate mlp layer, typically 4*d_residual
            mlp_drop (float, optional): dropout rate for the first mlp layer
        """
        super().__init__()
        
        self.fc_up = nn.Linear(d_residual, d_hidden)
        self.fc_dw = nn.Linear(d_hidden, d_residual)
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(mlp_drop)
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X (torch.Tensor): input residual stream, `dims[batch, seq_len, d_residual]`
        Returns:
            torch.Tensor: otuput residual stream, `dims[batch, seq_len, d_residual]`
        """
        
        X = self.fc_up(X)
        X = self.activation(X)
        X = self.dropout(X)
        X = self.fc_dw(X)
        
        return X

class LayerNorm(nn.Module):
    """ layer norm acts independent on each element of the sequence (of each batch) """
    
    def __init__(self, d_residual: int):
        """
        Args:
            d_residual (int): residual stream embedding dimension
        """
        super().__init__()
        
        self.eps   = 1e-8
        
        self.gamma = nn.Parameter(torch.ones([d_residual]))
        self.beta  = nn.Parameter(torch.zeros([d_residual]))
    

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X (torch.Tensor): input residual stream, `dims[batch, seq_len, d_residual]`
        Returns:
            torch.Tensor: output normalized residual stream, `dims[batch, seq_len, d_residual]`
        """
        
        X = X - eo.reduce(X, "B s e -> B s 1", "mean") # shift by mean
        X = X / (eo.reduce(X**2, "B s e -> B s 1", "mean") + self.eps)**0.5 # scale by variance
        X = X * self.gamma[None, None, :] + self.beta[None, None, :]
        
        return X
        
class AttentionBlock(nn.Module):
    def __init__(self, d_residual, d_k, d_v=None, att_drop=0.1, mlp_drop=0.1, res_drop=0.1) -> None:
        """
        Args:
            d_residual (_type_): residual stream embedding dimension 
            d_k (_type_):        query and key embedding dimension
            d_v (_type_):        value embedding dimension (typically = d_k)
            att_drop (float):    dropout rate for attention weights
            mlp_drop (float):    dropout rate for the first mlp layer
            res_drop (float):    dropout rate for residual connections
        """
        super().__init__()
        
        self.sdp_attention  = SDPAttentionKernel(d_residual, d_k, d_v=d_v, att_drop=att_drop)
        self.layer_norm_att = LayerNorm(d_residual)
        self.layer_norm_mlp = LayerNorm(d_residual)
        self.ff_mlp         = MLP(d_residual, 4*d_residual, mlp_drop=mlp_drop)
        self.dropout        = nn.Dropout(res_drop)
    
    def forward(self, RES: torch.Tensor, MSK: torch.Tensor=None) -> torch.Tensor:
        """ 
        Args:
            RES (torch.Tensor): input residual stream, `dims[batch, seq_len, d_residual]`
            MSK (torch.Tensor): optionsal bool mask, broadcastable to `dims[batch, seq_len, seq_len]`

        Returns:
            torch.Tensor: output residual stream, `dims[batch, seq_len, d_residual]`
        """
        
        X = self.layer_norm_att(RES)
        X = self.sdp_attention(X, MSK=MSK)
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
    
    sm = nn.Softmax(dim=2)
    A = torch.rand(2, 5, 5)
    A = sm(A)
    
    print(A)
    
    print(A.sum(dim=2))
    


