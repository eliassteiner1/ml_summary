import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops as eo

    
class SingleHeadAttentionKernel(nn.Module):
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

class MultiHeadAttentionKernel(nn.Module):
    """ just the dot product fc graph attention operation in isolation. can be used for all other attention layers """
    
    def __init__(self, d_residual: int, d_k: int, d_v: int=None, n_heads: int=2, att_drop: float=0.1):

        super().__init__()
        
        if d_v is None:
            d_v = d_k # this is the standard, but technically not constrained
        self.softmax_temp = d_k**0.5
        
        if d_k % n_heads != 0:
            raise ValueError()
        if d_v % n_heads != 0:
            raise ValueError()
        
        self.d_k     = d_k
        self.d_v     = d_v
        self.n_heads = n_heads
        self.d_kh    = int(d_k / n_heads)
        self.d_vh    = int(d_v / n_heads)
        
        self.w_q = nn.Linear(d_residual, d_k)
        self.w_k = nn.Linear(d_residual, d_k)
        self.w_v = nn.Linear(d_residual, d_v)
        self.w_o = nn.Linear(d_v, d_residual)
         
        self.dropout = nn.Dropout(att_drop)
        self.softmax = nn.Softmax(dim=3) # softmax is over rows of attention weights! #TODO: watch out for dim!
        
    def forward(self, X: torch.Tensor, MSK: torch.Tensor=None) -> torch.Tensor:
        # MSK broadcastable to [Batch, N_head, seq_len, seq_len]. likely this will be either 
        # [1, 1, seq_len, seq_len] or [Batch, 1, seq_len, seq_len] but different mask for heads never makes sense..

        Q = self.w_q(X)
        K = self.w_k(X)
        V = self.w_v(X)
        
        Q = eo.rearrange(Q, "B s (H d_kh) -> B H s d_kh", H=self.n_heads, d_kh=self.d_kh)
        K = eo.rearrange(K, "B s (H d_kh) -> B H s d_kh", H=self.n_heads, d_kh=self.d_kh)
        V = eo.rearrange(V, "B s (H d_vh) -> B H s d_vh", H=self.n_heads, d_vh=self.d_vh)
        
        ATTW = torch.einsum("BHid, BHjd -> BHij", [Q, K]) 
        ATTW = ATTW / self.softmax_temp
        if MSK is not None:
            ATTW = ATTW.masked_fill(MSK.bool(), float("-inf"))
        ATTW = self.softmax(ATTW)
        ATTW = self.dropout(ATTW)
        CTXT = torch.einsum("BHij, BHjk -> BHik", [ATTW, V])
        CTXT = eo.rearrange(CTXT, "B H s d_vh -> B s (H d_vh)")
        
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
        
        self.sdp_attention  = SingleHeadAttentionKernel(d_residual, d_k, d_v=d_v, att_drop=att_drop)
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


    
    

if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    



    
    

