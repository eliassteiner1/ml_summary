import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionVanilla(nn.Module):
    """ just as an overview, most basic vanilla attention block """
    
    def __init__(self, input_features, attention_dim, value_dim = None):
        super().__init__()
        
        if value_dim is None:
            value_dim = attention_dim # this is the standard, but technically not constrained
        
        self.input_features = input_features
        self.attention_dim = attention_dim
        self.value_dim = value_dim
        
        self.w_q = nn.Linear(input_features, attention_dim)
        self.w_k = nn.Linear(input_features, attention_dim)
        self.w_v = nn.Linear(input_features, value_dim)
        
        self.w_z = nn.Linear(value_dim, input_features)
        
        self.softmax = nn.Softmax(dim=2) # row wise!
        
    def forward(self, X):
        
        Q = self.w_q(X)
        K = self.w_k(X)
        V = self.w_v(X)
        
        ATTW = torch.einsum("Bij, Bkj -> Bik", [Q, K]) / (self.attention_dim**0.5)
        ATTW = self.softmax(ATTW)
        CTXT = torch.einsum("Bij, Bjk -> Bik", [ATTW, V])
        
        X = self.w_z(CTXT)
        
        return X