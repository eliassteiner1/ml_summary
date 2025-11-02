import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, input_feats_dim, latent_dim, output_feats_dim) -> None:
        """
        input_feats are the main data. latent_dim can be chosen.
        -> note, needs skip connection on the outside!
        -> note, this is modified from standard. Normally input_feats_dim = output_feats_dim!
        """
        super().__init__()
        
        # define dimensions
        self.input_feats_dim  = input_feats_dim
        self.latent_dim       = latent_dim
        self.output_feats_dim = output_feats_dim
        
        # define attention weights (= matrices Wq, Wk, Wv)
        self.query = nn.Linear(self.input_feats_dim, self.latent_dim)
        self.key   = nn.Linear(self.input_feats_dim, self.latent_dim)
        self.value = nn.Linear(self.input_feats_dim, self.latent_dim)
        
        # define last linear reshaping layer (= matrix Wout)
        self.linear_out = nn.Linear(self.latent_dim, self.output_feats_dim)
        
        # softmax (row-wise!)
        self.softmax = nn.Softmax(dim=2) 
        
        
    def forward(self, x, msk=None):
        """
        input x  : [BS (var, broadcastable) | input length (var) | feats (known, Architecture defining)]
        input msk: [BS (var, broadcastable) | input length (var) ]
        """

        # fallback for when no mask is provided -> [BS (var, broadcastable) | 1 | feats (known, architecture def.)]
        # (so the maskvector is a lying row vector times batch size)
        if msk is None:
            msk = torch.zeros(x.shape[0], x.shape[1]).to(DEVICE)
        
        # calculate the embedded vectors in latent space (= matrices Q, K, V)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # calculate the attention matrix -> normalize by latent dimension -> apply softmax -> and weigh it with V
        # mask has [BS, maskvec]. needs unsqueeze(1) so that it is a [BS, 1, maskvec] "lying" row vector -> broadcasted
        SCO = torch.bmm(Q, K.transpose(1, 2)) / (self.latent_dim**0.5)
        SCO = SCO.masked_fill(msk.unsqueeze(1) == 1, float("-inf"))
        ATT = self.softmax(SCO)
        WGH = torch.bmm(ATT, V)
        
        # apply last linear layer to reshape output
        x = self.linear_out(WGH)
        
        return x 


if __name__ == "__main__":
    print("done")