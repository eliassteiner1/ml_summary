import torch
import torch.nn as nn 
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, input_feats_dim, latent_dim):
        """
        input_feats are the main data. latent_dim can be chosen.
        """
        super(SelfAttention, self).__init__()
        
        # define dimensions
        self.input_feats_dim = input_feats_dim
        self.latent_dim      = latent_dim
        
        # define attention weights (= matrices Wq, Wk, Wv)
        self.query = nn.Linear(self.input_feats_dim, self.latent_dim)
        self.key   = nn.Linear(self.input_feats_dim, self.latent_dim)
        self.value = nn.Linear(self.input_feats_dim, self.latent_dim)
        
        # define last linear reshaping layer (= matrix Wout)
        self.linear_out = nn.Linear(self.latent_dim, self.input_feats_dim)
        
        # softmax (row-wise!)
        self.softmax = nn.Softmax(dim=2) 
        
    def forward(self, x):
        """
        inputs flattened to respect:   [BS (arbitrary)   | ? (arbitrary) | feats (known, Architecture defining)]
        """

        # calculate the embedded vectors in latent space (= matrices Q, K, V)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # calculate the attention matrix -> normalize by latent dimension -> apply softmax -> and weigh it with V
        SCO = torch.bmm(Q, K.transpose(1, 2)) / (self.latent_dim**0.5)
        ATT = self.softmax(SCO)
        WGH = torch.bmm(ATT, V)
        
        # apply last linear layer to reshape output
        x = self.linear_out(WGH)
        
        return x   

class CrossAttention(nn.Module):
    def __init__(self, input_feats_dim, input_class_dim, latent_dim):
        """
        input_feats are the main data & input_class is the regulating input. latent_dim can be chosen.
        """
        super(CrossAttention, self).__init__()
        
        # define dimensions
        self.input_feats_dim = input_feats_dim
        self.input_class_dim = input_class_dim
        self.latent_dim      = latent_dim
        
        # define attention weights (= matrices Wq, Wk, Wv)
        self.query = nn.Linear(self.input_feats_dim, self.latent_dim)
        self.key   = nn.Linear(self.input_class_dim, self.latent_dim)
        self.value = nn.Linear(self.input_class_dim, self.latent_dim)
        
        # define last linear reshaping layer (= matrix Wout)
        self.linear_out = nn.Linear(self.latent_dim, self.input_feats_dim)
        
        # softmax (row-wise!)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x, class_vector):
        """
        inputs flattened to respect:   [BS (arbitrary)   | ? (arbitrary) | channels (known, Architecture defining)]
        
        [input_x     ] has Dimensions: [BS=1 (arbitrary) | V (arbitrary) | n_feats=128 (defines Architecture)     ]
        [class_vector] has Dimensions: [BS=1 (arbitrary) | 1 (arbitrary) | nr_classes  (defines Architecture)     ]
        
        note: class vector should be unsqueezed to match the nr of dimensions of x (to make sure torch.bmm works)
        """

        # calculate the embedded vectors in latent space (= matrices Q, K, V)
        Q = self.query(x)
        K = self.key(class_vector)
        V = self.value(class_vector)

        # calculate the attention matrix -> normalize by latent dimension -> apply softmax -> and weigh it with V
        SCO = torch.bmm(Q, K.transpose(1, 2)) / (self.latent_dim**0.5)
        ATT = self.softmax(SCO)
        WGH = torch.bmm(ATT, V)
        
        # apply last linear layer to reshape output
        x = self.linear_out(WGH)
        
        return x

class SelfAttentionMultihead(nn.Module):
    def __init__(self, input_feats_dim, latent_dim, n_heads):
        """
        input_feats are the main data. latent_dim can be chosen. n_heads can be chosen. latent_dim will be distributed among all the n attention heads
        """
        super(SelfAttentionMultihead, self).__init__()
        
        # check that the latent_dim is actually divisible by n_heads
        assert latent_dim % n_heads == 0, "Choose the latent_dim as an integer multiple of n_heads"
        
        # define total dimensions 
        self.input_feats_dim = input_feats_dim
        self.latent_dim      = latent_dim
        self.n_heads         = n_heads
        
        # automatically determine latent dimensions of one head. (technically d_k and d_v don't have to be equal)
        self.d_k = latent_dim // n_heads
        self.d_v = latent_dim // n_heads
        
        # define attention weights (Query: Wq, Key: Wk, Value: Wv)
        self.W_Q = nn.Linear(self.input_feats_dim, self.n_heads * self.d_k)
        self.W_K = nn.Linear(self.input_feats_dim, self.n_heads * self.d_k)
        self.W_V = nn.Linear(self.input_feats_dim, self.n_heads * self.d_v)
               
        # define last linear reshaping layer (Wout)
        self.W_out = nn.Linear(self.latent_dim, self.input_feats_dim)
        
        # softmax (row-wise!)
        self.softmax = nn.Softmax(dim=3) 
        
    def forward(self, x):
        # Input Shape: [BS=1 V(=arbitrary), input_feats=128]
        
        # store original batch size (for reshaping)
        batch_size = x.shape[0]
        
        # calculate the embedded vectors in latent space
        # note: just using one big W matrix and reshaping is equivalent to using separate small W matrices for each head
        Q = self.W_Q(x).view(batch_size, -1, self.n_heads, self.d_k)
        K = self.W_K(x).view(batch_size, -1, self.n_heads, self.d_k)
        V = self.W_V(x).view(batch_size, -1, self.n_heads, self.d_v)
        # [BS=1, V, nr_heads, d_k (= latent per head)]
        
        # permute dimensions. For torch.matmul dotproduct, exactly the last two dimensions are considered!
        Q = Q.permute(dims=(0, 2, 1, 3))
        K = K.permute(dims=(0, 2, 1, 3))
        V = V.permute(dims=(0, 2, 1, 3))
        # [BS=1, nr_heads, V, d_k or d_v (= latent per head)]

        # apply dot product attention.
        # note: every operation is now carried out over all n_head dimensions too -> parallel heads!
        SCO = torch.matmul(Q, K.transpose(2, 3)) / (self.d_k**0.5)
        ATT = self.softmax(SCO)
        # [BS=1, nr_heads, V, V]
        WGH = torch.matmul(ATT, V)
        # [BS=1, nr_heads, V, d_v]

        # concatenate all the individual WGH matrices (along n_head dimension) (= stacking along latent dimension)
        WGH_list = [WGH[:, i, :, :] for i in range(self.n_heads)]
        # list of [BS=1, V, d_v]
        WGH_cat  = torch.cat(WGH_list, dim=2)
        # [BS=1, V, d_v*n_heads (= latent_dim)]
        
        # apply the final W_out matrix to "reshape" the output to match inputs_feats_dim
        x = self.W_out(WGH_cat)
        # [BS=1, V, input_feats=128]
        
        return x   

class CrossAttentionMultihead(nn.Module):
    def __init__(self, input_x_dim, input_y_dim, latent_dim, n_heads):
        """
        input_x is the main data. input_y is the "regulating" additional input -> "x attends to y"
        latent_dim can be chosen. n_heads can be chosen. latent_dim will be distributed among all the n attention heads.
        """
        super(CrossAttentionMultihead, self).__init__()
        
        # check that the latent_dim is actually divisible by n_heads
        assert latent_dim % n_heads == 0, "Choose the latent_dim as an integer multiple of n_heads"
        
        # define absolute model dimensions 
        self.input_x_dim = input_x_dim
        self.input_y_dim = input_y_dim
        self.latent_dim  = latent_dim
        self.n_heads     = n_heads
        
        # automatically determine latent dimensions of ONE head. (technically d_k and d_v don't HAVE to be equal)
        self.d_k = latent_dim // n_heads
        self.d_v = latent_dim // n_heads
        
        # define attention weights (Query: Wq, Key: Wk, Value: Wv)
        self.W_Q = nn.Linear(self.input_x_dim, self.n_heads * self.d_k)
        self.W_K = nn.Linear(self.input_y_dim, self.n_heads * self.d_k)
        self.W_V = nn.Linear(self.input_y_dim, self.n_heads * self.d_v)
        
        # softmax (row-wise!)
        self.softmax = nn.Softmax(dim=3)   
            
        # define last linear reshaping layer (Wout)
        self.W_out = nn.Linear(self.latent_dim, self.input_x_dim)
        
    # concatenation of all individual heads along latent dimension (d_v) 
    # using list comprehension - more clear and understandable approach
    def concatenate1(self, WGH):
        WGH_list = [WGH[:, i, :, :] for i in range(self.n_heads)]
        # --> list of [BS=1, V, d_v]
        WGH_cat  = torch.cat(WGH_list, dim=2)
        # --> [BS=1, V, d_k*n_heads (= latent_dim)]
        return WGH_cat
    
    # concatenation of all individual heads along latent dimension (d_v)  
    # using .view() is more efficient but very unclear to understand what's going on  
    def concatenate2(self, WGH):
        # move the dimension along which is concatenated (d_v latent features) to the last place
        # move the dimension which is eliminated by concatenation (n_heads) to the second last place
        # use contiguous to ensure continuous memory for the tensor after permutation
        WGH = WGH.permute(dims=(0, 2, 1, 3)).contiguous()
        # --> [BS=1, V, n_heads, d_v]
        WGH_cat = WGH.view(self.batch_size, -1, self.n_heads*self.d_v)
        # --> [BS=1, V, d_v*n_heads (= latent_dim)]
        return WGH_cat

        
    def forward(self, x, y):
        assert len(x.shape) == 3, "x: Expecting Tensor with 3 Dimensions: [BS (arbitrary), V (arbitrary), input_x_dim]"
        assert len(y.shape) == 3, "y: Expecting Tensor with 3 Dimensions: [BS (arbitrary), 1 (arbitrary), input_y_dim]"
        assert x.shape[2] == self.input_x_dim, f"last dimension of x (got {x.shape[2]}) must match input_x_dim"
        assert y.shape[2] == self.input_y_dim, f"last dimension of y (got {x.shape[2]}) must match input_y_dim"
        # --> Input Shape x: [BS=1, V (= arbitrary), input_feats=128]
        # --> input Shape y: [BS=1, 1 (= arbitrary), nr_classes     ]
        
        # store original batch size (for reshaping)
        self.batch_size = x.shape[0]
        
        # calculate the embedded vectors in latent space
        # note: just using one big W matrix and reshaping is equivalent to using separate small W matrices for each head
        Q = self.W_Q(x).view(self.batch_size, -1, self.n_heads, self.d_k)
        K = self.W_K(y).view(self.batch_size, -1, self.n_heads, self.d_k)
        V = self.W_V(y).view(self.batch_size, -1, self.n_heads, self.d_v)
        # --> Q:    [BS=1, V, nr_heads, d_k (= latent per head)       ]
        # --> K, V: [BS=1, 1, nr_heads, d_k or d_v (= latent per head)]
        
        # permute dimensions. For torch.matmul dotproduct, exactly the last two dimensions are considered!
        Q = Q.permute(dims=(0, 2, 1, 3))
        K = K.permute(dims=(0, 2, 1, 3))
        V = V.permute(dims=(0, 2, 1, 3))
        # --> Q:    [BS=1, nr_heads, V, d_k (= latent per head)       ]
        # --> K, V: [BS=1, nr_heads, 1, d_k or d_v (= latent per head)]

        # apply dot product attention.
        # note: every operation is now carried out over all n_head dimensions too -> parallel heads!
        SCO = torch.matmul(Q, K.transpose(2, 3)) / (self.d_k**0.5)
        ATT = self.softmax(SCO)
        # --> [BS=1, nr_heads, V, 1]
        WGH = torch.matmul(ATT, V)
        # --> [BS=1, nr_heads, V, d_v]

        # concatenate all the individual WGH matrices (along n_head dimension) (= stacking along latent dimension)
        blah    = self.concatenate1(WGH)
        WGH_cat = self.concatenate2(WGH)
        # --> [BS=1, V, d_v*n_heads (= latent_dim)]
        
        # apply the final W_out matrix to "reshape" the output to match inputs_feats_dim
        x = self.W_out(WGH_cat)
        # --> [BS=1, V, input_feats=128]
        
        return x

    
if __name__ == "__main__":
    model = CrossAttentionMultihead(input_x_dim=128, input_y_dim=5, latent_dim=64, n_heads=4)
    torchmodel = nn.MultiheadAttention(128, 4)

    input_x = torch.rand(1, 6600, 128)
    input_y = torch.rand(1, 1, 5)


    output = model(input_x, input_y)

