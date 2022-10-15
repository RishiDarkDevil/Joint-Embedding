import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def attention(q, k, v, d_k, mask=None, dropout=None):
    """
    Calculate attention
    :input:
        q:          query
        k:          key
        v:          value
        d_k:        scaled term
        mask:       whether to use masking attention
        dropout:    dropout rate
    :output:
    """

    # Query, Key matrix multiplication
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    
    # If mask, use masking attetion
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Softmax for scaling in range [0,1]
    scores = F.softmax(scores, dim=-1)
    
    # Dropout
    if dropout is not None:
        scores = dropout(scores)

    # Score, Value matrix multiplication
    output = torch.matmul(scores, v)
    return output, scores


class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position, device='mps'):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)
        self.device = torch.device(device)

    def forward(self, length_q, length_k):
        # length_q and length_k are nothing but just sequence length and is same in this case
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).to(self.device)
        embeddings = self.embeddings_table[final_mat].to(self.device)

        return embeddings

class MultiHeadAttention(nn.Module):
    """
    Calculate multihead attention with num_heads
    :input:
        heads:          number of attention heads
        d_model:        embedding dim
        dropout:        dropout rate
    :output:
    """
    def __init__(self, heads, d_model, dropout = 0.1, max_relative_position = 3, device='mps'):
        super().__init__()
        
        self.d_model = d_model
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.h = heads

        self.max_relative_position = 3

        self.relative_position_k = RelativePosition(self.d_k, self.max_relative_position)
        self.relative_position_v = RelativePosition(self.d_k, self.max_relative_position)
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        self.scale = torch.sqrt(torch.FloatTensor([self.d_k])).to(device)

        # For visualization
        self.attn = None
        self.device = torch.device(device)

    def forward(self, q, k, v, mask = None):
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
        k = k.to(self.device)
        q = q.to(self.device)
        v = v.to(self.device)

        bs = q.size(0)
        len_k = k.shape[1]
        len_q = q.shape[1]
        len_v = v.shape[1]

        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)

        # Usual Attention
        q1 = q.view(bs, -1, self.h, self.d_k).permute(0, 2, 1, 3)
        k1 = k.view(bs, -1, self.h, self.d_k).permute(0, 2, 1, 3)
        attn1 = torch.matmul(q1, k1.permute(0, 1, 3, 2)) 

        # Relative Positional Encoding Compatibility Function additional term calculate
        r_k = self.relative_position_k(len_q, len_k)
        r_k = r_k.permute(0, 2, 1)
        q2 = q1.permute(2, 0, 1, 3).reshape(len_q, -1, self.d_k)

        attn2 = torch.matmul(q2, r_k).view(len_q, bs, self.h, len_q)
        attn2 = attn2.permute(1, 2, 0, 3)
        attn = (attn1 + attn2) / self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)

        attn = self.dropout(torch.softmax(attn, dim = -1))

        #attn = [batch size, n heads, query len, key len]
        v1 = v.view(bs, -1, self.h, self.d_k).permute(0, 2, 1, 3) 
        weight1 = torch.matmul(attn, v1) # Usual attn-weighted value sum

        # Relative Positional Encoding Representation additional term calculate
        r_v = self.relative_position_v(len_q, len_v)
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, bs*self.h, len_k)
        weight2 = torch.matmul(weight2, r_v)
        weight2 = weight2.transpose(0, 1).contiguous().view(bs, self.h, len_q, self.d_k)

        x = weight1 + weight2
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(bs, -1, self.d_model)
        
        #x = [batch size, query len, hid dim]
        x = self.out(x)
        
        #x = [batch size, query len, hid dim]
        
        return x

    
    # def forward(self, q, k, v, mask=None):
        
    #     bs = q.size(0)
        
    #     # perform linear operation and split into N heads
    #     k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
    #     q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
    #     v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
    #     # transpose to get dimensions bs * N * sl * d_model
    #     k = k.transpose(1,2)
    #     q = q.transpose(1,2)
    #     v = v.transpose(1,2)
        

    #     # calculate attention 
    #     scores, self.attn = attention(q, k, v, self.d_k, mask, self.dropout)
    #     # concatenate heads and put through final linear layer
    #     concat = scores.transpose(1,2).contiguous()\
    #     .view(bs, -1, self.d_model)
    #     output = self.out(concat)
    
    #     return output

