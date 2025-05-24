import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class BASE_ATTENTION(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    # TODO: implement return_attentions
    # TODO: how does it work out for Bert ??? update: I removed flash attention and do the work manually!
    def forward(self, x, attention_mask=None, return_attentions=False):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        # y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # y = self.c_proj(y)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(head_size, dtype=torch.float32))  # (B,nh,T,T)
        
        # Apply mask
        if attention_mask is not None:  # only for Bert models
            attention_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -10000.0
            scores_masked = attn_scores + attention_mask
        else:  # only for GPT models
            causal_mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
            scores_masked = attn_scores + causal_mask

        attn_probs = F.softmax(scores_masked, dim=-1)
        y = torch.matmul(attn_probs, v)  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        # if return_attentions:
        #     mask = torch.tril(torch.ones(T, T), diagonal=0).to(x.device)
        #     scores = scores * mask
        #     return (y, scores)
        
        return y
