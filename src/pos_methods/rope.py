import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from rotary_embedding_torch import RotaryEmbedding


class ROPE(nn.Module):

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
        self.rotary_emb = RotaryEmbedding(dim = 64)
        # self.register_buffer(
        #     "mask", torch.tril(torch.ones(1, 1, config.block_size, config.block_size).cuda()), persistent=False
        # )

    def forward(self, x, attention_mask=None, return_attentions=False):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        ################## applying rotary PE ###################
        k = self.rotary_emb.rotate_queries_or_keys(k)
        q = self.rotary_emb.rotate_queries_or_keys(q)
        #########################################################

        d_k = q.size(-1)  # Embedding dimension
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        # Apply mask
        if attention_mask is not None:  # only for Bert models
            attention_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -10000.0
            scores_masked = scores + attention_mask
        else:  # only for GPT models
            causal_mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
            scores_masked = scores + causal_mask
        # mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        # scores_masked = scores.masked_fill(mask[:, :, :T, :T] == 0, float("-inf"))

        # Softmax normalization
        attention_weights = torch.softmax(scores_masked, dim=-1)

        # Weighted sum of values
        y = torch.matmul(attention_weights, v)
        ##############################################################################

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)

        # if return_attentions:
        #     mask = torch.tril(torch.ones(T, T), diagonal=0).to(x.device)
        #     scores = scores * mask
        #     return (y, scores)

        return y