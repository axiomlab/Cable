import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class T5Bias(nn.Module):

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
        self.block_size = config.block_size
        # self.causal = config.causal
        # self.dtype = torch.bfloat16
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.register_buffer(
        #     "mask", torch.tril(torch.ones(1, 1, config.block_size, config.block_size).cuda()), persistent=False
        # )

        ################# T5 bias implementation ######################
        self.max_distance = 256
        self.t5_position = nn.Embedding(2 * self.max_distance + 1, self.n_head)

    # TODO: implement return_attentions
    # TODO: make compatible with Bert models (attention mask) (I'm not sure about the current version)
    def forward(self, x, attention_mask=None, return_attentions=False):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        ###################################################################################################
        # position_ids = torch.arange(self.block_size, dtype=torch.long, device=self.device)
        position_ids = torch.arange(T, dtype=torch.long, device=self.device)
        relative_positions = position_ids.view(-1, 1) - position_ids.view(1, -1)

        # Clamp values to max_distance range
        relative_positions = torch.clamp(relative_positions, min=-self.max_distance, max=self.max_distance)
        relative_positions = relative_positions + self.max_distance  # Shift to make all values positive

        # Fetch the bias values from the embedding table
        bias = self.t5_position(relative_positions).permute(2, 0, 1).unsqueeze(0)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(C, dtype=torch.float32)) + bias[:, :, :T, :T]

        # Apply mask
        if attention_mask is not None:  # only for Bert models
            attention_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -10000.0
            scores_masked = scores + attention_mask
        else:  # only for GPT models
            causal_mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
            scores_masked = scores + causal_mask

        attention_weights = torch.softmax(scores_masked, dim=-1)
        y = torch.matmul(attention_weights, v)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)

        # if return_attentions:
        #     mask = torch.tril(torch.ones(T, T), diagonal=0).to(x.device)
        #     scores = scores * mask
        #     return (y, scores)

        return y