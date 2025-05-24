import torch
import torch.nn as nn
import torch.nn.functional as F
import math




class Kernel_CABLE6(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # biases projections for all heads, but in a batch
        self.cable_layer = nn.Linear(config.n_embd, config.n_head)
        self.cable_layer_scale = nn.Linear(config.n_embd, config.n_head)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.block_size = config.block_size
        self.n_embd = config.n_embd


    def forward(self, x, attention_mask=None, return_attentions=False):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # extracting query, key, and value vectors
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Relative cable biases for tokens on each head
        Sums = torch.cumsum(-F.relu(self.cable_layer(x)), dim=1).permute(0, 2, 1)  # (B,nh,T)

        # Apply mask
        if attention_mask is not None:  # only for Bert models
            cable_bias_old = -1.0 * torch.abs(Sums.unsqueeze(3) - Sums.unsqueeze(2))  # (B,nh,T,T)
            # Apply kernel
            cable_bias_old = -torch.log(cable_bias_old**2 + 1)
            bias_weights = F.softplus(self.cable_layer_scale(x)).permute(0, 2, 1)  # (B,nh,T)
            cable_bias = bias_weights.unsqueeze(-1) * cable_bias_old  # (B,nh,T,1) * (B,nh,T,T) -> (B,nh,T,T)
            attention_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -10000.0
            scores_masked = cable_bias + attention_mask
        else:  # only for GPT models
            cable_bias_old = (Sums.unsqueeze(3) - Sums.unsqueeze(2))  # (B,nh,T,T)
            # Apply kernel
            cable_bias_old = -torch.log(cable_bias_old**2 + 1)
            bias_weights = F.softplus(self.cable_layer_scale(x)).permute(0, 2, 1)  # (B,nh,T)
            cable_bias = bias_weights.unsqueeze(-1) * cable_bias_old  # (B,nh,T,1) * (B,nh,T,T) -> (B,nh,T,T)
            causal_mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
            scores_masked = cable_bias + causal_mask

        ###### Using torch.baddbmm for faster matrix mult with bias ########
        q_flat = q.reshape(B * self.n_head, T, head_size)  # (B*nh, T, hs)
        k_flat = k.reshape(B * self.n_head, T, head_size).transpose(-2, -1)  # (B*nh, hs, T)
        bias_flat = (scores_masked).reshape(B * self.n_head, T, T)  # (B*nh, T, T)
        scale = 1.0 / math.sqrt(head_size)
        att = torch.baddbmm(bias_flat, q_flat * scale, k_flat)  # (B*nh, T, T)
        att = att.reshape(B, self.n_head, T, T)  # (B,nh,T,T)
        ####################################################################

        # normal computation
        # att = q @ k.transpose(2,3)/math.sqrt(C/self.n_head) + cable_bias
        # att = att.masked_fill(mask[:,:,:T,:T] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        y = att @ v

        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.c_proj(y)

        # if return_attentions:
        #     mask = torch.tril(torch.ones(T, T), diagonal=0).to(x.device)
        #     scores = scores * mask
        #     return (y, scores)

        return y
    