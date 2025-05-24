import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# torch.manual_seed(42)
# # If using CUDA (GPU), also set these seeds
# torch.cuda.manual_seed(42)
# torch.cuda.manual_seed_all(42)  # For multi-GPU setups
# # Additional settings for better reproducibility (may impact performance)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


class Kernel_CABLE(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # print(self.c_attn.bias)

        # biases projections for all heads, but in a batch
        self.cable_layer = nn.Linear(config.n_embd, config.n_head)
        
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        
        self.n_head = config.n_head
        self.block_size = config.block_size
        self.n_embd = config.n_embd


        # Register buffers for causal mask and caching
        # self.register_buffer("causal_mask", torch.triu(torch.full((config.block_size, config.block_size), float('-inf')), 
        #                                                diagonal=1), persistent=False)  # persistent=False -> Won't be saved in state_dict
        # self.register_buffer("cached_bias", None)
        # self.register_buffer("cached_seq_len", None)

    # TODO: implement return_attentions
    def forward(self, x, attention_mask=None, return_attentions=False):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # extracting query, key, and value vectors
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)   # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)   # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)   # (B, nh, T, hs)
        
        # Relative cable biases for tokens on each head
        Sums = torch.cumsum(-F.softplus(self.cable_layer(x)), dim=1).permute(0, 2, 1)  # (B,nh,T)
        cable_bias = (Sums.unsqueeze(3) - Sums.unsqueeze(2))  # (B,nh,T,T)

        # Apply kernel
        cable_bias = -torch.log(cable_bias**2 + 1)

        # Apply mask
        if attention_mask is not None:  # only for Bert models
            attention_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -10000.0
            scores_masked = cable_bias + attention_mask
        else:  # only for GPT models
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
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))  # (B,nh,T,T)
        # att = att + cable_bias + self.causal_mask  # (B,nh,T,T)

        att = F.softmax(att, dim=-1)
        y = att @ v

        # Re-assemble all head outputs
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.c_proj(y)

        # if return_attentions:
        #     mask = torch.tril(torch.ones(T, T), diagonal=0).to(x.device)
        #     scores = scores * mask
        #     return (y, scores)
        
        return y



# class CableConfig:
#     block_size: int = 5 # max sequence length
#     vocab_size: int = 50257 
#     n_layer: int = 24 # number of layers
#     n_head: int = 4 # number of heads
#     n_embd: int = 12 # embedding dimension


# config = CableConfig
# cable = CABLE(config)

# B,T,C = 1,5,12

# x = torch.randn(B,T,C)
# print(x)

# cable(x)



class DAPE_Kernel_CABLE(nn.Module):
    def __init__(self, config, mlp_dape_width=32):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # print(self.c_attn.bias)

        # biases projections for all heads, but in a batch
        self.cable_layer = nn.Linear(config.n_embd, config.n_head)
        
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        
        self.n_head = config.n_head
        self.block_size = config.block_size
        self.n_embd = config.n_embd

        self.mlp_dape = nn.Sequential(
            nn.Linear(2 * config.n_head, mlp_dape_width),
            nn.LeakyReLU(),
            nn.Linear(mlp_dape_width, config.n_head)
        )

        # Register buffers for causal mask and caching
        self.register_buffer("causal_mask", torch.triu(torch.full((config.block_size, config.block_size), float('-inf')), 
                                                       diagonal=1), persistent=False)  # persistent=False -> Won't be saved in state_dict
        # self.register_buffer("cached_bias", None)
        # self.register_buffer("cached_seq_len", None)


    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # extracting query, key, and value vectors
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)   # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)   # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)   # (B, nh, T, hs)
        
        # Relative cable biases for tokens on each head
        Sums = torch.cumsum(-F.softplus(self.cable_layer(x)), dim=1).permute(0, 2, 1)  # (B,nh,T)
        cable_bias = (Sums.unsqueeze(3) - Sums.unsqueeze(2))  # (B,nh,T,T)

        # Apply kernel
        cable_bias = torch.log(cable_bias**2 + 1)

        # normal computation
        # TODO: Can we use torch.baddbmm ?
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))  # (B,nh,T,T)

        # Apply DAPE
        scores = torch.cat((att, cable_bias), dim=1).permute(0, 2, 3, 1)  # (B,T,T,2*nh)
        scores = self.mlp_dape(scores).permute(0, 3, 1, 2)  # (B,nh,T,T)

        att = att + scores + self.causal_mask  # (B,nh,T,T)

        att = F.softmax(att, dim=-1)
        y = att @ v

        # Re-assemble all head outputs
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.c_proj(y)
        # print(y)
        return y



# class CableConfig:
#     block_size: int = 5 # max sequence length
#     vocab_size: int = 50257 
#     n_layer: int = 24 # number of layers
#     n_head: int = 4 # number of heads
#     n_embd: int = 12 # embedding dimension


# config = CableConfig
# cable = DAPE_CABLE(config)

# B,T,C = 1,5,12

# x = torch.randn(B,T,C)
# print(x)

# cable(x)