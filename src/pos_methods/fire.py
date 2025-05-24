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


class FIRE(nn.Module):
    def __init__(self, config, mlp_fire_width=32, init_c=0.1, init_L=512., eps=1e-6):
        super().__init__()
        assert config.n_embd % config.n_head == 0
    
        # key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        self.dtype = torch.float32
        self.eps = eps

        # Define the MLP layers
        self.mlp_fire = nn.Sequential(
            nn.Linear(1, mlp_fire_width),
            nn.ReLU(),
            nn.Linear(mlp_fire_width, self.n_head))
        
        # Initialize c (log transformation parameter)
        self.c = nn.Parameter(torch.tensor(init_c))

        # Initialize L (threshold)
        self.init_L = nn.Parameter(torch.tensor(init_L), requires_grad=False)

        # Learn a multiplier to L
        self.L_multiplier = nn.Parameter(torch.tensor(1.0))

        # Register buffers for causal mask and caching
        # self.register_buffer("causal_mask", torch.triu(torch.full((config.block_size, config.block_size), float('-inf')), 
        #                                                diagonal=1), persistent=False)  # persistent=False -> Won't be saved in state_dict
        # self.register_buffer("cached_matrix", None)
        # self.register_buffer("cached_seq_len", None)


    # TODO: implement return_attentions
    def forward(self, x, attention_mask=None, return_attentions=False):
        B, T, C = x.size()  # batch size, sequence length, embedding dim

        # Calculate query, key, values
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)   # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)   # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)   # (B, nh, T, hs)

        causal_mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)

        # if self.cached_seq_len is None or self.cached_seq_len < T:
            # Create new bias matrix if sequence is longer than cache
        relative_positions = torch.tril(
            torch.arange(T, device=x.device).view(T, 1) + 
            torch.arange(0, -T, -1, device=x.device)).to(self.dtype)  # (T,T)

        #     self.cached_seq_len = torch.tensor(T, device=x.device)
        #     self.cached_matrix = relative_positions

        # else:
        #     # Use cached relative_positions
        #     relative_positions = self.cached_matrix

        threshold = torch.abs(self.L_multiplier * self.init_L)
        rel_distance_max = torch.max(torch.tril(relative_positions), dim=-1)[0]

        pos_normalizer = torch.max(rel_distance_max, threshold)
        pos_normalizer = pos_normalizer[:, None]

        rel_distance = torch.log(torch.abs(self.c * relative_positions) + 1)
        pos_normalizer = torch.log(torch.abs(self.c * pos_normalizer) + 1) + self.eps

        # Progressive interpolation
        normalized_distance = rel_distance / pos_normalizer
        normalized_distance = normalized_distance.to(x.dtype)

        fire_bias = self.mlp_fire(normalized_distance.unsqueeze(-1))
        fire_bias = fire_bias.permute(2, 0, 1)  # (nh,T,T)

        # broadcat bias along the batch dimension       
        fire_bias = torch.tile(fire_bias.unsqueeze(0), (B, 1, 1, 1))  # (1,nh,T,T) → (B,nh,T,T)
        
        ###### Using torch.baddbmm for faster matrix mult with bias ########
        q_flat = q.reshape(B * self.n_head, T, head_size)  # (B*nh, T, hs)
        k_flat = k.reshape(B * self.n_head, T, head_size).transpose(-2, -1)  # (B*nh, hs, T)
        bias_flat = (fire_bias + causal_mask).reshape(B * self.n_head, T, T)  # (B*nh, T, T)
        scale = 1.0 / math.sqrt(head_size)
        att = torch.baddbmm(bias_flat, q_flat * scale, k_flat)  # (B*nh, T, T)
        att = att.reshape(B, self.n_head, T, T)  # (B,nh,T,T)
        ####################################################################

        # normal computation
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))  # (B,nh,T,T)
        # att = att + fire_bias + self.causal_mask  # (B,nh,T,T)
        
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
# fire = FIRE(config)

# B,T,C = 2,5,12
# x = torch.randn(B,T,C)

# fire(x)



class DAPE_FIRE(nn.Module):
    def __init__(self, config, mlp_fire_width=32, mlp_dape_width=32, init_c=0.1, init_L=512., eps=1e-6):
        super().__init__()
        assert config.n_embd % config.n_head == 0
    
        # key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        self.dtype = torch.float32
        self.eps = eps

        # Define the MLP layers
        self.mlp_fire = nn.Sequential(
            nn.Linear(1, mlp_fire_width),
            nn.ReLU(),
            nn.Linear(mlp_fire_width, self.n_head))
        
        self.mlp_dape = nn.Sequential(
            nn.Linear(2 * config.n_head, mlp_dape_width),
            nn.LeakyReLU(),
            nn.Linear(mlp_dape_width, config.n_head)
        )
        
        # Initialize c (log transformation parameter)
        self.c = nn.Parameter(torch.tensor(init_c))

        # Initialize L (threshold)
        self.init_L = nn.Parameter(torch.tensor(init_L), requires_grad=False)

        # Learn a multiplier to L
        self.L_multiplier = nn.Parameter(torch.tensor(1.0))

        # Register buffers for causal mask and caching
        # self.register_buffer("causal_mask", torch.triu(torch.full((config.block_size, config.block_size), float('-inf')), 
        #                                                diagonal=1), persistent=False)  # persistent=False -> Won't be saved in state_dict
        # self.register_buffer("cached_matrix", None)
        # self.register_buffer("cached_seq_len", None)


    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dim

        # Calculate query, key, values
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)   # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)   # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)   # (B, nh, T, hs)

        causal_mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)

        # if self.cached_seq_len is None or self.cached_seq_len < T:
            # Create new bias matrix if sequence is longer than cache
        relative_positions = torch.tril(
            torch.arange(T, device=x.device).view(T, 1) + 
            torch.arange(0, -T, -1, device=x.device)).to(self.dtype)  # (T,T)

        #     self.cached_seq_len = torch.tensor(T, device=x.device)
        #     self.cached_matrix = relative_positions

        # else:
        #     # Use cached relative_positions
        #     relative_positions = self.cached_matrix

        threshold = torch.abs(self.L_multiplier * self.init_L)
        rel_distance_max = torch.max(torch.tril(relative_positions), dim=-1)[0]

        pos_normalizer = torch.max(rel_distance_max, threshold)
        pos_normalizer = pos_normalizer[:, None]

        rel_distance = torch.log(torch.abs(self.c * relative_positions) + 1)
        pos_normalizer = torch.log(torch.abs(self.c * pos_normalizer) + 1) + self.eps

        # Progressive interpolation
        normalized_distance = rel_distance / pos_normalizer
        normalized_distance = normalized_distance.to(x.dtype)

        fire_bias = self.mlp_fire(normalized_distance.unsqueeze(-1))
        fire_bias = fire_bias.permute(2, 0, 1)  # (nh,T,T)

        # broadcat bias along the batch dimension       
        pos_bias = torch.tile(fire_bias.unsqueeze(0), (B, 1, 1, 1))  # (1,nh,T,T) → (B,nh,T,T)

        # normal computation
        # TODO: Can we use torch.baddbmm ?
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))  # (B,nh,T,T)

        # Apply DAPE
        scores = torch.cat((att, pos_bias), dim=1).permute(0, 2, 3, 1)  # (B,T,T,2*nh)
        scores = self.mlp_dape(scores).permute(0, 3, 1, 2)  # (B,nh,T,T)

        att = att + scores + causal_mask  # (B,nh,T,T)
        
        att = F.softmax(att, dim=-1)
        y = att @ v
        
        # Re-assemble all head outputs
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.c_proj(y)
        return y




# class CableConfig:
#     block_size: int = 5 # max sequence length
#     vocab_size: int = 50257 
#     n_layer: int = 24 # number of layers
#     n_head: int = 4 # number of heads
#     n_embd: int = 12 # embedding dimension


# config = CableConfig
# dape_fire = DAPE_FIRE(config)

# B,T,C = 2,5,12
# x = torch.randn(B,T,C)

# dape_fire(x)