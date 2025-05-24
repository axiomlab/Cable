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


class AliBi(nn.Module):
    def __init__(self, config):
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
        
        # ALiBi-specific initialization
        self.register_buffer("slopes", self._get_alibi_slopes(self.n_head), persistent=False)  # (nh,1,1)
        
        # Register buffers for causal mask and caching
        # self.register_buffer("causal_mask", torch.triu(torch.full((config.block_size, config.block_size), float('-inf')), 
        #                                                diagonal=1), persistent=False)  # persistent=False -> Won't be saved in state_dict
        # self.register_buffer("cached_bias", None)
        # self.register_buffer("cached_seq_len", None)


    def _get_alibi_slopes(self, n):
        """Generate slopes for ALiBi attention heads"""
        x = (2 ** 8) ** (1 / n)
        return (
            torch.tensor([1 / x ** (i + 1) for i in range(n)])
            .unsqueeze(-1)
            .unsqueeze(-1)
            .cuda()
        )


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

        # causal_mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        
        # ALiBi bias computation
        relative_positions = -torch.tril(
            torch.arange(T, device=x.device).view(T, 1) + 
            torch.arange(0, -T, -1, device=x.device)).to(self.dtype)  # (T,T)
            
        # Apply mask
        if attention_mask is not None:  # only for Bert models
            attention_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -10000.0
            relative_positions = relative_positions + relative_positions.T - torch.diag(torch.diag(relative_positions))
            alibi_bias = self.slopes.to(x.device) * relative_positions  # (nh,1,1) * (T,T) -> (nh,T,T)
            # broadcat bias along the batch dimension        
            alibi_bias = torch.tile(alibi_bias.unsqueeze(0), (B, 1, 1, 1))  # (1,nh,T,T) → (B,nh,T,T)
            scores_masked = alibi_bias + attention_mask
        else:  # only for GPT models
            causal_mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
            alibi_bias = self.slopes.to(x.device) * relative_positions  # (nh,1,1) * (T,T) -> (nh,T,T)
            # broadcat bias along the batch dimension        
            alibi_bias = torch.tile(alibi_bias.unsqueeze(0), (B, 1, 1, 1))  # (1,nh,T,T) → (B,nh,T,T)
            scores_masked = alibi_bias + causal_mask

        # # broadcat bias along the batch dimension        
        # alibi_bias = torch.tile(alibi_bias.unsqueeze(0), (B, 1, 1, 1))  # (1,nh,T,T) → (B,nh,T,T)
        
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
        # att = att + alibi_bias + self.causal_mask  # (B,nh,T,T)
        
        att = F.softmax(att, dim=-1)
        y = att @ v
        
        # Re-assemble all head outputs
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.c_proj(y)

        # if return_attentions:
        #     mask = torch.tril(torch.ones(T, T), diagonal=0).to(x.device)
        #     if attention_mask is None:  # This is a GPT model, which needs causal masking
        #         scores = scores * mask
        #     return (y, scores)
        
        return y
    




# class CableConfig:
#     block_size: int = 5 # max sequence length
#     vocab_size: int = 50257 
#     n_layer: int = 24 # number of layers
#     n_head: int = 4 # number of heads
#     n_embd: int = 12 # embedding dimension


# config = CableConfig
# alibi = AliBi(config)

# B,T,C = 1,5,12
# x = torch.randn(B,T,C)

# alibi(x)


class DAPE_AliBi(nn.Module):
    def __init__(self, config, mlp_dape_width=32):
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
        
        self.mlp_dape = nn.Sequential(
            nn.Linear(2 * config.n_head, mlp_dape_width),
            nn.LeakyReLU(),
            nn.Linear(mlp_dape_width, config.n_head)
        )
        
        # ALiBi-specific initialization
        # slopes = torch.Tensor(self._get_alibi_slopes(self.n_head))
        self.register_buffer("slopes", self._get_alibi_slopes(self.n_head), persistent=False)  # (nh,1,1)
        
        # Register buffers for causal mask and caching
        self.register_buffer("causal_mask", torch.triu(torch.full((config.block_size, config.block_size), float('-inf')), 
                                                       diagonal=1), persistent=False)  # persistent=False -> Won't be saved in state_dict
        self.register_buffer("cached_bias", None)
        self.register_buffer("cached_seq_len", None)


    def _get_alibi_slopes(self, n):
        """Generate slopes for ALiBi attention heads"""
        x = (2 ** 8) ** (1 / n)
        return (
            torch.tensor([1 / x ** (i + 1) for i in range(n)])
            .unsqueeze(-1)
            .unsqueeze(-1)
            .cuda()
        )


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
        
        # ALiBi bias computation
        if self.cached_seq_len is None or self.cached_seq_len < T:
            # Create new bias matrix if sequence is longer than cache
            relative_positions = -torch.tril(
                torch.arange(T, device=x.device).view(T, 1) + 
                torch.arange(0, -T, -1, device=x.device)).to(self.dtype)  # (T,T)
            
            alibi_bias = self.slopes.to(x.device) * relative_positions  # (nh,1,1) * (T,T) -> (nh,T,T)

            # Cache for future use
            self.cached_seq_len = torch.tensor(T, device=x.device)
            self.cached_bias = alibi_bias

        else:
            # Use cached bias
            alibi_bias = self.cached_bias

        # broadcat bias along the batch dimension        
        pos_bias = torch.tile(alibi_bias.unsqueeze(0), (B, 1, 1, 1))  # (1,nh,T,T) → (B,nh,T,T)
        
        # normal computation
        # TODO: Can we use torch.baddbmm ?
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))  # (B,nh,T,T)

        # Apply DAPE
        scores = torch.cat((att, pos_bias), dim=1).permute(0, 2, 3, 1)  # (B,T,T,2*nh)
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
# dape_alibi = DAPE_AliBi(config)

# B,T,C = 1,5,12
# x = torch.randn(B,T,C)

# dape_alibi(x)

