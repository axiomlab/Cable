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


class AliBi2(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.alibi_scale = nn.Linear(config.n_embd, config.n_head)
        
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
            # alibi_bias_old = self.slopes.to(x.device) * relative_positions  # (nh,1,1) * (T,T) -> (nh,T,T)
            # broadcat bias along the batch dimension
            alibi_bias_old = torch.tile(relative_positions.unsqueeze(0).unsqueeze(0), (B, self.n_head, 1, 1))  # (1,1,T,T) → (B,nh,T,T)
            bias_weights = F.softplus(self.alibi_scale(x)).permute(0, 2, 1)  # (B,nh,T)
            alibi_bias = bias_weights.unsqueeze(-1) * alibi_bias_old  # (B,nh,T,1) * (B,nh,T,T) -> (B,nh,T,T)
            scores_masked = alibi_bias + attention_mask
        else:  # only for GPT models
            causal_mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
            # alibi_bias_old = self.slopes.to(x.device) * relative_positions  # (nh,1,1) * (T,T) -> (nh,T,T)
            # broadcat bias along the batch dimension        
            alibi_bias_old = torch.tile(relative_positions.unsqueeze(0).unsqueeze(0), (B, self.n_head, 1, 1))  # (1,1,T,T) → (B,nh,T,T)
            bias_weights = F.softplus(self.alibi_scale(x)).permute(0, 2, 1)  # (B,nh,T)
            alibi_bias = bias_weights.unsqueeze(-1) * alibi_bias_old  # (B,nh,T,1) * (B,nh,T,T) -> (B,nh,T,T)
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