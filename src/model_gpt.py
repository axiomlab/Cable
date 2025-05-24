import os
import math
import time
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from pos_methods.base import BASE_ATTENTION
from pos_methods.alibi import AliBi, DAPE_AliBi
from pos_methods.cable import CABLE, DAPE_CABLE
from pos_methods.fire import FIRE, DAPE_FIRE
from pos_methods.kerple import KERPLE, DAPE_KERPLE
from pos_methods.rope import ROPE
from pos_methods.t5bias import T5Bias
from pos_methods.cable5 import CABLE5
from pos_methods.cable6 import CABLE6
from pos_methods.cable7 import CABLE7
from pos_methods.rotali import ROTALI
from pos_methods.kcable import Kernel_CABLE
from pos_methods.kcable5 import Kernel_CABLE5
from pos_methods.kcable6 import Kernel_CABLE6
from pos_methods.alibi2 import AliBi2



class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)      
        self.attn = self._init_attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def _init_attention(self, config):
        """Initialize the appropriate attention mechanism based on config"""
        attention_classes = {
            'cable': DAPE_CABLE if config.use_dape else CABLE,
            'kcable': Kernel_CABLE,
            'cable5': CABLE5,
            'kcable5': Kernel_CABLE5,
            'cable6': CABLE6,
            'kcable6': Kernel_CABLE6,
            'cable7': CABLE7,
            'rotali':ROTALI,
            'alibi': DAPE_AliBi if config.use_dape else AliBi,
            'alibi2': AliBi2,
            'fire': DAPE_FIRE if config.use_dape else FIRE,
            'kerple': DAPE_KERPLE if config.use_dape else KERPLE,
            'learnable': BASE_ATTENTION,
            'sinusoidal': BASE_ATTENTION,
            'rope': ROPE,
            't5bias': T5Bias
        }
        if config.pos_method not in attention_classes:
            raise ValueError(f"Unknown position method: {config.pos_method}")
        return attention_classes[config.pos_method](config)
    
    def forward(self, x, return_attentions=False):
        attn_output = self.attn(self.ln_1(x), return_attentions=return_attentions)
        
        if return_attentions:
            attn_output, attentions = attn_output
            
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        
        return (x, attentions) if return_attentions else x


class Model(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        if config.pos_method == 'learnable':
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = nn.LayerNorm(config.n_embd),
            ))
        else:
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = nn.LayerNorm(config.n_embd),
            ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, return_attentions=False):
        # idx is of shape (B, T)
        B, T = idx.size()
        
        if self.config.pos_method == 'learnable':
            assert T <= self.config.block_size, f"Cannot forward sequence of length {T} in learnable positional encoing, block size is only {self.config.block_size}"
        
        # forward the token
        x = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)

        all_attentions = () if return_attentions else None
    
        # Handle learnable and sinusoidal positional encoding Seperately
        if self.config.pos_method == 'learnable':
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
            pos_emb = self.transformer.wpe(pos) # learnable position embeddings of shape (T, n_embd)
            x = x + pos_emb
        elif self.config.pos_method == 'sinusoidal':
            position = torch.arange(T).unsqueeze(1).to(idx.device)  # Shape: (max_len, 1)
            div_term = torch.exp(torch.arange(0, self.config.n_embd, 2) * (-math.log(10000.0) / self.config.n_embd)).to(idx.device)
            pe = torch.zeros(T, self.config.n_embd).to(idx.device)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            x = x + pe[:T, :].unsqueeze(0) * 0.1
        
        block_output = x
        # forward the blocks of the transformer
        for block in self.transformer.h:
            # x = block(x)
            block_output = block(block_output, return_attentions=return_attentions)
            if return_attentions:
                block_output, attentions = block_output
                all_attentions = all_attentions + (attentions,)
            
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(block_output)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        if return_attentions:
            return logits, loss, all_attentions
        else:
            return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type, master_process):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
