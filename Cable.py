import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

os.environ['PYTHONFAULTHANDLER'] = '1'
os.environ['NCCL_TIMEOUT'] = '200'  # Set to 5 minutes
os.environ['NCCL_DEBUG_SUBSYS'] = 'NET'

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd + config.n_head)

        # biases projections for all heads, but in a batch
        self.b_attn = nn.Linear(config.n_embd, config.n_head)
        
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        
        self.n_head = config.n_head
        self.block_size = config.block_size

        self.relu = nn.ReLU()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float32  # Use consistent dtype

        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        h = self.b_attn(x)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        zzz1 = (torch.cumsum(-F.softplus(h), dim=1))

        z1 = zzz1.view(B, T, self.n_head, 1).transpose(1, 2)

        ll = torch.tensor(math.sqrt(C)).cuda().expand(B, self.n_head, T, 1)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        k = torch.cat([k, -z1, ll], dim=-1)
        q = torch.cat([q, ll, z1], dim=-1)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att3 = torch.triu(
            torch.full((self.block_size, self.block_size), float('-inf'), dtype=self.dtype, device=self.device),
            diagonal=1)

        ###### Using torch.baddbmm for faster matrix mult with bias ########
        kt = k.transpose(2, 3)
        q_shape = q.shape
        qt = q.reshape(q_shape[0] * q_shape[1], q_shape[2], q_shape[3])
        kt_shape = kt.shape
        kt = kt.reshape(kt_shape[0] * kt_shape[1], kt_shape[2], kt_shape[3])
        
        base_attention = torch.baddbmm(att3, qt, kt)/math.sqrt(C)
        att = base_attention.reshape(B, self.n_head, T, T)
        ####################################################################

        y = F.softmax(att, dim=-1) @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)
        
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
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
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class CableConfig:
    block_size: int = 512 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 24 # number of layers
    n_head: int = 16 # number of heads
    n_embd: int = 1024 # embedding dimension

class Cable(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

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

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        # forward the token and posisition embeddings
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb 
        
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
            
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
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

# ------------------ Data Loader ----------------------------------
import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, path):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = path # Set your dataset path here
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # run the training loop
    from torch.distributed import init_process_group, destroy_process_group
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist
    import datetime
    from torch.utils.tensorboard import SummaryWriter
    import argparse

    parser = argparse.ArgumentParser(description="Cable relative positional encoding training")
    parser.add_argument('--model', type=str, default='medium',
                        choices=['large', 'medium', 'small', 'tiny'],
                        help='model size (default: medium)')

    parser.add_argument('--dataset-dir', type=str, default='edu_fineweb10B',
                        help='path to the tokenized dataset (default: edu_fineweb10B)')

    parser.add_argument('--writer-dir', type=str, default='run_logs',
                        help='dir for writing tensorboard outputs (default: run_logs)')

    parser.add_argument('--save-dir', type=str, default='Logs/Cable',
                        help='dir for saving logs and checkpoints (default: Logs/Cable)')

    parser.add_argument('--use_compile', action='store_true', default=True, help='using torch.compile')

    args = parser.parse_args()
    writer_dir_path = args.writer_dir
    run_group_name = 'Cable'
    run_No = datetime.datetime.now().strftime("%m%d%H%M")
    writer = SummaryWriter(log_dir=f"{writer_dir_path}/{run_group_name}/{run_No}")
    
    # set up DDP (distributed data parallel).
    # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    else:
        # vanilla, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        # attempt to autodetect device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")
    
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
    
    enc = tiktoken.get_encoding("gpt2")
    
    total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
    B = 128 # micro batch size
    T = 512 # sequence length
    assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
    
    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train", path=args.dataset_dir)
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val", path=args.dataset_dir)
    
    torch.set_float32_matmul_precision('high')
    
    # create model
    if args.model == 'large':
        model = Cable(CableConfig(vocab_size=50304, n_layer=36, n_head=20, n_embd=1280))
    elif args.model == 'medium':
        model = Cable(CableConfig(vocab_size=50304, n_layer=24, n_head=16, n_embd=1024))
    elif args.model == 'small':
        model = Cable(CableConfig(vocab_size=50304, n_layer=12, n_head=12, n_embd=768))
    elif args.model == 'tiny':
        model = Cable(CableConfig(vocab_size=50304, n_layer=6, n_head=8, n_embd=512))

    model.to(device)

    if args.use_compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model # always contains the "raw" unwrapped model
    
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 715
    max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
    
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_steps:
            return max_lr * (it+1) / warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > max_steps:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)
    
    # optimize!
    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)
    
    # create the log directory we will write checkpoints to and log to
    log_dir = args.save_dir
    os.makedirs(log_dir, exist_ok=True)
    
    from datetime import datetime
    log_file = os.path.join(log_dir, f"log_trial.txt")
    with open(log_file, "w") as f: # open for writing to clear the file
        pass
    num_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    
    if master_process:
        print(f'### starting training loop of gpt2 @{datetime.now()}; num of parameters:{num_params:_}; torch.compile:{use_compile};\n')
        print(f"=> Batch_size:{B}; Sequence_length:{T}; max_lr:{max_lr}; min_lr:{min_lr}")
        writer.add_text('Start of training...', f'### starting training loop of ali_model_sd @{datetime.now()}; num of parameters:{num_params:_}; torch.compile:{use_compile}; \n')
        writer.add_text('HyperParmeters', f"=> Batch_size:{B}; Sequence_length:{T}; max_lr:{max_lr}; min_lr:{min_lr}")
        with open(log_file, "a") as f:
            f.write(f'### starting training loop @{datetime.now()}; num of parameters:{num_params:_}\n')
            f.write(f"=> Batch_size:{B}; Sequence_length:{T}; max_lr:{max_lr}; min_lr:{min_lr}")
            
    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # once in a while evaluate our validation loss
        if step % 250 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

            perplexity = math.exp(val_loss_accum.item())

            if master_process:
                writer.add_scalar("loss/val", val_loss_accum.item(), step)
                writer.add_scalar('perplexity', perplexity, step)
                print(f"Step {step}: validation loss: {val_loss_accum.item():.4f}, perplexity: {perplexity:.2f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f} perplexity {perplexity:.2f}\n")

                # writer.flush()
                torch.cuda.empty_cache()
                if step > 0 and (step % 5000 == 0 or last_step):
                    # optionally write model checkpoints
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        'step': step,
                        'val_loss': val_loss_accum.item(),
                        'perplexity': perplexity
                    }
                    torch.save(checkpoint, checkpoint_path)

    
        # do one step of the optimization
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)

            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        
        if device_type == "cuda":
            torch.cuda.synchronize() # wait for the GPU to finish work
        t1 = time.time()
        dt = t1 - t0 # time difference in seconds
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        
        if master_process:
            writer.add_scalar("loss/train", loss_accum.item(), step)
            writer.add_scalar("norm", norm, step)
            writer.add_scalar("learning_rate", lr, step)
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}...")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}...\n")
            with open(log_file, "a") as f:
                f.write(f'### end of training loop @{datetime.now()}...\n')
                
    writer.close()
    print(f'### end of training loop @{datetime.now()}\n')
    
    if ddp:
        destroy_process_group()
