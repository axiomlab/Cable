from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import datetime
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass
import argparse
from Cable import Cable
from data_loader import DataLoaderLite
import os
import torch
import time
import math

@dataclass
class CableConfig:
    block_size: int = 512 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 6 # number of layers
    n_head: int = 8 # number of heads
    n_embd: int = 512 # embedding dimension


if __name__ == "__main__":
    

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

    parser.add_argument('--batch-size', type=int, default=16, help='batch size: 128 for wikitext and 32 for edu_fineweb')
    
    parser.add_argument('--sequence-length', type=int, default=1024, help='Sequence length (512/1024)')

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
        
    total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
    B = args.batch_size # micro batch size
    T = args.sequence_length # sequence length
    
    assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
    
    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train", path=args.dataset_dir, master_process=master_process)
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val", path=args.dataset_dir, master_process=master_process)
    
    torch.set_float32_matmul_precision('high')
    
    # create model
    if args.model == 'large':
        model = Cable(CableConfig(vocab_size=50304, n_layer=36, n_head=20, n_embd=1280, block_size=T))
    elif args.model == 'medium':
        model = Cable(CableConfig(vocab_size=50304, n_layer=24, n_head=16, n_embd=1024, block_size=T))
    elif args.model == 'small':
        model = Cable(CableConfig(vocab_size=50304, n_layer=12, n_head=12, n_embd=768, block_size=T))
    elif args.model == 'tiny':
        model = Cable(CableConfig(vocab_size=50304, n_layer=6, n_head=8, n_embd=512, block_size=T))

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
    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type, master_process=master_process)
    
    # create the log directory we will write checkpoints to and log to
    log_dir = args.save_dir
    os.makedirs(log_dir, exist_ok=True)
    
    from datetime import datetime
    log_file = os.path.join(log_dir, f"log_trial.txt")
    with open(log_file, "w") as f: # open for writing to clear the file
        pass
    num_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    
    if master_process:
        print(f'### starting training loop of gpt2 @{datetime.now()}; num of parameters:{num_params:_}; torch.compile:{args.use_compile};\n')
        print(f"=> Batch_size:{B}; Sequence_length:{T}; max_lr:{max_lr}; min_lr:{min_lr}")
        writer.add_text('Start of training...', f'### starting training loop of ali_model_sd @{datetime.now()}; num of parameters:{num_params:_}; torch.compile:{args.use_compile}; \n')
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
                    checkpoint = raw_model._orig_mod.state_dict() if hasattr(raw_model, '_orig_mod') else raw_model.state_dict()
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
