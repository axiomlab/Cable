# TODO: This script runs the valuation on multiple GPUs, however it seems 
# to be a discrepency of the outputs with the single GPU script!!!

import torch
import torch.multiprocessing as mp
from dataclasses import dataclass
import sys
import math
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
sys.path.append('src/')
from model_gpt import Model
from data_loader import DataLoaderLite



@dataclass
class ModelConfig:
    pos_method: str = None
    use_dape: bool = False
    block_size: int = 1024
    vocab_size: int = 50304 
    n_layer: int = 24
    n_head: int = 16 
    n_embd: int = 1024


def evaluate_model(model, model_name, batch_size, seq_len, eval_dataset_path, device, process_rank=0, num_processes=1):
    val_loader = DataLoaderLite(
        B=batch_size,
        T=seq_len,
        process_rank=process_rank,
        num_processes=num_processes,
        split="val",
        path=eval_dataset_path,
        master_process=(process_rank == 0),
    )
    
    total_loss = 0.0
    total_val_steps = val_loader.num_total_tokens // (batch_size * seq_len * num_processes)
    
    dataset_name = (
        'fineweb10B' if 'fineweb10B' in eval_dataset_path
        else 'wikitext-103' if 'wikitext-103' in eval_dataset_path
        else None
    )
    
    model.eval()
    with torch.no_grad():
        for _ in tqdm(range(total_val_steps), desc=f"GPU {process_rank} | {model_name} @ {dataset_name} @ {seq_len}"):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                _, loss = model(x, y)
            total_loss += loss.item()
            del x, y
            torch.cuda.empty_cache()
    
    avg_loss = total_loss / total_val_steps
    return avg_loss



def load_model(model_path, model_name, device):
    pos_method = model_name.split('_')[1]
    trained_seq_len = int(model_name.split('_')[-1])

    if 'tiny' in model_name:
        config = ModelConfig(pos_method=pos_method, vocab_size=50304, n_layer=6, n_head=8, n_embd=512, block_size=trained_seq_len)
    elif 'small' in model_name:
        config = ModelConfig(pos_method=pos_method, vocab_size=50304, n_layer=12, n_head=12, n_embd=768, block_size=trained_seq_len)
    elif 'medium' in model_name:
        config = ModelConfig(pos_method=pos_method, vocab_size=50304, n_layer=24, n_head=16, n_embd=1024, block_size=trained_seq_len)
    
    model = Model(config)
    state_dict = torch.load(model_path, map_location='cpu')
    keys_to_remove = [key for key in state_dict.keys() if "cached_bias" in key]
    keys_to_remove += [key for key in state_dict.keys() if "cached_seq_len" in key]
    keys_to_remove += [key for key in state_dict.keys() if "cached_matrix" in key]
    for key in keys_to_remove:
        del state_dict[key]
    model.load_state_dict(state_dict)
    model.to(device)
    return model, pos_method, trained_seq_len



def main_worker(rank, saved_models_path, eval_dataset_path, world_size):
    torch.cuda.set_device(rank)
    device = f'cuda:{rank}'
    
    eval_seq_lens = [512, 1024, 2048, 4096, 8192, 15872]
    eval_seq_lens = [16384]
    
    models_to_evaluate = {}
    for root, dirs, files in os.walk(saved_models_path):
        for file in files:
            if 'model' in file:
                full_path = os.path.abspath(os.path.join(root, file))
                if full_path.split('/')[-2].split('_')[1] in ['cable']:
                    models_to_evaluate.update({full_path.split('/')[-2]: full_path})
    
    results_per_gpu = {}
    
    for model_name, model_path in models_to_evaluate.items():
        model, pos_method, trained_seq_len = load_model(model_path, model_name, device)

        new_row = {}
        for eval_seq_len in eval_seq_lens:
            # learnable pos method does not have any extrapolation
            if pos_method == 'learnable' and eval_seq_len > trained_seq_len:
                ppl = None
            else:
                loss = evaluate_model(
                    model=model,
                    model_name=model_name,
                    batch_size=1,
                    seq_len=eval_seq_len,
                    eval_dataset_path=eval_dataset_path,
                    device=device,
                    process_rank=rank,
                    num_processes=world_size,
                )
                ppl = math.exp(loss)
            new_row[f'ppl_{eval_seq_len}'] = ppl
        
        results_per_gpu[model_name] = new_row
        torch.cuda.empty_cache()
        # print('='*50)
    
    # Save each GPU's results separately
    torch.save(results_per_gpu, f'evals/gpt_extrapolation/results_rank{rank}.pt')
    print(f"GPU {rank} finished.")



def multi_gpu_main(saved_models_path, eval_dataset_path):
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs to do evaluations!")
    mp.spawn(
        main_worker,
        args=(saved_models_path, eval_dataset_path, world_size),
        nprocs=world_size,
        join=True
    )

    all_results = []
    index_names = []
    for rank in range(world_size):
        gpu_results = torch.load(f'evals/gpt_extrapolation/results_rank{rank}.pt')
        for model_name, metrics in gpu_results.items():
            all_results.append(metrics)
            index_names.append(model_name)
    
    dataset_name = (
        'fineweb10B' if 'fineweb10B' in eval_dataset_path
        else 'wikitext-103' if 'wikitext-103' in eval_dataset_path
        else None
    )
    
    final_df = pd.DataFrame(all_results, index=index_names)
    final_df.to_pickle(f'evals/gpt_extrapolation/GPTs_extrapolation_{dataset_name}_2.pkl')
    return final_df




if __name__ == '__main__':
    multi_gpu_main(saved_models_path='/home/hmrz/Cable/Logs/', eval_dataset_path='/home/hmrz/wikitext-103')
    multi_gpu_main(saved_models_path='/home/hmrz/Cable/Logs/', eval_dataset_path='/home/hmrz/edu_fineweb10B')
