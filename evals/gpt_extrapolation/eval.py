import torch
from dataclasses import dataclass
import sys
sys.path.append('src/')
from model_gpt import Model
from train_gpt import ModelConfig
from data_loader import DataLoaderLite
import math
from tqdm import tqdm
import os
import pandas as pd
from typing import List




def _get_dataset_identifier(name: str) -> str:
    known_datasets = {
        'fineweb-edu-10B': 'fineweb-edu-10B',
        'fineweb-10B': 'fineweb-10B', 
        'wikitext-103': 'wikitext-103',
        'wikitext-2': 'wikitext-2'
    }
    for key, value in known_datasets.items():
        if key in name:
            return value
    return None


def evaluate_model(model, model_name, batch_size, seq_len, eval_dataset_path, device):

    val_loader = DataLoaderLite(
        B=batch_size,
        T=512,
        process_rank=0,  
        num_processes=1,  
        master_process=0,
        split="val",
        path=eval_dataset_path,
    )
    
    total_loss = 0.0
    # evaluation on fineweb takes too long if we don't limit total_val_steps
    total_val_steps = min(val_loader.num_total_tokens // (batch_size*seq_len), 1000)

    eval_dataset_name = _get_dataset_identifier(eval_dataset_path)
    with torch.no_grad():
        for _ in tqdm(range(total_val_steps), desc=f"{model_name} @ {eval_dataset_name} @ {seq_len}"):
            torch.cuda.empty_cache()
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                _, loss = model(x, y)
            total_loss += loss.item() / total_val_steps
            del x, y
            
    perplexity = math.exp(total_loss)
    print(f"loss={total_loss:.4f}, ppl={perplexity:.2f}")
    return perplexity


def main(saved_models_path: str, trained_dataset_name: str, eval_dataset_path: str):
    eval_seq_lens = [512, 1024, 2048, 4096, 8192, 15360]
    # eval_seq_lens = [15360]

    train_dataset_name = _get_dataset_identifier(trained_dataset_name)
    eval_dataset_name = _get_dataset_identifier(eval_dataset_path)

    if train_dataset_name != eval_dataset_name:
        # we are doing a out of domain generalization evaluation on the base sequence length
        eval_seq_lens = [1024]  # since our models were trained on 1024 sequence length

    results_path = f'evals/gpt_extrapolation/GPTs_extrapolation_{train_dataset_name}_{eval_dataset_name}.pkl'

    # Load existing results if available
    if os.path.exists(results_path):
        final_df = pd.read_pickle(results_path)
        existing_indices = set(final_df.index)
    else:
        final_df = pd.DataFrame(columns=[f'ppl_{eval_seq_len}' for eval_seq_len in eval_seq_lens])
        existing_indices = set()

    models_to_evaluate = {}
    for root, dirs, files in os.walk(saved_models_path):
        for file in files:
            if 'model' in file:
                full_path = os.path.abspath(os.path.join(root, file))
                if full_path.split('/')[-2].split('_')[1] not in ['cable2', 'cable3', 'cable4'] and train_dataset_name in full_path:
                    models_to_evaluate.update({full_path.split('/')[-2]: full_path})

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for model_name, model_path in models_to_evaluate.items():
        if model_name in existing_indices:
            print(f"Skipping evaluation of {model_name} on {eval_dataset_name} (already evaluated).")
            continue

        pos_method = model_name.split('_')[1]
        trained_seq_len = int(model_name.split('_')[-1])

        if 'tiny' in model_name:
            config = ModelConfig(pos_method=pos_method, vocab_size=50304, n_layer=6, n_head=8, n_embd=512, block_size=trained_seq_len)
        elif 'small' in model_name:
            config = ModelConfig(pos_method=pos_method, vocab_size=50304, n_layer=12, n_head=12, n_embd=768, block_size=trained_seq_len)
        elif 'medium' in model_name:
            config = ModelConfig(pos_method=pos_method, vocab_size=50304, n_layer=24, n_head=16, n_embd=1024, block_size=trained_seq_len)

        model = Model(config)
        state_dict = torch.load(model_path)
        keys_to_remove = [key for key in state_dict.keys() if "cached_bias" in key or "cached_seq_len" in key or "cached_matrix" in key]
        for key in keys_to_remove:
            del state_dict[key]
        model.load_state_dict(state_dict)
        model.eval().to(device)

        new_row = {}
        for eval_seq_len in eval_seq_lens:
            if 'medium' in model_name and (pos_method in ['cable6', 'kcable6', 'cable7'] and eval_seq_len == 15360):
                eval_seq_len = 14000
            if pos_method == 'learnable' and eval_seq_len > trained_seq_len:
                ppl = None
            else:
                ppl = evaluate_model(
                    model=model,
                    model_name=model_name,
                    batch_size=1,
                    seq_len=eval_seq_len,
                    eval_dataset_path=eval_dataset_path,
                    device=device
                )
            if eval_seq_len == 14000:
                eval_seq_len = 15360
            new_row.update({f'ppl_{eval_seq_len}': round(ppl, 2) if ppl else None})

        final_df.loc[model_name] = new_row
        final_df.to_pickle(results_path)
        torch.cuda.empty_cache()
        print('='*50)

    return final_df


if __name__ == '__main__':
    main(saved_models_path='/home/hmrz/Cable/Logs/', 
         trained_dataset_name='fineweb-edu-10B',
         eval_dataset_path='/home/hmrz/fineweb-edu-10B')

    main(saved_models_path='/home/hmrz/Cable/Logs/', 
         trained_dataset_name='fineweb-edu-10B',
         eval_dataset_path='/home/hmrz/fineweb-10B')
    
    main(saved_models_path='/home/hmrz/Cable/Logs/', 
         trained_dataset_name='fineweb-edu-10B',
        #  eval_dataset_path='/home/hmrz/Cable/data/wikitext-103'
         eval_dataset_path='/home/hmrz/wikitext-103')

    main(saved_models_path='/home/hmrz/Cable/Logs/', 
         trained_dataset_name='fineweb-edu-10B',
         eval_dataset_path='/home/hmrz/wikitext-2')
    
    main(saved_models_path='/home/hmrz/Cable/Logs/', 
         trained_dataset_name='wikitext-103',
         eval_dataset_path='/home/hmrz/wikitext-103')