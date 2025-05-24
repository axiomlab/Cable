"""
Downloads and evaluates HellaSwag in Python.
https://github.com/rowanz/hellaswag

Example HellaSwag json item:

{"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he", "ctx": "A man is sitting on a roof. he", "split": "val", "split_type": "indomain", "label": 3, "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "source_id": "activitynet~v_-JhWjGDPHMY"}

ind: dataset ID
activity_label: The ActivityNet or WikiHow label for this example
context: There are two formats. The full context is in ctx. When the context ends in an (incomplete) noun phrase, like for ActivityNet, this incomplete noun phrase is in ctx_b, and the context up until then is in ctx_a. This can be useful for models such as BERT that need the last sentence to be complete. However, it's never required. If ctx_b is nonempty, then ctx is the same thing as ctx_a, followed by a space, then ctx_b.
endings: a list of 4 endings. The correct index is given by label (0,1,2, or 3)
split: train, val, or test.
split_type: indomain if the activity label is seen during training, else zeroshot
source_id: Which video or WikiHow article this example came from

gpt2 (124M)
- eleuther harness reports acc 28.92%, acc_norm 31.14% (multiple choice style)
- this script: 10042 acc: 0.2859 acc_norm: 0.2955 (completion style)

gpt2-xl (1558M)
- eleuther harness reports acc 40.04%, acc_norm 50.89% (multiple choice style)
- this script: 10042 acc: 0.3842 acc_norm: 0.4893 (completion style)

The validation set of HellaSwag has a total of 10,042 examples.
"""

import os
import json
import requests
import tiktoken
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
# from transformers import GPT2LMHeadModel
import sys
sys.path.append('src/')
from model_gpt import Model
from train_gpt import ModelConfig
from dataclasses import dataclass
from tqdm import tqdm

# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")


def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")

def download(split):
    """Downloads HellaSwag DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)

def render_example(example):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    # data needed to reproduce this eval on the C size
    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    # gather up all the tokens
    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode(" " + end) # note: prepending " " because GPT-2 tokenizer
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label

def iterate_examples(split):
    # there are 10,042 examples in total in val
    download(split)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        for line in f:
            example = json.loads(line)
            yield example



@torch.no_grad()
def evaluate(saved_models_path: str):

    torch.set_float32_matmul_precision('high') # use tf32

    models_to_evaluate = {}
    for root, dirs, files in os.walk(saved_models_path):
        for file in files:
            if 'model' in file:
                full_path = os.path.abspath(os.path.join(root, file))
                # in-domain evaluation
                if full_path.split('/')[-2].split('_')[1] not in ['cable2', 'cable3', 'cable4']:
                    models_to_evaluate.update({full_path.split('/')[-2]: full_path})

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for model_name, model_path in models_to_evaluate.items():
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

        log_dir = "evals/gpt_hellaswag/reports/"

        # check for existance
        if f"{model_name}.log" in os.listdir(log_dir):
            print(f"Skipping {model_name} (already evaluated).")
            continue
        else:
            log_file =  log_dir + f"{model_name}.log"
            with open(log_file, "a") as f:
                f.write(f"============== Evaluating {model_name} on Hellaswag benchmark ============== \n")

        num_correct_norm = 0
        num_correct = 0
        num_total = 0
        for example in tqdm(iterate_examples("val"), desc=f"Evaluating {model_name} on Hellaswag benchmark \n"):
            data, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)

            # get the logits
            logits, _ = model(tokens)
            # evaluate the autoregressive loss at all positions
            shift_logits = (logits[..., :-1, :]).contiguous()
            shift_tokens = (tokens[..., 1:]).contiguous()
            flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_shift_tokens = shift_tokens.view(-1)
            shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
            shift_losses = shift_losses.view(tokens.size(0), -1)
            # now get the average loss just for the completion region (where mask == 1), in each row
            shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
            masked_shift_losses = shift_losses * shift_mask
            # sum and divide by the number of 1s in the mask
            sum_loss = masked_shift_losses.sum(dim=1)
            avg_loss = sum_loss / shift_mask.sum(dim=1)
            # now we have a loss for each of the 4 completions
            # the one with the lowest loss should be the most likely
            pred = sum_loss.argmin().item()
            pred_norm = avg_loss.argmin().item()

            # accumulate stats
            num_total += 1
            num_correct += int(pred == label)
            num_correct_norm += int(pred_norm == label)
            # print(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")
            log_line = f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}"
            print(log_line)
            with open(log_file, "a") as f:
                f.write(log_line + "\n")

            # debug: pretty print a few examples, and the losses in each case
            if num_total < 10:
                print("---")
                print(f"Context:\n {example['ctx']}")
                print(f"Endings:")
                for i, end in enumerate(example["endings"]):
                    print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
                print(f"predicted: {pred_norm}, actual: {label}")
        
        torch.cuda.empty_cache()
        print('='*50)



if __name__ == "__main__":
    evaluate(saved_models_path='/home/hmrz/Cable/Logs/')