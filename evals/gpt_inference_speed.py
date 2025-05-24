import torch
from dataclasses import dataclass
import tiktoken
import torch.nn.functional as F
import time
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
import sys

sys.path.append('src/')
from model_gpt import Model 


models_to_evaluate = {
    "medium_alibi_fineweb-edu-10B_1_524288_16_1024": "/home/hmrz/Cable/Logs/medium_alibi_fineweb-edu-10B_1_524288_16_1024/model_19072.pt",
    "medium_cable_fineweb-edu-10B_1_524288_16_1024": "/home/hmrz/Cable/Logs/medium_cable_fineweb-edu-10B_1_524288_16_1024/model_19072.pt",
    "medium_fire_fineweb-edu-10B_1_524288_16_1024": "/home/hmrz/Cable/Logs/medium_fire_fineweb-edu-10B_1_524288_16_1024/model_19072.pt",
    "medium_kerple_fineweb-edu-10B_1_524288_16_1024": "/home/hmrz/Cable/Logs/medium_kerple_fineweb-edu-10B_1_524288_16_1024/model_19072.pt",
    "medium_learnable_fineweb-edu-10B_1_524288_16_1024": "/home/hmrz/Cable/Logs/medium_learnable_fineweb-edu-10B_1_524288_16_1024/model_19072.pt",
    "medium_rope_fineweb-edu-10B_1_524288_16_1024": "/home/hmrz/Cable/Logs/medium_rope_fineweb-edu-10B_1_524288_16_1024/model_19072.pt",
    "medium_t5bias_fineweb-edu-10B_1_524288_16_1024": "/home/hmrz/Cable/Logs/medium_t5bias_fineweb-edu-10B_1_524288_16_1024/model_19072.pt",
    "medium_sinusoidal_fineweb-edu-10B_1_524288_16_1024": "/home/hmrz/Cable/Logs/medium_sinusoidal_fineweb-edu-10B_1_524288_16_1024/model_19072.pt",
}


@dataclass
class ModelConfig:
    pos_method: str = None
    use_dape: bool = False
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 1024


device = 'cuda' if torch.cuda.is_available() else 'cpu'
enc = tiktoken.get_encoding("gpt2")
csv_path = "evals/tps_results.csv"


with open(csv_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["model", "seq_len", "tokens_per_second"])


for model_name, model_path in models_to_evaluate.items():
    print(f"Evaluating {model_name}...")
    pos_method = model_name.split('_')[1]
    trained_seq_len = int(model_name.split('_')[-1])

    config = ModelConfig(pos_method=pos_method, block_size=trained_seq_len)
    if 'tiny' in model_name:
        config.n_layer, config.n_head, config.n_embd = 6, 8, 512
    elif 'small' in model_name:
        config.n_layer, config.n_head, config.n_embd = 12, 12, 768
    elif 'medium' in model_name:
        config.n_layer, config.n_head, config.n_embd = 24, 16, 1024

    model = Model(config)
    state_dict = torch.load(model_path)
    for k in [k for k in state_dict if "cached_" in k]:
        del state_dict[k]
    model.load_state_dict(state_dict)
    model.eval().to(device)

    # Inference
    num_return_sequences = 4
    max_length = 512
    prompt = "Hello, I'm a language model,"
    tokens = enc.encode(prompt)
    input_len = len(tokens)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device).manual_seed(42)

    start_time = time.time()
    while xgen.size(1) < max_length:
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.float32):
                logits, _ = model(xgen)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
            xcol = torch.gather(topk_indices, -1, ix)
            xgen = torch.cat((xgen, xcol), dim=1)
    end_time = time.time()

    # Compute TPS
    total_time = end_time - start_time
    tokens_generated_per_seq = max_length - input_len
    total_generated_tokens = num_return_sequences * tokens_generated_per_seq
    tokens_per_second = total_generated_tokens / total_time
    print(f"TPS: {tokens_per_second:.2f}")

    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([model_name, trained_seq_len, tokens_per_second])

