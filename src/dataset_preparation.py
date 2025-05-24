import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm
import argparse

# ------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataname", type=str, choices=["fineweb-10B", "fineweb-edu-10B", "wikitext-103", "wikitext-2"], required=True)
args = parser.parse_args()

if args.dataname == "fineweb-10B":
    local_dir = "../data/fineweb-10B"
    data_path = "HuggingFaceFW/fineweb"
    sample = "sample-10BT"
    shard_size = int(1e8) # 100M tokens per shard, total of 100 shards
elif args.dataname == "fineweb-edu-10B":
    local_dir = "../data/fineweb-edu-10B"
    data_path = "HuggingFaceFW/fineweb-edu"
    sample = "sample-10BT"
    shard_size = int(1e8) # 100M tokens per shard, total of 100 shards
# TODO: Not sure about wikitext-103 specifications!
elif args.dataname == "wikitext-103":
    local_dir = "../data/wikitext-103"
    data_path = "iohadrubin/wikitext-103-raw-v1"
    sample = None
    shard_size = int(5e7) # 50M tokens per shard, total of 4 shards
elif args.dataname == "wikitext-2":
    local_dir = "../data/wikitext-2"
    data_path = "wikitext"
    sample = "wikitext-2-v1"
    shard_size = int(25e3) # 25K tokens per shard, total of 98 shards
else:
    raise ValueError(f"Unknown dataname {args.dataname}!")

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
fw = load_dataset(data_path, sample, split="train")

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token
def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_count()//2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, fw, chunksize=16):

        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < shard_size:
            # simply append tokens to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"{args.dataname}_{split}_{shard_index:06d}")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

    # write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"{args.dataname}_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])
