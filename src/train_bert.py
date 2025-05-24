import os
import math
import torch
import pickle
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
import datasets
from transformers.utils import ModelOutput
from transformers import DataCollatorForLanguageModeling
from transformers import PreTrainedModel
from transformers import BertTokenizer, Trainer, TrainingArguments, BertConfig
from transformers.modeling_outputs import MaskedLMOutput
from model_bert import BertForMaskedLM
import os
import argparse
from setuptools._distutils.util import strtobool


##################
# use export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" instead of the following
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7" 
##################

parser = argparse.ArgumentParser(description="Cable relative positional encoding training")
parser.add_argument('--model', type=str, default='medium',
                    choices=['large', 'medium', 'small', 'tiny'],
                    help='model size (default: medium)')
parser.add_argument('--pos-method', type=str, default='cable',
                    choices=["cable", "cable5", "cable6", "kcable", "kcable5", "rotali", "fire", "kerple", "alibi", "rope", "t5bias", "sinusoidal", "learnable"],
                    help='positional encoding method (default: cable)')

parser.add_argument('--use-dape', type=strtobool, default=False,
                    help='determine if you want to use dape version of the positional encoding method (default: False)')

parser.add_argument('--dataset-dir', type=str, default='data/fineweb-edu-10B-Nima',
                    help='path to the tokenized dataset (default: fineweb-edu-10B-Nima)')

parser.add_argument('--writer-dir', type=str, default='run_logs_bert/',
                    help='dir for writing tensorboard outputs (default: run_logs_bert/)')

parser.add_argument('--save-dir', type=str, default='Logs_bert/',
                    help='dir for saving logs and checkpoints (default: Logs_bert/)')

parser.add_argument('--use-compile', type=strtobool, default=True, help='using torch.compile')

parser.add_argument('--num-epochs', type=int, default=1, help='Number of epochs for training')

parser.add_argument('--total-batch-size', type=int, default=None, help='Total batch size in tokens (gradient accumulation steps will be calculated as total_batch_size // (batch_size * seq_len * num_gpus))')

parser.add_argument('--batch-size', type=int, default=32, help='Micro batch size per GPU (samples processed in parallel)')

parser.add_argument('--sequence-length', type=int, default=512, help='Sequence length 512 for bert base')

args = parser.parse_args()


# TODO: I think we should make it 2**17. Look at the BERT paper!
if args.total_batch_size is None:
    args.total_batch_size = 2 ** 19

pos_method = args.pos_method + '-dape' if args.use_dape else args.pos_method
run_id = 'medium_' + pos_method + "_fineweb10B" + '_' + str(args.num_epochs) + '_' + str(args.total_batch_size) + '_' + str(args.batch_size) + '_' + str(args.sequence_length)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# print(args.dataset_dir)
ds = datasets.load_from_disk(args.dataset_dir)
train_dataset = ds['train']
test_dataset = ds['test']
# print(test_dataset[0].keys())
train_dataset = train_dataset.remove_columns(['text', 'id', 'dump', 'url', 'date', 'file_path', 'language', 'language_score', 'token_count', 'token_type_ids'] )
test_dataset= test_dataset.remove_columns(['text', 'id', 'dump', 'url', 'date', 'file_path', 'language', 'language_score', 'token_count', 'token_type_ids'] )


# TODO: add more configs
if args.model == 'medium':
    config = BertConfig(  # default config of bert base (~ 110M params, ~ 0.4 GB checkpoint)
        vocab_size=tokenizer.vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        attention_probs_dropout_prob=0.2,
        hidden_dropout_prob=0.2,
        layer_norm_eps=1e-12,
        initializer_range=0.02,
        pad_token_id=tokenizer.pad_token_id
    )
else:
    raise ValueError('Currently only supports Bert Medium!')


# ------ my added configs for compatibilty with pos methods codes
config.pos_method = args.pos_method
config.use_dape = args.use_dape
config.n_embd = config.hidden_size
config.n_head = config.num_attention_heads
config.block_size = config.max_position_embeddings
# ---------------------------------------------------------------

model = BertForMaskedLM(config)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15 
)

total_corpus_token = len(train_dataset) * 512
token_per_update = args.total_batch_size
# TODO: I think we should train it with more epochs. Look at the BERT paper!
step_count = args.num_epochs * (total_corpus_token // token_per_update)
print("total step count: %d" % step_count)
max_len = 512
batch_size = args.batch_size
world_count = torch.cuda.device_count()
grad_accum = token_per_update // (batch_size * world_count * max_len)
assert token_per_update % (batch_size * world_count * max_len) == 0, "error for batch_size"

# def compute_metrics(eval_pred):
#     # Unpack the tuple
#     loss, logits = eval_pred[0] if isinstance(eval_pred[0], tuple) else (None, eval_pred[0])
#     labels = eval_pred[1]

#     # If loss wasn't returned directly from the model, calculate it

#     if loss is None and labels is not None:
#         loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
#         loss = loss_fct(
#             torch.tensor(logits).view(-1, config.vocab_size),
#             torch.tensor(labels).view(-1)
#         ).item()

#     # Ensure it's a Python float, not a tensor
#     if hasattr(loss, 'item'):
#         loss = loss.item()

#     return {
#         "eval_loss": loss,
#     }


# TODO: use custom training instead of huggingface trainer
training_args = TrainingArguments(
    # output_dir='./results%d' % run_id,
    load_best_model_at_end=True,
    eval_strategy="steps",
    eval_steps=5000,
    learning_rate=1e-4,
    adam_epsilon=1e-8,
    max_grad_norm=1.0,
    warmup_steps=10000,
    dataloader_num_workers=4,
    seed=1337,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    adam_beta1=0.9,
    adam_beta2=0.999,
    max_steps=step_count,
    weight_decay=0.01,
    torch_compile=True,
    # torch_compile_backend="eager" if args.use_compile else None,
    # torch_compile_mode="reduce-overhead" if args.use_compile else None,
    logging_dir=args.writer_dir + run_id,
    logging_steps=10,
    save_steps=10000,
    label_names=["labels"],
    save_safetensors=True,
    gradient_accumulation_steps=grad_accum,
    report_to="tensorboard",
    do_eval=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    #compute_metrics=compute_metrics
)
trainer.train()

model.save_pretrained(args.save_dir + run_id)
tokenizer.save_pretrained(args.save_dir + run_id)
# os.environ["HF_TOKEN"] = 'HF_TOKEN'  # replace if wanted
# trainer.push_to_hub()

results = trainer.evaluate()
print("Evaluation results:", results)

