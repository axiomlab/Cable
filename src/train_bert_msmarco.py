# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0
# https://github.com/AnswerDotAI/ModernBERT/blob/8c57a0f01c12c4953ead53d398a36f81a4ba9e38/examples/train_st.py

import torch.nn as nn
import os
import json
import argparse
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    models 
)
from model_bert import BertModel  
from transformers import AutoConfig, AutoTokenizer
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers




class CustomBERTModule(nn.Module):
    def __init__(self, model_ckpt):
        super().__init__()
        config = AutoConfig.from_pretrained(model_ckpt)
        self.custom_bert = BertModel.from_pretrained(model_ckpt, config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    def tokenize(self, texts):
        """
        Tokenizes a list of texts into input_ids, attention_mask, token_type_ids (if applicable).
        This output is passed to the forward() method later.
        """
        # Handle both single and paired inputs
        if isinstance(texts[0], tuple):
            texts_a, texts_b = zip(*texts)
            encoded = self.tokenizer(
                list(texts_a),
                list(texts_b),
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=True
            )
        else:
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=True
            )
        return encoded

    def forward(self, features):
        """
        `features` is a dict with keys:
            - input_ids
            - token_type_ids (optional)
            - attention_mask (optional)
        Must return a dict with:
            - token_embeddings: [batch_size, seq_len, hidden_size]
            - cls_token_embeddings: [batch_size, hidden_size]
            - attention_mask: [batch_size, seq_len]
        """
        input_ids = features['input_ids']
        attention_mask = features.get('attention_mask', None)
        token_type_ids = features.get('token_type_ids', None)

        sequence_output, pooled_output = self.custom_bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

        features['token_embeddings'] = sequence_output
        features['cls_token_embeddings'] = pooled_output
        features['attention_mask'] = attention_mask
        return features

    def get_word_embedding_dimension(self):
        return self.custom_bert.config.hidden_size
    



def main():
    # parse the lr & model name
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--model_ckpt", type=str)
    p_args = parser.parse_args()

    lr = p_args.lr
    model_name = p_args.model_ckpt
    model_shortname = model_name.split("/")[-1]

    bert_module = CustomBERTModule(p_args.model_ckpt)
    pooling = models.Pooling(
        word_embedding_dimension=bert_module.get_word_embedding_dimension(),
        pooling_mode='mean'  # or 'cls', 'max'
    )

    model = SentenceTransformer(modules=[bert_module, pooling])

    # 2. Load a dataset to finetune on
    dataset = load_dataset(
        "sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1",
        "triplet-hard",
        split="train",
    )
    dataset_dict = dataset.train_test_split(test_size=1_000, seed=12)
    train_dataset = dataset_dict["train"].select(range(1_250_000))
    # train_dataset = dataset_dict["train"].select(range(12500))
    eval_dataset = dataset_dict["test"]

    # 3. Define a loss function
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=128)  # Increase mini_batch_size if you have enough VRAM

    run_name = f"{model_shortname}-DPR-{lr}"
    # 4. (Optional) Specify training arguments
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        # output_dir="ignore_output",  # required dummy path
        # output_dir=f"evals/bert_mldr/finetuned_models/{model_shortname}/{run_name}",
        # Optional training parameters:
        num_train_epochs=1,
        per_device_train_batch_size=512,
        per_device_eval_batch_size=512,
        warmup_ratio=0.05,
        fp16=False,  # Set to False if GPU can't handle FP16
        bf16=True,  # Set to True if GPU supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # (Cached)MultipleNegativesRankingLoss benefits from no duplicates
        learning_rate=lr,
        # Optional tracking/debugging parameters:
        # save_strategy="steps",
        save_strategy="no",
        # save_steps=500,
        # save_total_limit=2,
        # eval_strategy="steps",
        # eval_steps=20,
        logging_steps=10,
        logging_dir=f"evals/bert_mldr/finetuned_models_logs/{run_name}",
        run_name=run_name, 
        report_to="tensorboard",
        do_eval=True,
    )

    # 5. (Optional) Create an evaluator & evaluate the base model
    dev_evaluator = TripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
        name="msmarco-co-condenser-dev",
    )
    dev_evaluator(model)

    # 6. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )
    trainer.train()

    # 7. (Optional) Evaluate the trained model on the evaluator after training
    dev_evaluator(model)

    # 8. Save the model manually + copy tokenizer files
    # TODO: work out manually coping tokenizer files
    model[0].custom_bert.save_pretrained(f"evals/bert_mldr/finetuned_models/{run_name}")
    import shutil
    shutil.copy(f"{p_args.model_ckpt}/special_tokens_map.json", f"evals/bert_mldr/finetuned_models/{run_name}/special_tokens_map.json")
    shutil.copy(f"{p_args.model_ckpt}/tokenizer_config.json", f"evals/bert_mldr/finetuned_models/{run_name}/tokenizer_config.json")
    shutil.copy(f"{p_args.model_ckpt}/vocab.txt", f"evals/bert_mldr/finetuned_models/{run_name}/vocab.txt")


if __name__ == "__main__":
    main()