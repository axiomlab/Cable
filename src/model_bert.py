import os
import math
import torch
import pickle
import time
import numpy as np
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
import datasets
from transformers.utils import ModelOutput
from transformers import DataCollatorForLanguageModeling
from transformers import PreTrainedModel
from transformers.models.bert import BertPreTrainedModel
#from transformers.models.bert.modeling_bert import BertForMaskedLM, BertModel, BertEmbeddings
from transformers import BertTokenizer, Trainer, TrainingArguments, BertConfig
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput
from pos_methods.cable import CABLE, DAPE_CABLE
from pos_methods.cable5 import CABLE5
from pos_methods.alibi import AliBi, DAPE_AliBi
from pos_methods.rope import ROPE
from pos_methods.alibi import AliBi
from pos_methods.kcable import Kernel_CABLE
from pos_methods.kerple import KERPLE, DAPE_KERPLE
from pos_methods.fire import FIRE, DAPE_FIRE
from pos_methods.base import BASE_ATTENTION
from pos_methods.cable6 import CABLE6



# class BidirectionalSelfAttention(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         assert config.hidden_size % config.num_attention_heads == 0
        
#         self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size)
#         self.h_attn = nn.Linear(config.hidden_size, config.num_attention_heads)
        
#         self.c_proj = nn.Linear(config.hidden_size, config.hidden_size)
#         self.c_proj.BERT_SCALE_INIT = 1  # Similar to NANOGPT_SCALE_INIT
        
#         self.n_head = config.num_attention_heads
#         self.max_position_embeddings = config.max_position_embeddings
#         self.dtype = torch.float32  # Use consistent dtype
#         self.hidden_size = config.hidden_size
        
#         self.attn_dropout = nn.Dropout(config.attention_probs_dropout_prob)
#         self.resid_dropout = nn.Dropout(config.hidden_dropout_prob)
        
#     def forward(self, x, attention_mask=None):
#         B, T, C = x.size()  # batch size, sequence length, embedding dimensionality
        
#         qkv = self.c_attn(x)
#         q, k, v = qkv.split(self.hidden_size, dim=2)
#         b = self.h_attn(x).permute(0, 2, 1)  # (B, nh, T)
        
#         k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
#         q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
#         v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        
#         Bias = -1.0 * torch.abs(b.unsqueeze(3) - b.unsqueeze(2))  # (B, nh, T, T))
        
#         if attention_mask is not None:
#             attention_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -10000.0
#             Bias = Bias + attention_mask
        
#         b_shape = Bias.shape
#         bt = Bias.reshape(b_shape[0] * b_shape[1], b_shape[2], b_shape[3])
        
#         q_shape = q.shape
#         qt = q.reshape(q_shape[0] * q_shape[1], q_shape[2], q_shape[3])
#         k_shape = k.shape
#         kt = k.reshape(k_shape[0] * k_shape[1], k_shape[2], k_shape[3])
#         kt = kt.transpose(1, 2)
        
#         att_scores = torch.baddbmm(bt, qt/math.sqrt(C/self.n_head), kt/math.sqrt(C/self.n_head))
#         att_scores = att_scores.reshape(B, self.n_head, T, T)
#         att_scores = att_scores
        
#         att_probs = F.softmax(att_scores, dim=-1)
#         att_probs = self.attn_dropout(att_probs)
        
#         y = att_probs @ v
#         y = y.transpose(1, 2).contiguous().view(B, T, C)
#         y = self.c_proj(y)
#         y = self.resid_dropout(y)
#         return y


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        if config.pos_method == 'learnable':
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        
    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        if self.config.pos_method == 'learnable':
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = words_embeddings + token_type_embeddings + position_embeddings
        elif self.config.pos_method == 'sinusoidal':
            position = torch.arange(seq_length).unsqueeze(1).to(input_ids.device)  # Shape: (max_len, 1)
            div_term = torch.exp(torch.arange(0, self.config.n_embd, 2) * (-math.log(10000.0) / self.config.n_embd)).to(input_ids.device)
            pe = torch.zeros(seq_length, self.config.n_embd).to(input_ids.device)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            embeddings = words_embeddings + token_type_embeddings + pe[:seq_length, :].unsqueeze(0) * 0.1
        else:
            embeddings = words_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings



class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = self._init_attention(config)
        
        self.LayerNorm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.intermediate = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU()
        )
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.LayerNorm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        

    def _init_attention(self, config):
        """Initialize the appropriate attention mechanism based on config"""
        attention_classes = {
            'cable': DAPE_CABLE if config.use_dape else CABLE,
            'cable5': CABLE5,
            'cable6': CABLE6,
            # 'rotali':ROTALI,
            'alibi': DAPE_AliBi if config.use_dape else AliBi,
            'fire': DAPE_FIRE if config.use_dape else FIRE,
            'kerple': DAPE_KERPLE if config.use_dape else KERPLE,
            'learnable': BASE_ATTENTION,
            'sinusoidal': BASE_ATTENTION,
            'rope': ROPE,
            # 't5bias': T5Bias
        }
        if config.pos_method not in attention_classes:
            raise ValueError(f"Unknown position method: {config.pos_method}")
        return attention_classes[config.pos_method](config)
    

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask=attention_mask)
        hidden_states = self.LayerNorm1(hidden_states + attention_output)
        
        intermediate_output = self.intermediate(hidden_states)
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        output = self.LayerNorm2(hidden_states + layer_output)
        return output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
    
    def forward(self, hidden_states, attention_mask=None):
        all_encoder_layers = []
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers



class BertMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        
    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class BertForMaskedLM(BertPreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"
    def __init__(self, config):
        super().__init__(config)
        self.config = config  # Make sure to store the config

        self.bert = BertModel(config)
        self.mlm_head = BertMLMHead(config)

        # Initialize weights but DON'T tie weights here
        self.init_weights()

    def get_output_embeddings(self):
        return self.mlm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.mlm_head.decoder = new_embeddings

    def get_input_embeddings(self):
        return self.bert.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.bert.set_input_embeddings(value)

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        """
        output_embeddings = self.get_output_embeddings()
        input_embeddings = self.get_input_embeddings()
        output_embeddings.weight = input_embeddings.weight

        if getattr(output_embeddings, "bias", None) is not None:
            output_embeddings.bias.data = nn.functional.pad(
                output_embeddings.bias.data,
                (0, output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0]),
                "constant",
                0,
            )

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
            position_ids=None, labels=None, masked_lm_labels=None):

        if labels is not None and masked_lm_labels is None:
            masked_lm_labels = labels

        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids
        )

        sequence_output = outputs[0]
        prediction_scores = self.mlm_head(sequence_output)

        masked_lm_loss = None
        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)  # -100 index = padding token
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1)
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
        )


class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        
        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh()
        )
    
    # Add these two methods
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
        
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids)
        encoder_outputs = self.encoder(embedding_output, attention_mask)
        sequence_output = encoder_outputs[-1]
        first_token_tensor = sequence_output[:, 0]
        pooled_output = self.pooler(first_token_tensor)
        
        return sequence_output, pooled_output
    


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        # head_mask: Optional[torch.Tensor] = None,
        # inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        # output_attentions: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
        # return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            # head_mask=head_mask,
            # inputs_embeds=inputs_embeds,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

        # return SequenceClassifierOutput(
        #     loss=loss,
        #     logits=logits,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )
    