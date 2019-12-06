from __future__ import absolute_import, division, print_function

import os
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchcrf import CRF

class GloveLSTMCRF(nn.Module):
    def __init__(self, config, embedding_path, label_path, emb_non_trainable=False, use_crf=True):
        super(GloveLSTMCRF, self).__init__()

        seq_size = config['n_ctx']
        token_emb_dim = config['token_emb_dim']
        lstm_hidden_dim = config['lstm_hidden_dim']
        lstm_num_layers = config['lstm_num_layers']
        lstm_dropout = config['lstm_dropout']

        # glove embedding layer
        weights_matrix = self.__load_embedding(embedding_path)
        self.embed = self.__create_embedding_layer(weights_matrix, non_trainable=emb_non_trainable)

        self.dropout = nn.Dropout(config['dropout'])

        # BiLSTM layer
        self.lstm = nn.LSTM(input_size=token_emb_dim,
                            hidden_size=lstm_hidden_dim,
                            num_layers=lstm_num_layers,
                            dropout=lstm_dropout,
                            bidirectional=True,
                            batch_first=True)

        # projection layer
        self.labels = self.__load_label(label_path)
        self.label_size = len(self.labels)
        self.linear = nn.Linear(lstm_hidden_dim*2, self.label_size)

        # CRF layer
        self.use_crf = use_crf
        if self.use_crf:
            self.crf = CRF(num_tags=self.label_size, batch_first=True)

    def __load_embedding(self, input_path):
        weights_matrix = np.load(input_path)
        weights_matrix = torch.tensor(weights_matrix)
        return weights_matrix

    def __create_embedding_layer(self, weights_matrix, non_trainable=False):
        vocab_size, emb_dim = weights_matrix.size()
        emb_layer = nn.Embedding(vocab_size, emb_dim)
        emb_layer.load_state_dict({'weight': weights_matrix})
        if non_trainable:
            emb_layer.weight.requires_grad = False
        return emb_layer

    def __load_label(self, input_path):
        labels = {}
        with open(input_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                toks = line.strip().split()
                label = toks[0]
                label_id = int(toks[1])
                labels[label_id] = label
        return labels

    def forward(self, x, tags=None):
        # x : [batch_size, seq_size]
        # tags : [batch_size, seq_size]
        embed_out = self.embed(x)
        # embed_out : [batch_size, seq_size, token_emb_dim]

        lstm_out, (h_n, c_n) = self.lstm(embed_out)
        # lstm_out : [batch_size, seq_size, lstm_hidden_dim*2]

        lstm_out = self.dropout(lstm_out)

        logits = self.linear(lstm_out)
        # logits : [batch_size, seq_size, label_size]
     
        if not self.use_crf: return logits

        if tags is not None: # given golden ys(answer)
            log_likelihood = self.crf(logits, tags, mask=None, reduction='mean')
            prediction = self.crf.decode(logits)
            # prediction : [batch_size, seq_size]
            return logits, log_likelihood, prediction
        else:
            prediction = self.crf.decode(logits)
            return logits, prediction

