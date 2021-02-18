from __future__ import absolute_import, division, print_function

import os
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from torchcrf import CRF

class BaseModel(nn.Module):
    def __init__(self, config=None):
        super(BaseModel, self).__init__()
        if config and hasattr(config['opt'], 'seed'):
            self.set_seed(config['opt'])

    def set_seed(self, opt):
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)

    def load_embedding(self, input_path):
        weights_matrix = np.load(input_path)
        weights_matrix = torch.as_tensor(weights_matrix)
        return weights_matrix

    def create_embedding_layer(self, vocab_dim, emb_dim, weights_matrix=None, non_trainable=True, padding_idx=0):
        emb_layer = nn.Embedding(vocab_dim, emb_dim, padding_idx=padding_idx)
        if torch.is_tensor(weights_matrix):
            emb_layer.load_state_dict({'weight': weights_matrix})
        if non_trainable:
            emb_layer.weight.requires_grad = False
        return emb_layer

    def load_dict(self, input_path):
        dic = {}
        with open(input_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                toks = line.strip().split()
                _key = toks[0]
                _id = int(toks[1])
                dic[_id] = _key
        return dic

    def forward(self, x):
        return x

class TextCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes):
        super(TextCNN, self).__init__()
        convs = []
        for ks in kernel_sizes:
            convs.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=ks))
        self.convs = nn.ModuleList(convs)

    def forward(self, x):
        # x : [batch_size, seq_size, emb_dim]
        # num_filters == out_channels
        x = x.permute(0, 2, 1)
        # x : [batch_size, emb_dim, seq_size]
        conved = [F.relu(conv(x)) for conv in self.convs]
        # conved : [ [batch_size, num_filters, *], [batch_size, num_filters, *], [batch_size, num_filters, *] ]
        # for ONNX conversion, do not use F.max_pool1d(),
        pooled = [torch.max(cv, dim=2)[0] for cv in conved]
        # pooled : [ [batch_size, num_filters], [batch_size, num_filters], [batch_size, num_filters] ]
        cat = torch.cat(pooled, dim = 1)
        # cat : [batch_size, len(kernel_sizes) * num_filters]
        return cat

class DenseNet(nn.Module):
    def __init__(self, densenet_kernels, emb_dim, first_num_filters, num_filters, last_num_filters, activation=F.relu):
        super(DenseNet, self).__init__()
        self.activation = activation
        self.densenet_kernels = densenet_kernels
        self.densenet_width = len(densenet_kernels[0])
        self.densenet_block = []
        for i, kss in enumerate(self.densenet_kernels): # densenet depth
            if i == 0:
                in_channels = emb_dim
                out_channels = first_num_filters
            else:
                in_channels = first_num_filters + num_filters * (i-1)
                out_channels = num_filters
            convs = []
            for j, ks in enumerate(kss):                # densenet width
                padding = (ks - 1)//2
                conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=ks, padding=padding)
                convs.append(conv)
            convs = nn.ModuleList(convs)
            self.densenet_block.append(convs)
        self.densenet_block = nn.ModuleList(self.densenet_block)
        ks = 1
        in_channels = emb_dim + num_filters * self.densenet_width
        out_channels = last_num_filters
        padding = (ks - 1)//2
        self.conv_last = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=ks, padding=padding)
        self.last_dim = last_num_filters

    def forward(self, x, mask):
        # x     : [batch_size, seq_size, emb_dim]
        # mask  : [batch_size, seq_size]
        x = x.permute(0, 2, 1)
        # x     : [batch_size, emb_dim, seq_size]
        masks = mask.unsqueeze(2).to(torch.float)
        # masks : [batch_size, seq_size, 1]
        masks = masks.permute(0, 2, 1)
        # masks : [batch_size, 1, seq_size]

        merge_list = []
        for j in range(self.densenet_width):
            conv_results = []
            for i, kss in enumerate(self.densenet_kernels):
                if i == 0: conv_in = x
                else: conv_in  = torch.cat(conv_results, dim=-2)
                conv_out = self.densenet_block[i][j](conv_in)
                # conv_out first : [batch_size, first_num_filters, seq_size]
                # conv_out other : [batch_size, num_filters, seq_size]
                conv_out *= masks # masking, auto broadcasting along with second dimension
                conv_out = self.activation(conv_out)
                conv_results.append(conv_out)
            merge_list.append(conv_results[-1]) # last one only

        conv_last = self.conv_last(torch.cat([x] + merge_list, dim=-2))
        conv_last *= masks
        conv_last = F.relu(conv_last)
        # conv_last : [batch_size, last_num_filters, seq_size]
        conv_last = conv_last.permute(0, 2, 1)
        # conv_last : [batch_size, seq_size, last_num_filters]
        return conv_last

class DSA(nn.Module):
    def __init__(self, config, dsa_num_attentions, dsa_input_dim, dsa_dim, dsa_r=3):
        super(DSA, self).__init__()
        self.config = config
        self.device = config['opt'].device
        dsa = []
        for i in range(dsa_num_attentions):
            dsa.append(nn.Linear(dsa_input_dim, dsa_dim))
        self.dsa = nn.ModuleList(dsa)
        self.dsa_r = dsa_r # r iterations
        self.last_dim = dsa_num_attentions * dsa_dim

    def __self_attention(self, x, mask, r=3):
        # x    : [batch_size, seq_size, dsa_dim]
        # mask : [batch_size, seq_size]
        # r    : r iterations
        # initialize
        mask = mask.to(torch.float)
        inv_mask = mask.eq(0.0)
        # inv_mask : [batch_size, seq_size], ex) [False, ..., False, True, ..., True]
        softmax_mask = mask.masked_fill(inv_mask, -1e20)
        # softmax_mask : [batch_size, seq_size], ex) [1., 1., 1.,  ..., -1e20, -1e20, -1e20] 
        q = torch.zeros(mask.shape[0], mask.shape[-1], requires_grad=False).to(torch.float).to(self.device)
        # q : [batch_size, seq_size]
        z_list = []
        # iterative computing attention
        for idx in range(r):
            # softmax masking
            q *= softmax_mask
            # attention weights
            a = torch.softmax(q.detach().clone(), dim=-1) # preventing from unreachable variable at gradient computation. 
            # a : [batch_size, seq_size]
            a *= mask
            a = a.unsqueeze(2)
            # a : [batch_size, seq_size, 1]
            # element-wise multiplication(broadcasting) and summation along 1 dim
            s = (a * x).sum(1)
            # s : [batch_size, dsa_dim]
            z = torch.tanh(s)
            # z : [batch_size, dsa_dim]
            z_list.append(z)
            # update q
            m = z.unsqueeze(2)
            # m : [batch_size, dsa_dim, 1]
            q += torch.matmul(x, m).squeeze(2)
            # q : [batch_size, seq_size]
        return z_list[-1]

    def forward(self, x, mask):
        # x     : [batch_size, seq_size, dsa_input_dim]
        # mak   : [batch_size, seq_size]
        z_list = []
        for p in self.dsa: # dsa_num_attentions
            # projection to dsa_dim
            p_out = F.leaky_relu(p(x))
            # p_out : [batch_size, seq_size, dsa_dim]
            z_j = self.__self_attention(p_out, mask, r=self.dsa_r)
            # z_j : [batch_size, dsa_dim]
            z_list.append(z_j)
        z = torch.cat(z_list, dim=-1)
        # z : [batch_size, dsa_num_attentions * dsa_dim]
        return z

class CharCNN(BaseModel):
    def __init__(self, config):
        super().__init__(config=config)
        self.config = config
        self.device = config['opt'].device
        self.seq_size = config['n_ctx']
        self.char_n_ctx = config['char_n_ctx']
        char_vocab_size = config['char_vocab_size']
        self.char_emb_dim = config['char_emb_dim']
        char_num_filters = config['char_num_filters']
        char_kernel_sizes = config['char_kernel_sizes']

        self.char_padding_idx = config['char_padding_idx']
        self.embed_char = super().create_embedding_layer(char_vocab_size, self.char_emb_dim, weights_matrix=None, non_trainable=False, padding_idx=self.char_padding_idx)
        self.textcnn = TextCNN(self.char_emb_dim, char_num_filters, char_kernel_sizes)
        self.last_dim = len(char_kernel_sizes) * char_num_filters

    def forward(self, x):
        # x : [batch_size, seq_size, char_n_ctx]

        char_ids = x
        # char_ids : [batch_size, seq_size, char_n_ctx]
        mask = char_ids.view(-1, self.char_n_ctx).ne(self.char_padding_idx) # broadcasting
        # mask : [batch_size*seq_size, char_n_ctx]
        mask = mask.unsqueeze(2).to(torch.float)
        # mask : [batch_size*seq_size, char_n_ctx, 1]

        char_embed_out = self.embed_char(char_ids)
        # char_embed_out : [batch_size, seq_size, char_n_ctx, char_emb_dim]
        char_embed_out = char_embed_out.view(-1, self.char_n_ctx, self.char_emb_dim)
        # char_embed_out : [batch_size*seq_size, char_n_ctx, char_emb_dim]
        char_embed_out *= mask # masking, auto-broadcasting

        charcnn_out = self.textcnn(char_embed_out)
        # charcnn_out : [batch_size*seq_size, last_dim]
        charcnn_out = charcnn_out.view(-1, self.seq_size, charcnn_out.shape[-1])
        # charcnn_out : [batch_size, seq_size, last_dim]
        return charcnn_out

class GloveLSTMCRF(BaseModel):
    def __init__(self, config, embedding_path, label_path, pos_path, emb_non_trainable=True, use_crf=False, use_char_cnn=False, use_mha=False):
        super().__init__(config=config)

        self.config = config
        self.device = config['opt'].device
        self.seq_size = config['n_ctx']
        pos_emb_dim = config['pos_emb_dim']
        lstm_hidden_dim = config['lstm_hidden_dim']
        lstm_num_layers = config['lstm_num_layers']
        lstm_dropout = config['lstm_dropout']
        self.use_char_cnn = use_char_cnn
        self.use_crf = use_crf
        self.use_mha = use_mha
        mha_num_attentions = config['mha_num_attentions']

        # glove embedding layer
        weights_matrix = super().load_embedding(embedding_path)
        vocab_dim, token_emb_dim = weights_matrix.size()
        padding_idx = config['pad_token_id']
        self.embed_token = super().create_embedding_layer(vocab_dim, token_emb_dim, weights_matrix=weights_matrix, non_trainable=emb_non_trainable, padding_idx=padding_idx)

        # pos embedding layer
        self.poss = super().load_dict(pos_path)
        self.pos_vocab_size = len(self.poss)
        padding_idx = config['pad_pos_id']
        self.embed_pos = super().create_embedding_layer(self.pos_vocab_size, pos_emb_dim, weights_matrix=None, non_trainable=False, padding_idx=padding_idx)

        emb_dim = token_emb_dim + pos_emb_dim
        # char embedding layer
        if self.use_char_cnn:
            self.charcnn = CharCNN(config)
            emb_dim = token_emb_dim + pos_emb_dim + self.charcnn.last_dim

        # BiLSTM layer
        self.lstm = nn.LSTM(input_size=emb_dim,
                            hidden_size=lstm_hidden_dim,
                            num_layers=lstm_num_layers,
                            dropout=lstm_dropout,
                            bidirectional=True,
                            batch_first=True)
        self.lstm_dim = lstm_hidden_dim*2

        self.dropout = nn.Dropout(config['dropout'])

        # Multi-Head Attention layer
        self.mha_dim = self.lstm_dim
        if self.use_mha:
            self.mha = nn.MultiheadAttention(self.lstm_dim, num_heads=mha_num_attentions)
            # self.layernorm_mha = nn.LayerNorm(self.mha_dim)

        # projection layer
        self.labels = super().load_dict(label_path)
        self.label_size = len(self.labels)
        self.linear = nn.Linear(self.mha_dim, self.label_size)

        # CRF layer
        if self.use_crf:
            self.crf = CRF(num_tags=self.label_size, batch_first=True)

    def forward(self, x):
        # x[0, 1] : [batch_size, seq_size]
        # x[2]    : [batch_size, seq_size, char_n_ctx]
        token_ids = x[0]
        pos_ids = x[1]

        mask = torch.sign(torch.abs(token_ids)).to(torch.uint8).to(self.device)
        # mask : [batch_size, seq_size]
        lengths = torch.sum(mask.to(torch.long), dim=1)
        # lengths : [batch_size]

        # 1. Embedding
        token_embed_out = self.embed_token(token_ids)
        # token_embed_out : [batch_size, seq_size, token_emb_dim]
        pos_embed_out = self.embed_pos(pos_ids)
        # pos_embed_out : [batch_size, seq_size, pos_emb_dim]
        if self.use_char_cnn:
            char_ids = x[2]
            # char_ids : [batch_size, seq_size, char_n_ctx]
            charcnn_out = self.charcnn(char_ids)
            # charcnn_out : [batch_size, seq_size, self.charcnn.last_dim]
            embed_out = torch.cat([token_embed_out, pos_embed_out, charcnn_out], dim=-1)
            # embed_out : [batch_size, seq_size, emb_dim]
        else:
            embed_out = torch.cat([token_embed_out, pos_embed_out], dim=-1)
            # embed_out : [batch_size, seq_size, emb_dim]
        embed_out = self.dropout(embed_out)

        # 2. LSTM
        # FIXME : pytorch 1.7.0 bug https://github.com/pytorch/pytorch/issues/43227 , lengths.cpu()
        packed_embed_out = torch.nn.utils.rnn.pack_padded_sequence(embed_out, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, (h_n, c_n) = self.lstm(packed_embed_out)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=self.seq_size)
        # lstm_out : [batch_size, seq_size, self.lstm_dim == lstm_hidden_dim*2]
        lstm_out = self.dropout(lstm_out)

        # 3. MHA
        if self.use_mha:
            # reference : https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
            #             https://discuss.pytorch.org/t/how-to-add-padding-mask-to-nn-transformerencoder-module/63390/3
            query = lstm_out.permute(1, 0, 2)
            # query : [seq_size, batch_size, self.lstm_dim]
            key = query
            value = query
            key_padding_mask = mask.ne(1) # attention_mask => mask = [[1, 1, ..., 0, ...]] => [[False, False, ..., True, ...]]
            attn_output, attn_output_weights = self.mha(query, key, value, key_padding_mask=key_padding_mask)
            # attn_output : [seq_size, batch_size, self.mha_dim]
            mha_out = attn_output.permute(1, 0, 2)
            # mha_out : [batch_size, seq_size, self.mha_dim]
            # residual, layernorm, dropout
            # mha_out = self.layernorm_mha(mha_out + lstm_out)
            mha_out = self.dropout(mha_out)
        else:
            mha_out = lstm_out
            # mha_out : [batch_size, seq_size, self.mha_dim]

        # 4. Output
        logits = self.linear(mha_out)
        # logits : [batch_size, seq_size, label_size]
        if not self.use_crf: return logits
        prediction = self.crf.decode(logits)
        prediction = torch.as_tensor(prediction, dtype=torch.long)
        # prediction : [batch_size, seq_size]
        return logits, prediction

class GloveDensenetCRF(BaseModel):
    def __init__(self, config, embedding_path, label_path, pos_path, emb_non_trainable=True, use_crf=False, use_char_cnn=False, use_mha=False):
        super().__init__(config=config)

        self.config = config
        self.device = config['opt'].device
        self.seq_size = config['n_ctx']
        pos_emb_dim = config['pos_emb_dim']
        self.use_crf = use_crf
        self.use_char_cnn = use_char_cnn
        self.use_mha = use_mha
        mha_num_attentions = config['mha_num_attentions']

        # glove embedding layer
        weights_matrix = super().load_embedding(embedding_path)
        vocab_dim, token_emb_dim = weights_matrix.size()
        padding_idx = config['pad_token_id']
        self.embed_token = super().create_embedding_layer(vocab_dim, token_emb_dim, weights_matrix=weights_matrix, non_trainable=emb_non_trainable, padding_idx=padding_idx)

        # pos embedding layer
        self.poss = super().load_dict(pos_path)
        self.pos_vocab_size = len(self.poss)
        padding_idx = config['pad_pos_id']
        self.embed_pos = super().create_embedding_layer(self.pos_vocab_size, pos_emb_dim, weights_matrix=None, non_trainable=False, padding_idx=padding_idx)

        emb_dim = token_emb_dim + pos_emb_dim
        # char embedding layer
        if self.use_char_cnn:
            self.charcnn = CharCNN(config)
            emb_dim = token_emb_dim + pos_emb_dim + self.charcnn.last_dim

        # Densenet layer
        densenet_kernels = config['densenet_kernels']
        first_num_filters = config['densenet_first_num_filters']
        num_filters = config['densenet_num_filters']
        last_num_filters = config['densenet_last_num_filters']
        self.densenet = DenseNet(densenet_kernels, emb_dim, first_num_filters, num_filters, last_num_filters, activation=F.relu)
        self.layernorm_densenet = nn.LayerNorm(self.densenet.last_dim)

        self.dropout = nn.Dropout(config['dropout'])

        # Multi-Head Attention layer
        self.mha_dim = self.densenet.last_dim
        if self.use_mha:
            self.mha = nn.MultiheadAttention(self.densenet.last_dim, num_heads=mha_num_attentions)

        # projection layer
        self.labels = super().load_dict(label_path)
        self.label_size = len(self.labels)
        self.linear = nn.Linear(self.mha_dim, self.label_size)

        # CRF layer
        if self.use_crf:
            self.crf = CRF(num_tags=self.label_size, batch_first=True)

    def forward(self, x):
        # x[0, 1] : [batch_size, seq_size]
        # x[2]    : [batch_size, seq_size, char_n_ctx]
        token_ids = x[0]
        pos_ids = x[1]

        mask = torch.sign(torch.abs(token_ids)).to(torch.uint8).to(self.device)
        # mask : [batch_size, seq_size]

        # 1. Embedding
        token_embed_out = self.embed_token(token_ids)
        # token_embed_out : [batch_size, seq_size, token_emb_dim]
        pos_embed_out = self.embed_pos(pos_ids)
        # pos_embed_out   : [batch_size, seq_size, pos_emb_dim]
        if self.use_char_cnn:
            char_ids = x[2]
            # char_ids : [batch_size, seq_size, char_n_ctx]
            charcnn_out = self.charcnn(char_ids)
            # charcnn_out : [batch_size, seq_size, self.charcnn.last_dim]
            embed_out = torch.cat([token_embed_out, pos_embed_out, charcnn_out], dim=-1)
            # embed_out : [batch_size, seq_size, emb_dim]
        else:
            embed_out = torch.cat([token_embed_out, pos_embed_out], dim=-1)
            # embed_out : [batch_size, seq_size, emb_dim]
        embed_out = self.dropout(embed_out)

        # 2. DenseNet
        densenet_out = self.densenet(embed_out, mask)
        # densenet_out : [batch_size, seq_size, self.densenet.last_dim == last_num_filters]
        densenet_out = self.layernorm_densenet(densenet_out)
        densenet_out = self.dropout(densenet_out)

        # 3. MHA
        if self.use_mha:
            # reference : https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
            #             https://discuss.pytorch.org/t/how-to-add-padding-mask-to-nn-transformerencoder-module/63390/3
            query = densenet_out.permute(1, 0, 2)
            # query : [seq_size, batch_size, self.densenet.last_dim]
            key = query
            value = query
            key_padding_mask = mask.ne(1) # attention_mask => mask = [[1, 1, ..., 0, ...]] => [[False, False, ..., True, ...]]
            attn_output, attn_output_weights = self.mha(query, key, value, key_padding_mask=key_padding_mask)
            # attn_output : [seq_size, batch_size, self.mha_dim]
            mha_out = attn_output.permute(1, 0, 2)
            # mha_out : [batch_size, seq_size, self.mha_dim]
            mha_out = self.dropout(mha_out)
        else:
            mha_out = lstm_out
            # mha_out : [batch_size, seq_size, self.mha_dim]

        # 4. Output
        logits = self.linear(mha_out)
        # logits : [batch_size, seq_size, label_size]
        if not self.use_crf: return logits
        prediction = self.crf.decode(logits)
        prediction = torch.as_tensor(prediction, dtype=torch.long)
        # prediction : [batch_size, seq_size]
        return logits, prediction

class BertLSTMCRF(BaseModel):
    def __init__(self, config, bert_config, bert_model, bert_tokenizer, label_path, pos_path, use_crf=False, use_pos=False, use_mha=False, disable_lstm=False, feature_based=False):
        super().__init__(config=config)

        self.config = config
        self.device = config['opt'].device
        self.seq_size = config['n_ctx']
        pos_emb_dim = config['pos_emb_dim']
        lstm_hidden_dim = config['lstm_hidden_dim']
        lstm_num_layers = config['lstm_num_layers']
        lstm_dropout = config['lstm_dropout']
        self.use_crf = use_crf
        self.use_pos = use_pos
        self.use_mha = use_mha
        mha_num_attentions = config['mha_num_attentions']
        self.disable_lstm = disable_lstm

        # bert embedding layer
        self.bert_config = bert_config
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
        self.bert_feature_based = feature_based
        self.bert_hidden_size = bert_config.hidden_size
        self.bert_num_layers = bert_config.num_hidden_layers

        # DSA layer for bert_feature_based
        dsa_num_attentions = config['dsa_num_attentions']
        dsa_input_dim = self.bert_hidden_size
        dsa_dim = config['dsa_dim']
        dsa_r = config['dsa_r']
        self.dsa = DSA(config, dsa_num_attentions, dsa_input_dim, dsa_dim, dsa_r=dsa_r)
        self.layernorm_dsa = nn.LayerNorm(self.dsa.last_dim)

        bert_emb_dim = self.bert_hidden_size
        if self.bert_feature_based:
            '''
            # 1) last layer, 2) mean pooling
            bert_emb_dim = self.bert_hidden_size
            '''
            # 3) DSA pooling
            bert_emb_dim = self.dsa.last_dim

        # pos embedding layer
        self.poss = super().load_dict(pos_path)
        self.pos_vocab_size = len(self.poss)
        padding_idx = config['pad_pos_id']
        self.embed_pos = super().create_embedding_layer(self.pos_vocab_size, pos_emb_dim, weights_matrix=None, non_trainable=False, padding_idx=padding_idx)
        if self.use_pos:
            emb_dim = bert_emb_dim + pos_emb_dim
        else:
            emb_dim = bert_emb_dim

        # BiLSTM layer
        self.lstm_dim = emb_dim
        if not self.disable_lstm:
            self.lstm = nn.LSTM(input_size=emb_dim,
                                hidden_size=lstm_hidden_dim,
                                num_layers=lstm_num_layers,
                                dropout=lstm_dropout,
                                bidirectional=True,
                                batch_first=True)
            self.lstm_dim = lstm_hidden_dim*2

        self.dropout = nn.Dropout(config['dropout'])

        # Multi-Head Attention layer
        self.mha_dim = self.lstm_dim 
        if self.use_mha:
            self.mha = nn.MultiheadAttention(self.lstm_dim, num_heads=mha_num_attentions)

        # projection layer
        self.labels = super().load_dict(label_path)
        self.label_size = len(self.labels)
        self.linear = nn.Linear(self.mha_dim, self.label_size)

        # CRF layer
        if self.use_crf:
            self.crf = CRF(num_tags=self.label_size, batch_first=True)

    def _compute_bert_embedding(self, x, head_mask=None):
        params = {
            'input_ids': x[0],
            'attention_mask': x[1],
            'output_hidden_states': True,
            'output_attentions': True,
            'return_dict': True
        }
        if self.bert_model.config.model_type not in ['bart', 'distilbert']:
            params['token_type_ids'] = None if self.bert_model.config.model_type in ['roberta'] else x[2] # RoBERTa don't use segment_ids
        if head_mask is not None:
            params['head_mask'] = head_mask
        if self.bert_feature_based:
            # feature-based
            with torch.no_grad():
                bert_outputs = self.bert_model(**params)
                if self.bert_model.config.model_type in ['bart']:
                    all_hidden_states = bert_outputs.decoder_hidden_states
                else:
                    all_hidden_states = bert_outputs.hidden_states
                '''
                # 1) last layer
                embedded = bert_outputs.last_hidden_state
                # embedded : [batch_size, seq_size, bert_hidden_size]
                '''
                '''
                # 2) mean pooling
                stack = torch.stack(all_hidden_states, dim=-1)
                embedded = torch.mean(stack, dim=-1)
                # ([batch_size, seq_size, bert_hidden_size], ..., [batch_size, seq_size, bert_hidden_size])
                # -> stack(-1) -> [batch_size, seq_size, bert_hidden_size, *], ex) * == 25 for bert large
                # -> max/mean(-1) ->  [batch_size, seq_size, bert_hidden_size]
                '''
                # 3) DSA pooling
                stack = torch.stack(all_hidden_states, dim=-2)
                # stack : [batch_size, seq_size, *, bert_hidden_size]
                stack = stack.view(-1, self.bert_num_layers + 1, self.bert_hidden_size)
                # stack : [*, bert_num_layers, bert_hidden_size]
                dsa_mask = torch.ones(stack.shape[0], stack.shape[1]).to(self.device)
                # dsa_mask : [*, bert_num_layers]
                dsa_out = self.dsa(stack, dsa_mask)
                # dsa_out : [*, self.dsa.last_dim]
                dsa_out = self.layernorm_dsa(dsa_out)
                embedded = dsa_out.view(-1, self.seq_size, self.dsa.last_dim)
                # embedded : [batch_size, seq_size, self.dsa.last_dim]
        else:
            # fine-tuning
            # x[0], x[1], x[2] : [batch_size, seq_size]
            bert_outputs = self.bert_model(**params)
            embedded = bert_outputs.last_hidden_state
            # embedded : [batch_size, seq_size, bert_hidden_size]
        return embedded, bert_outputs

    def forward(self, x, head_mask=None, freeze_bert=False):
        # x[0,1,2] : [batch_size, seq_size]

        mask = x[1].to(torch.uint8).to(self.device)
        # mask == attention_mask : [batch_size, seq_size]
        lengths = torch.sum(mask.to(torch.long), dim=1)
        # lengths : [batch_size]

        # 1. Embedding
        if freeze_bert:
            # freeze_bert is the runtime option which has the same effect to the static option `feature_based`.
            with torch.no_grad():
                bert_embed_out, bert_outputs = self._compute_bert_embedding(x, head_mask=head_mask)
        else:
            bert_embed_out, bert_outputs = self._compute_bert_embedding(x, head_mask=head_mask)
        # bert_embed_out : [batch_size, seq_size, *]
        pos_ids = x[3]
        pos_embed_out = self.embed_pos(pos_ids)
        # pos_embed_out : [batch_size, seq_size, pos_emb_dim]
        if self.use_pos:
            embed_out = torch.cat([bert_embed_out, pos_embed_out], dim=-1)
        else:
            embed_out = bert_embed_out
        # embed_out : [batch_size, seq_size, emb_dim]
        embed_out = self.dropout(embed_out)

        # 2. LSTM
        if not self.disable_lstm:
            # FIXME : pytorch 1.7.0 bug https://github.com/pytorch/pytorch/issues/43227 , lengths.cpu()
            packed_embed_out = torch.nn.utils.rnn.pack_padded_sequence(embed_out, lengths.cpu(), batch_first=True, enforce_sorted=False)
            lstm_out, (h_n, c_n) = self.lstm(packed_embed_out)
            lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=self.seq_size)
            # lstm_out : [batch_size, seq_size, self.lstm_dim == lstm_hidden_dim*2]
            lstm_out = self.dropout(lstm_out)
        else:
            lstm_out = embed_out
            # lstm_out : [batch_size, seq_size, self.lstm_dim == emb_dim]

        # 3. MHA
        if self.use_mha:
            # reference : https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
            #             https://discuss.pytorch.org/t/how-to-add-padding-mask-to-nn-transformerencoder-module/63390/3
            query = lstm_out.permute(1, 0, 2)
            # query : [seq_size, batch_size, self.lstm_dim]
            key = query
            value = query
            key_padding_mask = mask.ne(1) # attention_mask => mask = [[1, 1, ..., 0, ...]] => [[False, False, ..., True, ...]]
            attn_output, attn_output_weights = self.mha(query, key, value, key_padding_mask=key_padding_mask)
            # attn_output : [seq_size, batch_size, self.mha_dim]
            mha_out = attn_output.permute(1, 0, 2)
            # mha_out : [batch_size, seq_size, self.mha_dim]
            mha_out = self.dropout(mha_out)
        else:
            mha_out = lstm_out
            # mha_out : [batch_size, seq_size, self.mha_dim]

        # 4. Output
        logits = self.linear(mha_out)
        # logits : [batch_size, seq_size, label_size]
        if not self.use_crf: return logits
        prediction = self.crf.decode(logits)
        prediction = torch.as_tensor(prediction, dtype=torch.long)
        # prediction : [batch_size, seq_size]
        return logits, prediction

class ElmoLSTMCRF(BaseModel):
    def __init__(self, config, elmo_model, embedding_path, label_path, pos_path, emb_non_trainable=True, use_crf=False, use_char_cnn=False, use_mha=False):
        super().__init__(config=config)

        self.config = config
        self.device = config['opt'].device
        self.seq_size = config['n_ctx']
        pos_emb_dim = config['pos_emb_dim']
        elmo_emb_dim = config['elmo_emb_dim']
        lstm_hidden_dim = config['lstm_hidden_dim']
        lstm_num_layers = config['lstm_num_layers']
        lstm_dropout = config['lstm_dropout']
        self.use_crf = use_crf
        self.use_char_cnn = use_char_cnn
        self.use_mha = use_mha
        mha_num_attentions = config['mha_num_attentions']

        # elmo embedding
        self.elmo_model = elmo_model

        # glove embedding layer
        weights_matrix = super().load_embedding(embedding_path)
        vocab_dim, token_emb_dim = weights_matrix.size()
        padding_idx = config['pad_token_id']
        self.embed_token = super().create_embedding_layer(vocab_dim, token_emb_dim, weights_matrix=weights_matrix, non_trainable=emb_non_trainable, padding_idx=padding_idx)

        # pos embedding layer
        self.poss = super().load_dict(pos_path)
        self.pos_vocab_size = len(self.poss)
        padding_idx = config['pad_pos_id']
        self.embed_pos = super().create_embedding_layer(self.pos_vocab_size, pos_emb_dim, weights_matrix=None, non_trainable=False, padding_idx=padding_idx)

        emb_dim = elmo_emb_dim + token_emb_dim + pos_emb_dim
        # char embedding layer
        if self.use_char_cnn:
            self.charcnn = CharCNN(config)
            emb_dim = elmo_emb_dim + token_emb_dim + pos_emb_dim + self.charcnn.last_dim

        # BiLSTM layer
        self.lstm = nn.LSTM(input_size=emb_dim,
                            hidden_size=lstm_hidden_dim,
                            num_layers=lstm_num_layers,
                            dropout=lstm_dropout,
                            bidirectional=True,
                            batch_first=True)
        self.lstm_dim = lstm_hidden_dim*2

        self.dropout = nn.Dropout(config['dropout'])

        # Multi-Head Attention layer
        self.mha_dim = self.lstm_dim
        if self.use_mha:
            self.mha = nn.MultiheadAttention(self.lstm_dim, num_heads=mha_num_attentions)

        # projection layer
        self.labels = super().load_dict(label_path)
        self.label_size = len(self.labels)
        self.linear = nn.Linear(self.mha_dim, self.label_size)

        # CRF layer
        if self.use_crf:
            self.crf = CRF(num_tags=self.label_size, batch_first=True)

    def forward(self, x):
        # x[0,1] : [batch_size, seq_size]
        # x[2]   : [batch_size, seq_size, max_characters_per_token]
        token_ids = x[0]
        pos_ids = x[1]
        char_ids = x[2]

        mask = torch.sign(torch.abs(token_ids)).to(torch.uint8).to(self.device)
        # mask : [batch_size, seq_size]
        lengths = torch.sum(mask.to(torch.long), dim=1)
        # lengths : [batch_size]

        # 1. Embedding
        elmo_embed_out = self.elmo_model(char_ids)['elmo_representations'][0]
        # elmo_embed_out  : [batch_size, seq_size, elmo_emb_dim]
        '''
        masks = mask.unsqueeze(2).to(torch.float)
        # masks : [batch_size, seq_size, 1]
        elmo_embed_out *= masks # auto-braodcasting
        '''
        token_embed_out = self.embed_token(token_ids)
        # token_embed_out : [batch_size, seq_size, token_emb_dim]
        pos_embed_out = self.embed_pos(pos_ids)
        # pos_embed_out   : [batch_size, seq_size, pos_emb_dim]
        if self.use_char_cnn:
            char_ids = x[2]
            # char_ids : [batch_size, seq_size, char_n_ctx]
            charcnn_out = self.charcnn(char_ids)
            # charcnn_out : [batch_size, seq_size, self.charcnn.last_dim]
            embed_out = torch.cat([elmo_embed_out, token_embed_out, pos_embed_out, charcnn_out], dim=-1)
            # embed_out : [batch_size, seq_size, emb_dim]
        else:
            embed_out = torch.cat([elmo_embed_out, token_embed_out, pos_embed_out], dim=-1)
            # embed_out : [batch_size, seq_size, emb_dim]
        embed_out = self.dropout(embed_out)

        # 2. LSTM
        # FIXME : pytorch 1.7.0 bug https://github.com/pytorch/pytorch/issues/43227 , lengths.cpu()
        packed_embed_out = torch.nn.utils.rnn.pack_padded_sequence(embed_out, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, (h_n, c_n) = self.lstm(packed_embed_out)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=self.seq_size)
        # lstm_out : [batch_size, seq_size, lstm_hidden_dim*2]
        lstm_out = self.dropout(lstm_out)

        # 3. MHA
        if self.use_mha:
            # reference : https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
            #             https://discuss.pytorch.org/t/how-to-add-padding-mask-to-nn-transformerencoder-module/63390/3
            query = lstm_out.permute(1, 0, 2)
            # query : [seq_size, batch_size, self.lstm_dim]
            key = query
            value = query
            key_padding_mask = mask.ne(1) # attention_mask => mask = [[1, 1, ..., 0, ...]] => [[False, False, ..., True, ...]]
            attn_output, attn_output_weights = self.mha(query, key, value, key_padding_mask=key_padding_mask)
            # attn_output : [seq_size, batch_size, self.mha_dim]
            mha_out = attn_output.permute(1, 0, 2)
            # mha_out : [batch_size, seq_size, self.mha_dim]
            mha_out = self.dropout(mha_out)
        else:
            mha_out = lstm_out
            # mha_out : [batch_size, seq_size, self.mha_dim]

        # 4. Output
        logits = self.linear(mha_out)
        # logits : [batch_size, seq_size, label_size]
        if not self.use_crf: return logits
        prediction = self.crf.decode(logits)
        prediction = torch.as_tensor(prediction, dtype=torch.long)
        # prediction : [batch_size, seq_size]
        return logits, prediction

