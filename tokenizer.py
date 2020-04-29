from __future__ import absolute_import, division, print_function

import os
import pdb

class Tokenizer():
    def __init__(self, vocab, config, chars=None):
        self.vocab = vocab
        self.config = config
        self.chars  = chars
        self.n_ctx = config['n_ctx']
        if 'char_n_ctx' in config: self.char_n_ctx = config['char_n_ctx']
        self.pad_token = config['pad_token']
        self.unk_token = config['unk_token']
        self.pad_id = config['pad_token_id']
        self.unk_id = config['unk_token_id']
        self.lowercase = False
        if 'lowercase' in config: self.lowercase = config['lowercase']

    def update_vocab(self, vocab):
        self.vocab = vocab

    def update_chars(self, chars):
        self.chars = chars

    def tokenize(self, sent):
        tokens = sent.split()
        return tokens

    def convert_tokens_to_ids(self, tokens):
        ids = []
        vocab = self.vocab
        for token in tokens:
            if self.lowercase: token = token.lower()
            d = vocab[token] if token in vocab else self.unk_id
            ids.append(d)
        # padding
        padding_length = self.n_ctx - len(ids)
        ids += [self.pad_id] * padding_length
        ids = ids[:self.n_ctx]
        return ids

    def convert_tokens_to_cids(self, tokens):
        assert self.chars is not None
        ids = []
        vocab = self.chars
        for token in tokens:
            cids = []
            for ch in token:
                d = vocab[ch] if ch in vocab else self.unk_id
                cids.append(d)
            # padding cids
            padding_length = self.char_n_ctx - len(cids)
            cids += [self.pad_id] * padding_length
            cids = cids[:self.char_n_ctx]
            ids.append(cids)
        # padding
        padding_length = self.n_ctx - len(ids)
        ids += [self.pad_id] * padding_length
        ids = ids[:self.n_ctx]
        return ids
