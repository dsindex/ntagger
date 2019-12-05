from __future__ import absolute_import, division, print_function

import os
import pdb

_PAD_TOKEN = '<pad>'
_UNK_TOKEN = '<unk>'
_PAD_ID  = 0
_UNK_ID  = 1

class Tokenizer():
    def __init__(self, vocab, config):
        self.vocab = vocab
        self.config = config
        self.pad_token = _PAD_TOKEN
        self.unk_token = _UNK_TOKEN
        self.pad_id = _PAD_ID
        self.unk_id = _UNK_ID

    def tokenize(self, sent):
        """Default white-space tokenizer
        """

        tokens = sent.split()
        return tokens

    def convert_tokens_to_ids(self, tokens):
        ids = []
        vocab = self.vocab
        for token in tokens:
            if self.config['lowercase']: token = token.lower()
            d = vocab[token] if token in vocab else self.unk_id
            ids.append(d)
        return ids
            
    @staticmethod
    def get_pad_token():
        return _PAD_TOKEN

    @staticmethod
    def get_unk_token():
        return _UNK_TOKEN

    @staticmethod
    def get_pad_id():
        return _PAD_ID

    @staticmethod
    def get_unk_id():
        return _UNK_ID

