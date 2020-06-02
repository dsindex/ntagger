from __future__ import absolute_import, division, print_function

import os
import pdb

class Tokenizer():
    def __init__(self, vocab, config):
        self.vocab = vocab
        self.config = config
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

    def tokenize(self, sent):
        tokens = sent.split()
        return tokens

    def convert_tokens_to_ids(self, tokens, pad_sequence=True, min_seq_size=10):
        """
        Args:
          pad_sequence, min_seq_size:
            if pad_sequence is True, pad the sequence up to n_ctx(max_seq_size).
            else do not pad basically. however, since the sequence size should be larger than min_seq_size.
            we pad the sequence additionally.
        """
        ids = []
        vocab = self.vocab
        for token in tokens:
            if self.lowercase: token = token.lower()
            d = vocab[token] if token in vocab else self.unk_id
            ids.append(d)
        if pad_sequence:
            padding_length = self.n_ctx - len(ids)
            if padding_length > 0:
                ids += [self.pad_id] * padding_length
        else:
            padding_length = min_seq_size - len(ids)
            if padding_length > 0:
                ids += [self.pad_id] * padding_length
        ids = ids[:self.n_ctx]
        return ids

    def convert_tokens_to_cids(self, tokens, pad_sequence=True, min_seq_size=10):
        """
        Args:
          pad_sequence, min_seq_size:
            if pad_sequence is True, pad the sequence up to n_ctx(max_seq_size).
            else do not pad basically. however, since the sequence size should be larger than min_seq_size.
            we pad the sequence additionally.
        """
        from allennlp.modules.elmo import batch_to_ids
        pad_cids = [[self.pad_id] * self.char_n_ctx]
        ids = batch_to_ids([tokens])[0].detach().cpu().numpy().tolist()
        # padding
        if pad_sequence:
            padding_length = self.n_ctx - len(ids)
            if padding_length > 0:
                ids += pad_cids * padding_length
        else:
            padding_length = min_seq_size - len(ids)
            if padding_length > 0:
                ids += pad_cids * padding_length
        ids = ids[:self.n_ctx]
        return ids
