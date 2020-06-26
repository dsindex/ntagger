#!/usr/bin/env python
#-*- coding: utf8 -*-

import sys
import os
import argparse
from tqdm import tqdm
import logging
import truecase
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_corpus(filename):

    def spill(bucket, data, seq):
        result = {}
        entries = []
        for line in bucket:
            tokens = line.split()
            word = tokens[0]
            tag  = tokens[1]
            chunk = tokens[2]
            label = tokens[3]
            entry = {'word': word, 'tag': tag, 'chunk': chunk, 'label': label}
            entries.append(entry)
        data[seq] = entries
        return True

    data = {}
    bucket = []
    seq = 1
    with open(filename, 'r') as fd:
        for line in tqdm(fd, desc='Reading corpus'):
            if not line: break
            line = line.strip()
            if not line and len(bucket) >= 1:
                spill(bucket, data, seq)
                bucket = []
                seq += 1
                continue
            if line: bucket.append(line)
        if len(bucket) != 0:
            spill(bucket, data, seq)
            seq += 1
    return data

def to_truecase(tokens):
    """
    # code from https://github.com/google-research/bert/issues/223#issuecomment-649619302

    # original tokens
    #['FULL', 'FEES', '1.875', 'REOFFER', '99.32', 'SPREAD', '+20', 'BP']

    # truecased tokens
    #['Full', 'fees', '1.875', 'Reoffer', '99.32', 'spread', '+20', 'BP']
    """
    word_lst = [(w, idx) for idx, w in enumerate(tokens) if all(c.isalpha() for c in w)]
    lst = [w for w, _ in word_lst if re.match(r'\b[A-Z\.\-]+\b', w)]

    if len(lst) and len(lst) == len(word_lst):
        parts = truecase.get_true_case(' '.join(lst)).split()

        # the trucaser have its own tokenization ...
        # skip if the number of word dosen't match
        if len(parts) != len(word_lst): return tokens

        for (w, idx), nw in zip(word_lst, parts):
            tokens[idx] = nw

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    opt = parser.parse_args()

    gdic = read_corpus(opt.input_path)
    glist = sorted(gdic.items(), key=lambda x : x, reverse=False)
    fd = sys.stdout
    for seq, entries in tqdm(glist, desc='Processing'):
        # convert to truecase
        tokens = []
        for entry in entries:
            word = entry['word']
            tokens.append(word)
        to_truecase(tokens)
        for token, entry in zip(tokens, entries):
            word = token
            tag = entry['tag']
            chunk = entry['chunk']
            label = entry['label']
            tp = [word, tag, chunk, label]
            fd.write(' '.join(tp) + '\n')
        fd.write('\n')
    fd.close()

