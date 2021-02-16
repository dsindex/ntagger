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

    def spill(bucket, data):
        entries = []
        for line in bucket:
            tokens = line.split()
            word = tokens[0]
            tag  = tokens[1]
            chunk = tokens[2]
            label = tokens[3]
            entry = {'word': word, 'tag': tag, 'chunk': chunk, 'label': label}
            entries.append(entry)
        data.append(entries)
        return True

    data = []
    bucket = []
    with open(filename, 'r') as fd:
        for line in tqdm(fd, desc='Reading corpus'):
            if not line: break
            line = line.strip()
            if not line and len(bucket) >= 1:
                spill(bucket, data)
                bucket = []
                continue
            if line: bucket.append(line)
        if len(bucket) != 0:
            spill(bucket, data)
    return data

'''
since CoNLL++ test.txt has incorrect chunk tags,
combine CoNLL 2003 test.txt with CoNLL++ test.txt.
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conll2003', type=str)
    parser.add_argument('--conllpp', type=str)
    opt = parser.parse_args()

    glist_2003 = read_corpus(opt.conll2003)
    glist_pp = read_corpus(opt.conllpp)
    fd = sys.stdout
    for seq, entries_2003 in enumerate(tqdm(glist_2003, desc='Processing')):
        entries_pp = glist_pp[seq]
        for entry_2003, entry_pp in zip(entries_2003, entries_pp):
            word = entry_2003['word']
            tag = entry_2003['tag']
            chunk = entry_2003['chunk']
            label = entry_2003['label']
            label_pp = entry_pp['label']
            if label != label_pp:
                logger.info('%s, %s -> %s' % (word, label, label_pp))
                label = label_pp
            tp = [word, tag, chunk, label]
            fd.write(' '.join(tp) + '\n')
        fd.write('\n')
    fd.close()

