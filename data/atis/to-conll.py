import sys
import os
import argparse
import random
import time
import json
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# data : https://github.com/monologg/JointBERT/tree/master/data/atis

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_label', type=str, default='label')
    parser.add_argument('--input_seq_in', type=str, default='seq.in')
    parser.add_argument('--input_seq_out', type=str, default='seq.out')

    opt = parser.parse_args()
    
    logger.info("%s", opt)

    labels = []
    with open(opt.input_label, 'r') as fid:
        for line in fid:
            line = line.strip()
            labels.append(line)

    sentences = []
    with open(opt.input_seq_in, 'r') as fid:
        for line in fid:
            line = line.strip()
            sentences.append(line.split())

    token_labels = []
    with open(opt.input_seq_out, 'r') as fid:
        for line in fid:
            line = line.strip()
            token_labels.append(line.split())

    tot_num = len(labels)
    for i in range(tot_num):
        label = labels[i]
        sentence = sentences[i]
        token_label = token_labels[i]
        print(label, '-', '-', 'O')
        for token, tag in zip(sentence, token_label):
            print(token, '-', '-', tag)
        print('')
