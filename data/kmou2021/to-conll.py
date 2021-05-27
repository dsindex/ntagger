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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--words', type=str, default='train.words.txt')
    parser.add_argument('--tags', type=str, default='train.tags.txt')
    parser.add_argument('--pos', type=str, default='train.pos.txt')

    opt = parser.parse_args()
    
    logger.info("%s", opt)

    all_words = []
    with open(opt.words, 'r') as fid:
        for line in fid:
            line = line.strip()
            all_words.append(line.split())

    all_tags = []
    with open(opt.tags, 'r') as fid:
        for line in fid:
            line = line.strip()
            all_tags.append(line.split())

    all_pos = []
    with open(opt.pos, 'r') as fid:
        for line in fid:
            line = line.strip()
            all_pos.append(line.split())

    tot_num = len(all_words)
    for i in range(tot_num):
        words = all_words[i]
        tags = all_tags[i]
        pos = all_pos[i]

        for word, tag, p in zip(words, tags, pos):
            # append '다' for 'VV', 'VA', ...
            if p in ['VV', 'VA', 'VX', 'XSV', 'XSA']:
                word = word + '다'
            print(word, p, '-', tag)
        print('')
