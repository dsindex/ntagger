import sys
import os
import argparse
import random
import time
import json
from tqdm import tqdm
import logging
from unicodedata import category

logging.basicConfig(level=logging.DEBUG)
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
        
        if len(words) != len(pos) or len(words) != len(tags):
            logger.error('%s %s', len(words), ' '.join(words))
            logger.error('%s %s', len(pos), ' '.join(pos))
            logger.error('%s %s', len(tags), ' '.join(pos))
            sys.exit(1)

        valid = True
        for j, word in enumerate(words):
            if category(word[0]) in ['Cc', 'Zs', 'Cf']:
                logger.error('Control/Space Character %s-th word in %s', j, ' '.join(words))
                valid = False
        if not valid: continue

        for word, tag, p in zip(words, tags, pos):
            # append '다' for 'VV', 'VA', ...
            if p in ['VV', 'VA', 'VX', 'XSV', 'XSA']:
                word = word + '다'
            print(word, p, '-', tag)
        print('')
