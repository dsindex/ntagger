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
    parser.add_argument('--file', type=str, default='test.txt.pred')

    opt = parser.parse_args()
    
    logger.info("%s", opt)

    with open(opt.file, 'r') as fid:
        for line in fid:
            line = line.strip()
            if not line:
                print()
                continue
            tokens = line.split()
            token = tokens[0]
            pos = tokens[1]
            pt = tokens[2]
            tag = tokens[3]
            pred = tokens[4]

            for idx, char in enumerate(token):
                if tag != 'O':
                    prefix, suffix = tag.split('-')
                    if prefix == 'B' and idx != 0:
                        tag = 'I-' + suffix
                if pred != 'O':
                    prefix, suffix = pred.split('-')
                    if prefix == 'B' and idx != 0:
                        pred = 'I-' + suffix
                t = [char, pos, pt, tag, pred]
                print(' '.join(t))


