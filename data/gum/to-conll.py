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

def read_corpus(file_path) :

    def spill(bucket):
        etagged = []
        for line in bucket:
            row = line.split('\t')
            size = len(row)
            if size != 2:
                logger.warn('{}: {}: need to check'.format(seq, line))
                sys.exit(1)
            word = row[0]
            tag  = '-'
            chunk = '-'
            label = row[1]
            if 'object' in label or 'abstract' in label:
                label = 'O'
            etagged.append({'word': word,
                            'tag': tag,
                            'chunk': chunk,
                            'label': label})
        return etagged

    data = []
    bucket = []
    with open(file_path, 'r') as fd:
        iterator = tqdm(fd, desc='Readding')
        for seq, line in enumerate(iterator):
            line = line.strip()
            if not line:
                # spill
                if len(bucket) >= 1:
                    etagged = spill(bucket)
                    data.append(etagged)
                    bucket = []
            else:
                # add
                bucket.append(line)
        if len(bucket) != 0 :
            etagged = spill(bucket)
            data.append(etagged)
    return data

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_train', type=str, default='gum-train.conll')
    parser.add_argument('--input_test', type=str, default='gum-test.conll')
    parser.add_argument('--train', type=str, default='train.txt')
    parser.add_argument('--valid', type=str, default='valid.txt')
    parser.add_argument('--test', type=str, default='test.txt')

    opt = parser.parse_args()
    
    logger.info("%s", opt)

    data_train = read_corpus(opt.input_train)
    data_test = read_corpus(opt.input_test)

    with open(opt.train, 'w') as ftrain, open(opt.valid, 'w') as fvalid:
        tot_num = len(data_train)
        train_num = tot_num * 0.9
        iterator = tqdm(data_train, desc='Writing')
        for seq, etagged in enumerate(iterator) :
            bulk = '' 
            for item in etagged:
                tp = [item['word'], item['tag'], item['chunk'], item['label']]
                out = ' '.join(tp)
                out += '\n'
                bulk += out
            bulk += '\n'
            if seq < train_num:
                ftrain.write(bulk)
            else:
                fvalid.write(bulk)

    with open(opt.test, 'w') as ftest:
        iterator = tqdm(data_test, desc='Writing')
        for seq, etagged in enumerate(iterator) :
            bulk = '' 
            for item in etagged:
                tp = [item['word'], item['tag'], item['chunk'], item['label']]
                out = ' '.join(tp)
                out += '\n'
                bulk += out
            bulk += '\n'
            ftest.write(bulk)
