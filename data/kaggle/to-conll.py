import sys
import os
import argparse
import random
import time
import json
import csv
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_csv(path, skip_header=True):
    with open(path) as f:
        reader = csv.reader(f, delimiter=',')
        if skip_header:
            next(reader)
        # row := [tmp, word, tag, label]
        data = [row if len(row) == 4 else None for row in reader]
    return data

def read_corpus(csv_data) :

    def spill(bucket):
        etagged = []
        for row in bucket:
            size = len(row)
            if size != 4:
                line = ','.join(row)
                logger.warn('{}: {}: need to check'.format(seq, line))
                sys.exit(1)
            word = row[1]
            tag  = row[2]
            # FIXME, broken characters, ex) [,<85>,",",O], [<85>, :, -, O]
            if tag == ',': word = ','
            if tag == ':': word = '-'
            label = row[3]
            etagged.append({'word': word,
                            'tag': tag,
                            'label': label})
        return etagged

    data = []
    bucket = []
    iterator = tqdm(csv_data, desc='Readding')
    for seq, row in enumerate(iterator):
        if 'Sentence:' in row[0]:
            # spill
            if len(bucket) >= 1:
                etagged = spill(bucket)
                data.append(etagged)
                bucket = []
            # add
            bucket.append(row)
        else:
            # add
            bucket.append(row)
    if len(bucket) != 0 :
        etagged = spill(bucket)
        data.append(etagged)
    return data

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='ner_dataset.csv.utf')
    parser.add_argument('--train', type=str, default='train.txt')
    parser.add_argument('--valid', type=str, default='valid.txt')

    opt = parser.parse_args()
    
    logger.info("%s", opt)

    csv_data = load_csv(opt.input)

    data = read_corpus(csv_data)

    with open(opt.train, 'w') as ftrain, open(opt.valid, 'w') as fvalid:
        tot_num = len(data)
        train_num = tot_num * 0.9
        iterator = tqdm(data, desc='Writing')
        for seq, etagged in enumerate(iterator) :
            bulk = '' 
            for item in etagged:
                tp = [item['word'], item['tag'], '-', item['label']]
                out = ' '.join(tp)
                out += '\n'
                bulk += out
            bulk += '\n'
            if seq < train_num:
                ftrain.write(bulk)
            else:
                fvalid.write(bulk)
