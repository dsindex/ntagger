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

from khaiii import KhaiiiApi
khaiii_api = KhaiiiApi()

def to_eojs(sentence):
    text = []
    for char, _ in sentence:
        if char == '': text.append(' ')
        else: text.append(char)
    text = ''.join(text)
    eojs = text.split()
    return eojs

def segment(eojs):
    segmented = []
    for eoj_idx, eoj in enumerate(eojs):
        if eoj_idx != 0: segmented.append(('_', 'O'))
        khaiii_sent = khaiii_api.analyze(eoj)
        for khaiii_word in khaiii_sent:
            morphs = []
            found_e = False
            for khaiii_morph in khaiii_word.morphs:
                morphs.append(khaiii_morph.lex)
                if khaiii_morph.tag in ['EC', 'EP', 'EF']:
                    found_e = True
            matched = False
            if ''.join(morphs) == khaiii_word.lex and not found_e:
                matched = True
            # print(khaiii_word, matched)
            if matched:
                for khaiii_morph in khaiii_word.morphs:
                    for idx, char in enumerate(khaiii_morph.lex):
                        if idx == 0:
                            segmented.append((char, 'B'))
                        else:
                            segmented.append((char, 'I'))
            else:
                for idx, char in enumerate(eoj):
                    if idx == 0:
                        segmented.append((char, 'B'))
                    else:
                        segmented.append((char, 'I'))
    return segmented

def get_etc(char):
    pos  = 'pos'
    if char == '':
        char = '_'
        pos  = '_'
    pt = 'pt'
    return char, pos, pt

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='klue-ner-v1.1_train.tsv')
    parser.add_argument('--segmentation', action='store_true') 

    opt = parser.parse_args()
    
    logger.info("%s", opt)

    data = []
    sentence = []
    with open(opt.file, 'r') as fid:
        for line in fid:
            if line[0:2] == '##': continue
            if not line.strip():
                data.append(sentence)
                sentence = []
                continue
            tokens = line.split('\t')
            char = tokens[0].strip()
            tag = tokens[1].strip()
            sentence.append((char, tag))

    for sentence in data:
        if opt.segmentation:
            eojs = to_eojs(sentence)
            segmented = segment(eojs)
            bucket = []
            for (char, tag), (_, mark) in zip(sentence, segmented):
                char, pos, pt = get_etc(char)
                if mark == 'B':
                    if bucket:
                        print(' '.join(bucket))
                    bucket = [char, pos, pt, tag]
                elif mark == 'O':
                    if bucket:
                        print(' '.join(bucket))
                        bucket = []
                        print(' '.join([char, pos, pt, tag]))
                    else:
                        logger.error('[1] incomplete in ', ' '.join(eojs))
                        sys.exit(1)
                elif mark == 'I':
                    if bucket:
                        bucket[0] += char
                    else:
                        logger.error('[2] incomplete in ', ' '.join(eojs))
                        sys.exit(1)
            if bucket:
                print(' '.join(bucket))
        else:
            for char, tag in sentence:
                char, pos, pt = get_etc(char)
                print(char + ' ' + pos + ' ' + pt + ' ' + tag)
        print()
