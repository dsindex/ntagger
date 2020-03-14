#!/usr/bin/env python
#-*- coding: utf8 -*-

import sys
import os
from optparse import OptionParser
import random

# global variable
VERBOSE = 0

def open_file(filename, mode) :
    try : fid = open(filename, mode)
    except :
        sys.stderr.write("open_file(), file open error : %s\n" % (filename))
        sys.exit(1)
    else :
        return fid

def close_file(fid) :
    fid.close()

def read_corpus(filename) :

    def spill(bucket, data, seq) :
        sent = None
        tagged_sent = None
        result = {}
        etagged = []
        idx = 0
        for line in bucket :
            tokens = line.split('\t')
            size = len(tokens)
            if idx <= 2 :
                if line[0:2] != '##':
                    sys.stderr.write(str(seq) + ' : need to check format : ' + line + '\n')
                    return False
                if idx == 1:
                    sent = line[3:]
                if idx == 2:
                    tagged_sent = line[3:]
            else :
                if size != 4:
                    sys.stderr.write(str(seq) + ' : need to check format : ' + line + '\n')
                    return False
                morphi = tokens[0]
                morphs = tokens[1]
                tags   = tokens[2]
                etype = tokens[3]
                etagged.append({'morphi': morphi,
                                'morphs': morphs,
                                'tags': tags,
                                'etype': etype,
                                'reserved': None})
            idx += 1
        result['seq'] = seq
        result['sent'] = sent
        result['tagged_sent'] = tagged_sent
        result['etagged'] = etagged
        data[seq] = result
        return True

    data = {}
    bucket = []
    seq = 1
    fid = open_file(filename, 'r')
    for line in fid :
        if not line : break
        line = line.strip()
        if not line and len(bucket) >= 1 :
            spill(bucket, data, seq)
            bucket = []
            seq += 1
            continue
        if line : bucket.append(line)
    if len(bucket) != 0 :
        spill(bucket, data, seq)
        seq += 1
    close_file(fid)
    return data

if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("--verbose", action="store_const", const=1, dest="verbose", help="verbose mode")
    parser.add_option("-g", "--golden", dest="golden", help="golden tagged data")

    (options, args) = parser.parse_args()

    if options.verbose : VERBOSE = 1
    golden_path = options.golden
    if not golden_path :
        parser.print_help()
        sys.exit(1)
    golden_dic = read_corpus(golden_path)
    golden_list = sorted(golden_dic.items(), key=lambda x : x[1]['seq'], reverse=False)
    for key, entry in golden_list :
        seq = entry['seq']
        sent = entry['sent']
        tagged_sent = entry['tagged_sent']
        etagged = entry['etagged']
        num_morphs = len(etagged)
        out_list = []
        valid = True
        for idx in range(num_morphs) :
            entry = etagged[idx]
            morphi = entry['morphi']
            morphs = entry['morphs']
            tags = entry['tags']
            etype = entry['etype']
            '''
            if morphi == '_' and tags == '_': continue
            '''
            tags = tags.split('+')
            morphs = morphs.split('+')
            if len(morphs) == 0: # literal '+' case
                morphs = ['+']
            for tag, morph in zip(tags, morphs):
                # append '다' for 'VV', 'VA', ...
                if tag in ['VV', 'VA', 'VX', 'XSV', 'XSA']:
                    morph = morph + '다'
                morph = morph.lower()
                if not morph or not tag or not etype:
                    sys.stderr.write('format error on ' + sent + '\n')
                    valid = False
                out = morph + ' ' + tag + ' ' + '-' + ' ' + etype
                out_list.append(out)
        if valid:
            out = '\n'.join(out_list)
            sys.stdout.write(out + '\n')
            sys.stdout.write('\n')
