#!/usr/bin/env python
#-*- coding: utf8 -*-

import sys
import os
from optparse import OptionParser

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

    def spill(bucket, data, seq, linenum) :
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
                    sys.stderr.write(str(linenum) + ' : need to check format : ' + line + '\n')
                    return False
                if idx == 1:
                    sent = line[3:]
                if idx == 2:
                    tagged_sent = line[3:]
            else :
                if size != 4:
                    sys.stderr.write(str(linenum) + ' : need to check format : ' + line + '\n')
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
    linenum = 0
    seq = 1
    fid = open_file(filename, 'r')
    for line in fid :
        if not line : break
        line = line.strip()
        linenum += 1
        if not line and len(bucket) >= 1 :
            spill(bucket, data, seq, linenum)
            bucket = []
            seq += 1
            continue
        if line : bucket.append(line)
    if len(bucket) != 0 :
        spill(bucket, data, seq, linenum)
        seq += 1
    close_file(fid)
    return data

def apply_correction_rules(morphi, morphs, tags):
    if morphi == u'\xad':
        morphi = '_'
        morphs = '_'
        tags   = '_'
    if tags[0] == '+':
        if morphi == '봤':
            morphs = '보+았'
            tags   = 'VV+EP'
        if morphi == '됐':
            morphs = '되+었'
            tags   = 'VV+EP'
        if morphi == '되':
            morphs = '되'
            tags   = 'VV'
        if morphi == '했':
            morphs = '하+였'
            tags   = 'VV+EP'
        if morphi == '했었':
            morphs = '하+였었'
            tags   = 'VV+EP'
        if morphi == '왔':
            morphs = '오+았'
            tags   = 'VV+EP'
        if morphi == '왔었':
            morphs = '오+았었'
            tags   = 'VV+EP'
        if morphi == '와':
            morphs = '오+아'
            tags   = 'VV+EC'
        if morphi == '와야':
            morphs = '오+아야'
            tags   = 'VV+EC'
        if morphi == '와서':
            morphs = '오+아서'
            tags   = 'VV+EC'
        if morphi == '컸':
            morphs = '크+었'
            tags   = 'VA+EP'
        if morphi == '커서':
            morphs = '크+어서'
            tags   = 'VA+EC'
        if morphi == '커':
            morphs = '크+어'
            tags   = 'VA+EC'
        if morphi == '줬':
            morphs = '주+었'
            tags   = 'VV+EP'
        if morphi == '졌':
            morphs = '지+었'
            tags   = 'VX+EP'
        if morphi == '써야':
            morphs = '쓰+어야'
            tags   = 'VV+EC'
        if morphi == '써서':
            morphs = '쓰+어서'
            tags   = 'VV+EC'
        if morphi == '써':
            morphs = '쓰+어'
            tags   = 'VV+EC'
        if morphi == '써도':
            morphs = '쓰+어도'
            tags   = 'VV+EC'
        if morphi == '썼':
            morphs = '쓰+었'
            tags   = 'VV+EP'
        if morphi == '쐈':
            morphs = '쏘+았'
            tags   = 'VV+EP'
        if morphi == '꿨':
            morphs = '꾸+었'
            tags   = 'VV+EP'
        if morphi == '쳤':
            morphs = '치+었'
            tags   = 'VV+EP'
        if morphi == '췄':
            morphs = '추+었'
            tags   = 'VV+EP'
        if morphi == '놨':
            morphs = '노+았'
            tags   = 'VV+EP'
        if morphi == '겠':
            morphs = '겠'
            tags   = 'EP'
        if morphi == '퍼':
            morphs = '푸+어'
            tags   = 'VV+EC'
        if morphi == '뒀':
            morphs = '두+었'
            tags   = 'VV+EP'
        if morphi == '꼈':
            morphs = '끼+었'
            tags   = 'VV+EP'
        if morphi == '떴':
            morphs = '뜨+었'
            tags   = 'VV+EP'
        if morphi == '떠':
            morphs = '뜨+어'
            tags   = 'VV+EC'

    return morphi, morphs, tags

if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("--verbose", action="store_const", const=1, dest="verbose", help="verbose mode")
    parser.add_option("-g", "--golden", dest="golden", help="golden tagged data")

    (options, args) = parser.parse_args()

    if options.verbose : VERBOSE = 1
    golden_path = options.golden
    if not golden_path:
        parser.print_help()
        sys.exit(1)
    golden_dic = read_corpus(golden_path)
    golden_list = sorted(golden_dic.items(), key=lambda x : x[1]['seq'], reverse=False)
    total = len(golden_list)
    fd = sys.stdout
    for key, entry in golden_list :
        seq = entry['seq']
        sent = entry['sent']
        tagged_sent = entry['tagged_sent']
        out = '##' + ' ' + str(seq)
        fd.write(out +'\n')
        out = '##' + ' ' + sent
        fd.write(out +'\n')
        out = '##' + ' ' + tagged_sent
        fd.write(out +'\n')
        etagged = entry['etagged']
        num_morphs = len(etagged)
        for idx in range(num_morphs) :
            entry = etagged[idx]
            morphi = entry['morphi']
            morphs = entry['morphs']
            tags = entry['tags']
            etype = entry['etype']

            morphi, morphs, tags = apply_correction_rules(morphi, morphs, tags)
                
            out = morphi + '\t' + morphs + '\t' + tags + '\t' + etype
            fd.write(out + '\n')
        fd.write('\n')    
