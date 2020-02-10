#!/usr/bin/env python
#-*- coding: utf8 -*-

from __future__ import print_function, division, absolute_import, unicode_literals

import os
from optparse import OptionParser

# global variable
VERBOSE = 0

import sys

if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("--verbose", action="store_const", const=1, dest="verbose", help="verbose mode")
    (options, args) = parser.parse_args()

    while 1:
        try:
            line = sys.stdin.readline()
        except KeyboardInterrupt:
            break
        if not line:
            break
        line = line.strip()
        if not line:
            print('')
            continue
        line = line.replace(' ', '\t')
        toks = line.split('\t')
        term = toks[0]
        mtag = toks[1]
        chunk = toks[2]
        label = toks[3]
        predict = toks[4]
        if 'X-' in mtag: continue
        print(term + ' ' + mtag + ' ' + chunk + ' ' + label + ' ' + predict)
