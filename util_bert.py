from __future__ import absolute_import, division, print_function

import os
import pdb

from tqdm import tqdm

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------- #
# BERT
#   reference
#     https://github.com/huggingface/transformers/blob/master/examples/utils_ner.py
# ---------------------------------------------------------------------------- #

class InputExample(object):
    def __init__(self, guid, words, labels):
        self.guid = guid
        self.words = words
        self.labels = labels

class InputFeature(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids

def read_examples_from_file(file_path, mode='train'):
    guid_index = 1
    examples = []
    tot_num_line = sum(1 for _ in open(file_path, 'r'))
    with open(file_path, encoding="utf-8") as f:
        bucket = []
        for idx, line in enumerate(tqdm(f, total=tot_num_line)):
            line = line.strip()
            if line == "":
                tokens = []
                labelseq = []
                for entry in bucket:
                    token = entry[0]
                    pos = entry[1]
                    pt = entry[2]
                    label = entry[3]
                    tokens.append(token)
                    labelseq.append(label)
                examples.append(InputExample(guid="{}-{}".format(mode, guid_index),
                                             words=tokens,
                                             labels=labelseq))
                guid_index += 1
                bucket = []
            else:
                entry = line.split()
                assert(len(entry) == 4)
                bucket.append(entry)
        if len(bucket) != 0:
            tokens = []
            labelseq = []
            for entry in bucket:
                token = entry[0]
                pos = entry[1]
                pt = entry[2]
                label = entry[3]
                tokens.append(token)
                labelseq.append(label)
            examples.append(InputExample(guid="{}-{}".format(mode, guid_index),
                                         words=tokens,
                                         labels=labelseq))
            guid_index += 1

    return examples

def convert_single_example_to_feature(example,
                                      label_map,
                                      max_seq_length,
                                      tokenizer,
                                      cls_token="[CLS]",
                                      cls_token_segment_id=0,
                                      sep_token="[SEP]",
                                      pad_token=0,
                                      pad_token_label_id=0,
                                      pad_token_segment_id=0,
                                      sequence_a_segment_id=0,
                                      ex_index=-1):

    tokens = []
    label_ids = []
    for word, label in zip(example.words, example.labels):
        word_tokens = tokenizer.tokenize(word)
        tokens.extend(word_tokens)
        label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

    special_tokens_count = 2
    if len(tokens) > max_seq_length - special_tokens_count:
        tokens = tokens[:(max_seq_length - special_tokens_count)]
        label_ids = label_ids[:(max_seq_length - special_tokens_count)]

    # convention in BERT:
    # for single sequences:
    #  tokens:     [CLS] the dog is hairy . [SEP]
    #  input_ids:    x   x   x   x  x     x   x   0  0  0 ...
    #  segment_ids:  0   0   0   0  0     0   0   0  0  0 ...
    #  input_mask:   1   1   1   1  1     1   1   0  0  0 ...

    tokens += [sep_token]
    label_ids += [pad_token_label_id]
    segment_ids = [sequence_a_segment_id] * len(tokens)

    tokens = [cls_token] + tokens
    label_ids = [pad_token_label_id] + label_ids
    segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    input_ids += ([pad_token] * padding_length)
    input_mask += ([0] * padding_length)
    segment_ids += ([pad_token_segment_id] * padding_length)
    label_ids += ([pad_token_label_id] * padding_length)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

    if ex_index != -1 and ex_index < 5:
        logger.info("*** Example ***")
        logger.info("guid: %s", example.guid)
        logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
        logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
        logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
        logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

    feature = InputFeature(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_ids=label_ids)
    return feature

def convert_examples_to_features(examples,
                                 label_map,
                                 max_seq_length,
                                 tokenizer,
                                 cls_token="[CLS]",
                                 cls_token_segment_id=0,
                                 sep_token="[SEP]",
                                 pad_token=0,
                                 pad_token_label_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0):

    features = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        if ex_index % 1000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        feature = convert_single_example_to_feature(example,
                                                    label_map,
                                                    max_seq_length,
                                                    tokenizer,
                                                    cls_token=cls_token,
                                                    cls_token_segment_id=cls_token_segment_id,
                                                    sep_token=sep_token,
                                                    pad_token=pad_token,
                                                    pad_token_label_id=pad_token_label_id,
                                                    pad_token_segment_id=pad_token_segment_id,
                                                    sequence_a_segment_id=sequence_a_segment_id,
                                                    ex_index=ex_index)
        features.append(feature)
    return features
