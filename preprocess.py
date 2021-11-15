import sys
import os
import argparse
import numpy as np
import random
import math
import json
from collections import Counter 
import pdb
import logging

from tqdm import tqdm
from util import load_config, Tokenizer, read_examples_from_file, convert_examples_to_features
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_TRAIN_FILE = 'train.txt'
_VALID_FILE = 'valid.txt'
_TEST_FILE  = 'test.txt'
_SUFFIX = '.ids'
_VOCAB_FILE = 'vocab.txt'
_EMBED_FILE = 'embedding.npy'
_POS_FILE = 'pos.txt'
_CHAR_FILE = 'char.txt'
_LABEL_FILE = 'label.txt'
_GLABEL_FILE = 'glabel.txt'
_FSUFFIX = '.fs'

def build_dict(input_path, config, extra_path=None):
    logger.info("\n[building dict]")
    args = config['args']
    poss = {}
    chars = {}
    labels = {}
    glabels = {}

    # add pad/unk info, set base id
    poss[config['pad_pos']] = config['pad_pos_id']
    pos_id = 1

    chars[config['pad_token']] = config['pad_token_id']   # 0
    chars[config['unk_token']] = config['unk_token_id']   # 1
    char_id = 2

    labels[config['pad_label']] = config['pad_label_id']  # 0
    label_id = 1

    glabels[config['pad_label']] = config['pad_label_id'] # 0
    glabel_id = 1

    for path in [input_path, extra_path]:
        if not path: break
        is_next_bos = True
        tot_num_line = sum(1 for _ in open(path, 'r')) 
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(tqdm(f, total=tot_num_line)):
                line = line.strip()
                if line == "":
                    is_next_bos = True
                    continue
                toks = line.split()
                try:
                    assert(len(toks) == 4)
                except Exception as e:
                    logger.error(str(idx) + '\t' + line + '\t' + str(e))
                    sys.exit(1)

                if args.bert_use_mtl and is_next_bos:
                    glabel = toks[0]
                    if glabel not in glabels:
                        glabels[glabel] = glabel_id
                        glabel_id += 1
                    is_next_bos = False
                    continue

                word = toks[0]
                pos = toks[1]
                label = toks[-1]
                if pos not in poss:
                    poss[pos] = pos_id
                    pos_id += 1
                for ch in word:
                    if ch not in chars:
                        chars[ch] = char_id
                        char_id += 1
                if label not in labels:
                    labels[label] = label_id
                    label_id += 1
                    # for preventing 'key not found error' for 'B-', 'I-'
                    p = label.find('-')
                    if p != -1:
                        if 'B-' in label:
                            label = 'I-' + label[p+1:]
                        if 'I-' in label:
                            label = 'B-' + label[p+1:]
                        if label not in labels:
                            labels[label] = label_id
                            label_id += 1
                        
    if args.use_ncrf:
        # NCRF takes actual number of labels + 2, see model/crf.py
        # START_TAG = -2
        # END_TAG = -1
        labels['START_TAG'] = label_id # init_transitions[:,START_TAG] = -10000.0
        label_id += 1
        labels['STOP_TAG'] = label_id # init_transitions[STOP_TAG,:] = -10000.0
        label_id += 1

    logger.info("\nUnique poss, chars, labels, glabels : {}, {}, {}, {}".format(len(poss), len(chars), len(labels), len(glabels)))
    return poss, chars, labels, glabels

def write_dict(dic, output_path):
    logger.info("\n[Writing dict]")
    f_write = open(output_path, 'w', encoding='utf-8')
    for idx, item in enumerate(tqdm(dic.items())):
        _key = item[0]
        _id = item[1]
        f_write.write(_key + ' ' + str(_id))
        f_write.write('\n')
    f_write.close()

# ---------------------------------------------------------------------------- #
# Glove
# ---------------------------------------------------------------------------- #

def build_init_vocab(config):
    init_vocab = {}
    init_vocab[config['pad_token']] = config['pad_token_id']
    init_vocab[config['unk_token']] = config['unk_token_id']
    return init_vocab

def build_vocab_from_embedding(input_path, vocab, config):
    logger.info("\n[Building vocab from pretrained embedding]")
    # build embedding as numpy array
    embedding = []
    # <pad>
    vector = np.array([float(0) for i in range(config['token_emb_dim'])]).astype(float)
    embedding.append(vector)
    # <unk>
    vector = np.array([random.random() for i in range(config['token_emb_dim'])]).astype(float)
    embedding.append(vector)
    tot_num_line = sum(1 for _ in open(input_path, 'r'))
    tid = len(vocab)
    with open(input_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(tqdm(f, total=tot_num_line)):
            toks = line.strip().split()
            word = toks[0]
            vector = np.array(toks[1:]).astype(float)
            assert(config['token_emb_dim'] == len(vector))
            vocab[word] = tid
            embedding.append(vector)
            tid += 1
    embedding = np.array(embedding)
    return vocab, embedding
    
def build_data(input_path, tokenizer):
    logger.info("\n[Tokenizing and building data]")
    vocab = tokenizer.vocab
    config = tokenizer.config
    data = []
    all_tokens = Counter()
    _long_data = 0
    tot_num_line = sum(1 for _ in open(input_path, 'r')) 
    with open(input_path, 'r', encoding='utf-8') as f:
        bucket = []
        for idx, line in enumerate(tqdm(f, total=tot_num_line)):
            line = line.strip()
            if line == "":
                tokens = []
                posseq = []
                labelseq = []
                for entry in bucket:
                    token = entry[0]
                    pos = entry[1]
                    pt = entry[2]
                    label = entry[3]
                    tokens.append(token)
                    posseq.append(pos)
                    labelseq.append(label)
                if len(tokens) > config['n_ctx']:
                    t = ' '.join(tokens)
                    logger.info("\n# Data over text length limit : {:,} / {:,}, {}".format(len(tokens), config['n_ctx'], t))
                    tokens = tokens[:config['n_ctx']]
                    posseq = posseq[:config['n_ctx']]
                    labelseq = labelseq[:config['n_ctx']]
                    _long_data += 1
                for token in tokens:
                    all_tokens[token] += 1
                data.append((tokens, posseq, labelseq))
                bucket = []
            else:
                entry = line.split()
                try:
                    assert(len(entry) == 4)
                except Exception as e:
                    logger.error(str(idx) + '\t' + line + '\t' + str(e))
                    sys.exit(1)
                bucket.append(entry)
        if len(bucket) != 0:
            tokens = []
            posseq = []
            labelseq = []
            for entry in bucket:
                token = entry[0]
                pos = entry[1]
                pt = entry[2]
                label = entry[3]
                tokens.append(token)
                posseq.append(pos)
                labelseq.append(label)
            if len(tokens) > config['n_ctx']:
                tokens = tokens[:config['n_ctx']]
                posseq = posseq[:config['n_ctx']]
                labelseq = labelseq[:config['n_ctx']]
                _long_data += 1
            for token in tokens:
                all_tokens[token] += 1
            data.append((tokens, posseq, labelseq))

    logger.info("\n# Data over text length limit : {:,}".format(_long_data))
    logger.info("\nTotal unique tokens : {:,}".format(len(all_tokens)))
    logger.info("Vocab size : {:,}".format(len(vocab)))
    total_token_cnt = sum(all_tokens.values())
    cover_token_cnt = 0
    for item in all_tokens.most_common():
        token = item[0]
        if tokenizer.config['lowercase']: token = token.lower()
        if token in vocab:
            cover_token_cnt += item[1]
    logger.info("Total tokens : {:,}".format(total_token_cnt))
    logger.info("Vocab coverage : {:.2f}%\n".format(cover_token_cnt/total_token_cnt*100.0))
    return data

def write_data(args, data, output_path, tokenizer, poss, labels):
    logger.info("\n[Writing data]")
    config = tokenizer.config
    pad_id = tokenizer.pad_id
    default_label = config['default_label']
    num_tok_per_sent = []
    f_write = open(output_path, 'w', encoding='utf-8')
    for idx, item in enumerate(tqdm(data)):
        tokens, posseq, labelseq = item[0], item[1], item[2]
        if len(tokens) == 0:
            logger.info("\nData Error!! : {}", idx)
            continue
        assert(len(tokens) == len(posseq))
        assert(len(tokens) == len(labelseq))
        # token ids
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        for _ in range(config['n_ctx'] - len(token_ids)):
            token_ids.append(pad_id)
        token_ids_str = ' '.join([str(d) for d in token_ids])
        # pos ids
        pos_ids = []
        for pos in posseq:
            pos_id = poss[pos]
            pos_ids.append(pos_id)
        for _ in range(config['n_ctx'] - len(pos_ids)):
            pos_ids.append(config['pad_pos_id'])
        pos_ids_str = ' '.join([str(d) for d in pos_ids])
        # label ids
        label_ids = []
        for label in labelseq:
            if label in labels:
                label_id = labels[label]
            else:
                logger.warn("Unknown label: {}".format(label))
                label_id = labels[default_label]
            label_ids.append(label_id)
        for _ in range(config['n_ctx'] - len(label_ids)):
            label_ids.append(config['pad_label_id'])
        label_ids_str = ' '.join([str(d) for d in label_ids])
        tokens_str = ' '.join(tokens)
        # format: label list \t token list \t pos list \t word list
        f_write.write(label_ids_str + '\t' + token_ids_str + '\t' + pos_ids_str + '\t' + tokens_str)
        num_tok_per_sent.append(len(tokens))
        f_write.write('\n')
    f_write.close()
    ntps = np.array(num_tok_per_sent)
    logger.info("\nMEAN : {:.2f}, MAX:{}, MIN:{}, MEDIAN:{}\n".format(\
            np.mean(ntps), int(np.max(ntps)), int(np.min(ntps)), int(np.median(ntps))))

def write_vocab(vocab, output_path):
    logger.info("\n[Writing vocab]")
    f_write = open(output_path, 'w', encoding='utf-8')
    for idx, item in enumerate(tqdm(vocab.items())):
        tok = item[0]
        tok_id = item[1]
        f_write.write(tok + ' ' + str(tok_id))
        f_write.write('\n')
    f_write.close()

def write_embedding(embedding, output_path):
    logger.info("\n[Writing embedding]")
    np.save(output_path, embedding)

def preprocess_glove_or_elmo(config):
    args = config['args']

    # vocab, embedding
    init_vocab = build_init_vocab(config)
    vocab, embedding = build_vocab_from_embedding(args.embedding_path, init_vocab, config)

    # build poss, chars, labels
    path = os.path.join(args.data_dir, _TRAIN_FILE)
    poss, chars, labels, _ = build_dict(path, config)

    tokenizer = Tokenizer(vocab, config)

    # build data
    path = os.path.join(args.data_dir, _TRAIN_FILE)
    train_data = build_data(path, tokenizer)

    path = os.path.join(args.data_dir, _VALID_FILE)
    valid_data = build_data(path, tokenizer)

    path = os.path.join(args.data_dir, _TEST_FILE)
    test_data = build_data(path, tokenizer)

    # write data, vocab, embedding, poss, labels
    path = os.path.join(args.data_dir, _TRAIN_FILE + _SUFFIX)
    write_data(args, train_data, path, tokenizer, poss, labels)

    path = os.path.join(args.data_dir, _VALID_FILE + _SUFFIX)
    write_data(args, valid_data, path, tokenizer, poss, labels)

    path = os.path.join(args.data_dir, _TEST_FILE + _SUFFIX)
    write_data(args, test_data, path, tokenizer, poss, labels)

    path = os.path.join(args.data_dir, _VOCAB_FILE)
    write_vocab(vocab, path)

    path = os.path.join(args.data_dir, _EMBED_FILE)
    write_embedding(embedding, path)

    path = os.path.join(args.data_dir, _POS_FILE)
    write_dict(poss, path)

    path = os.path.join(args.data_dir, _LABEL_FILE)
    write_dict(labels, path)

# ---------------------------------------------------------------------------- #
# BERT
# ---------------------------------------------------------------------------- #

def build_features(input_path, tokenizer, poss, labels, config, mode='train', w_tokenizer=None, glabels={}):

    logger.info("[Creating features from file] %s", input_path)
    examples = read_examples_from_file(config, input_path, mode=mode)
    features = convert_examples_to_features(config, examples, poss, labels, config['n_ctx'], tokenizer,
                                            cls_token=tokenizer.cls_token,
                                            cls_token_segment_id=0,
                                            sep_token=tokenizer.sep_token,
                                            sep_token_extra=bool(config['emb_class'] in ['roberta']),
                                            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                            pad_token=tokenizer.pad_token,
                                            pad_token_id=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_pos_id=config['pad_pos_id'],
                                            pad_token_label_id=config['pad_label_id'],
                                            pad_token_segment_id=0,
                                            sequence_a_segment_id=0,
                                            glabel_map=glabels,
                                            w_tokenizer=w_tokenizer)
    return features

def write_features(features, output_path):
    import torch

    logger.info("[Saving features into file] %s", output_path)
    torch.save(features, output_path)
   
def preprocess_bert(config):
    args = config['args']

    w_tokenizer = None
    if args.bert_use_subword_pooling and args.bert_use_word_embedding:
        args = config['args']
        # vocab, embedding
        init_vocab = build_init_vocab(config)
        vocab, embedding = build_vocab_from_embedding(args.embedding_path, init_vocab, config)
        w_tokenizer = Tokenizer(vocab, config)
        # write embedding
        path = os.path.join(args.data_dir, _EMBED_FILE)
        write_embedding(embedding, path)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name_or_path, revision=args.bert_revision)
    # build poss, chars, labels, glabels
    path = os.path.join(args.data_dir, _TRAIN_FILE)
    poss, chars, labels, glabels = build_dict(path, config)

    # build features
    path = os.path.join(args.data_dir, _TRAIN_FILE)
    train_features = build_features(path, tokenizer, poss, labels, config, mode='train', w_tokenizer=w_tokenizer, glabels=glabels)

    path = os.path.join(args.data_dir, _VALID_FILE)
    valid_features = build_features(path, tokenizer, poss, labels, config, mode='valid', w_tokenizer=w_tokenizer, glabels=glabels)

    path = os.path.join(args.data_dir, _TEST_FILE)
    test_features = build_features(path, tokenizer, poss, labels, config, mode='test', w_tokenizer=w_tokenizer, glabels=glabels)

    # write features
    path = os.path.join(args.data_dir, _TRAIN_FILE + _FSUFFIX)
    write_features(train_features, path)

    path = os.path.join(args.data_dir, _VALID_FILE + _FSUFFIX)
    write_features(valid_features, path)

    path = os.path.join(args.data_dir, _TEST_FILE + _FSUFFIX)
    write_features(test_features, path)

    # write poss, labels, glabels
    path = os.path.join(args.data_dir, _POS_FILE)
    write_dict(poss, path)
    path = os.path.join(args.data_dir, _LABEL_FILE)
    write_dict(labels, path)
    path = os.path.join(args.data_dir, _GLABEL_FILE)
    write_dict(glabels, path)

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', type=str, default='configs/config-glove.json')
    parser.add_argument('--data_dir', type=str, default='data/conll2003')
    parser.add_argument('--embedding_path', type=str, default='embeddings/glove.6B.300d.txt')
    parser.add_argument("--seed", default=5, type=int)
    parser.add_argument('--use_ncrf', action='store_true', help="Use NCRF instead of pytorch-crf. in this case, '<bos>, <eos>' labels will be added.")
    # for BERT
    parser.add_argument("--bert_model_name_or_path", type=str, default='bert-base-uncased',
                        help="Path to pre-trained model or shortcut name(ex, bert-base-uncased)")
    parser.add_argument('--bert_revision', type=str, default='main')
    parser.add_argument('--bert_use_sub_label', action='store_true',
                        help="Set this flag to use sub label instead of using pad label for sub tokens.")
    parser.add_argument('--bert_use_subword_pooling', action='store_true',
                        help="Set this flag for bert subword pooling.")
    parser.add_argument('--bert_use_word_embedding', action='store_true',
                        help="Set this flag to use word embedding(eg, GloVe). it should be used with --bert_use_subword_pooling.")
    parser.add_argument('--bert_use_doc_context', action='store_true',
                        help="Set this flag to use document-level context.")
    parser.add_argument("--bert_doc_separator", type=str, default='-DOCSTART-',
                        help="Path to pre-trained model or shortcut name(ex, bert-base-uncased)")
    parser.add_argument("--bert_doc_context_option", default=1, type=int,
                        help="1: prev one example, cur example, next examples, 2: prev examples, cur example, next examples")
    parser.add_argument('--bert_use_mtl', action='store_true',
                        help="Set this flag to use multi-task learning of token and sentence classification.")
    args = parser.parse_args()

    # set seed
    random.seed(args.seed)

    # set config
    config = load_config(args)
    config['args'] = args
    logger.info("%s", config)

    if config['emb_class'] == 'glove':
        preprocess_glove_or_elmo(config)
    elif config['emb_class'] == 'elmo':
        preprocess_glove_or_elmo(config)
    else:
        preprocess_bert(config)


if __name__ == '__main__':
    main()
