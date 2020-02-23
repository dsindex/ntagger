from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
import json
import time
import pdb
import logging

import torch
import torch.nn as nn
from model import GloveLSTMCRF, BertLSTMCRF, ElmoLSTMCRF
from dataset import CoNLLGloveDataset, CoNLLBertDataset, CoNLLElmoDataset
from torch.utils.data import DataLoader
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(opt):
    try:
        with open(opt.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        config = dict()
    return config

def prepare_dataset(config, filepath, DatasetClass, shuffle=False, num_workers=1):
    dataset = DatasetClass(config, filepath)
    sampler = None
    loader = DataLoader(dataset, batch_size=config['opt'].batch_size, \
            shuffle=shuffle, num_workers=num_workers, sampler=sampler, pin_memory=True)
    logger.info("[{} data loaded]".format(filepath))
    return loader

def to_device(x, device):
    if type(x) != list: # torch.tensor
        x = x.to(device)
    else:               # list of torch.tensor
        for i in range(len(x)):
            x[i] = x[i].to(device)
    return x

def to_numpy(x):
    if type(x) != list: # torch.tensor
        x = x.detach().cpu().numpy()
    else:               # list of torch.tensor
        for i in range(len(x)):
            x[i] = x[i].detach().cpu().numpy()
    return x

def write_prediction(opt, ys, preds, labels, pad_label_id, default_label):
    # load test data
    tot_num_line = sum(1 for _ in open(opt.test_path, 'r')) 
    with open(opt.test_path, 'r', encoding='utf-8') as f:
        data = []
        bucket = []
        for idx, line in enumerate(tqdm(f, total=tot_num_line)):
            line = line.strip()
            if line == "":
                data.append(bucket)
                bucket = []
            else:
                entry = line.split()
                assert(len(entry) == 4)
                bucket.append(entry)
        if len(bucket) != 0:
            data.append(bucket)
    # write prediction
    try:
        pred_path = opt.test_path + '.pred'
        with open(pred_path, 'w', encoding='utf-8') as f:
            for i, bucket in enumerate(data):      # foreach sentence
                # from preds
                j_bucket = 0
                for j in range(ys.shape[1]):       # foreach token
                    pred_label = default_label
                    if ys[i][j] != pad_label_id:
                        pred_label = labels[preds[i][j]]
                        entry = bucket[j_bucket]
                        entry.append(pred_label)
                        f.write(' '.join(entry) + '\n')
                        j_bucket += 1
                # remained
                for j, entry in enumerate(bucket): # foreach remained token
                    if j < j_bucket: continue
                    pred_label = default_label
                    entry = bucket[j]
                    entry.append(pred_label)
                    f.write(' '.join(entry) + '\n')
                f.write('\n')
    except Exception as e:
        logger.warn(str(e))

def evaluate(opt):
    # set config
    config = load_config(opt)
    config['device'] = opt.device
    config['opt'] = opt

    # set path
    if config['emb_class'] == 'glove':
        opt.data_path = os.path.join(opt.data_dir, 'test.txt.ids')
    if config['emb_class'] == 'bert':
        opt.data_path = os.path.join(opt.data_dir, 'test.txt.fs')
    if config['emb_class'] == 'elmo':
        opt.data_path = os.path.join(opt.data_dir, 'test.txt.ids')
    opt.embedding_path = os.path.join(opt.data_dir, 'embedding.npy')
    opt.label_path = os.path.join(opt.data_dir, 'label.txt')
    opt.pos_path = os.path.join(opt.data_dir, 'pos.txt')
    opt.test_path = os.path.join(opt.data_dir, 'test.txt')

    test_data_path = opt.data_path
    batch_size = opt.batch_size
    device = opt.device
    torch.set_num_threads(opt.num_thread)

    # prepare test dataset
    if config['emb_class'] == 'glove':
        test_loader = prepare_dataset(config, test_data_path, CoNLLGloveDataset, shuffle=False, num_workers=1)
    if config['emb_class'] == 'bert':
        test_loader = prepare_dataset(config, test_data_path, CoNLLBertDataset, shuffle=False, num_workers=1)
    if config['emb_class'] == 'elmo':
        test_loader = prepare_dataset(config, test_data_path, CoNLLElmoDataset, shuffle=False, num_workers=1)
 
    # load pytorch model checkpoint
    logger.info("[Loading model...]")
    if device == 'cpu':
        checkpoint = torch.load(opt.model_path, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(opt.model_path)

    # prepare model and load parameters
    if config['emb_class'] == 'glove':
        model = GloveLSTMCRF(config, opt.embedding_path, opt.label_path, opt.pos_path,
                             emb_non_trainable=True, use_crf=opt.use_crf)
    if config['emb_class'] == 'bert':
        from transformers import BertTokenizer, BertConfig, BertModel
        bert_tokenizer = BertTokenizer.from_pretrained(opt.bert_output_dir,
                                                       do_lower_case=opt.bert_do_lower_case)
        bert_model = BertModel.from_pretrained(opt.bert_output_dir)
        bert_config = bert_model.config
        ModelClass = BertLSTMCRF
        model = ModelClass(config, bert_config, bert_model, opt.label_path, opt.pos_path,
                           use_crf=opt.use_crf, use_pos=opt.bert_use_pos, disable_lstm=opt.bert_disable_lstm)
    if config['emb_class'] == 'elmo':
        from allennlp.modules.elmo import Elmo
        elmo_model = Elmo(opt.elmo_options_file, opt.elmo_weights_file, 2, dropout=0)
        model = ElmoLSTMCRF(config, elmo_model, opt.embedding_path, opt.label_path, opt.pos_path,
                             emb_non_trainable=True, use_crf=opt.use_crf)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    logger.info("[Loaded]")
 
    # evaluation
    model.eval()
    preds = None
    ys    = None
    n_batches = len(test_loader)
    total_examples = 0
    whole_st_time = time.time()
    with torch.no_grad():
        for i, (x,y) in enumerate(tqdm(test_loader, total=n_batches)):
            x = to_device(x, device)
            y = to_device(y, device)
            if opt.use_crf:
                logits, prediction = model(x)
            else:
                logits = model(x)
            if preds is None:
                if opt.use_crf:
                    preds = prediction
                else:
                    preds = to_numpy(logits)
                ys = to_numpy(y)
            else:
                if opt.use_crf:
                    preds = np.append(preds, prediction, axis=0)
                else:
                    preds = np.append(preds, to_numpy(logits), axis=0)
                ys = np.append(ys, to_numpy(y), axis=0)
            cur_examples = y.size(0)
            total_examples += cur_examples
    whole_time = int((time.time()-whole_st_time)*1000)
    avg_time = whole_time / total_examples
    if not opt.use_crf: preds = np.argmax(preds, axis=2)
    # compute measure using seqeval
    labels = model.labels
    ys_lbs = [[] for _ in range(ys.shape[0])]
    preds_lbs = [[] for _ in range(ys.shape[0])]
    pad_label_id = config['pad_label_id']
    for i in range(ys.shape[0]):     # foreach sentence
        for j in range(ys.shape[1]): # foreach token
            if ys[i][j] != pad_label_id:
                ys_lbs[i].append(labels[ys[i][j]])
                preds_lbs[i].append(labels[preds[i][j]])
    ret = {
        "precision": precision_score(ys_lbs, preds_lbs),
        "recall": recall_score(ys_lbs, preds_lbs),
        "f1": f1_score(ys_lbs, preds_lbs)
    }
    f1 = ret['f1']
    # write predicted labels to file
    default_label = config['default_label']
    write_prediction(opt, ys, preds, labels, pad_label_id, default_label)

    logger.info("[F1] : {}, {}".format(f1, total_examples))
    logger.info("[Elapsed Time] : {}ms, {}ms on average".format(whole_time, avg_time))

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, default='data/conll2003')
    parser.add_argument('--config', type=str, default='config-glove.json')
    parser.add_argument('--model_path', type=str, default='pytorch-model.pt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_thread', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--use_crf', action='store_true', help="add CRF layer")
    # for BERT
    parser.add_argument('--bert_do_lower_case', action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--bert_output_dir', type=str, default='bert-checkpoint',
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--bert_disable_lstm', action='store_true',
                        help="disable lstm layer")
    parser.add_argument('--bert_use_pos', action='store_true', help="add Part-Of-Speech features")
    # for ELMo
    parser.add_argument('--elmo_options_file', type=str, default='embeddings/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json')
    parser.add_argument('--elmo_weights_file', type=str, default='embeddings/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5')

    opt = parser.parse_args()

    evaluate(opt) 

if __name__ == '__main__':
    main()
