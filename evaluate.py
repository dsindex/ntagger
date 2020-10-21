from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
import json
import time
import pdb
import logging

import torch
import torch.quantization
import torch.nn as nn
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

from tqdm import tqdm
from util import load_config, to_device, to_numpy
from model import GloveLSTMCRF, GloveDensenetCRF, BertLSTMCRF, ElmoLSTMCRF
from dataset import prepare_dataset, CoNLLGloveDataset, CoNLLBertDataset, CoNLLElmoDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_path(config):
    opt = config['opt']
    if config['emb_class'] in ['glove', 'elmo']:
        opt.data_path = os.path.join(opt.data_dir, 'test.txt.ids')
    else:
        opt.data_path = os.path.join(opt.data_dir, 'test.txt.fs')
    opt.embedding_path = os.path.join(opt.data_dir, 'embedding.npy')
    opt.label_path = os.path.join(opt.data_dir, 'label.txt')
    opt.pos_path = os.path.join(opt.data_dir, 'pos.txt')
    opt.test_path = os.path.join(opt.data_dir, 'test.txt')
    opt.vocab_path = os.path.join(opt.data_dir, 'vocab.txt')

def load_checkpoint(config):
    opt = config['opt']
    if opt.device == 'cpu':
        checkpoint = torch.load(opt.model_path, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(opt.model_path)
    logger.info("[Loading checkpoint done]")
    return checkpoint

def load_model(config, checkpoint):
    opt = config['opt']
    if config['emb_class'] == 'glove':
        if config['enc_class'] == 'bilstm':
            model = GloveLSTMCRF(config, opt.embedding_path, opt.label_path, opt.pos_path,
                                 emb_non_trainable=True, use_crf=opt.use_crf, use_char_cnn=opt.use_char_cnn)
        if config['enc_class'] == 'densenet':
            model = GloveDensenetCRF(config, opt.embedding_path, opt.label_path, opt.pos_path,
                                     emb_non_trainable=True, use_crf=opt.use_crf, use_char_cnn=opt.use_char_cnn)
    elif config['emb_class'] == 'elmo':
        from allennlp.modules.elmo import Elmo
        elmo_model = Elmo(opt.elmo_options_file, opt.elmo_weights_file, 2, dropout=0)
        model = ElmoLSTMCRF(config, elmo_model, opt.embedding_path, opt.label_path, opt.pos_path,
                            emb_non_trainable=True, use_crf=opt.use_crf, use_char_cnn=opt.use_char_cnn)
    else:
        from transformers import AutoTokenizer, AutoConfig, AutoModel
        bert_config = AutoConfig.from_pretrained(opt.bert_output_dir)
        bert_tokenizer = AutoTokenizer.from_pretrained(opt.bert_output_dir)
        bert_model = AutoModel.from_config(bert_config)
        ModelClass = BertLSTMCRF
        model = ModelClass(config, bert_config, bert_model, bert_tokenizer, opt.label_path, opt.pos_path,
                           use_crf=opt.use_crf, use_pos=opt.bert_use_pos, disable_lstm=opt.bert_disable_lstm,
                           feature_based=opt.bert_use_feature_based)
    model.load_state_dict(checkpoint)
    model = model.to(opt.device)
    logger.info("[Loaded]")
    return model

def convert_onnx(config, torch_model, x):
    opt = config['opt']
    import torch.onnx

    if config['emb_class'] in ['glove', 'elmo']:
        input_names = ['token_ids', 'pos_ids', 'char_ids']
        output_names = ['logits']
        dynamic_axes = {'token_ids': {0: 'batch', 1: 'sequence'},
                        'pos_ids':   {0: 'batch', 1: 'sequence'},
                        'char_ids' : {0: 'batch', 1: 'sequence'},
                        'logits':    {0: 'batch', 1: 'sequence'}}
        if opt.use_crf:
            output_names += ['prediction']
            dynamic_axes['prediction'] = {0: 'batch', 1: 'sequence'}
    else:
        input_names = ['input_ids', 'input_mask', 'segment_ids', 'pos_ids']
        output_names = ['logits']
        dynamic_axes = {'input_ids':   {0: 'batch', 1: 'sequence'},
                        'input_mask':  {0: 'batch', 1: 'sequence'},
                        'segment_ids': {0: 'batch', 1: 'sequence'},
                        'pos_ids':     {0: 'batch', 1: 'sequence'},
                        'logits':      {0: 'batch', 1: 'sequence'}}
        if opt.use_crf:
            output_names += ['prediction']
            dynamic_axes['prediction'] = {0: 'batch', 1: 'sequence'}
        
    with torch.no_grad():
        torch.onnx.export(torch_model,                  # model being run
                          x,                            # model input (or a tuple for multiple inputs)
                          opt.onnx_path,                # where to save the model (can be a file or file-like object)
                          export_params=True,           # store the trained parameter weights inside the model file
                          opset_version=opt.onnx_opset, # the ONNX version to export the model to
                          do_constant_folding=True,     # whether to execute constant folding for optimization
                          verbose=True,
                          input_names=input_names,      # the model's input names
                          output_names=output_names,    # the model's output names
                          dynamic_axes=dynamic_axes)    # variable length axes

# ------------------------------------------------------------------------------ #
# source code from https://github.com/huggingface/transformers/blob/master/src/transformers/convert_graph_to_onnx.py#L374
# ------------------------------------------------------------------------------ #
def quantize_onnx(onnx_path, quantized_onnx_path):
    """
    Quantize the weights of the model from float32 to in8 to allow very efficient inference on modern CPU.
    """
    import onnx
    from onnxruntime.quantization import QuantizationMode, quantize

    onnx_model = onnx.load(onnx_path)

    # Discussed with @yufenglee from ONNX runtime, this will be address in the next release of onnxruntime
    print(
        "As of onnxruntime 1.4.0, models larger than 2GB will fail to quantize due to protobuf constraint.\n"
        "This limitation will be removed in the next release of onnxruntime."
    )

    quantized_model = quantize(
        model=onnx_model,
        quantization_mode=QuantizationMode.IntegerOps,
        force_fusions=True,
        symmetric_weight=True,
    )

    # Save model
    onnx.save_model(quantized_model, quantized_onnx_path)

def check_onnx(config):
    opt = config['opt']
    import onnx
    onnx_model = onnx.load(opt.onnx_path)
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))

# ---------------------------------------------------------------------------- #
# Evaluation
# ---------------------------------------------------------------------------- #

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
                if i >= ys.shape[0]:
                    logger.info("Stop to write predictions: %s" % (i))
                    break
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

def prepare_datasets(config):
    opt = config['opt']
    if config['emb_class'] == 'glove':
        DatasetClass = CoNLLGloveDataset
    elif config['emb_class'] == 'elmo':
        DatasetClass = CoNLLElmoDataset
    else:
        DatasetClass = CoNLLBertDataset
    test_loader = prepare_dataset(config, opt.data_path, DatasetClass, sampling=False, num_workers=1)
    return test_loader

def evaluate(opt):
    # set config
    config = load_config(opt)
    if opt.num_threads > 0: torch.set_num_threads(opt.num_threads)
    config['opt'] = opt
    logger.info("%s", config)

    # set path
    set_path(config)

    # prepare test dataset
    test_loader = prepare_datasets(config)
 
    # load pytorch model checkpoint
    checkpoint = load_checkpoint(config)

    # prepare model and load parameters
    model = load_model(config, checkpoint)
    model.eval()

    # convert to onnx format
    if opt.convert_onnx:
        (x, y) = next(iter(test_loader))
        x = to_device(x, opt.device)
        y = to_device(y, opt.device)
        convert_onnx(config, model, x)
        check_onnx(config)
        logger.info("[ONNX model saved at {}".format(opt.onnx_path))
        # quantize onnx
        if opt.quantize_onnx:
            quantize_onnx(opt.onnx_path, opt.quantized_onnx_path)
            logger.info("[Quantized ONNX model saved at {}".format(opt.quantized_onnx_path))
        return

    # load onnx model for using onnxruntime
    if opt.enable_ort:
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = opt.num_threads
        sess_options.intra_op_num_threads = opt.num_threads
        ort_session = ort.InferenceSession(opt.onnx_path, sess_options=sess_options)
    
    # enable to use dynamic quantized model (pytorch>=1.3.0)
    if opt.enable_dqm and opt.device == 'cpu':
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        print(model)
 
    # evaluation
    preds = None
    ys    = None
    n_batches = len(test_loader)
    total_examples = 0
    whole_st_time = time.time()
    first_time = time.time()
    first_examples = 0
    total_duration_time = 0.0
    with torch.no_grad():
        for i, (x,y) in enumerate(tqdm(test_loader, total=n_batches)):
            start_time = time.time()
            x = to_device(x, opt.device)
            y = to_device(y, opt.device)

            if opt.enable_ort:
                x = to_numpy(x)
                if config['emb_class'] in ['glove', 'elmo']:
                    ort_inputs = {ort_session.get_inputs()[0].name: x[0],
                                  ort_session.get_inputs()[1].name: x[1]}
                    if opt.use_char_cnn:
                        ort_inputs[ort_session.get_inputs()[2].name] = x[2]
                else:
                    if config['emb_class'] in ['distilbert', 'bart']:
                        ort_inputs = {ort_session.get_inputs()[0].name: x[0],
                                      ort_session.get_inputs()[1].name: x[1]}
                    else:
                        ort_inputs = {ort_session.get_inputs()[0].name: x[0],
                                      ort_session.get_inputs()[1].name: x[1],
                                      ort_session.get_inputs()[2].name: x[2]}
                    if opt.bert_use_pos:
                        ort_inputs[ort_session.get_inputs()[3].name] = x[3]
                if opt.use_crf:
                    logits, prediction = ort_session.run(None, ort_inputs)
                    prediction = to_device(torch.tensor(prediction), opt.device)
                    logits = to_device(torch.tensor(logits), opt.device)
                else:
                    logits = ort_session.run(None, ort_inputs)[0]
                    logits = to_device(torch.tensor(logits), opt.device)
            else:
                if opt.use_crf: logits, prediction = model(x)
                else: logits = model(x)

            if preds is None:
                if opt.use_crf: preds = to_numpy(prediction)
                else: preds = to_numpy(logits)
                ys = to_numpy(y)
            else:
                if opt.use_crf: preds = np.append(preds, to_numpy(prediction), axis=0)
                else: preds = np.append(preds, to_numpy(logits), axis=0)
                ys = np.append(ys, to_numpy(y), axis=0)
            cur_examples = y.size(0)
            total_examples += cur_examples
            if i == 0: # first one may take longer time, so ignore in computing duration.
                first_time = float((time.time()-first_time)*1000)
                first_examples = cur_examples
            if opt.num_examples != 0 and total_examples >= opt.num_examples:
                logger.info("[Stop Evaluation] : up to the {} examples".format(total_examples))
                break
            duration_time = float((time.time()-start_time)*1000)
            if i != 0: total_duration_time += duration_time
            '''
            logger.info("[Elapsed Time] : {}ms".format(duration_time))
            '''
    whole_time = float((time.time()-whole_st_time)*1000)
    avg_time = (whole_time - first_time) / (total_examples - first_examples)
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
        "f1": f1_score(ys_lbs, preds_lbs),
        "report": classification_report(ys_lbs, preds_lbs, digits=4),
    }
    print(ret['report'])
    f1 = ret['f1']
    # write predicted labels to file
    default_label = config['default_label']
    write_prediction(opt, ys, preds, labels, pad_label_id, default_label)

    logger.info("[F1] : {}, {}".format(f1, total_examples))
    logger.info("[Elapsed Time] : {} examples, {}ms, {}ms on average".format(total_examples, whole_time, avg_time))
    logger.info("[Elapsed Time(total_duration_time, average)] : {}ms, {}ms".format(total_duration_time, total_duration_time/(total_examples-1)))

# ---------------------------------------------------------------------------- #
# Inference : not yet implemented
# ---------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', type=str, default='configs/config-glove.json')
    parser.add_argument('--data_dir', type=str, default='data/conll2003')
    parser.add_argument('--model_path', type=str, default='pytorch-model-glove.pt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_threads', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_examples', default=0, type=int, help="Number of examples to evaluate, 0 means all of them.")
    parser.add_argument('--use_crf', action='store_true', help="Add CRF layer")
    parser.add_argument('--use_char_cnn', action='store_true', help="Add Character features")
    # for BERT
    parser.add_argument('--bert_output_dir', type=str, default='bert-checkpoint',
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--bert_use_feature_based', action='store_true',
                        help="Use BERT as feature-based, default fine-tuning")
    parser.add_argument('--bert_disable_lstm', action='store_true',
                        help="Disable lstm layer")
    parser.add_argument('--bert_use_pos', action='store_true', help="Add Part-Of-Speech features")
    # for ELMo
    parser.add_argument('--elmo_options_file', type=str, default='embeddings/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json')
    parser.add_argument('--elmo_weights_file', type=str, default='embeddings/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5')
    # for ONNX
    parser.add_argument('--convert_onnx', action='store_true',
                        help="Set this flag to convert to onnx format.")
    parser.add_argument('--enable_ort', action='store_true',
                        help="Set this flag to evaluate using onnxruntime.")
    parser.add_argument('--onnx_path', type=str, default='pytorch-model.onnx')
    parser.add_argument('--onnx_opset', default=11, type=int, help="ONNX opset version.")
    parser.add_argument('--quantize_onnx', action='store_true',
                        help="Set this flag to quantize ONNX.")
    parser.add_argument('--quantized_onnx_path', type=str, default='pytorch-model.onnx-quantized')
    # for Quantization
    parser.add_argument('--enable_dqm', action='store_true',
                        help="Set this flag to use dynamic quantized model.")

    opt = parser.parse_args()

    evaluate(opt) 

if __name__ == '__main__':
    main()
