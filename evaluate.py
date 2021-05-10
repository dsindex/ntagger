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
from sklearn.metrics import classification_report as sequence_classification_report, confusion_matrix

from tqdm import tqdm
from util import load_checkpoint, load_config, load_dict, to_device, to_numpy
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
    opt.glabel_path = os.path.join(opt.data_dir, 'glabel.txt')
    opt.pos_path = os.path.join(opt.data_dir, 'pos.txt')
    opt.test_path = os.path.join(opt.data_dir, 'test.txt')
    opt.vocab_path = os.path.join(opt.data_dir, 'vocab.txt')

def load_model(config, checkpoint):
    opt = config['opt']
    labels = load_dict(opt.label_path)
    label_size = len(labels)
    config['labels'] = labels
    config['label_size'] = label_size
    glabels = load_dict(opt.glabel_path)
    glabel_size = len(glabels)
    config['glabels'] = glabels
    config['glabel_size'] = glabel_size
    poss = load_dict(opt.pos_path)
    pos_size = len(poss)
    config['poss'] = poss
    config['pos_size'] = pos_size
    if config['emb_class'] == 'glove':
        if config['enc_class'] == 'bilstm':
            model = GloveLSTMCRF(config, opt.embedding_path, label_size, pos_size,
                                 emb_non_trainable=True, use_crf=opt.use_crf,
                                 use_char_cnn=opt.use_char_cnn, use_mha=opt.use_mha)
        if config['enc_class'] == 'densenet':
            model = GloveDensenetCRF(config, opt.embedding_path, label_size, pos_size,
                                     emb_non_trainable=True, use_crf=opt.use_crf,
                                     use_char_cnn=opt.use_char_cnn, use_mha=opt.use_mha)
    elif config['emb_class'] == 'elmo':
        from allennlp.modules.elmo import Elmo
        elmo_model = Elmo(opt.elmo_options_file, opt.elmo_weights_file, 2, dropout=0)
        model = ElmoLSTMCRF(config, elmo_model, opt.embedding_path, label_size, pos_size,
                            emb_non_trainable=True, use_crf=opt.use_crf,
                            use_char_cnn=opt.use_char_cnn, use_mha=opt.use_mha)
    else:
        from transformers import AutoTokenizer, AutoConfig, AutoModel
        bert_config = AutoConfig.from_pretrained(opt.bert_output_dir)
        bert_tokenizer = AutoTokenizer.from_pretrained(opt.bert_output_dir)
        bert_model = AutoModel.from_config(bert_config)
        ModelClass = BertLSTMCRF
        model = ModelClass(config, bert_config, bert_model, bert_tokenizer, label_size, glabel_size, pos_size,
                           use_crf=opt.use_crf, use_pos=opt.bert_use_pos,
                           use_char_cnn=opt.use_char_cnn, use_mha=opt.use_mha,
                           use_subword_pooling=opt.bert_use_subword_pooling, use_word_embedding=opt.bert_use_word_embedding,
                           embedding_path=opt.embedding_path, emb_non_trainable=True,
                           use_doc_context=opt.bert_use_doc_context,
                           disable_lstm=opt.bert_disable_lstm,
                           feature_based=opt.bert_use_feature_based,
                           use_mtl=opt.bert_use_mtl)
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
        input_names = ['input_ids', 'input_mask', 'segment_ids', 'pos_ids', 'char_ids']
        output_names = ['logits']
        dynamic_axes = {'input_ids':   {0: 'batch', 1: 'sequence'},
                        'input_mask':  {0: 'batch', 1: 'sequence'},
                        'segment_ids': {0: 'batch', 1: 'sequence'},
                        'pos_ids':     {0: 'batch', 1: 'sequence'},
                        'char_ids':    {0: 'batch', 1: 'sequence'},
                        'logits':      {0: 'batch', 1: 'sequence'}}
        if opt.bert_use_doc_context:
            input_name += ['doc2sent_idx', 'doc2sent_mask']
            dynamic_axes['doc2sent_idx'] = {0: 'batch', 1: 'sequence'}
            dynamic_axes['doc2sent_mask'] = {0: 'batch', 1: 'sequence'}
        if opt.bert_use_subword_pooling:
            input_names += ['word2token_idx', 'word2token_mask']
            dynamic_axes['word2token_idx'] = {0: 'batch', 1: 'sequence'}
            dynamic_axes['word2token_mask'] = {0: 'batch', 1: 'sequence'}
            if opt.bert_use_word_embedding:
                input_names += ['word_ids']
                dynamic_axes['word_ids'] = {0: 'batch', 1: 'sequence'}
        if opt.use_crf:
            output_names += ['prediction']
            dynamic_axes['prediction'] = {0: 'batch', 1: 'sequence'}
        if opt.bert_use_mtl:
            output_names += ['glogits']
            dynamic_axes['glogits'] = {0: 'batch'}
       
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

def quantize_onnx(onnx_path, quantized_onnx_path):
    import onnx
    from onnxruntime.quantization import QuantizationMode, quantize

    onnx_model = onnx.load(onnx_path)

    quantized_model = quantize(
        model=onnx_model,
        quantization_mode=QuantizationMode.IntegerOps,
        force_fusions=True,
        symmetric_weight=True,
    )

    onnx.save_model(quantized_model, quantized_onnx_path)

def check_onnx(config):
    opt = config['opt']
    import onnx
    onnx_model = onnx.load(opt.onnx_path)
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))

def build_onnx_input(config, ort_session, x):
    opt = config['opt']
    x = to_numpy(x)
    if config['emb_class'] in ['glove', 'elmo']:
        ort_inputs = {ort_session.get_inputs()[0].name: x[0],
                      ort_session.get_inputs()[1].name: x[1]}
        if opt.use_char_cnn:
            ort_inputs[ort_session.get_inputs()[2].name] = x[2]
    else:
        # x order must be sync with x parameter of BertLSTMCRF.forward().
        # x[0,1,2] : [batch_size, seq_size], input_ids / input_mask / segment_ids == input_ids / attention_mask / token_type_ids
        # x[3] :     [batch_size, seq_size], pos_ids
        # x[4] :     [batch_size, seq_size, char_n_ctx], char_ids

        # with --bert_use_doc_context
        # x[5] :     [batch_size, seq_size], doc2sent_idx
        # x[6] :     [batch_size, seq_size], doc2sent_mask
        # x[7] :     [batch_size, seq_size], word2token_idx  with --bert_use_subword_pooling
        # x[8] :     [batch_size, seq_size], word2token_mask with --bert_use_subword_pooling
        # x[9] :     [batch_size, seq_size], word_ids        with --bert_use_word_embedding

        # without --bert_use_doc_context
        # x[5] :     [batch_size, seq_size], word2token_idx  with --bert_use_subword_pooling
        # x[6] :     [batch_size, seq_size], word2token_mask with --bert_use_subword_pooling
        # x[7] :     [batch_size, seq_size], word_ids        with --bert_use_word_embedding
        if config['emb_class'] in ['roberta', 'distilbert', 'bart']:
            ort_inputs = {ort_session.get_inputs()[0].name: x[0],
                          ort_session.get_inputs()[1].name: x[1]}
        else:
            ort_inputs = {ort_session.get_inputs()[0].name: x[0],
                          ort_session.get_inputs()[1].name: x[1],
                          ort_session.get_inputs()[2].name: x[2]}
        if opt.bert_use_pos:
            ort_inputs[ort_session.get_inputs()[3].name] = x[3]
        if opt.use_char_cnn:
            ort_inputs[ort_session.get_inputs()[4].name] = x[4]
        base_idx = 5
        if opt.bert_use_doc_context:
            ort_inputs[ort_session.get_inputs()[base_idx].name] = x[base_idx]
            ort_inputs[ort_session.get_inputs()[base_idx+1].name] = x[base_idx+1]
            base_idx += 2
        if opt.bert_use_subword_pooling:
            ort_inputs[ort_session.get_inputs()[base_idx].name] = x[base_idx]
            ort_inputs[ort_session.get_inputs()[base_idx+1].name] = x[base_idx+1]
            if opt.bert_use_word_embedding:
                ort_inputs[ort_session.get_inputs()[base_idx+2].name] = x[base_idx+2]
    return ort_inputs

# ---------------------------------------------------------------------------- #
# Evaluation
# ---------------------------------------------------------------------------- #

def write_prediction(config, model, ys, preds, labels):
    opt = config['opt']
    pad_label_id = config['pad_label_id']
    default_label = config['default_label']

    # load test data
    tot_num_line = sum(1 for _ in open(opt.test_path, 'r')) 
    with open(opt.test_path, 'r', encoding='utf-8') as f:
        data = []
        bucket = []
        for idx, line in enumerate(tqdm(f, total=tot_num_line)):
            line = line.strip()
            if line == "":
                if opt.bert_use_mtl:
                    bucket = bucket[1:]
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
                    logger.info("[Stop to write predictions] : %s" % (i))
                    break
                use_subtoken = False
                ys_idx = 0
                if config['emb_class'] not in ['glove', 'elmo']:
                    use_subtoken = True
                    ys_idx = 1 # account '[CLS]'
                if opt.bert_use_subword_pooling:
                    use_subtoken = False
                for j, entry in enumerate(bucket): # foreach token
                    entry = bucket[j]
                    pred_label = default_label
                    if ys_idx < ys.shape[1]:
                        pred_label = labels[preds[i][ys_idx]]
                    entry.append(pred_label)
                    f.write(' '.join(entry) + '\n')
                    if use_subtoken:
                        word = entry[0]
                        word_tokens = model.bert_tokenizer.tokenize(word)
                        ys_idx += len(word_tokens)
                    else:
                        ys_idx += 1
                f.write('\n')
    except Exception as e:
        logger.warn(str(e))

def write_gprediction(opt, gpreds, glabels):
    # load test data
    tot_num_line = sum(1 for _ in open(opt.test_path, 'r')) 
    with open(opt.test_path, 'r', encoding='utf-8') as f:
        data = []
        is_next_bos = True
        for idx, line in enumerate(tqdm(f, total=tot_num_line)):
            line = line.strip()
            if line == "":
                is_next_bos = True
                continue
            tokens = line.split()
            if is_next_bos:
                glabel = tokens[0]
            is_next_bos = False
            data.append(glabel)
    # write prediction
    try:
        gpred_path = opt.test_path + '.gpred'
        with open(gpred_path, 'w', encoding='utf-8') as f:
            for glabel, gpred in zip(data, gpreds):
                gpred_id = np.argmax(gpred)
                gpred_label = glabels[gpred_id]
                f.write(glabel + '\t' + gpred_label + '\n')
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
    checkpoint = load_checkpoint(opt.model_path, device=opt.device)

    # prepare model and load parameters
    model = load_model(config, checkpoint)
    model.eval()

    # convert to onnx format
    if opt.convert_onnx:
        # FIXME not working for --use_crf
        batch = next(iter(test_loader))
        if config['emb_class'] not in ['glove', 'elmo']:
            x, y, gy = batch
        else:
            x, y = batch
        x = to_device(x, opt.device)
        convert_onnx(config, model, x)
        check_onnx(config)
        logger.info("[ONNX model saved] : {}".format(opt.onnx_path))
        # quantize onnx
        if opt.quantize_onnx:
            quantize_onnx(opt.onnx_path, opt.quantized_onnx_path)
            logger.info("[Quantized ONNX model saved] : {}".format(opt.quantized_onnx_path))
        return

    # load onnx model for using onnxruntime
    if opt.enable_ort:
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = opt.num_threads
        sess_options.intra_op_num_threads = opt.num_threads
        ort_session = ort.InferenceSession(opt.onnx_path, sess_options=sess_options)
    
    # enable to use dynamic quantized model (pytorch>=1.3.0)
    if opt.enable_dqm:
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        print(model)
 
    # evaluation
    preds = None
    ys    = None
    gpreds = None
    gys    = None
    n_batches = len(test_loader)
    total_examples = 0
    whole_st_time = time.time()
    first_time = time.time()
    first_examples = 0
    total_duration_time = 0.0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, total=n_batches)):
            start_time = time.time()
            if config['emb_class'] not in ['glove', 'elmo']:
                x, y, gy = batch
                gy = to_device(gy, opt.device)
            else:
                x, y = batch

            x = to_device(x, opt.device)
            y = to_device(y, opt.device)
            if opt.enable_ort:
                ort_inputs = build_onnx_input(config, ort_session, x)
                if opt.use_crf:
                    # FIXME not working for --use_crf
                    if opt.bert_use_mtl:
                        logits, prediction, glogits = ort_session.run(None, ort_inputs)
                        glogits = to_device(torch.tensor(glogits), opt.device)
                    else:
                        logits, prediction = ort_session.run(None, ort_inputs)
                    prediction = to_device(torch.tensor(prediction), opt.device)
                    logits = to_device(torch.tensor(logits), opt.device)
                else:
                    if opt.bert_use_mtl:
                        logits, glogits = ort_session.run(None, ort_inputs)
                        glogits = to_device(torch.tensor(glogits), opt.device)
                    else:
                        logits = ort_session.run(None, ort_inputs)[0]
                    logits = to_device(torch.tensor(logits), opt.device)
                    logits = torch.softmax(logits, dim=-1)
            else:
                if opt.use_crf:
                    if opt.bert_use_mtl:
                        logits, prediction, glogits = model(x)
                    else:
                        logits, prediction = model(x)
                else:
                    if opt.bert_use_mtl:
                        logits, glogits = model(x)
                    else:
                        logits = model(x)
                    logits = torch.softmax(logits, dim=-1)

            if preds is None:
                if opt.use_crf: preds = to_numpy(prediction)
                else: preds = to_numpy(logits)
                ys = to_numpy(y)
            else:
                if opt.use_crf: preds = np.append(preds, to_numpy(prediction), axis=0)
                else: preds = np.append(preds, to_numpy(logits), axis=0)
                ys = np.append(ys, to_numpy(y), axis=0)

            if opt.bert_use_mtl:
                glogits = torch.softmax(glogits, dim=-1)
                if gpreds is None:
                    gpreds = to_numpy(glogits)
                    gys = to_numpy(gy)
                else:
                    gpreds = np.append(gpreds, to_numpy(glogits), axis=0)
                    gys = np.append(gys, to_numpy(gy), axis=0)

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

    # generate report for token classification
    if not opt.use_crf: preds = np.argmax(preds, axis=2)
    # compute measure using seqeval
    labels = config['labels']
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
    # write predicted labels to file
    write_prediction(config, model, ys, preds, labels)

    # generate report for sequence classification
    if opt.bert_use_mtl:
        glabels = config['glabels']
        glabel_names = [v for k, v in sorted(glabels.items(), key=lambda x: x[0])]
        glabel_ids = [k for k in glabels.keys()]
        gpreds_ids = np.argmax(gpreds, axis=1)
        try:
            g_report = sequence_classification_report(gys, gpreds_ids, target_names=glabel_names, labels=glabel_ids, digits=4)
            g_report_dict = sequence_classification_report(gys, gpreds_ids, target_names=glabel_names, labels=glabel_ids, output_dict=True)
            g_matrix = confusion_matrix(gys, gpreds_ids)
            ret['g_report'] = g_report
            ret['g_report_dict'] = g_report_dict
            ret['g_f1'] = g_report_dict['micro avg']['f1-score']
            ret['g_matrix'] = g_matrix
        except Exception as e:
            logger.warn(str(e))
        print(ret['g_report'])
        print(ret['g_f1'])
        print(ret['g_matrix'])
        logger.info("[sequence classification F1] : {}, {}".format(ret['g_f1'], total_examples))
        # write predicted glabels to file
        write_gprediction(opt, gpreds, glabels)

    logger.info("[token classification F1] : {}, {}".format(ret['f1'], total_examples))
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
    parser.add_argument('--use_mha', action='store_true', help="Add Multi-Head Attention layer.")
    # for BERT
    parser.add_argument('--bert_output_dir', type=str, default='bert-checkpoint',
                        help="The checkpoint directory of fine-tuned BERT model.")
    parser.add_argument('--bert_use_feature_based', action='store_true',
                        help="Use BERT as feature-based, default fine-tuning")
    parser.add_argument('--bert_disable_lstm', action='store_true',
                        help="Disable lstm layer")
    parser.add_argument('--bert_use_pos', action='store_true', help="Add Part-Of-Speech features")
    parser.add_argument('--bert_use_subword_pooling', action='store_true',
                        help="Set this flag for bert subword pooling.")
    parser.add_argument('--bert_use_word_embedding', action='store_true',
                        help="Set this flag to use word embedding(eg, GloVe). it should be used with --bert_use_subword_pooling.")
    parser.add_argument('--bert_use_doc_context', action='store_true',
                        help="Set this flag to use document-level context.")
    parser.add_argument('--bert_use_mtl', action='store_true',
                        help="Set this flag to use multi-task learning of token and sentence classification.")
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
