from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
import time
import pdb
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    pass
import numpy as np
import random
from seqeval.metrics import precision_score, recall_score, f1_score

from util    import load_config, to_device, to_numpy
from model   import GloveLSTMCRF, GloveDensenetCRF, BertLSTMCRF, ElmoLSTMCRF
from dataset import prepare_dataset, CoNLLGloveDataset, CoNLLBertDataset, CoNLLElmoDataset
from progbar import Progbar # instead of tqdm
from early_stopping import EarlyStopping

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(opt):
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

def set_apex_and_distributed(opt):
    if not APEX_AVAILABLE: opt.use_amp = False
    opt.distributed = False
    if 'WORLD_SIZE' in os.environ and opt.use_amp:
        opt.world_size = int(os.environ['WORLD_SIZE'])
        opt.distributed = int(os.environ['WORLD_SIZE']) > 1
    if opt.distributed:
        torch.cuda.set_device(opt.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        opt.word_size = torch.distributed.get_world_size()

def train_epoch(model, config, train_loader, val_loader, epoch_i):
    device = config['device']
    opt = config['opt']

    optimizer = config['optimizer']
    scheduler = config['scheduler']
    writer = config['writer']
    pad_label_id = config['pad_label_id']

    local_rank = opt.local_rank
    use_amp = opt.use_amp
    criterion = nn.CrossEntropyLoss(ignore_index=pad_label_id).to(device)
    n_batches = len(train_loader)
    prog = Progbar(target=n_batches)
    # train one epoch
    model.train()
    train_loss = 0.
    st_time = time.time()
    for local_step, (x,y) in enumerate(train_loader):
        global_step = (len(train_loader) * epoch_i) + local_step
        x = to_device(x, device)
        y = to_device(y, device)
        if opt.use_crf:
            logits, log_likelihood, prediction = model(x, tags=y)
            loss = -1 * log_likelihood
        else:
            logits = model(x)
            # reshape for computing loss
            logits_view = logits.view(-1, model.label_size)
            y_view = y.view(-1)
            loss = criterion(logits_view, y_view)
        # back-propagation - begin
        optimizer.zero_grad()
        if use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        if scheduler and epoch_i > opt.warmup_epoch and local_step == 0: # after warmup, for every epoch
            scheduler.step()
        # back-propagation - end
        train_loss += loss.item()
        if local_rank == 0 and writer:
            writer.add_scalar('Loss/train', loss.item(), global_step)
        curr_lr = scheduler.get_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
        prog.update(local_step+1,
                    [('global step', global_step),
                     ('train curr loss', loss.item()),
                     ('lr', curr_lr)])
    train_loss = train_loss / n_batches

    # evaluate
    eval_ret = evaluate(model, config, val_loader, device)
    eval_loss = eval_ret['loss']
    eval_f1 = eval_ret['f1']
    curr_time = time.time()
    elapsed_time = (curr_time - st_time) / 60
    st_time = curr_time
    curr_lr = scheduler.get_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
    if local_rank == 0:
        logger.info('{:3d} epoch | {:5d}/{:5d} | train loss : {:10.6f}, valid loss {:10.6f}, valid f1 {:.4f}| lr :{:7.6f} | {:5.2f} min elapsed'.\
                format(epoch_i, local_step+1, len(train_loader), train_loss, eval_loss, eval_f1, curr_lr, elapsed_time)) 
        if writer:
            writer.add_scalar('Loss/valid', eval_loss, global_step)
            writer.add_scalar('F1/valid', eval_f1, global_step)
            writer.add_scalar('LearningRate/train', curr_lr, global_step)
    return eval_loss, eval_f1
 
def evaluate(model, config, val_loader, device):
    model.eval()
    opt = config['opt']
    pad_label_id = config['pad_label_id']

    eval_loss = 0.
    criterion = nn.CrossEntropyLoss(ignore_index=pad_label_id).to(device)
    n_batches = len(val_loader)
    prog = Progbar(target=n_batches)
    preds = None
    ys    = None
    with torch.no_grad():
        for i, (x,y) in enumerate(val_loader):
            x = to_device(x, device)
            y = to_device(y, device)
            if opt.use_crf:
                logits, log_likelihood, prediction = model(x, y)
                loss = -1 * log_likelihood
            else:
                logits = model(x)
                loss = criterion(logits.view(-1, model.label_size), y.view(-1))
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
            eval_loss += loss.item()
            prog.update(i+1,
                        [('eval curr loss', loss.item())])
    eval_loss = eval_loss / n_batches
    if not opt.use_crf: preds = np.argmax(preds, axis=2)
    # compute measure using seqeval
    labels = model.labels
    ys_lbs = [[] for _ in range(ys.shape[0])]
    preds_lbs = [[] for _ in range(ys.shape[0])]
    for i in range(ys.shape[0]):     # foreach sentence 
        for j in range(ys.shape[1]): # foreach token
            if ys[i][j] != pad_label_id:
                ys_lbs[i].append(labels[ys[i][j]])
                preds_lbs[i].append(labels[preds[i][j]])
    ret = {
        "loss": eval_loss,
        "precision": precision_score(ys_lbs, preds_lbs),
        "recall": recall_score(ys_lbs, preds_lbs),
        "f1": f1_score(ys_lbs, preds_lbs)
    }
    return ret

def save_model(model, opt, config):
    checkpoint_path = opt.save_path
    with open(checkpoint_path, 'wb') as f:
        if opt.use_amp:
            checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'amp': amp.state_dict()
                    }
        else:
            checkpoint = model.state_dict()
        torch.save(checkpoint,f)

def train(opt):
    device = torch.device(opt.device)
    set_seed(opt)
    set_apex_and_distributed(opt)

    # set config
    config = load_config(opt)
    config['device'] = device
    config['opt'] = opt
    logger.info("%s", config)
  
    # prepare train, valid dataset
    if config['emb_class'] == 'glove':
        filepath = os.path.join(opt.data_dir, 'train.txt.ids')
        train_loader = prepare_dataset(config, filepath, CoNLLGloveDataset, shuffle=True, num_workers=2)
        filepath = os.path.join(opt.data_dir, 'valid.txt.ids')
        valid_loader = prepare_dataset(config, filepath, CoNLLGloveDataset, shuffle=False, num_workers=2)
    if config['emb_class'] == 'bert':
        filepath = os.path.join(opt.data_dir, 'train.txt.fs')
        train_loader = prepare_dataset(config, filepath, CoNLLBertDataset, shuffle=True, num_workers=2)
        filepath = os.path.join(opt.data_dir, 'valid.txt.fs')
        valid_loader = prepare_dataset(config, filepath, CoNLLBertDataset, shuffle=False, num_workers=2)
    if config['emb_class'] == 'elmo':
        filepath = os.path.join(opt.data_dir, 'train.txt.ids')
        train_loader = prepare_dataset(config, filepath, CoNLLElmoDataset, shuffle=True, num_workers=2)
        filepath = os.path.join(opt.data_dir, 'valid.txt.ids')
        valid_loader = prepare_dataset(config, filepath, CoNLLElmoDataset, shuffle=False, num_workers=2)

    label_path = os.path.join(opt.data_dir, opt.label_filename)
    pos_path = os.path.join(opt.data_dir, opt.pos_filename)
    # prepare model
    if config['emb_class'] == 'glove':
        if config['enc_class'] == 'bilstm':
            embedding_path = os.path.join(opt.data_dir, opt.embedding_filename)
            emb_non_trainable = not opt.embedding_trainable
            model = GloveLSTMCRF(config, embedding_path, label_path, pos_path,
                                 emb_non_trainable=emb_non_trainable, use_crf=opt.use_crf)
        if config['enc_class'] == 'densenet':
            embedding_path = os.path.join(opt.data_dir, opt.embedding_filename)
            emb_non_trainable = not opt.embedding_trainable
            model = GloveDensenetCRF(config, embedding_path, label_path, pos_path,
                                     emb_non_trainable=emb_non_trainable, use_crf=opt.use_crf)
    if config['emb_class'] == 'bert':
        from transformers import BertTokenizer, BertConfig, BertModel
        bert_tokenizer = BertTokenizer.from_pretrained(opt.bert_model_name_or_path,
                                                       do_lower_case=opt.bert_do_lower_case)
        bert_model = BertModel.from_pretrained(opt.bert_model_name_or_path,
                                               from_tf=bool(".ckpt" in opt.bert_model_name_or_path))
        bert_config = bert_model.config
        ModelClass = BertLSTMCRF
        model = ModelClass(config, bert_config, bert_model, label_path, pos_path,
                           use_crf=opt.use_crf, use_pos=opt.bert_use_pos, disable_lstm=opt.bert_disable_lstm, feature_based=opt.bert_use_feature_based)
    if config['emb_class'] == 'elmo':
        from allennlp.modules.elmo import Elmo
        embedding_path = os.path.join(opt.data_dir, opt.embedding_filename)
        emb_non_trainable = not opt.embedding_trainable
        elmo_model = Elmo(opt.elmo_options_file, opt.elmo_weights_file, 2, dropout=0)
        model = ElmoLSTMCRF(config, elmo_model, embedding_path, label_path, pos_path,
                            emb_non_trainable=emb_non_trainable, use_crf=opt.use_crf)
    model.to(device)
    print(model)
    logger.info("[Model prepared]")

    # create optimizer, scheduler, summary writer
    logger.info("[Creating optimizer, scheduler, summary writer...]")
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2norm)
    if opt.use_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt.opt_level)
    if opt.distributed:
        model = DDP(model, delay_allreduce=True)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=opt.decay_rate)
    try:
        writer = SummaryWriter(log_dir=opt.log_dir)
    except:
        writer = None
    logger.info("[Ready]")

    # training

    # additional config setting for parameter passing
    config['optimizer'] = optimizer
    config['scheduler'] = scheduler
    config['writer'] = writer

    early_stopping = EarlyStopping(logger, patience=10, measure='f1', verbose=1)
    best_val_loss = float('inf')
    best_val_f1 = -float('inf')
    for epoch_i in range(opt.epoch):
        epoch_st_time = time.time()
        eval_loss, eval_f1 = train_epoch(model, config, train_loader, valid_loader, epoch_i)
        if early_stopping.validate(eval_f1, measure='f1'): break
        if opt.local_rank == 0 and eval_f1 > best_val_f1:
            best_val_f1 = eval_f1
            if opt.save_path:
                logger.info("[Best model saved] : {:10.6f}".format(best_val_f1))
                save_model(model, opt, config)
                if config['emb_class'] == 'bert':
                    if not os.path.exists(opt.bert_output_dir):
                        os.makedirs(opt.bert_output_dir)
                    bert_tokenizer.save_pretrained(opt.bert_output_dir)
                    bert_model.save_pretrained(opt.bert_output_dir)
            early_stopping.reset(best_val_f1)
        early_stopping.status()

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, default='data/conll2003')
    parser.add_argument('--embedding_filename', type=str, default='embedding.npy')
    parser.add_argument('--label_filename', type=str, default='label.txt')
    parser.add_argument('--pos_filename', type=str, default='pos.txt')
    parser.add_argument('--config', type=str, default='config-glove.json')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--decay_rate', type=float, default=1.0)
    parser.add_argument('--warmup_epoch', type=int, default=4)
    parser.add_argument('--l2norm', type=float, default=1e-6)
    parser.add_argument('--save_path', type=str, default='pytorch-model-glove.pt')
    parser.add_argument('--tmax',type=int, default=-1)
    parser.add_argument('--opt-level', type=str, default='O1')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--log_dir', type=str, default='runs')
    parser.add_argument('--seed', default=5, type=int)
    parser.add_argument('--use_crf', action='store_true', help="add CRF layer")
    parser.add_argument('--embedding_trainable', action='store_true')
    # for BERT
    parser.add_argument('--bert_model_name_or_path', type=str, default='bert-base-uncased',
                        help="Path to pre-trained model or shortcut name(ex, bert-base-uncased)")
    parser.add_argument('--bert_do_lower_case', action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--bert_output_dir', type=str, default='bert-checkpoint',
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--bert_use_feature_based', action='store_true',
                        help="use BERT as feature-based, default fine-tuning")
    parser.add_argument('--bert_disable_lstm', action='store_true',
                        help="disable lstm layer")
    parser.add_argument('--bert_use_pos', action='store_true', help="add Part-Of-Speech features")
    # for ELMo
    parser.add_argument('--elmo_options_file', type=str, default='embeddings/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json')
    parser.add_argument('--elmo_weights_file', type=str, default='embeddings/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5')

    opt = parser.parse_args()

    train(opt)
 
if __name__ == '__main__':
    main()
