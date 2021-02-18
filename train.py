from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
import time
import pdb
import json
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torch.autograd.profiler as profiler

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    pass
import numpy as np
import random
import json
from tqdm import tqdm

from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from util    import load_checkpoint, load_config, to_device, to_numpy
from model   import GloveLSTMCRF, GloveDensenetCRF, BertLSTMCRF, ElmoLSTMCRF
from dataset import prepare_dataset, CoNLLGloveDataset, CoNLLBertDataset, CoNLLElmoDataset
from early_stopping import EarlyStopping
import optuna
from datasets.metric import temp_seed 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_epoch(model, config, train_loader, valid_loader, epoch_i, best_eval_f1):
    opt = config['opt']

    optimizer = None
    scheduler = None
    optimizer_1st = config['optimizer']
    scheduler_1st = config['scheduler']
    optimizer_2nd = config['optimizer_2nd']
    scheduler_2nd = config['scheduler_2nd']
    writer = config['writer']
    scaler = config['scaler']
    pad_label_id = config['pad_label_id']
    optimizer = optimizer_1st
    scheduler = scheduler_1st
    freeze_bert = False
    if opt.bert_freezing_epoch > 0:
        # apply second optimizer/scheduler during freezing epochs
        if epoch_i < opt.bert_freezing_epoch:
            optimizer = optimizer_2nd
            scheduler = scheduler_2nd
            freeze_bert = True

    criterion = nn.CrossEntropyLoss(ignore_index=pad_label_id).to(opt.device)
    n_batches = len(train_loader)

    # train one epoch
    train_loss = 0.
    avg_loss = 0.
    local_best_eval_loss = float('inf')
    local_best_eval_f1 = 0
    st_time = time.time()
    optimizer.zero_grad()
    epoch_iterator = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch_i}")
    for local_step, (x,y) in enumerate(epoch_iterator):
        model.train()
        global_step = (len(train_loader) * epoch_i) + local_step
        x = to_device(x, opt.device)
        y = to_device(y, opt.device)
        if opt.use_crf:
            with autocast(enabled=opt.use_amp):
                if opt.use_profiler:
                    with profiler.profile(profile_memory=True, record_shapes=True) as prof:
                        logits, prediction = model(x)
                    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
                else:
                    if config['emb_class'] not in ['glove', 'elmo']:
                        logits, prediction = model(x, freeze_bert=freeze_bert)
                    else:
                        logits, prediction = model(x)
                mask = torch.sign(torch.abs(x[0])).to(torch.uint8).to(opt.device)
                log_likelihood = model.crf(logits, y, mask=mask, reduction='mean')
                loss = -1 * log_likelihood
                if opt.gradient_accumulation_steps > 1:
                    loss = loss / opt.gradient_accumulation_steps
        else:
            with autocast(enabled=opt.use_amp):
                if opt.use_profiler:
                    with profiler.profile(profile_memory=True, record_shapes=True) as prof:
                        logits = model(x)
                    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
                else:
                    if config['emb_class'] not in ['glove', 'elmo']:
                        logits = model(x, freeze_bert=freeze_bert)
                    else:
                        logits = model(x)
                # reshape for computing loss
                logits_view = logits.view(-1, model.label_size)
                y_view = y.view(-1)
                loss = criterion(logits_view, y_view)
                if opt.gradient_accumulation_steps > 1:
                    loss = loss / opt.gradient_accumulation_steps
        # back-propagation - begin
        if opt.device == 'cpu':
            loss.backward()
        else:
            scaler.scale(loss).backward()
        if (local_step + 1) % opt.gradient_accumulation_steps == 0:
            if opt.device == 'cpu':
                optimizer.step()
            else:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            curr_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
            epoch_iterator.set_description(f"Epoch {epoch_i}, local_step: {local_step}, loss: {loss:.3f}, curr_lr: {curr_lr:.7f}")
            if opt.eval_and_save_steps > 0 and global_step != 0 and global_step % opt.eval_and_save_steps == 0:
                # evaluate
                eval_ret = evaluate(model, config, valid_loader)
                eval_loss = eval_ret['loss']
                eval_f1 = eval_ret['f1']
                if local_best_eval_loss > eval_loss: local_best_eval_loss = eval_loss
                if local_best_eval_f1 < eval_f1: local_best_eval_f1 = eval_f1
                if writer:
                    writer.add_scalar('Loss/valid', eval_loss, global_step)
                    writer.add_scalar('F1/valid', eval_f1, global_step)
                    writer.add_scalar('LearningRate/train', curr_lr, global_step)
                if eval_f1 > best_eval_f1:
                    best_eval_f1 = eval_f1
                    if opt.save_path and not opt.hp_search_optuna:
                        logger.info("[Best model saved] : {}, {}".format(eval_loss, eval_f1))
                        save_model(config, model)
                        # save finetuned bert model/config/tokenizer
                        if config['emb_class'] not in ['glove', 'elmo']:
                            if not os.path.exists(opt.bert_output_dir):
                                os.makedirs(opt.bert_output_dir)
                            model.bert_tokenizer.save_pretrained(opt.bert_output_dir)
                            model.bert_model.save_pretrained(opt.bert_output_dir)
        # back-propagation - end
        train_loss += loss.item()
        if writer: writer.add_scalar('Loss/train', loss.item(), global_step)
    avg_loss = train_loss / n_batches

    # evaluate at the end of epoch
    eval_ret = evaluate(model, config, valid_loader)
    eval_loss = eval_ret['loss']
    eval_f1 = eval_ret['f1']
    if local_best_eval_loss > eval_loss: local_best_eval_loss = eval_loss
    if local_best_eval_f1 < eval_f1: local_best_eval_f1 = eval_f1
    if writer:
        writer.add_scalar('Loss/valid', eval_loss, global_step)
        writer.add_scalar('F1/valid', eval_f1, global_step)
        writer.add_scalar('LearningRate/train', curr_lr, global_step)
    if eval_f1 > best_eval_f1:
        best_eval_f1 = eval_f1
        if opt.save_path and not opt.hp_search_optuna:
            logger.info("[Best model saved] : {}, {}".format(eval_loss, eval_f1))
            save_model(config, model)
            # save finetuned bert model/config/tokenizer
            if config['emb_class'] not in ['glove', 'elmo']:
                if not os.path.exists(opt.bert_output_dir):
                    os.makedirs(opt.bert_output_dir)
                model.bert_tokenizer.save_pretrained(opt.bert_output_dir)
                model.bert_model.save_pretrained(opt.bert_output_dir)

    curr_time = time.time()
    elapsed_time = (curr_time - st_time) / 60
    st_time = curr_time
    logs = {'epoch': epoch_i,
           'local_step': local_step+1,
           'epoch_step': len(train_loader),
           'avg_loss': avg_loss,
           'local_best_eval_loss': local_best_eval_loss,
           'local_best_eval_f1': local_best_eval_f1,
           'best_eval_f1': best_eval_f1,
           'elapsed_time': elapsed_time
    }
    logger.info(json.dumps(logs, indent=4, ensure_ascii=False, sort_keys=True))

    return local_best_eval_loss, local_best_eval_f1, best_eval_f1
 
def evaluate(model, config, valid_loader, eval_device=None):
    opt = config['opt']
    device = opt.device
    if eval_device != None: device = eval_device
    pad_label_id = config['pad_label_id']

    eval_loss = 0.
    criterion = nn.CrossEntropyLoss(ignore_index=pad_label_id).to(device)
    n_batches = len(valid_loader)
    preds = None
    ys    = None
    with torch.no_grad():
        iterator = tqdm(valid_loader, total=len(valid_loader), desc=f"Evaluate")
        for i, (x,y) in enumerate(iterator):
            model.eval()
            x = to_device(x, device)
            y = to_device(y, device)
            if opt.use_crf:
                logits, prediction = model(x)
                mask = torch.sign(torch.abs(x[0])).to(torch.uint8).to(device)
                log_likelihood = model.crf(logits, y, mask=mask, reduction='mean')
                loss = -1 * log_likelihood
            else:
                logits = model(x)
                loss = criterion(logits.view(-1, model.label_size), y.view(-1))
                # softmax after computing cross entropy loss
                logits = torch.softmax(logits, dim=-1)
            if preds is None:
                if opt.use_crf: preds = to_numpy(prediction)
                else: preds = to_numpy(logits)
                ys = to_numpy(y)
            else:
                if opt.use_crf: preds = np.append(preds, to_numpy(prediction), axis=0)
                else: preds = np.append(preds, to_numpy(logits), axis=0)
                ys = np.append(ys, to_numpy(y), axis=0)
            eval_loss += loss.item()
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
        "f1": f1_score(ys_lbs, preds_lbs),
        "report": classification_report(ys_lbs, preds_lbs, digits=4),
    }
    print(ret['report'])
    return ret

def save_model(config, model, save_path=None):
    opt = config['opt']
    optimizer = config['optimizer']
    checkpoint_path = opt.save_path
    if save_path: checkpoint_path = save_path
    with open(checkpoint_path, 'wb') as f:
        checkpoint = model.state_dict()
        torch.save(checkpoint,f)

def set_path(config):
    opt = config['opt']
    if config['emb_class'] in ['glove', 'elmo']:
        opt.train_path = os.path.join(opt.data_dir, 'train.txt.ids')
        opt.valid_path = os.path.join(opt.data_dir, 'valid.txt.ids')
    else:
        opt.train_path = os.path.join(opt.data_dir, 'train.txt.fs')
        opt.valid_path = os.path.join(opt.data_dir, 'valid.txt.fs')
    opt.label_path = os.path.join(opt.data_dir, opt.label_filename)
    opt.pos_path = os.path.join(opt.data_dir, opt.pos_filename)
    opt.embedding_path = os.path.join(opt.data_dir, opt.embedding_filename)

def prepare_datasets(config, hp_search_bsz=None, train_path=None, valid_path=None):
    opt = config['opt']
    default_train_path = opt.train_path
    default_valid_path = opt.valid_path
    if train_path: default_train_path = train_path
    if valid_path: default_valid_path = valid_path
    if config['emb_class'] == 'glove':
        DatasetClass = CoNLLGloveDataset
    elif config['emb_class'] == 'elmo':
        DatasetClass = CoNLLElmoDataset
    else:
        DatasetClass = CoNLLBertDataset
    train_loader = prepare_dataset(config,
            default_train_path,
            DatasetClass,
            sampling=True,
            num_workers=2,
            hp_search_bsz=hp_search_bsz)
    valid_loader = prepare_dataset(config,
            default_valid_path,
            DatasetClass,
            sampling=False,
            num_workers=2,
            batch_size=opt.eval_batch_size)
    return train_loader, valid_loader

def get_bert_embed_layer_list(config, bert_model):
    opt = config['opt']
    embed_list = list(bert_model.embeddings.parameters())
    # note that 'distilbert' has no encoder.layer, so don't use bert_remove_layers for distilbert.
    layer_list = bert_model.encoder.layer
    return embed_list, layer_list

def reduce_bert_model(config, bert_model, bert_config):
    opt = config['opt']
    remove_layers = opt.bert_remove_layers
    # drop layers
    if remove_layers is not "":
        embed_list, layer_list = get_bert_embed_layer_list(config, bert_model)
        layer_indexes = [int(x) for x in remove_layers.split(",")]
        layer_indexes.sort(reverse=True)
        for layer_idx in layer_indexes:
            if layer_idx < 0 or layer_idx >= bert_config.num_hidden_layers: continue
            del(layer_list[layer_idx])
            logger.info("[layer removed] : %s" % (layer_idx))
        if len(layer_indexes) > 0:
            bert_config.num_hidden_layers = len(layer_list)

def prepare_model(config):
    opt = config['opt']
    emb_non_trainable = not opt.embedding_trainable
    if config['emb_class'] == 'glove':
        if config['enc_class'] == 'bilstm':
            model = GloveLSTMCRF(config, opt.embedding_path, opt.label_path, opt.pos_path,
                                 emb_non_trainable=emb_non_trainable, use_crf=opt.use_crf,
                                 use_char_cnn=opt.use_char_cnn, use_mha=opt.use_mha)
        if config['enc_class'] == 'densenet':
            model = GloveDensenetCRF(config, opt.embedding_path, opt.label_path, opt.pos_path,
                                     emb_non_trainable=emb_non_trainable, use_crf=opt.use_crf,
                                     use_char_cnn=opt.use_char_cnn, use_mha=opt.use_mha)
    elif config['emb_class'] == 'elmo':
        from allennlp.modules.elmo import Elmo
        elmo_model = Elmo(opt.elmo_options_file, opt.elmo_weights_file, 2, dropout=0)
        model = ElmoLSTMCRF(config, elmo_model, opt.embedding_path, opt.label_path, opt.pos_path,
                            emb_non_trainable=emb_non_trainable, use_crf=opt.use_crf,
                            use_char_cnn=opt.use_char_cnn, use_mha=opt.use_mha)
    else:
        from transformers import AutoTokenizer, AutoConfig, AutoModel
        bert_tokenizer = AutoTokenizer.from_pretrained(opt.bert_model_name_or_path)
        bert_model = AutoModel.from_pretrained(opt.bert_model_name_or_path,
                                               from_tf=bool(".ckpt" in opt.bert_model_name_or_path))
        bert_config = bert_model.config
        # bert model reduction
        reduce_bert_model(config, bert_model, bert_config)
        ModelClass = BertLSTMCRF
        model = ModelClass(config, bert_config, bert_model, bert_tokenizer, opt.label_path, opt.pos_path,
                           use_crf=opt.use_crf, use_pos=opt.bert_use_pos, use_mha=opt.use_mha,
                           disable_lstm=opt.bert_disable_lstm,
                           feature_based=opt.bert_use_feature_based)
    if opt.restore_path:
        checkpoint = load_checkpoint(opt.restore_path, device=opt.device)
        model.load_state_dict(checkpoint)
    model.to(opt.device)
    logger.info("[model] :\n{}".format(model.__str__()))
    logger.info("[model prepared]")
    return model

def prepare_osws(config, model, train_loader, lr=None, weight_decay=None):
    opt = config['opt']

    default_lr = opt.lr
    if lr: default_lr = lr
    default_weight_decay = opt.weight_decay
    if weight_decay: default_weight_decay = weight_decay

    from transformers import AdamW, get_linear_schedule_with_warmup
    num_training_steps_for_epoch = len(train_loader) // opt.gradient_accumulation_steps
    num_training_steps = num_training_steps_for_epoch * opt.epoch
    num_warmup_steps = num_training_steps_for_epoch * opt.warmup_epoch
    logger.info("(num_training_steps_for_epoch, num_training_steps, num_warmup_steps): ({}, {}, {})".\
        format(num_training_steps_for_epoch, num_training_steps, num_warmup_steps))        
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': default_weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=default_lr, eps=opt.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps)
    try:
        writer = SummaryWriter(log_dir=opt.log_dir)
    except:
        writer = None
    scaler = GradScaler()
    logger.info("[Creating optimizer, scheduler, summary writer, scaler]")
    return optimizer, scheduler, writer, scaler

def train(opt):
    if torch.cuda.is_available():
        logger.info("%s", torch.cuda.get_device_name(0))

    # set etc
    torch.autograd.set_detect_anomaly(True)

    # set config
    config = load_config(opt)
    config['opt'] = opt
    logger.info("%s", config)
 
    # set path
    set_path(config)
  
    # prepare train, valid dataset
    train_loader, valid_loader = prepare_datasets(config)

    with temp_seed(opt.seed):
        # prepare model
        model = prepare_model(config)

        # create optimizer, scheduler, summary writer, scaler
        optimizer, scheduler, writer, scaler = prepare_osws(config, model, train_loader)
        # create secondary optimizer, scheduler
        optimizer_2nd, scheduler_2nd, _, _ = prepare_osws(config, model, train_loader, lr=opt.bert_lr_during_freezing)
        config['optimizer'] = optimizer
        config['scheduler'] = scheduler
        config['optimizer_2nd'] = optimizer_2nd
        config['scheduler_2nd'] = scheduler_2nd
        config['writer'] = writer
        config['scaler'] = scaler

        # training
        early_stopping = EarlyStopping(logger, patience=opt.patience, measure='f1', verbose=1)
        local_worse_epoch = 0
        best_eval_f1 = -float('inf')
        for epoch_i in range(opt.epoch):
            epoch_st_time = time.time()
            eval_loss, eval_f1, best_eval_f1 = train_epoch(model, config, train_loader, valid_loader, epoch_i, best_eval_f1)
            # early stopping
            if early_stopping.validate(eval_f1, measure='f1'): break
            if eval_f1 == best_eval_f1:
                early_stopping.reset(best_eval_f1)
            early_stopping.status()

# for optuna, global for passing opt 
gopt = None

def hp_search_optuna(trial: optuna.Trial):
    if torch.cuda.is_available():
        logger.info("%s", torch.cuda.get_device_name(0))

    global gopt
    opt = gopt
    # set config
    config = load_config(opt)
    config['opt'] = opt
    logger.info("%s", config)

    # set path
    set_path(config)

    # set search spaces
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    bsz = trial.suggest_categorical('batch_size', [32, 64, 128])
    seed = trial.suggest_int('seed', 17, 42)
    epochs = trial.suggest_int('epochs', 1, opt.epoch)

    # prepare train, valid dataset
    train_loader, valid_loader = prepare_datasets(config, hp_search_bsz=bsz)

    with temp_seed(seed):
        # prepare model
        model = prepare_model(config)
        # create optimizer, scheduler, summary writer, scaler
        optimizer, scheduler, writer, scaler = prepare_osws(config, model, train_loader, lr=lr)
        # create secondary optimizer, scheduler
        optimizer_2nd, scheduler_2nd, _, _ = prepare_osws(config, model, train_loader, lr=opt.bert_lr_during_freezing)
        config['optimizer'] = optimizer
        config['scheduler'] = scheduler
        config['optimizer_2nd'] = optimizer_2nd
        config['scheduler_2nd'] = scheduler_2nd
        config['writer'] = writer
        config['scaler'] = scaler

        early_stopping = EarlyStopping(logger, patience=opt.patience, measure='f1', verbose=1)
        best_eval_f1 = -float('inf')
        for epoch in range(epochs):
            eval_loss, eval_f1, best_eval_f1 = train_epoch(model, config, train_loader, valid_loader, epoch, best_eval_f1)

            # early stopping
            if early_stopping.validate(eval_f1, measure='f1'): break
            if eval_f1 == best_eval_f1:
                early_stopping.reset(best_eval_f1)
            early_stopping.status()

            trial.report(eval_f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return eval_f1

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', type=str, default='configs/config-glove.json')
    parser.add_argument('--data_dir', type=str, default='data/conll2003')
    parser.add_argument('--embedding_filename', type=str, default='embedding.npy')
    parser.add_argument('--label_filename', type=str, default='label.txt')
    parser.add_argument('--pos_filename', type=str, default='pos.txt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--eval_and_save_steps', type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup_epoch', type=int, default=0,  help="Number of warmup epoch")
    parser.add_argument('--patience', default=7, type=int, help="Max number of epoch to be patient for early stopping.")
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument('--save_path', type=str, default='pytorch-model-glove.pt')
    parser.add_argument('--restore_path', type=str, default='')
    parser.add_argument('--log_dir', type=str, default='runs')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--use_crf', action='store_true', help="Add CRF layer")
    parser.add_argument('--embedding_trainable', action='store_true', help="Set word embedding(Glove) trainable.")
    parser.add_argument('--use_char_cnn', action='store_true', help="Add Character features.")
    parser.add_argument('--use_mha', action='store_true', help="Add Multi-Head Attention layer.")
    parser.add_argument('--use_amp', action='store_true', help="Use automatic mixed precision.")
    parser.add_argument('--use_profiler', action='store_true', help="Use profiler.")
    # for BERT
    parser.add_argument('--bert_model_name_or_path', type=str, default='bert-base-uncased',
                        help="Path to pre-trained model or shortcut name(ex, bert-base-uncased)")
    parser.add_argument('--bert_output_dir', type=str, default='bert-checkpoint',
                        help="The output directory where the BERT model checkpoints will be written.")
    parser.add_argument('--bert_use_feature_based', action='store_true',
                        help="Use BERT as feature-based, default fine-tuning")
    parser.add_argument('--bert_disable_lstm', action='store_true',
                        help="Disable lstm layer")
    parser.add_argument('--bert_use_pos', action='store_true', help="Add Part-Of-Speech features")
    parser.add_argument('--bert_remove_layers', type=str, default='',
                        help="Specify layer numbers to remove during finetuning e.g. 8,9,10,11 to remove last 4 layers from BERT base(12 layers)")
    parser.add_argument('--bert_freezing_epoch', default=0, type=int,
                        help="Number of freezing epoch for BERT.")
    parser.add_argument('--bert_lr_during_freezing', type=float, default=1e-3,
                        help="The learning rate during freezing BERT.")
    # for ELMo
    parser.add_argument('--elmo_options_file', type=str, default='embeddings/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json')
    parser.add_argument('--elmo_weights_file', type=str, default='embeddings/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5')
    # for Optuna
    parser.add_argument('--hp_search_optuna', action='store_true',
                        help="Set this flag to use hyper-parameter search by Optuna.")
    parser.add_argument('--hp_trials', default=24, type=int,
                        help="Number of trials for hyper-parameter search.")

    opt = parser.parse_args()

    if opt.hp_search_optuna:
        global gopt
        gopt = opt
        study = optuna.create_study(direction='maximize')
        study.optimize(hp_search_optuna, n_trials=opt.hp_trials)
        df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
        print(df)
        logger.info("[study.best_params] : %s", study.best_params)
        logger.info("[study.best_value] : %s", study.best_value)
        logger.info("[study.best_trial] : %s", study.best_trial) # for all, study.trials
    else:
        train(opt)
 
if __name__ == '__main__':
    main()
