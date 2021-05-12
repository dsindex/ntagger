import sys
import os
import argparse
import time
import pdb
import json
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from accelerate import Accelerator
from transformers import AdamW, get_linear_schedule_with_warmup

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    pass

import numpy as np
import random
import json
from tqdm import tqdm

from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import classification_report as sequence_classification_report, confusion_matrix
from util    import load_checkpoint, load_config, load_dict
from model   import GloveLSTMCRF, GloveDensenetCRF, BertLSTMCRF, ElmoLSTMCRF
from dataset import prepare_dataset, CoNLLGloveDataset, CoNLLBertDataset, CoNLLElmoDataset
from early_stopping import EarlyStopping
from label_smoothing import LabelSmoothingCrossEntropy
import optuna
from datasets.metric import temp_seed 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_epoch(model, config, train_loader, valid_loader, epoch_i, best_eval_f1):
    opt = config['opt']
    accelerator = config['accelerator'] 

    optimizer = None
    scheduler = None
    optimizer_1st = config['optimizer']
    scheduler_1st = config['scheduler']
    optimizer_2nd = config['optimizer_2nd']
    scheduler_2nd = config['scheduler_2nd']
    writer = config['writer']
    pad_label_id = config['pad_label_id']
    optimizer = optimizer_1st
    scheduler = scheduler_1st
    freeze_bert = False
    if opt.bert_freezing_epoch > 0:
        # apply second optimizer/scheduler during freezing epochs
        if epoch_i < opt.bert_freezing_epoch and optimizer_2nd != None and scheduler_2nd != None:
            optimizer = optimizer_2nd
            scheduler = scheduler_2nd
            freeze_bert = True

    if opt.criterion == 'LabelSmoothingCrossEntropy':
        criterion = LabelSmoothingCrossEntropy(ignore_index=pad_label_id, reduction='sum')
        g_criterion = LabelSmoothingCrossEntropy(ignore_index=pad_label_id, reduction='sum')
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=pad_label_id)
        g_criterion = nn.CrossEntropyLoss(ignore_index=pad_label_id)

    n_batches = len(train_loader)

    # train one epoch
    train_loss = 0.
    avg_loss = 0.
    local_best_eval_loss = float('inf')
    local_best_eval_f1 = 0
    st_time = time.time()
    optimizer.zero_grad()
    epoch_iterator = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch_i}")
    for local_step, batch in enumerate(epoch_iterator):
        if config['emb_class'] not in ['glove', 'elmo']:
            x, y, gy = batch
        else:
            x, y = batch

        model.train()
        global_step = (len(train_loader) * epoch_i) + local_step
        gloss = 0.
        if opt.use_crf:
            mask = torch.sign(torch.abs(x[1])).to(torch.uint8)
            if config['emb_class'] not in ['glove', 'elmo']:
                if opt.bert_use_mtl:
                    logits, prediction, glogits = model(x, freeze_bert=freeze_bert)
                    gloss = g_criterion(glogits, gy)
                else:
                    logits, prediction = model(x, freeze_bert=freeze_bert)
            else:
                logits, prediction = model(x)
            log_likelihood = model.crf(logits, y, mask=mask, reduction='mean')
            loss = -1 * log_likelihood
            loss = loss + gloss
        else:
            if config['emb_class'] not in ['glove', 'elmo']:
                if opt.bert_use_mtl:
                    logits, glogits = model(x, freeze_bert=freeze_bert)
                    gloss = g_criterion(glogits, gy)
                else:
                    logits = model(x, freeze_bert=freeze_bert)
            else:
                logits = model(x)
            # reshape for computing loss
            logits_view = logits.view(-1, config['label_size'])
            y_view = y.view(-1)
            loss = criterion(logits_view, y_view)
            loss = loss + gloss

        if opt.gradient_accumulation_steps > 1:
           loss = loss / opt.gradient_accumulation_steps

        # back-propagation - begin
        accelerator.backward(loss)
        if (local_step + 1) % opt.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            curr_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
            epoch_iterator.set_description(f"Epoch {epoch_i}, global_step: {global_step}, local_step: {local_step}, loss: {loss:.3f}, gloss: {gloss:.3f},curr_lr: {curr_lr:.7f}")
            if accelerator.is_main_process and opt.eval_and_save_steps > 0 and global_step != 0 and global_step % opt.eval_and_save_steps == 0:
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
                        unwrapped_model = accelerator.unwrap_model(model)
                        save_model(config, unwrapped_model)
                        logger.info("[Best model saved] : {}, {}".format(eval_loss, eval_f1))
                        # save finetuned bert model/config/tokenizer
                        if config['emb_class'] not in ['glove', 'elmo']:
                            if not os.path.exists(opt.bert_output_dir):
                                os.makedirs(opt.bert_output_dir)
                            unwrapped_model.bert_tokenizer.save_pretrained(opt.bert_output_dir)
                            unwrapped_model.bert_model.save_pretrained(opt.bert_output_dir)
                            logger.info("[Pretrained bert model saved] : {}, {}".format(eval_loss, eval_f1))
        # back-propagation - end
        train_loss += loss.item()
        if writer: writer.add_scalar('Loss/train', loss.item(), global_step)
    avg_loss = train_loss / n_batches

    # evaluate at the end of epoch
    if accelerator.is_main_process:
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
                unwrapped_model = accelerator.unwrap_model(model)
                save_model(config, unwrapped_model)
                logger.info("[Best model saved] : {}, {}".format(eval_loss, eval_f1))
                # save finetuned bert model/config/tokenizer
                if config['emb_class'] not in ['glove', 'elmo']:
                    if not os.path.exists(opt.bert_output_dir):
                        os.makedirs(opt.bert_output_dir)
                    unwrapped_model.bert_tokenizer.save_pretrained(opt.bert_output_dir)
                    unwrapped_model.bert_model.save_pretrained(opt.bert_output_dir)
                    logger.info("[Pretrained bert model saved] : {}, {}".format(eval_loss, eval_f1))

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
 
def evaluate(model, config, valid_loader):
    opt = config['opt']
    pad_label_id = config['pad_label_id']

    eval_loss = 0.
    criterion = nn.CrossEntropyLoss(ignore_index=pad_label_id)
    g_criterion = nn.CrossEntropyLoss(ignore_index=pad_label_id)
    n_batches = len(valid_loader)
    preds = None
    ys    = None
    gpreds = None
    gys    = None
    with torch.no_grad():
        iterator = tqdm(valid_loader, total=len(valid_loader), desc=f"Evaluate")
        for i, batch in enumerate(iterator):
            if config['emb_class'] not in ['glove', 'elmo']:
                x, y, gy = batch
            else:
                x, y = batch

            model.eval()
            mask = torch.sign(torch.abs(x[1])).to(torch.uint8)
            gloss = 0.
            if opt.use_crf:
                if opt.bert_use_mtl:
                    logits, prediction, glogits = model(x)
                    gloss = g_criterion(glogits, gy)
                else:
                    logits, prediction = model(x)
                log_likelihood = model.crf(logits, y, mask=mask, reduction='mean')
                loss = -1 * log_likelihood
                loss = loss + gloss
                logits = logits.cpu().numpy()
                prediction = prediction.cpu().numpy()
            else:
                if opt.bert_use_mtl:
                    logits, glogits = model(x)
                    gloss = g_criterion(glogits, gy)
                else:
                    logits = model(x)
                loss = criterion(logits.view(-1, config['label_size']), y.view(-1))
                loss = loss + gloss
                # softmax after computing cross entropy loss
                logits = torch.softmax(logits, dim=-1)
                logits = logits.cpu().numpy()

            y = y.cpu().numpy()
            if preds is None:
                if opt.use_crf: preds = prediction
                else: preds = logits
                ys = y
            else:
                if opt.use_crf: preds = np.append(preds, prediction, axis=0)
                else: preds = np.append(preds, logits, axis=0)
                ys = np.append(ys, y, axis=0)

            if opt.bert_use_mtl:
                glogits = torch.softmax(glogits, dim=-1)
                glogits = glogits.cpu().numpy()
                gy = gy.cpu().numpy()
                if gpreds is None:
                    gpreds = glogits
                    gys = gy
                else:
                    gpreds = np.append(gpreds, glogits, axis=0)
                    gys = np.append(gys, gy, axis=0)

            eval_loss += loss.item()

    # generate report for token classification
    eval_loss = eval_loss / n_batches
    if not opt.use_crf: preds = np.argmax(preds, axis=2)
    # compute measure using seqeval
    labels = config['labels']
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
    opt.glabel_path = os.path.join(opt.data_dir, opt.glabel_filename)
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
    emb_non_trainable = not opt.embedding_trainable
    if config['emb_class'] == 'glove':
        if config['enc_class'] == 'bilstm':
            model = GloveLSTMCRF(config, opt.embedding_path, label_size, pos_size,
                                 emb_non_trainable=emb_non_trainable, use_crf=opt.use_crf,
                                 use_char_cnn=opt.use_char_cnn, use_mha=opt.use_mha)
        if config['enc_class'] == 'densenet':
            model = GloveDensenetCRF(config, opt.embedding_path, label_size, pos_size,
                                     emb_non_trainable=emb_non_trainable, use_crf=opt.use_crf,
                                     use_char_cnn=opt.use_char_cnn, use_mha=opt.use_mha)
    elif config['emb_class'] == 'elmo':
        from allennlp.modules.elmo import Elmo
        elmo_model = Elmo(opt.elmo_options_file, opt.elmo_weights_file, 2, dropout=0)
        model = ElmoLSTMCRF(config, elmo_model, opt.embedding_path, label_size, pos_size,
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
        model = ModelClass(config, bert_config, bert_model, bert_tokenizer, label_size, glabel_size, pos_size,
                           use_crf=opt.use_crf, use_pos=opt.bert_use_pos,
                           use_char_cnn=opt.use_char_cnn, use_mha=opt.use_mha,
                           use_subword_pooling=opt.bert_use_subword_pooling, use_word_embedding=opt.bert_use_word_embedding,
                           embedding_path=opt.embedding_path, emb_non_trainable=emb_non_trainable,
                           use_doc_context=opt.bert_use_doc_context,
                           disable_lstm=opt.bert_disable_lstm,
                           feature_based=opt.bert_use_feature_based,
                           use_mtl=opt.bert_use_mtl)
    if opt.restore_path:
        checkpoint = load_checkpoint(opt.restore_path)
        model.load_state_dict(checkpoint)
    logger.info("[model] :\n{}".format(model.__str__()))
    logger.info("[model prepared]")
    return model

def prepare_others(config, model, data_loader, lr=None, weight_decay=None):
    opt = config['opt']
    accelerator = config['accelerator']

    default_lr = opt.lr
    if lr: default_lr = lr
    default_weight_decay = opt.weight_decay
    if weight_decay: default_weight_decay = weight_decay

    num_update_steps_per_epoch = math.ceil(len(data_loader) / opt.gradient_accumulation_steps)
    if opt.max_train_steps is None:
        opt.max_train_steps = opt.epoch * num_update_steps_per_epoch
    else:
        opt.epoch = math.ceil(opt.max_train_steps / num_update_steps_per_epoch)
    if opt.num_warmup_steps is None: 
        if opt.warmup_ratio:
            opt.num_warmup_steps = opt.max_train_steps * opt.warmup_epoch
        if opt.warmup_epoch:
            opt.num_warmup_steps = num_update_steps_per_epoch * opt.warmup_epoch

    logger.info(f"(num_update_steps_per_epoch, max_train_steps, num_warmup_steps): ({num_update_steps_per_epoch}, {opt.max_train_steps}, {opt.num_warmup_steps})")

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': default_weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=default_lr, eps=opt.adam_epsilon)

    model, optimizer = accelerator.prepare(model, optimizer)

    scheduler = get_linear_schedule_with_warmup(optimizer,
        num_warmup_steps=opt.num_warmup_steps,
        num_training_steps=opt.max_train_steps)

    try:
        writer = SummaryWriter(log_dir=opt.log_dir)
    except:
        writer = None

    return model, optimizer, scheduler, writer

def train(opt):

    # set etc
    torch.autograd.set_detect_anomaly(True)

    # set config
    config = load_config(opt)
    config['opt'] = opt
    logger.info("%s", config)
 
    # set path
    set_path(config)

    # create accelerator
    accelerator = Accelerator()
    config['accelerator'] = accelerator
    opt.device = accelerator.device

    # prepare train, valid dataset
    train_loader, valid_loader = prepare_datasets(config)

    with temp_seed(opt.seed):
        # prepare model
        model = prepare_model(config)

        # create optimizer, scheduler, summary writer
        model, optimizer, scheduler, writer = prepare_others(config, model, train_loader)
        # create secondary optimizer, scheduler
        _, optimizer_2nd, scheduler_2nd, _= prepare_others(config, model, train_loader, lr=opt.bert_lr_during_freezing)
        train_loader = accelerator.prepare(train_loader)
        valid_loader = accelerator.prepare(valid_loader)

        config['optimizer'] = optimizer
        config['scheduler'] = scheduler
        config['optimizer_2nd'] = optimizer_2nd
        config['scheduler_2nd'] = scheduler_2nd
        config['writer'] = writer

        total_batch_size = opt.batch_size * accelerator.num_processes * opt.gradient_accumulation_steps
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_loader)}")
        logger.info(f"  Num Epochs = {opt.epoch}")
        logger.info(f"  Instantaneous batch size per device = {opt.batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {opt.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {opt.max_train_steps}")

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

    global gopt
    opt = gopt
    # set config
    config = load_config(opt)
    config['opt'] = opt
    logger.info("%s", config)

    # set path
    set_path(config)

    # create accelerator
    accelerator = Accelerator()
    config['accelerator'] = accelerator
    opt.device = accelerator.device

    # set search spaces
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    bsz = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    seed = trial.suggest_int('seed', 17, 42)
    epochs = trial.suggest_int('epochs', 1, opt.epoch)

    # prepare train, valid dataset
    train_loader, valid_loader = prepare_datasets(config, hp_search_bsz=bsz)

    with temp_seed(seed):
        # prepare model
        model = prepare_model(config)

        # create optimizer, scheduler, summary writer
        model, optimizer, scheduler, writer = prepare_others(config, model, train_loader, lr=lr)
        # create secondary optimizer, scheduler
        _, optimizer_2nd, scheduler_2nd, _ = prepare_others(config, model, train_loader, lr=opt.bert_lr_during_freezing)
        train_loader = accelerator.prepare(train_loader)
        valid_loader = accelerator.prepare(valid_loader)

        config['optimizer'] = optimizer
        config['scheduler'] = scheduler
        config['optimizer_2nd'] = optimizer_2nd
        config['scheduler_2nd'] = scheduler_2nd
        config['writer'] = writer

        total_batch_size = opt.batch_size * accelerator.num_processes * opt.gradient_accumulation_steps
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_loader)}")
        logger.info(f"  Num Epochs = {opt.epoch}")
        logger.info(f"  Instantaneous batch size per device = {opt.batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {opt.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {opt.max_train_steps}")

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
    parser.add_argument('--glabel_filename', type=str, default='glabel.txt')
    parser.add_argument('--pos_filename', type=str, default='pos.txt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--max_train_steps', type=int, default=None)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--eval_and_save_steps', type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_warmup_steps', type=int, default=None)
    parser.add_argument('--warmup_epoch', type=int, default=0,  help="Number of warmup epoch")
    parser.add_argument('--warmup_ratio', type=float, default=0.0, help="Ratio for warmup over total number of training steps.")
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
    parser.add_argument('--criterion', type=str, default='CrossEntropyLoss', help="training objective, 'CrossEntropyLoss' | 'LabelSmoothingCrossEntropy', default 'CrossEntropyLoss'")
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
