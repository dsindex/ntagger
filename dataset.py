import os
import pdb

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_dataset(config, filepath, DatasetClass, sampling=False, num_workers=1, batch_size=0, hp_search_bsz=None):
    opt = config['opt']
    dataset = DatasetClass(config, filepath)

    if sampling:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    bz = opt.batch_size
    if batch_size > 0: bz = batch_size
    # for optuna
    if hp_search_bsz: bz = hp_search_bsz

    loader = DataLoader(dataset, batch_size=bz, num_workers=num_workers, sampler=sampler)
    logger.info("[{} data loaded]".format(filepath))
    return loader

class CoNLLGloveDataset(Dataset):
    def __init__(self, config, path):
        from allennlp.modules.elmo import batch_to_ids
        pad_ids = [config['pad_token_id']] * config['char_n_ctx']
        all_token_ids = []
        all_pos_ids = []
        all_char_ids = []
        all_label_ids = []
        with open(path,'r',encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                items = line.split('\t')
                token_ids = [int(d) for d in items[1].split()]
                pos_ids   = [int(d) for d in items[2].split()]
                # using ELMo.batch_to_ids, compute character ids: ex) 'The' [259, 85, 105, 102, 260, 261, 261, ...]
                # (actually byte-based, char_vocab_size == 262, char_padding_idx == 261)
                tokens    = items[3].split()
                char_ids  = batch_to_ids([tokens])[0].detach().cpu().numpy().tolist()
                for _ in range(len(token_ids) - len(char_ids)):
                    char_ids.append(pad_ids)
                label_ids = [int(d) for d in items[0].split()]
                all_token_ids.append(token_ids)
                all_pos_ids.append(pos_ids)
                all_char_ids.append(char_ids)
                all_label_ids.append(label_ids)
        all_token_ids = torch.tensor(all_token_ids, dtype=torch.long)
        all_pos_ids = torch.tensor(all_pos_ids, dtype=torch.long)
        all_char_ids = torch.tensor(all_char_ids, dtype=torch.long)
        all_label_ids = torch.tensor(all_label_ids, dtype=torch.long)

        self.x = TensorDataset(all_token_ids, all_pos_ids, all_char_ids)
        self.y = all_label_ids
 
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class CoNLLBertDataset(Dataset):
    def __init__(self, config, path):
        features = torch.load(path)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_pos_ids = torch.tensor([f.pos_ids for f in features], dtype=torch.long)
        all_char_ids = torch.tensor([f.char_ids for f in features], dtype=torch.long)
        all_word2token_idx = None
        all_word2token_mask = None
        if hasattr(features[0], 'word2token_idx'):
            all_word2token_idx = torch.tensor([f.word2token_idx for f in features], dtype=torch.long)
            all_word2token_mask = torch.tensor([f.word2token_mask for f in features], dtype=torch.long)
        all_word_ids = None
        if hasattr(features[0], 'word_ids'):
            all_word_ids = torch.tensor([f.word_ids for f in features], dtype=torch.long)
        all_doc2sent_idx = None
        if hasattr(features[0], 'doc2sent_idx'):
            all_doc2sent_idx = torch.tensor([f.doc2sent_idx for f in features], dtype=torch.long)
            all_doc2sent_mask = torch.tensor([f.doc2sent_mask for f in features], dtype=torch.long)

        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

        all_glabel_ids = torch.tensor([f.glabel_id for f in features], dtype=torch.long)
        
        # argument order must be sync with x parameter of BertLSTMCRF.forward().
        args = [all_input_ids, all_input_mask, all_segment_ids, all_pos_ids, all_char_ids]
        if all_doc2sent_idx != None:
            args += [all_doc2sent_idx, all_doc2sent_mask]
        if all_word2token_idx != None:
            if all_word_ids != None:
                args += [all_word2token_idx, all_word2token_mask, all_word_ids]
            else:
                args += [all_word2token_idx, all_word2token_mask]

        self.x = TensorDataset(*args)
        self.y = all_label_ids
        self.gy = all_glabel_ids
 
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.gy[idx]


class CoNLLElmoDataset(Dataset):
    def __init__(self, config, path):
        from allennlp.modules.elmo import batch_to_ids
        pad_ids = [config['pad_token_id']] * config['char_n_ctx']
        all_token_ids = []
        all_pos_ids = []
        all_char_ids = []
        all_label_ids = []
        with open(path,'r',encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                items = line.split('\t')
                token_ids = [int(d) for d in items[1].split()]
                pos_ids   = [int(d) for d in items[2].split()]
                # compute ELMo character ids
                tokens    = items[3].split()
                char_ids  = batch_to_ids([tokens])[0].detach().cpu().numpy().tolist()
                for _ in range(len(token_ids) - len(char_ids)):
                    char_ids.append(pad_ids)
                label_ids = [int(d) for d in items[0].split()]
                all_token_ids.append(token_ids)
                all_pos_ids.append(pos_ids)
                all_char_ids.append(char_ids)
                all_label_ids.append(label_ids)
        all_token_ids = torch.tensor(all_token_ids, dtype=torch.long)
        all_pos_ids = torch.tensor(all_pos_ids, dtype=torch.long)
        all_char_ids = torch.tensor(all_char_ids, dtype=torch.long)
        all_label_ids = torch.tensor(all_label_ids, dtype=torch.long)

        self.x = TensorDataset(all_token_ids, all_pos_ids, all_char_ids)
        self.y = all_label_ids
 
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
