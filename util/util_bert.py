import sys
import os
import pdb

from allennlp.modules.elmo import batch_to_ids
from tqdm import tqdm

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------- #
# base code from
#     https://github.com/huggingface/transformers/blob/master/examples/utils_ner.py
# ---------------------------------------------------------------------------- #

class InputExample(object):
    def __init__(self, guid, words, poss, labels, glabel=None):
        self.guid   = guid
        self.words  = words   # word sequence
        self.poss   = poss    # pos sequence
        self.labels = labels  # label sequence
        self.glabel = glabel  # global label

class InputFeature(object):
    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 pos_ids,
                 char_ids,
                 label_ids,
                 glabel_id,
                 word2token_idx=[],
                 word2token_mask=[],
                 word_ids=[],
                 doc2sent_idx=[],
                 doc2sent_mask=[]):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.pos_ids = pos_ids
        self.char_ids = char_ids
        self.label_ids = label_ids
        self.glabel_id = glabel_id
        if word2token_idx:
            self.word2token_idx = word2token_idx
            self.word2token_mask = word2token_mask
        if word_ids:
            self.word_ids = word_ids
        if doc2sent_idx:
            self.doc2sent_idx = doc2sent_idx
            self.doc2sent_mask = doc2sent_mask

def read_examples_from_file(config, file_path, mode='train'):
    args = config['args']
    guid_index = 1
    examples = []
    tot_num_line = sum(1 for _ in open(file_path, 'r'))
    with open(file_path, encoding="utf-8") as f:
        bucket = []
        for idx, line in enumerate(tqdm(f, total=tot_num_line)):
            line = line.strip()
            if line == "":
                tokens = []
                posseq = []
                labelseq = []
                glabel = None
                if args.bert_use_mtl:
                    glabel = bucket[0][0] # first token means global label.
                    bucket = bucket[1:]
                for entry in bucket:
                    token = entry[0]
                    pos = entry[1]
                    pt = entry[2]
                    label = entry[3]
                    tokens.append(token)
                    posseq.append(pos)
                    labelseq.append(label)
                examples.append(InputExample(guid="{}-{}".format(mode, guid_index),
                    words=tokens,
                    poss=posseq,
                    labels=labelseq,
                    glabel=glabel))
                guid_index += 1
                bucket = []
            else:
                entry = line.split()
                assert(len(entry) == 4)
                bucket.append(entry)
        if len(bucket) != 0:
            tokens = []
            posseq = []
            labelseq = []
            glabel = None
            if args.bert_use_mtl:
                glabel = bucket[0][0] # first token means global label.
                bucket = bucket[1:]
            for entry in bucket:
                token = entry[0]
                pos = entry[1]
                pt = entry[2]
                label = entry[3]
                tokens.append(token)
                posseq.append(pos)
                labelseq.append(label)
            examples.append(InputExample(guid="{}-{}".format(mode, guid_index),
                words=tokens,
                poss=posseq,
                labels=labelseq,
                glabel=glabel))
            guid_index += 1

    return examples

def build_document_context(config,
        example,
        max_seq_length,
        tokenizer,
        cls_token="[CLS]",
        cls_token_segment_id=0,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_token="[PAD]",
        pad_token_id=0,
        pad_token_segment_id=0,
        sequence_a_segment_id=0,
        examples=None,
        ex_index=-1):
    """
      ---------------------------------------------------------------------------
      with --bert_doc_context_option=1:
                           prev example   example     next examples   
      tokens:        [CLS] p1 p2 p3 p4 p5 x1 x2 x3 x4 n1 n2 n3 n4  m1 m2 m3 ...
      token_idx:     0     1  2  3  4  5  6  7  8  9  10 11 12 13  14 15 16 ...
      input_ids:     x     x  x  x  x  x  x  x  x  x  x  x  x  x   x  x  x  ...
      segment_ids:   0     0  0  0  0  0  0  0  0  0  0  0  0  0   0  0  0  ...
      input_mask:    1     1  1  1  1  1  1  1  1  1  1  1  1  1   1  1  1  ...
      doc2sent_idx:  0     6  7  8  9  0  0  0  0  0  0  0  0  0   0  0  0  ...
      doc2sent_mask: 1     1  1  1  1  0  0  0  0  0  0  0  0  0   0  0  0  ...

      with --bert_doc_context_option=2:
                           prev examples  example     next examples   

      input_ids, segment_ids, input_maks are replaced to document-level.
      and doc2sent_idx will be used to slice input_ids, segment_ids, input_mask.
      ---------------------------------------------------------------------------
    """

    args = config['args']

    doc2sent_idx = []
    doc2sent_mask = []

    # ---------------------------------------------------------------------------
    # build context
    # ---------------------------------------------------------------------------
    doc_start = args.bert_doc_separator # eg, '-DOCSTART-'
    start_ex_index = ex_index
    end_ex_index = ex_index
    if doc_start in example.words[0]:
        for ex in examples[ex_index+1:]:
            if doc_start in ex.words[0]: break
            else: end_ex_index += 1
    else:
        for ex in examples[ex_index::-1]:
            if doc_start in ex.words[0]: break
            else: start_ex_index -= 1
        for ex in examples[ex_index+1:]:
            if doc_start in ex.words[0]: break
            else: end_ex_index += 1
    if ex_index < 5:
        logger.info("document start index, end index: ({}, {})".format(start_ex_index, end_ex_index))
        logger.info("document start: %s", " ".join([str(x) for x in examples[start_ex_index].words]))
        logger.info("document end: %s", " ".join([str(x) for x in examples[end_ex_index].words]))

    csize = config['prev_context_size'] # previous max context size

    if args.bert_doc_context_option == 1:
        prev_example = None
        if ex_index > 0:
            prev_example = examples[ex_index-1]
        n_examples = len(examples)
        next_examples = None
        if ex_index+1 < n_examples:
            next_examples = examples[ex_index+1:end_ex_index+1]
    
        prev_words = []
        if prev_example == None:
            prev_words = [pad_token]
        else:
            prev_csize = csize # eg, csize: 64
            if prev_csize >= len(prev_example.words):
                prev_words = prev_example.words
            else:
                # preserve right-most words of previous example if the length exceeds previous max context size.
                prev_words = prev_example.words[len(prev_example.words)-prev_csize:]
        prev_words = [sep_token] + prev_words + [sep_token]

        next_words = []
        if next_examples == None:
            next_words = [pad_token]
        else:
            for idx, next_example in enumerate(next_examples):
                n_next_words = len(next_words)
                next_csize = csize*4 # prevent too long, eg, csize: 64 -> next context size: 256
                if n_next_words + len(next_example.words) + 1 > next_csize: break
                if idx == 0: next_words += next_example.words
                else: next_words += [sep_token] + next_example.words
        next_words = [sep_token] + next_words + [sep_token]

        words = prev_words + example.words + next_words
        bos = len(prev_words)
        eos = bos + len(example.words)

    if args.bert_doc_context_option == 2:
        prev_examples = examples[start_ex_index:ex_index]
        next_examples = examples[ex_index+1:end_ex_index+1]

        prev_words = []
        if prev_examples != []:
            for idx, prev_example in enumerate(prev_examples):
                if idx == 0: prev_words += prev_example.words
                else: prev_words += [sep_token] + prev_example.words
        next_words = []
        if next_examples != []:
            for idx, next_example in enumerate(next_examples):
                if idx == 0: next_words += next_example.words
                else: next_words += [sep_token] + next_example.words

        prev_csize = csize*2 # eg, csize: 64 -> prev context size: 128
        if prev_csize < len(prev_words):
            # preserve right-most words of previous examples if the length exceeds previous max context size.
            prev_words = prev_words[len(prev_words)-prev_csize:]

        words = prev_words + [sep_token] + example.words + [sep_token] + next_words
        bos = len(prev_words) + 1
        eos = bos + len(example.words)
    # ---------------------------------------------------------------------------

    tokens = []
    token_idx = 1 # consider first sub-token is '[CLS]'
    for word_idx, word in enumerate(words):
        word_tokens = tokenizer.tokenize(word)
        tokens.extend(word_tokens)             
        for token in word_tokens:
            if word_idx >= bos and word_idx < eos:
                # token_idx must be less than max_seq_length.
                if token_idx < max_seq_length:
                    doc2sent_idx.append(token_idx)
            token_idx += 1

    # for [CLS] and [SEP]
    special_tokens_count = 3 if sep_token_extra else 2
    if len(tokens) > max_seq_length - special_tokens_count:
        tokens = tokens[:(max_seq_length - special_tokens_count)]

    tokens += [sep_token]
    if sep_token_extra:
        # roberta uses an extra separator b/w pairs of sentences
        tokens += [sep_token]
    segment_ids = [sequence_a_segment_id] * len(tokens)

    tokens = [cls_token] + tokens
    segment_ids = [cls_token_segment_id] + segment_ids
    doc2sent_idx = [0] + doc2sent_idx

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    doc2sent_mask = [1] * len(doc2sent_idx)

    # zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    input_ids += ([pad_token_id] * padding_length)
    input_mask += ([0] * padding_length)
    segment_ids += ([pad_token_segment_id] * padding_length)

    padding_length_for_doc2sent_idx = max_seq_length - len(doc2sent_idx)
    # 0 padding means that the first token embedding('[CLS]') will be used as dummy.
    doc2sent_idx += ([0] * padding_length_for_doc2sent_idx)
    doc2sent_mask += ([0] * padding_length_for_doc2sent_idx)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(doc2sent_idx) == max_seq_length
    assert len(doc2sent_mask) == max_seq_length

    if ex_index != -1 and ex_index < 5:
        logger.info("*** Example(contextualized) ***")
        logger.info("guid: %s", example.guid)
        logger.info("tokens(context): %s", " ".join([str(x) for x in tokens]))
        logger.info("input_ids(context): %s", " ".join([str(x) for x in input_ids]))
        logger.info("input_mask(context): %s", " ".join([str(x) for x in input_mask]))
        logger.info("segment_ids(context): %s", " ".join([str(x) for x in segment_ids]))
        logger.info("doc2sent_idx: %s", " ".join([str(x) for x in doc2sent_idx]))
        logger.info("doc2sent_mask: %s", " ".join([str(x) for x in doc2sent_mask]))

    return input_ids, input_mask, segment_ids, doc2sent_idx, doc2sent_mask

def convert_single_example_to_feature(config,
        example,
        pos_map,
        label_map,
        max_seq_length,
        tokenizer,
        cls_token="[CLS]",
        cls_token_segment_id=0,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_token="[PAD]",
        pad_token_id=0,
        pad_token_pos_id=0,
        pad_token_label_id=0,
        pad_token_segment_id=0,
        sequence_a_segment_id=0,
        w_tokenizer=None,
        examples=None,
        glabel_map={},
        ex_index=-1):

    """
    convention in BERT:
    for single sequence:
      word      : the dog is hairy .
      word_idx  : 0   1   2  3     4                                          | params
      ----------------------------------------------------------------------- | -------------- |
      tokens:        [CLS] the dog is ha ##iry . [SEP] [PAD] [PAD] [PAD] ...  |                |
      token_idx:       0   1   2   3  4  5     6   7     8     9     10  ...  |                |
      input_ids:       x   x   x   x  x  x     x   x     0     0     0   ...  | input_ids      |
      segment_ids:     0   0   0   0  0  0     0   0     0     0     0   ...  | token_type_ids |
      input_mask:      1   1   1   1  1  1     1   1     0     0     0   ...  | attention_mask |
      label_ids:       0   1   1   1  1  0     1   0     0     0     0   ...  |                |
      glabel_id:       0                                                      |                |
      ----------------------------------------------------------------------- |                |
      pos_ids:         0   10  2   ...
      char_ids:        [0,..., 0] [259, ..., 261] ...
      ---------------------------------------------------------------------------

      ---------------------------------------------------------------------------
      with --bert_use_subword_pooling:

      word2token_idx:  0   1   2     3   4      6  0  0  0 ...
      word2token_mask: 1   1   1     1   1      1  0  0  0 ...

      additionally, 'label_ids, pos_ids, char_ids' are generated at word-level. 
      ---------------------------------------------------------------------------

      ---------------------------------------------------------------------------
      with --bert_use_subword_pooling --bert_use_word_embedding:

      word_ids:        0   2   2928  16  23223  4  0  0 ...
      ---------------------------------------------------------------------------
    """

    args = config['args']

    glabel = example.glabel
    glabel_id = pad_token_label_id
    if glabel is not None and glabel in glabel_map:
        glabel_id = glabel_map[glabel]

    tokens = []
    pos_ids = []
    char_ids = []
    pad_char_ids = [config['pad_token_id']] * config['char_n_ctx']
    label_ids = []
    word2token_idx = []
    token_idx = 1 # consider first sub-token is '[CLS]'
    word2token_mask = []
    word_ids = []

    for word, pos, label in zip(example.words, example.poss, example.labels):
        # word extension
        word_tokens = tokenizer.tokenize(word)
        tokens.extend(word_tokens)
        if args.bert_use_subword_pooling:
            # build word2token_idx, save the first token's idx of sub-tokens for the word.
            # token_idx must be less than max_seq_length.
            if token_idx < max_seq_length:
                word2token_idx.append(token_idx)
                token_idx += len(word_tokens)
        # pos extension
        pos_id = pos_map[pos]
        if args.bert_use_subword_pooling:
            pos_ids.extend([pos_id])
        else:
            # set same pod_id
            pos_ids.extend([pos_id] + [pos_id] * (len(word_tokens) - 1))
        # char extension
        if args.bert_use_subword_pooling:
            c_ids = batch_to_ids([word])[0].detach().cpu().numpy().tolist()
            char_ids.extend(c_ids)
        else:
            c_ids = batch_to_ids([word_tokens])[0].detach().cpu().numpy().tolist()
            char_ids.extend(c_ids)
        # label extension
        label_id = pad_token_label_id
        if label in label_map:
            label_id = label_map[label]
        if args.bert_use_subword_pooling:
            label_ids.extend([label_id])
        else:
            if args.bert_use_sub_label:
                if label == config['default_label']:
                    # ex) 'round', '##er' -> 1/'O', 1/'O'
                    sub_token_label = label
                    sub_token_label_id = label_map[sub_token_label]
                    label_ids.extend([label_id] + [sub_token_label_id] * (len(word_tokens) - 1))
                else:
                    # ex) 'BR', '##US', '##SE', '##LS' -> 6/'B-LOC', 9/'I-LOC', 9/'I-LOC', 9/'I-LOC'
                    sub_token_label = label
                    prefix, suffix = label.split('-', maxsplit=1)
                    if prefix == 'B': sub_token_label = 'I-' + suffix
                    sub_token_label_id = label_map[sub_token_label]
                    label_ids.extend([label_id] + [sub_token_label_id] * (len(word_tokens) - 1))
            else:
                # set pad_token_label_id
                label_ids.extend([label_id] + [pad_token_label_id] * (len(word_tokens) - 1))

    # build word ids
    if args.bert_use_subword_pooling and args.bert_use_word_embedding:
        word_ids = w_tokenizer.convert_tokens_to_ids(example.words) 

    if not args.bert_use_subword_pooling and len(tokens) != len(pos_ids):
        # tokenizer returns empty result, ex) [<96>, ;, -, O], [<94>, ``, -, O]
        logger.info("guid: %s", example.guid)
        logger.info("words: %s", " ".join([str(x) for x in example.words]))
        logger.info('len(words): ' + str(len(example.words)))
        logger.info("poss: %s", " ".join([str(x) for x in example.poss]))
        logger.info('len(poss): ' + str(len(example.poss)))
        logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
        logger.info('len(tokens): ' + str(len(tokens)))
        logger.info("pos_ids: %s", " ".join([str(x) for x in pos_ids]))
        logger.info('len(pos_ids): ' + str(len(pos_ids)))
        logger.warning('number of tokens != number of pos_ids, size mismatch!')
        return None

    # for [CLS] and [SEP]
    special_tokens_count = 3 if sep_token_extra else 2
    if len(tokens) > max_seq_length - special_tokens_count:
        tokens = tokens[:(max_seq_length - special_tokens_count)]
    if len(pos_ids) > max_seq_length - special_tokens_count:
        pos_ids = pos_ids[:(max_seq_length - special_tokens_count)]
    if len(char_ids) > max_seq_length - special_tokens_count:
        char_ids = char_ids[:(max_seq_length - special_tokens_count)]
    if len(label_ids) > max_seq_length - special_tokens_count:
        label_ids = label_ids[:(max_seq_length - special_tokens_count)]
    if word_ids:
        if len(word_ids) > max_seq_length - special_tokens_count:
            word_ids = word_ids[:(max_seq_length - special_tokens_count)]

    tokens += [sep_token]
    pos_ids += [pad_token_pos_id]
    char_ids += [pad_char_ids]
    label_ids += [pad_token_label_id]
    if word_ids:
        word_ids += [w_tokenizer.pad_id]
    if sep_token_extra:
        # roberta uses an extra separator b/w pairs of sentences
        tokens += [sep_token]
        pos_ids += [pad_token_pos_id]
        char_ids += [pad_char_ids]
        label_ids += [pad_token_label_id]
        if word_ids: word_ids += [w_tokenizer.pad_id] 
    segment_ids = [sequence_a_segment_id] * len(tokens)

    tokens = [cls_token] + tokens
    segment_ids = [cls_token_segment_id] + segment_ids
    pos_ids = [pad_token_pos_id] + pos_ids
    char_ids = [pad_char_ids] + char_ids
    label_ids = [pad_token_label_id] + label_ids
    if word2token_idx:
        word2token_idx = [0] + word2token_idx
    if word_ids:
        word_ids = [w_tokenizer.pad_id] + word_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    if word2token_idx:
        word2token_mask = [1] * len(word2token_idx)

    # zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    input_ids += ([pad_token_id] * padding_length)
    input_mask += ([0] * padding_length)
    segment_ids += ([pad_token_segment_id] * padding_length)

    padding_length = max_seq_length - len(pos_ids)
    pos_ids += ([pad_token_pos_id] * padding_length)

    padding_length = max_seq_length - len(char_ids)
    char_ids += ([pad_char_ids] * padding_length)

    padding_length = max_seq_length - len(label_ids)
    label_ids += ([pad_token_label_id] * padding_length)

    if word2token_idx:
        padding_length_for_word2token_idx = max_seq_length - len(word2token_idx)
        # 0 padding means that the first token embedding('[CLS]') will be used as dummy.
        word2token_idx += ([0] * padding_length_for_word2token_idx)
        word2token_mask += ([0] * padding_length_for_word2token_idx)
    if word_ids:
        padding_length = max_seq_length - len(word_ids)
        word_ids += ([w_tokenizer.pad_id] * padding_length)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(pos_ids) == max_seq_length
    assert len(char_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    if word2token_idx:
        assert len(word2token_idx) == max_seq_length
        assert len(word2token_mask) == max_seq_length
    if word_ids:
        assert len(word_ids) == max_seq_length

    if ex_index != -1 and ex_index < 5:
        logger.info("*** Example ***")
        logger.info("guid: %s", example.guid)
        logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
        logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
        logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
        logger.info("pos_ids: %s", " ".join([str(x) for x in pos_ids]))
        logger.info("char_ids: %s ...", " ".join([str(x) for x in char_ids][:3]))
        logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
        logger.info("glabel_id(glabel): %s(%s)", glabel_id, glabel)
        if word2token_idx:
            logger.info("word2token_idx: %s", " ".join([str(x) for x in word2token_idx]))
            logger.info("word2token_mask: %s", " ".join([str(x) for x in word2token_mask]))
        if word_ids:
            logger.info("word_ids: %s", " ".join([str(x) for x in word_ids]))

    doc2sent_idx = []
    doc2sent_mask = []
    if args.bert_use_doc_context:
        input_ids, input_mask, segment_ids, doc2sent_idx, doc2sent_mask = build_document_context(config,
            example,
            max_seq_length,
            tokenizer,
            cls_token=cls_token,
            cls_token_segment_id=cls_token_segment_id,
            sep_token=sep_token,
            sep_token_extra=sep_token_extra,
            pad_token=pad_token,
            pad_token_id=pad_token_id,
            pad_token_segment_id=pad_token_segment_id,
            sequence_a_segment_id=sequence_a_segment_id,
            examples=examples,
            ex_index=ex_index)

    feature = InputFeature(input_ids=input_ids,
                           input_mask=input_mask,
                           segment_ids=segment_ids,
                           pos_ids=pos_ids,
                           char_ids=char_ids,
                           label_ids=label_ids,
                           glabel_id=glabel_id,
                           word2token_idx=word2token_idx,
                           word2token_mask=word2token_mask,
                           word_ids=word_ids,
                           doc2sent_idx=doc2sent_idx,
                           doc2sent_mask=doc2sent_mask)
    return feature

def convert_examples_to_features(config,
                                 examples,
                                 pos_map,
                                 label_map,
                                 max_seq_length,
                                 tokenizer,
                                 cls_token="[CLS]",
                                 cls_token_segment_id=0,
                                 sep_token="[SEP]",
                                 sep_token_extra=False,
                                 pad_token="[PAD]",
                                 pad_token_id=0,
                                 pad_token_pos_id=0,
                                 pad_token_label_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 w_tokenizer=None,
                                 glabel_map={}):

    features = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        feature = convert_single_example_to_feature(config,
                                                    example,
                                                    pos_map,
                                                    label_map,
                                                    max_seq_length,
                                                    tokenizer,
                                                    cls_token=cls_token,
                                                    cls_token_segment_id=cls_token_segment_id,
                                                    sep_token=sep_token,
                                                    sep_token_extra=sep_token_extra,
                                                    pad_token=pad_token,
                                                    pad_token_id=pad_token_id,
                                                    pad_token_pos_id=pad_token_pos_id,
                                                    pad_token_label_id=pad_token_label_id,
                                                    pad_token_segment_id=pad_token_segment_id,
                                                    sequence_a_segment_id=sequence_a_segment_id,
                                                    w_tokenizer=w_tokenizer,
                                                    examples=examples,
                                                    glabel_map=glabel_map,
                                                    ex_index=ex_index)

        if feature:
            features.append(feature)

    return features
