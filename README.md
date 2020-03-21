## ntagger

reference pytorch code for named entity tagging.
- embedding
  - word : Glove, BERT, RoBERTa, ELMo
  - character : CNN
  - pos : look-up
- encoding
  - BiLSTM
  - DenseNet
    - [Dynamic Self-Attention: Computing Attention over Words Dynamically for Sentence Embedding](https://arxiv.org/pdf/1808.07383.pdf)
    - a slightly modified DenseNet for longer dependency.
- decoding
  - Softmax, CRF

## requirements

- python >= 3.6

- pip install -r requirements.txt

- pretrained embedding
  - glove
    - [download Glove6B](http://nlp.stanford.edu/data/glove.6B.zip) and unzip to 'embeddings' dir
    ```
    $ mkdir embeddings
    $ ls embeddings
    glove.6B.zip
    $ unzip glove.6B.zip 
    ```
  - BERT(huggingface's [transformers](https://github.com/huggingface/transformers.git))
  ```
  $ pip install tensorflow-gpu==2.0.0
  $ pip install transformers
  ```
  - ELMo([allennlp](https://github.com/allenai/allennlp))
  ```
  $ pip install allennlp==0.9.0
  $ cd embeddings
  $ curl -OL https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5
  $ curl -OL https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json
  ```

- data
  - CoNLL 2003 (english)
    - from [etagger](https://github.com/dsindex/etagger)
      - data/conll2003
    - [SOTA on CoNLL 2003 english](https://paperswithcode.com/sota/named-entity-recognition-ner-on-conll-2003)
  - Naver NER 2019 (Korean)
    - from [HanBert-NER](https://github.com/monologg/HanBert-NER)
      - data/clova2019
        - converted to CoNLL data format.
        ```
        이기범 eoj - B-PER
        한두 eoj - O
        쪽을 eoj - O
        먹고 eoj - O
        10분 eoj - B-TIM
        후쯤 eoj - I-TIM
        화제인을 eoj - B-CVL
        먹는 eoj - O
        것이 eoj - O
        좋다고 eoj - O
        한다 eoj - O
        . eoj - O
        ```
      - data/clova2019_morph
        - tokenized by morphological analyzer and converted to CoNLL data format.
        ```
        이기범 NNP - B-PER
        한두 NNP - O
        쪽 NNB - O
        을 X-JKO - O
        먹다 VV - O
        고 X-EC - O
        10 SN - B-TIM
        분 X-NNB - I-TIM
        후 NNG - I-TIM
        쯤 X-XSN - I-TIM
        화제 NNG - B-CVL
        인 X-NNG - I-CVL
        을 X-JKO - I-CVL
        먹다 VV - O
        ...
        ```
        - 'X-' prefix is prepending to POS(Part of Speech) tag of inside morphs for distinguishing following morphs.
        - we can evaluate the predicted result morph-by-morph or eojeol by eojeol manner(every lines having 'X-' POS tag are removed).
        ```
        이기범 NNP - B-PER
        한두 NNP - O
        쪽 NNB - O
        먹다 VV - O
        10 SN - B-TIM
        후 NNG - I-TIM
        화제 NNG - B-CVL
        먹다 VV - O
        ...
        ```
      - there is no test set. so, set valid.txt as test.txt.
    - Korean BERT and Glove were described [here](https://github.com/dsindex/iclassifier/blob/master/KOR_EXPERIMENTS.md)
      - bpe : `pytorch.all.bpe.4.8m_step` (inhouse)
      - dha-bpe : `pytorch.all.dha_s2.9.4_d2.9.27_bpe.4m_step` (inhouse)
      - dha : `pytorch.all.dha.2.5m_step`, `pytorch.all.dha_s2.9.4_d2.9.27.10m_step` (inhouse)
      - `kor.glove.300k.300d.txt`   (inhouse)
        - training corpus is the same as the data for Korean BERT.
    - Korean ELMo was described [here](https://github.com/dsindex/bilm-tf)
      - `kor_elmo_2x4096_512_2048cnn_2xhighway_1000k_weights.hdf5`, `kor_elmo_2x4096_512_2048cnn_2xhighway_1000k_options.json` (inhouse)
        - training corpus is the same as the data for Korean BERT.
  - KMOU NER 2019 (Korean)
    - from [KMOU NER](https://github.com/kmounlp/NER)
      - data/kmou2019
        - build train.raw, valid.raw
          - data version : https://github.com/kmounlp/NER/commit/0b32e066870bda9f65cc190f5e89c2edc6cf8f6d
          - same as [pytorch-bert-crf-ner](https://github.com/eagle705/pytorch-bert-crf-ner)
            - train.raw : 00002_NER.txt, ..., EXOBRAIN_NE_CORPUS_007.txt (1,425 files)
            - valid.raw : EXOBRAIN_NE_CORPUS_009.txt, EXOBRAIN_NE_CORPUS_010.txt (2 files)
        - apply correction and converting to CoNLL data format
        ```
        $ cd data/kmou2019
        $ python correction.py -g train.raw > t
        $ python to-conll.py -g t > train.txt
  
        ex)
        마음	마음	NNG	B-POH
        ’	’	SS	O
        에	에	JKB	O
        _	_	_	O
        담긴	담기+ㄴ	VV+ETM	O
        ->
        마음 NNG - B-POH
        ’ SS - O
        에 JKB - O
        _ _ - O
        담기다 VV - O
        ㄴ ETM - O

        $ python correction.py -g valid.raw > t
        $ python to-conll.py -g t > valid.txt
        ```
        - set valid.txt as test.txt
        ```
        $ cp -rf valid.txt test.txt
        ```

## CoNLL 2003 (english)

### experiments summary

- ntagger, measured by conlleval.pl (micro F1)

|                                 | F1 (%)       | Features             | Elapsed time / example (ms) |
| ------------------------------- | -----------  | -------------------- | --------------------------- |
| Glove, BiLSTM-CRF               | 88.49        | word, pos            | 32.8716 |
| Glove, BiLSTM-CRF               | 89.80        | word, character, pos | 36.7597 |
| Glove, DenseNet-CRF             | 88.23        | word, pos            | 24.7893 |
| Glove, DenseNet-CRF             | 88.48        | word, character, pos | 28.5851 |
| BERT-tiny, BiLSTM               | -            | word                 |         |
| BERT-mini, BiLSTM               | 81.55        | word                 | 21.4632 |
| BERT-small, BiLSTM              | 86.35        | word                 | 22.6087 |
| BERT-medium, BiLSTM             | 88.29        | word                 | 27.0486 |
| BERT-base, BiLSTM               | 90.55        | word                 | 30.5631 |
| BERT-large, BiLSTM              | 91.29        | word                 | 36.6495 |
| RoBERTa-base, BiLSTM            | 90.03        | word                 | 19.2503 |
| RoBERTa-large, BiLSTM           | 91.83        | word                 | 28.5525 |
| ELMo, BiLSTM                    | -            | word, pos            | -       |
| ELMo, Glove, BiLSTM             | **92.23**    | word, pos            | 79.9896 |
| ELMo, Glove, BiLSTM             | 91.97        | word, character, pos | 79.0648 |

- [etagger](https://github.com/dsindex/etagger), measured by conlleval (micro F1)

|                                 | F1 (%)            | Features                              | Etc        |
| ------------------------------- | ----------------  | ------------------------------------- | ---------- |
| Glove, BiLSTM-CRF               | 87.91             | word                                  |            |
| Glove, BiLSTM-CRF               | 89.20             | word, pos                             |            |
| Glove, BiLSTM-CRF               | 90.06             | word, character, pos                  |            |
| Glove, BiLSTM-CRF               | 90.85             | word, character, pos, chunk           |            |
| BERT-large, BiLSTM-CRF          | 90.22             | word, BERT as feature-based           |            |
| BERT-large, Glove, BiLSTM-CRF   | 91.83             | word, BERT as feature-based           |            |
| BERT-large, Glove, BiLSTM-CRF   | 91.19             | word trainable, BERT as feature-based |            |
| ELMo, BiLSTM-CRF                | -                 | word, pos                             |            |
| ELMo, BiLSTM-CRF                | -                 | word, character, pos                  |            |
| ELMo, BiLSTM-CRF                | -                 | word, character, pos, chunk           |            |
| ELMo, Glove, BiLSTM-CRF         | 91.78             | word, pos                             |            |
| ELMo, Glove, BiLSTM-CRF         | 92.38             | word, character, pos                  |            |
| ELMo, Glove, BiLSTM-CRF         | -                 | word, character, pos, chunk           |            |
| ELMo, Glove, BiLSTM-CRF         | 92.83             | word, character, pos, chunk           | Glove-100d |

- [CoNLL 2003(English) learderboard](https://paperswithcode.com/sota/named-entity-recognition-ner-on-conll-2003), measured by span-level F1 (same as micro F1)

|                                 | F1 (%) |
| ------------------------------- | ------ |
| CNN Large + fine-tune           | 93.5   |
| GCDT + BERT-L                   | 93.47  |
| LSTM-CRF+ELMo+BERT+Flair        | 93.38  |
| Hierarchical + BERT             | 93.37  |
| Flair embeddings + Pooling      | 93.18  |
| BERT Large                      | 92.8   |
| BERT Base                       | 92.4   |
| BiLSTM-CRF+ELMo                 | 92.22  |

### emb_class=glove, enc_class=bilstm

- train
```
* token_emb_dim in configs/config-glove.json == 300 (ex, glove.6B.300d.txt )
$ python preprocess.py
* --use_crf for adding crf layer, --embedding_trainable for fine-tuning pretrained word embedding
$ python train.py --use_crf

* tensorboardX
$ rm -rf runs
$ tensorboard --logdir runs/ --port ${port} --bind_all
```

- evaluation
```
$ python evaluate.py --use_crf
* seqeval.metrics supports IOB2(BIO) format, so FB1 from conlleval.pl should be similar value with.
$ cd data/conll2003; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.8848570669970794, 3684
INFO:__main__:[Elapsed Time] : 121099ms, 32.87160694896851ms on average
accuracy:  97.61%; precision:  88.46%; recall:  88.51%; FB1:  88.49

* --use_char_cnn --decay_rate=0.9
INFO:__main__:[F1] : 0.8979916836238168, 3684
INFO:__main__:[Elapsed Time] : 135526ms, 36.75970676079283ms on average
accuracy:  97.85%; precision:  89.74%; recall:  89.85%; FB1:  89.80
```

### emb_class=glove, enc_class=densenet

- train
```
* token_emb_dim in configs/config-glove.json == 300 (ex, glove.6B.300d.txt )
$ python preprocess.py --config=configs/config-densenet.json
* --use_crf for adding crf layer, --embedding_trainable for fine-tuning pretrained word embedding
$ python train.py --config=configs/config-densenet.json --save_path=pytorch-model-densenet.pt --use_crf --warmup_steps=13 --decay_rate=0.8 --epoch=64
```

- evaluation
```
$ python evaluate.py --config=configs/config-densenet.json --model_path=pytorch-model-densenet.pt --use_crf
$ cd data/conll2003; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.8823112371499469, 3684
INFO:__main__:[Elapsed Time] : 91415ms, 24.789302199294053ms on average
accuracy:  97.53%; precision:  88.33%; recall:  88.14%; FB1:  88.23

* --use_char_cnn
INFO:__main__:[F1] : 0.8847577092511013, 3684
INFO:__main__:[Elapsed Time] : 105399ms, 28.585120825414066ms on average
accuracy:  97.60%; precision:  88.06%; recall:  88.90%; FB1:  88.48
```

### emb_class=bert, enc_class=bilstm

- train
```
* n_ctx size should be less than 512
$ python preprocess.py --config=configs/config-bert.json --bert_model_name_or_path=./embeddings/bert-large-cased
* --use_crf for adding crf layer
* --bert_use_pos for adding Part-Of-Speech features
* --bert_use_feature_based for feature-based
* --bert_disable_lstm for removing lstm layer
$ python train.py --config=configs/config-bert.json --save_path=pytorch-model-bert.pt --bert_model_name_or_path=./embeddings/bert-large-cased --bert_output_dir=bert-checkpoint --batch_size=16 --lr=1e-5 --epoch=10
```

- evaluation
```
$ python evaluate.py --config=configs/config-bert.json --model_path=pytorch-model-bert.pt --data_dir=data/conll2003 --bert_output_dir=bert-checkpoint
$ cd data/conll2003; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.9129208531641106, 3684
INFO:__main__:[Elapsed Time] : 135017ms, 36.64956568946797ms on average
accuracy:  98.30%; precision:  90.89%; recall:  91.70%; FB1:  91.29

* --bert_use_pos
INFO:__main__:[F1] : 0.9111325554873234, 3684
INFO:__main__:[Elapsed Time] : 141093ms, 38.29885993485342ms on average
accuracy:  98.24%; precision:  90.30%; recall:  91.94%; FB1:  91.11

* --use_crf
* it seems that the F1 score is going worse with '--use_crf' for wp/bpe BERT.
INFO:__main__:[F1] : 0.9058430130235833, 3684
INFO:__main__:[Elapsed Time] : 218823ms, 59.398208469055376ms on average
accuracy:  98.12%; precision:  90.44%; recall:  91.13%; FB1:  90.78

* --bert_model_name_or_path=./embedings/bert-base-uncased --bert_do_lower_case
INFO:__main__:[F1] : 0.9054532577903682, 3684
INFO:__main__:[Elapsed Time] : 112704ms, 30.563127884876458ms on average
accuracy:  98.00%; precision:  90.55%; recall:  90.55%; FB1:  90.55

* --bert_model_name_or_path=./embedings/pytorch.uncased_L-8_H-512_A-8 --bert_do_lower_case
INFO:__main__:[F1] : 0.8829307140960977, 3684
INFO:__main__:[Elapsed Time] : 99730ms, 27.048601683410265ms on average
accuracy:  97.61%; precision:  88.25%; recall:  88.33%; FB1:  88.29

* --bert_model_name_or_path=./embedings/pytorch.uncased_L-4_H-512_A-8 --bert_do_lower_case
INFO:__main__:[F1] : 0.8634692805881324, 3684
INFO:__main__:[Elapsed Time] : 83385ms, 22.60874287265816ms on average
accuracy:  97.23%; precision:  85.38%; recall:  87.34%; FB1:  86.35

* --bert_model_name_or_path=./embedings/pytorch.uncased_L-4_H-256_A-4 --bert_do_lower_case
INFO:__main__:[F1] : 0.8155101324677603, 3684
INFO:__main__:[Elapsed Time] : 79173ms, 21.463209340211783ms on average
accuracy:  96.32%; precision:  80.82%; recall:  82.29%; FB1:  81.55

* --bert_model_name_or_path=./embedings/pytorch.uncased_L-2_H-128_A-2 --bert_do_lower_case

```

### emb_class=roberta, enc_class=bilstm

- train
```
* n_ctx size should be less than 512
$ python preprocess.py --config=configs/config-roberta.json --bert_model_name_or_path=./embeddings/roberta-large
$ python train.py --config=configs/config-roberta.json --save_path=pytorch-model-roberta.pt --bert_model_name_or_path=./embeddings/roberta-large --bert_output_dir=bert-checkpoint --batch_size=16 --lr=1e-5 --epoch=10
$ python train.py --config=configs/config-roberta.json --save_path=pytorch-model-roberta.pt --bert_model_name_or_path=./embeddings/roberta-large --bert_output_dir=bert-checkpoint --batch_size=32 --lr=1e-5 --epoch=10
$ python train.py --config=configs/config-roberta.json --save_path=pytorch-model-roberta.pt --bert_model_name_or_path=./embeddings/roberta-large --bert_output_dir=bert-checkpoint --batch_size=32 --lr=1e-5 --epoch=10 --bert_disable_lstm
$ python train.py --config=configs/config-roberta.json --save_path=pytorch-model-roberta.pt --bert_model_name_or_path=./embeddings/roberta-large --bert_output_dir=bert-checkpoint --batch_size=16 --lr=1e-5 --epoch=10 --use_crf
$ python train.py --config=configs/config-roberta.json --save_path=pytorch-model-roberta.pt --bert_model_name_or_path=./embeddings/roberta-large --bert_output_dir=bert-checkpoint --batch_size=16 --lr=1e-5 --epoch=10 --decay_rate=0.9
$ python train.py --config=configs/config-roberta.json --save_path=pytorch-model-roberta.pt --bert_model_name_or_path=./embeddings/roberta-large --bert_output_dir=bert-checkpoint --batch_size=16 --lr=1e-5 --epoch=10 --decay_rate=0.9 --bert_use_pos
```

- evaluation
```
$ python evaluate.py --config=configs/config-roberta.json --model_path=pytorch-model-roberta.pt --data_dir=data/conll2003 --bert_output_dir=bert-checkpoint
$ cd data/conll2003; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.9119915848527349, 3684
INFO:__main__:[Elapsed Time] : 145234ms, 39.37578061363019ms on average
accuracy:  98.27%; precision:  90.31%; recall:  92.10%; FB1:  91.20

* --batch_size=32
INFO:__main__:[F1] : 0.9182367890631846, 3684
INFO:__main__:[Elapsed Time] : 147221ms, 39.92967689383654ms on average

* --bert_disable_lstm
INFO:__main__:[F1] : 0.9183817062445031, 3684
INFO:__main__:[Elapsed Time] : 105306ms, 28.55253869128428ms on average

* --use_crf
INFO:__main__:[F1] : 0.9013054830287206, 3684
INFO:__main__:[Elapsed Time] : 221208ms, 60.01221830029867ms on average

* --decay_rate=0.9
INFO:__main__:[F1] : 0.9145419377527695, 3684
INFO:__main__:[Elapsed Time] : 152493ms, 41.35270160195493ms on average

* --decay_rate=0.9 --bert_use_pos
INFO:__main__:[F1] : 0.914000175330937, 3684
INFO:__main__:[Elapsed Time] : 153930ms, 41.748574531631824ms on average

* --bert_model_name_or_path=./embeddings/roberta-base --bert_disable_lstm
INFO:__main__:[F1] : 0.9002973587545915, 3684
INFO:__main__:[Elapsed Time] : 71015ms, 19.25033939723052ms on average

```

### emb_class=elmo, enc_class=bilstm

- train
```
* token_emb_dim in configs/config-elmo.json == 300 (ex, glove.6B.300d.txt )
* elmo_emb_dim  in configs/config-elmo.json == 1024 (ex, elmo_2x4096_512_2048cnn_2xhighway_5.5B_* )
$ python preprocess.py --config=configs/config-elmo.json --embedding_path=embeddings/glove.6B.300d.txt
* --use_crf for adding crf layer, --embedding_trainable for fine-tuning pretrained word embedding
$ python train.py --config=configs/config-elmo.json --save_path=pytorch-model-elmo.pt --use_crf
```

- evaluation
```
$ python evaluate.py --config=configs/config-elmo.json --model_path=pytorch-model-elmo.pt --use_crf
$ cd data/conll2003; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.9219494967331803, 3684
INFO:__main__:[Elapsed Time] : 239919ms, 65.12459283387622ms on average
accuracy:  98.29%; precision:  91.95%; recall:  92.44%; FB1:  92.19

* --decay_rate=0.9
INFO:__main__:[F1] : 0.922342119228728, 3684
INFO:__main__:[Elapsed Time] : 294772ms, 79.98968232419223ms on average
accuracy:  98.35%; precision:  92.15%; recall:  92.32%; FB1:  92.23

* --use_char_cnn
INFO:__main__:[F1] : 0.9137136782423814, 3684
INFO:__main__:[Elapsed Time] : 280253ms, 76.05566114580505ms on average
accuracy:  98.15%; precision:  91.44%; recall:  91.31%; FB1:  91.37

* --user_char_cnn --decay_rate=0.9
INFO:__main__:[F1] : 0.9196578181497487, 3684
INFO:__main__:[Elapsed Time] : 291372ms, 79.06489275047515ms on average
accuracy:  98.26%; precision:  91.62%; recall:  92.32%; FB1:  91.97
```

## Naver NER 2019 (Korean)

### experiments summary

#### clova2019(eoj-based)

- ntagger, measured by conlleval.pl (micro F1)

|                              | F1 (%)      | Features | Elapsed time / example (ms) |
| ---------------------------- | ------------| -------- | --------------------------- |    
| bpe BERT(4.8m), BiLSTM-CRF   | **85.26**   | eoj      | 44.094  |

- [HanBert-NER](https://github.com/monologg/HanBert-NER#results), measured by seqeval (same as conlleval, micro F1)

|                       | F1 (%)        | Features |
| --------------------- | ------------- | -------- |
| BiLSTM-CRF            | 76.45         | eoj      |
| Bert-multilingual     | 81.78         | eoj      |
| KoBERT                | 84.23         | eoj      |
| HanBert               | 84.84         | eoj      |

#### clova2019_morph(morph-based)

- ntagger, measured by conlleval.pl (micro F1)

|                                | m-by-m F1 (%) | e-by-e F1 (%)  | Features              | Elapsed time / example (ms) |
| ------------------------------ | ------------- | -------------- | --------------------- | --------------------------- |
| Glove, BiLSTM-CRF              | 84.29         | 84.29          | morph, pos            | 30.0968 |
| Glove, BiLSTM-CRF              | 84.76         | 84.76          | morph, character, pos | 32.9187 |
| Glove, DenseNet-CRF            | 83.44         | 83.49          | morph, pos            | 25.8059 |
| Glove, DenseNet-CRF            | 83.96         | 83.98          | morph, character, pos | 28.4051 |
| dha BERT(2.5m), BiLSTM-CRF     | 83.78         | 84.13          | morph, pos            | 41.8604 |
| dha-bpe BERT(4m),  BiLSTM-CRF  | 82.83         | 83.83          | morph, pos            | 42.4347 |
| dha BERT(10m),  BiLSTM-CRF     | 83.29         | 83.57          | morph, pos            | 44.4813 |
| ELMo, BiLSTM-CRF               | -             | -              | morph, pos            | -       |
| ELMo, BiLSTM-CRF               | -             | -              | morph, character, pos | -       |
| ELMo, Glove, BiLSTM-CRF        | 86.37         | 86.37          | morph, pos            | 82.7731 |
| ELMo, Glove, BiLSTM-CRF        | 86.46         | **86.47**      | morph, character, pos | 109.155 |

- [etagger](https://github.com/dsindex/etagger), measured by conlleval (micro F1)

|                              | m-by-m F1 (%) | e-by-e F1 (%)  | Features                          |
| ---------------------------- | ------------- | -------------- | --------------------------------- |
| Glove, BiLSTM-CRF            | 85.51         | 85.51          | morph, character, pos             |
| dha BERT(2.5m), BiLSTM-CRF   | 81.25         | 81.39          | morph, pos, BERT as feature-based |
| ELMo, Glove, BiLSTM-CRF      | 86.75         | 86.75          | morph, character, pos             |

### emb_class=glove, enc_class=bilstm

- train
```
* token_emb_dim in configs/config-glove.json == 300 (ex, kor.glove.300k.300d.txt )
$ python preprocess.py --data_dir data/clova2019_morph --embedding_path embeddings/kor.glove.300k.300d.txt
* --use_crf for adding crf layer, --embedding_trainable for fine-tuning pretrained word embedding.
$ python train.py --save_path=pytorch-model-glove-kor-morph.pt --data_dir data/clova2019_morph --use_crf --embedding_trainable

```

- evaluation
```
$ python evaluate.py --model_path=pytorch-model-glove-kor-morph.pt --data_dir data/clova2019_morph --use_crf
* seqeval.metrics supports IOB2(BIO) format, so FB1 from conlleval.pl should be similar value with.
$ cd data/clova2019_morph; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.8434531044045398, 9000
INFO:__main__:[Elapsed Time] : 270872ms, 30.096888888888888ms on average
accuracy:  93.80%; precision:  84.82%; recall:  83.76%; FB1:  84.29
  ** evaluation eoj-by-eoj
  $ cd data/clova2019_morph ; python to-eoj.py < test.txt.pred > test.txt.pred.eoj ; perl ../../etc/conlleval.pl < test.txt.pred.eoj ; cd ../..
  accuracy:  93.37%; precision:  84.83%; recall:  83.76%; FB1:  84.29

* --use_char_cnn
INFO:__main__:[F1] : 0.8481546211576025, 9000
INFO:__main__:[Elapsed Time] : 296373ms, 32.918768752083565ms on average
accuracy:  93.96%; precision:  85.38%; recall:  84.14%; FB1:  84.76
  ** evaluation eoj-by-eoj
  accuracy:  93.55%; precision:  85.38%; recall:  84.15%; FB1:  84.76

```

### emb_class=glove, enc_class=densenet

- train
```
* token_emb_dim in configs/config-glove.json == 300 (ex, kor.glove.300k.300d.txt )
$ python preprocess.py --config=configs/config-densenet.json --data_dir data/clova2019_morph --embedding_path embeddings/kor.glove.300k.300d.txt
* --use_crf for adding crf layer, --embedding_trainable for fine-tuning pretrained word embedding.
$ python train.py --config=configs/config-densenet.json --save_path=pytorch-model-densenet-kor-morph.pt --data_dir data/clova2019_morph --use_crf --embedding_trainable
```

- evaluation
```
$ python evaluate.py --config=configs/config-densenet.json --model_path=pytorch-model-densenet-kor-morph.pt --data_dir data/clova2019_morph --use_crf
* seqeval.metrics supports IOB2(BIO) format, so FB1 from conlleval.pl should be similar value with.
$ cd data/clova2019_morph; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.8350127432612621, 9000
INFO:__main__:[Elapsed Time] : 232331ms, 25.805978442049117ms on average
accuracy:  93.42%; precision:  82.80%; recall:  84.10%; FB1:  83.44
  ** evaluation eoj-by-eoj
  $ cd data/clova2019_morph ; python to-eoj.py < test.txt.pred > test.txt.pred.eoj ; perl ../../etc/conlleval.pl < test.txt.pred.eoj ; cd ../..
  accuracy:  92.96%; precision:  82.86%; recall:  84.13%; FB1:  83.49

* --user_char_cnn --decay_rate=0.9
INFO:__main__:[F1] : 0.8402205267380136, 9000
INFO:__main__:[Elapsed Time] : 255785ms, 28.405156128458717ms on average
accuracy:  93.66%; precision:  84.25%; recall:  83.68%; FB1:  83.96
  ** evaluation eoj-by-eoj
  accuracy:  93.24%; precision:  84.28%; recall:  83.69%; FB1:  83.98
```

### emb_class=bert, enc_class=bilstm, bpe BERT(4.8m), dha BERT(2.5m)

- train
```
* n_ctx size should be less than 512

* for clova2019_morph

$ python preprocess.py --config=configs/config-bert.json --data_dir data/clova2019_morph --bert_model_name_or_path=./embeddings/pytorch.all.dha.2.5m_step
$ python train.py --config=configs/config-bert.json --save_path=pytorch-model-bert-kor-morph.pt --bert_model_name_or_path=./embeddings/pytorch.all.dha.2.5m_step --bert_output_dir=bert-checkpoint --batch_size=32 --lr=5e-5 --epoch=20 --data_dir data/clova2019_morph --use_crf --bert_use_pos

* for clova2019

$ python preprocess.py --config=configs/config-bert.json --data_dir data/clova2019 --bert_model_name_or_path=./embeddings/pytorch.all.bpe.4.8m_step
$ python train.py --config=configs/config-bert.json --save_path=pytorch-model-bert-kor-eoj.pt --bert_model_name_or_path=./embeddings/pytorch.all.bpe.4.8m_step --bert_output_dir=bert-checkpoint --batch_size=32 --lr=5e-5 --epoch=20 --data_dir data/clova2019 --use_crf
```

- evaluation
```
* for clova2019_morph

$ python evaluate.py --config=configs/config-bert.json --model_path=pytorch-model-bert-kor-morph.pt --data_dir=data/clova2019_morph --bert_output_dir=bert-checkpoint --use_crf --bert_use_pos
$ cd data/clova2019_morph; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.838467292261662, 9000
INFO:__main__:[Elapsed Time] : 376744ms, 41.86044444444445ms on average
accuracy:  94.01%; precision:  83.72%; recall:  83.84%; FB1:  83.78
  *** evaluation eoj-by-eoj
  $ cd data/clova2019_morph ; python to-eoj.py < test.txt.pred > test.txt.pred.eoj ; perl ../../etc/conlleval.pl < test.txt.pred.eoj ; cd ../..
  accuracy:  93.47%; precision:  84.26%; recall:  84.01%; FB1:  84.13

** bert_outputs[2][-7], --decay_rate=0.9
INFO:__main__:[F1] : 0.8296454550078846, 9000
INFO:__main__:[Elapsed Time] : 376186ms, 41.786642960328926ms on average
accuracy:  93.73%; precision:  82.62%; recall:  83.17%; FB1:  82.90
  *** evaluation eoj-by-eoj
  accuracy:  93.19%; precision:  83.25%; recall:  83.34%; FB1:  83.29


* for clova2019

$ python evaluate.py --config=configs/config-bert.json --model_path=pytorch-model-bert-kor-eoj.pt --data_dir data/clova2019 --bert_output_dir=bert-checkpoint --use_crf
$ cd data/clova2019; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.8524098438884723, 9000
INFO:__main__:[Elapsed Time] : 396846ms, 44.094ms on average
accuracy:  93.81%; precision:  85.49%; recall:  85.02%; FB1:  85.26
```

### emb_class=bert, enc_class=bilstm, dha-bpe BERT(4m), dha BERT(10m)

- train
```
* n_ctx size should be less than 512

* for clova2019_morph

** dha-bpe
$ python preprocess.py --config=configs/config-bert.json --data_dir data/clova2019_morph --bert_model_name_or_path=./embeddings/pytorch.all.dha_s2.9.4_d2.9.27_bpe.4m_step
$ python train.py --config=configs/config-bert.json --save_path=pytorch-model-bert-kor-morph.pt --bert_model_name_or_path=./embeddings/pytorch.all.dha_s2.9.4_d2.9.27_bpe.4m_step --bert_output_dir=bert-checkpoint --batch_size=32 --lr=5e-5 --epoch=20 --data_dir data/clova2019_morph --use_crf --bert_use_pos

** dha
$ python preprocess.py --config=configs/config-bert.json --data_dir data/clova2019_morph --bert_model_name_or_path=./embeddings/pytorch.all.dha_s2.9.4_d2.9.27.10m_step
$ python train.py --config=configs/config-bert.json --save_path=pytorch-model-bert-kor-morph.pt --bert_model_name_or_path=./embeddings/pytorch.all.dha_s2.9.4_d2.9.27.10m_step --bert_output_dir=bert-checkpoint --batch_size=32 --lr=5e-5 --epoch=20 --data_dir data/clova2019_morph --use_crf --bert_use_pos

```

- evaluation
```
* for clova2019_morph

** dha-bpe
$ python evaluate.py --config=configs/config-bert.json --model_path=pytorch-model-bert-kor-morph.pt --data_dir=data/clova2019_morph --bert_output_dir=bert-checkpoint --use_crf --bert_use_pos
$ cd data/clova2019_morph; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.8295019157088124, 9000
INFO:__main__:[Elapsed Time] : 382042ms, 42.434714968329814ms on average
accuracy:  93.77%; precision:  81.78%; recall:  83.91%; FB1:  82.83
  *** evaluation eoj-by-eoj
  $ cd data/clova2019_morph ; python to-eoj.py < test.txt.pred > test.txt.pred.eoj ; perl ../../etc/conlleval.pl < test.txt.pred.eoj ; cd ../..
  accuracy:  93.37%; precision:  83.34%; recall:  84.33%; FB1:  83.83

** dha
$ python evaluate.py --config=configs/config-bert.json --model_path=pytorch-model-bert-kor-morph.pt --data_dir=data/clova2019_morph --bert_output_dir=bert-checkpoint --use_crf --bert_use_pos
$ cd data/clova2019_morph; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.8336304521299173, 9000
INFO:__main__:[Elapsed Time] : 400446ms, 44.48138682075786ms on average
accuracy:  93.58%; precision:  83.12%; recall:  83.46%; FB1:  83.29
  *** evaluation eoj-by-eoj
  $ cd data/clova2019_morph ; python to-eoj.py < test.txt.pred > test.txt.pred.eoj ; perl ../../etc/conlleval.pl < test.txt.pred.eoj ; cd ../..
  accuracy:  93.06%; precision:  83.55%; recall:  83.59%; FB1:  83.57

```

### emb_class=elmo, enc_class=bilstm

- train
```
* token_emb_dim in configs/config-elmo.json == 300 (ex, kor.glove.300k.300d.txt )
* elmo_emb_dim  in configs/config-elmo.json == 1024 (ex, kor_elmo_2x4096_512_2048cnn_2xhighway_1000k* )
$ python preprocess.py --config=configs/config-elmo.json --data_dir=data/clova2019_morph --embedding_path=embeddings/kor.glove.300k.300d.txt
$ python train.py --config=configs/config-elmo.json --save_path=pytorch-model-elmo-kor-morph.pt --data_dir=data/clova2019_morph --elmo_options_file=embeddings/kor_elmo_2x4096_512_2048cnn_2xhighway_1000k_options.json --elmo_weights_file=embeddings/kor_elmo_2x4096_512_2048cnn_2xhighway_1000k_weights.hdf5 --use_crf
```

- evaluation
```
$ python evaluate.py --config=configs/config-elmo.json --model_path=pytorch-model-elmo-kor-morph.pt --data_dir=data/clova2019_morph --elmo_options_file=embeddings/kor_elmo_2x4096_512_2048cnn_2xhighway_1000k_options.json --elmo_weights_file=embeddings/kor_elmo_2x4096_512_2048cnn_2xhighway_1000k_weights.hdf5 --use_crf
$ cd data/clova2019_morph; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

* --embedding_trainable
INFO:__main__:[F1] : 0.8642865647270933, 9000
INFO:__main__:[Elapsed Time] : 744958ms, 82.7731111111111ms on average
accuracy:  94.63%; precision:  86.36%; recall:  86.38%; FB1:  86.37
  ** evluation eoj-by-eoj
  $ cd data/clova2019_morph ; python to-eoj.py < test.txt.pred > test.txt.pred.eoj ; perl ../../etc/conlleval.pl < test.txt.pred.eoj ; cd ../..
  accuracy:  94.26%; precision:  86.37%; recall:  86.38%; FB1:  86.37

* --use_char_cnn --embedding_trainable
INFO:__main__:[F1] : 0.8651979978889305, 9000
INFO:__main__:[Elapsed Time] : 982480ms, 109.15546171796866ms on average
accuracy:  94.70%; precision:  86.53%; recall:  86.39%; FB1:  86.46
  ** evaluation eoj-by-eoj
  accuracy:  94.33%; precision:  86.54%; recall:  86.39%; FB1:  86.47
```

## KMOU NER 2019 (Korean)

### experiments summary

- ntagger, measured by conlleval.pl / token_eval.py (micro F1)

|                                | span / token F1 (%)    | Features              | Etc                   | Elapsed time / example (ms) |
| ------------------------------ | ---------------------- | --------------------- | --------------------- | --------------------------- |    
| Glove, BiLSTM-CRF              | 84.38 / 86.02          | morph, pos            |                       | 34.2192 |
| Glove, BiLSTM-CRF              | 85.76 / 87.04          | morph, character, pos |                       | 37.2386 |
| Glove, DenseNet-CRF            | 82.98 / 84.79          | morph, pos            |                       | 23.3758 |
| Glove, DenseNet-CRF            | 84.32 / 85.75          | morph, character, pos |                       | 22.6004 |
| dha BERT(2.5m), BiLSTM-CRF     | 85.47 / 87.31          | morph, pos            |                       | 44.3250 |
| dha BERT(10m), BiLSTM-CRF      | 85.24 / 87.35          | morph, pos            |                       | 37.7829 |
| dha-bpe BERT(4m), BiLSTM-CRF   | 85.18 / 88.01          | morph, pos            |                       | 39.0183 |
| ELMo, BiLSTM-CRF               | -     / -              | morph, pos            |                       | -       |
| ELMo, BiLSTM-CRF               | -     / -              | morph, character, pos |                       | -       |
| ELMo, Glove, BiLSTM-CRF        | **88.18** / 89.22      | morph, pos            |                       | 132.933 |
| ELMo, Glove, BiLSTM-CRF        | 87.86 / 88.75          | morph, character, pos |                       | 110.277 |

- [etagger](https://github.com/dsindex/etagger), measured by conlleval (micro F1)

|                              | span / token F1 (%) | Features              |
| ---------------------------- | ------------------- | --------------------- |
| ELMo, Glove, BiLSTM-CRF      | 89.09 / 89.90       | morph, character, pos |

- [Pytorch-BERT-CRF-NER](https://github.com/eagle705/pytorch-bert-crf-ner), measured by sklearn.metrics (token-level F1)

|                       | token-level macro / micro F1 (%)   | Features |
| --------------------- | ---------------------------------- | -------- | 
| KoBERT+CRF            | 87.56 / 89.70                      | morph    |

### emb_class=glove, enc_class=bilstm

- train
```
* token_emb_dim in configs/config-glove.json == 300 (ex, kor.glove.300k.300d.txt )
$ python preprocess.py --data_dir data/kmou2019 --embedding_path embeddings/kor.glove.300k.300d.txt
* --use_crf for adding crf layer, --embedding_trainable for fine-tuning pretrained word embedding.
$ python train.py --save_path=pytorch-model-glove-kor-morph.pt --data_dir data/kmou2019 --use_crf --embedding_trainable

```

- evaluation
```
$ python evaluate.py --model_path=pytorch-model-glove-kor-morph.pt --data_dir data/kmou2019 --use_crf
* seqeval.metrics supports IOB2(BIO) format, so FB1 from conlleval.pl should be similar value with.
$ cd data/kmou2019; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..
* token-level micro F1
$ cd data/kmou2019; python ../../etc/token_eval.py < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.8438235294117648, 927
INFO:__main__:[Elapsed Time] : 31811ms, 34.21922246220302ms on average
accuracy:  96.67%; precision:  84.51%; recall:  84.26%; FB1:  84.38
token_eval micro F1: 0.8602867105772956

* --use_char_cnn
INFO:__main__:[F1] : 0.8576455065239701, 927
INFO:__main__:[Elapsed Time] : 34629ms, 37.23866090712743ms on average
accuracy:  96.88%; precision:  85.63%; recall:  85.90%; FB1:  85.76
token_eval micro F1: 0.8704952336665891

```

### emb_class=glove, enc_class=densenet

- train
```
* token_emb_dim in configs/config-glove.json == 300 (ex, kor.glove.300k.300d.txt )
$ python preprocess.py --config=configs/config-densenet.json --data_dir data/kmou2019 --embedding_path embeddings/kor.glove.300k.300d.txt
* --use_crf for adding crf layer, --embedding_trainable for fine-tuning pretrained word embedding.
$ python train.py --config=configs/config-densenet.json --save_path=pytorch-model-densenet-kor-morph.pt --data_dir data/kmou2019 --use_crf --embedding_trainable
```

- evaluation
```
$ python evaluate.py --config=configs/config-densenet.json --model_path=pytorch-model-densenet-kor-morph.pt --data_dir data/kmou2019 --use_crf
* seqeval.metrics supports IOB2(BIO) format, so FB1 from conlleval.pl should be similar value with.
$ cd data/kmou2019; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..
* token-level micro F1
$ cd data/kmou2019; python ../../etc/token_eval.py < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.8298437039221469, 927
INFO:__main__:[Elapsed Time] : 21763ms, 23.375809935205183ms on average
accuracy:  96.35%; precision:  83.33%; recall:  82.64%; FB1:  82.98
token_eval micro F1: 0.8479336205520983

* --use_char_cnn
INFO:__main__:[F1] : 0.8432147562582345, 927
INFO:__main__:[Elapsed Time] : 21040ms, 22.600431965442766ms on average
accuracy:  96.53%; precision:  84.06%; recall:  84.58%; FB1:  84.32
token_eval micro F1: 0.8575865817673552

```

### emb_class=bert, enc_class=bilstm, dha BERT(2.5m), dha BERT(10m)

- train
```
* n_ctx size should be less than 512

* dha (2.5m)
$ python preprocess.py --config=configs/config-bert.json --data_dir data/kmou2019 --bert_model_name_or_path=./embeddings/pytorch.all.dha.2.5m_step
$ python train.py --config=configs/config-bert.json --save_path=pytorch-model-bert-kor-morph.pt --bert_model_name_or_path=./embeddings/pytorch.all.dha.2.5m_step --bert_output_dir=bert-checkpoint --batch_size=32 --lr=5e-5 --epoch=20 --data_dir data/kmou2019 --use_crf --bert_use_pos

* dha (10m)
$ python preprocess.py --config=configs/config-bert.json --data_dir data/kmou2019 --bert_model_name_or_path=./embeddings/pytorch.all.dha_s2.9.4_d2.9.27.10m_step
$ python train.py --config=configs/config-bert.json --save_path=pytorch-model-bert-kor-morph.pt --bert_model_name_or_path=./embeddings/pytorch.all.dha_s2.9.4_d2.9.27.10m_step --bert_output_dir=bert-checkpoint --batch_size=32 --lr=5e-5 --epoch=20 --data_dir data/kmou2019 --use_crf --bert_use_pos

```

- evaluation
```
* dha (2.5m)
$ python evaluate.py --config=configs/config-bert.json --model_path=pytorch-model-bert-kor-morph.pt --data_dir=data/kmou2019 --bert_output_dir=bert-checkpoint --use_crf --bert_use_pos
$ cd data/kmou2019; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..
$ cd data/kmou2019; python ../../etc/token_eval.py < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.8546656869948568, 927
INFO:__main__:[Elapsed Time] : 41296ms, 44.32505399568034ms on average
accuracy:  96.83%; precision:  85.53%; recall:  85.40%; FB1:  85.47
token_eval micro F1: 0.8731952291274326

* dha(10m)
$ python evaluate.py --config=configs/config-bert.json --model_path=pytorch-model-bert-kor-morph.pt --data_dir=data/kmou2019 --bert_output_dir=bert-checkpoint --use_crf --bert_use_pos
$ cd data/kmou2019; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..
$ cd data/kmou2019; python ../../etc/token_eval.py < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.8519003931847969, 927
INFO:__main__:[Elapsed Time] : 35123ms, 37.7829373650108ms on average
accuracy:  96.83%; precision:  84.59%; recall:  85.90%; FB1:  85.24
token_eval micro F1: 0.8735865242143024

```

### emb_class=bert, enc_class=bilstm, dha-bpe BERT(4m)

- train
```
* n_ctx size should be less than 512

$ python preprocess.py --config=configs/config-bert.json --data_dir data/kmou2019 --bert_model_name_or_path=./embeddings/pytorch.all.dha_s2.9.4_d2.9.27_bpe.4m_step
$ python train.py --config=configs/config-bert.json --save_path=pytorch-model-bert-kor-morph.pt --bert_model_name_or_path=./embeddings/pytorch.all.dha_s2.9.4_d2.9.27_bpe.4m_step --bert_output_dir=bert-checkpoint --batch_size=32 --lr=5e-5 --epoch=20 --data_dir data/kmou2019 --use_crf --bert_use_pos

```

- evaluation
```
$ python evaluate.py --config=configs/config-bert.json --model_path=pytorch-model-bert-kor-morph.pt --data_dir=data/kmou2019 --bert_output_dir=bert-checkpoint --use_crf --bert_use_pos
$ cd data/kmou2019; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..
$ cd data/kmou2019; python ../../etc/token_eval.py < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.8517251211861989, 927
INFO:__main__:[Elapsed Time] : 36267ms, 39.018358531317496ms on average
accuracy:  97.14%; precision:  82.79%; recall:  87.72%; FB1:  85.18
token_eval micro F1: 0.8801729462631254

```

### emb_class=elmo, enc_class=bilstm

- train
```
* token_emb_dim in configs/config-elmo.json == 300 (ex, kor.glove.300k.300d.txt )
* elmo_emb_dim  in configs/config-elmo.json == 1024 (ex, kor_elmo_2x4096_512_2048cnn_2xhighway_1000k* )
$ python preprocess.py --config=configs/config-elmo.json --data_dir=data/kmou2019 --embedding_path=embeddings/kor.glove.300k.300d.txt
$ python train.py --config=configs/config-elmo.json --save_path=pytorch-model-elmo-kor-morph.pt --data_dir=data/kmou2019 --elmo_options_file=embeddings/kor_elmo_2x4096_512_2048cnn_2xhighway_1000k_options.json --elmo_weights_file=embeddings/kor_elmo_2x4096_512_2048cnn_2xhighway_1000k_weights.hdf5 --use_crf
```

- evaluation
```
$ python evaluate.py --config=configs/config-elmo.json --model_path=pytorch-model-elmo-kor-morph.pt --data_dir=data/kmou2019 --elmo_options_file=embeddings/kor_elmo_2x4096_512_2048cnn_2xhighway_1000k_options.json --elmo_weights_file=embeddings/kor_elmo_2x4096_512_2048cnn_2xhighway_1000k_weights.hdf5 --use_crf
$ cd data/kmou2019; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..
$ cd data/kmou2019; python ../../etc/token_eval.py < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.8803757522383678, 927
INFO:__main__:[Elapsed Time] : 126180ms, 135.92548596112312ms on average
accuracy:  97.30%; precision:  88.00%; recall:  88.08%; FB1:  88.04
token_eval micro F1: 0.8880736809241336

* --batch_size=64 --decay_rate=0.9
INFO:__main__:[F1] : 0.8817518248175182, 927
INFO:__main__:[Elapsed Time] : 123401ms, 132.93304535637148ms on average
accuracy:  97.37%; precision:  87.66%; recall:  88.69%; FB1:  88.18
token_eval micro F1: 0.8922982036851438

* --embedding_trainable
INFO:__main__:[F1] : 0.8755125951962508, 927
INFO:__main__:[Elapsed Time] : 125665ms, 135.366090712743ms on average
accuracy:  97.33%; precision:  87.32%; recall:  87.78%; FB1:  87.55
token_eval micro F1: 0.8897515527950312

* --use_char_cnn --batch_size=64 --decay_rate=0.9
INFO:__main__:[F1] : 0.8786279683377308, 927
INFO:__main__:[Elapsed Time] : 102389ms, 110.27753779697625ms on average
accuracy:  97.29%; precision:  87.71%; recall:  88.02%; FB1:  87.86
token_eval micro F1: 0.8875822050290135

```

## references

- [transformers_examples](https://github.com/dsindex/transformers_examples)
- [macro and micro precision/recall/f1 score](https://datascience.stackexchange.com/a/24051)
