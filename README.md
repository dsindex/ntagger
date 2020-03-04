## ntagger

reference pytorch code for named entity tagging.
- embedding
  - Glove, BERT, ELMo
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
  $ pip install tensorflow-gpu==2.0
  $ pip install git+https://github.com/huggingface/transformers.git
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
      - `pytorch.all.bpe.4.8m_step` (inhouse)
      - `pytorch.all.dha.2.5m_step` (inhouse)
      - `kor.glove.300k.300d.txt`   (inhouse)
        - training corpus is the same as the data for Korean BERT.
    - Korean ELMo was described [here](https://github.com/dsindex/bilm-tf)
      - `kor_elmo_2x4096_512_2048cnn_2xhighway_1000k_weights.hdf5`, `kor_elmo_2x4096_512_2048cnn_2xhighway_1000k_options.json` (inhouse)
        - training corpus is the same as the data for Korean BERT.

## CoNLL 2003 (english)

### experiments summary

- ntagger

|                                 | F1 (%)                 | features  |
| ------------------------------- | ---------------------  | --------- |
| Glove, BiLSTM-CRF               | 88.49                  | word, pos |
| Glove, DenseNet-CRF             | 88.23                  | word, pos |
| BERT(large), BiLSTM             | 91.29                  | word      |
| ELMo, Glove, BiLSTM             | **92.19**              | word, pos |

- [etagger](https://github.com/dsindex/etagger)

|                                 | F1 (%)                 | features  |
| ------------------------------- | ---------------------  | --------- |
| Glove, BiLSTM-CRF               | 86.48                  | word      |
| Glove, BiLSTM-CRF               | 90.47 ~ 90.85          | word, character, pos, chunk |
| BERT(large), BiLSTM-CRF         | 90.22                  | word, BERT as feature-based |
| BERT(large), Glove, BiLSTM-CRF  | 91.83                  | word, BERT as feature-based |
| BERT(large), Glove, BiLSTM-CRF  | 91.19                  | word trainable, BERT as feature-based |
| ELMo, Glove, BiLSTM-CRF         | 92.45(avg), 92.83      | word, character, pos, chunk |

- [CoNLL 2003(English) learderboard](https://paperswithcode.com/sota/named-entity-recognition-ner-on-conll-2003)

|                                 | F1 (%) |
| ------------------------------- | -------|
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
```

### emb_class=glove, enc_class=densenet

- train
```
* token_emb_dim in configs/config-glove.json == 300 (ex, glove.6B.300d.txt )
$ python preprocess.py --config=configs/config-densenet.json
* --use_crf for adding crf layer, --embedding_trainable for fine-tuning pretrained word embedding
$ python train.py --config=configs/config-densenet.json --save_path=pytorch-model-densenet.pt --use_crf --epoch=50 --warmup_epoch=13 --decay_rate=0.8
```

- evaluation
```
$ python evaluate.py --config=configs/config-densenet.json --model_path=pytorch-model-densenet.pt --use_crf
$ cd data/conll2003; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.8823112371499469, 3684
INFO:__main__:[Elapsed Time] : 91415ms, 24.789302199294053ms on average
accuracy:  97.53%; precision:  88.33%; recall:  88.14%; FB1:  88.23
```

### emb_class=bert, enc_class=bilstm

- train
```
* n_ctx size should be less than 512
* download 'bert-large-cased' to './embeddings'
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
```

## Naver NER 2019 (Korean)

### experiments summary

#### clova2019(eoj-based)

- ntagger

|                       | F1 (%)        | features |
| --------------------- | ------------- | -------- |
| BERT(bpe), BiLSTM-CRF | **85.26**     | eoj      |

- [HanBert-NER](https://github.com/monologg/HanBert-NER#results)

|                       | F1 (%)        | features |
| --------------------- | ------------- | -------- |
| BiLSTM-CRF            | 76.45         | eoj      |
| KoBERT                | 84.23         | eoj      |
| HanBert               | 84.84         | eoj      |

#### clova2019_morph(morph-based)

- ntagger

|                              | m-by-m F1 (%) | e-by-e F1 (%)  | features     |
| ---------------------------- | ------------- | -------------- | ------------ |
| Glove, BiLSTM-CRF            | 84.29         | 84.29          | morph, pos   |
| Glove, DenseNet-CRF          | 83.44         | 83.49          | morph, pos   |
| BERT(dha), BiLSTM-CRF        | 83.78         | 84.13          | morph, pos   |
| ELMo, Glove, BiLSTM-CRF      | 86.37         | **86.37**      | morph, pos   |

- [etagger](https://github.com/dsindex/etagger)

|                              | m-by-m F1 (%) | e-by-e F1 (%)  | features         |
| ---------------------------- | ------------- | -------------- | ---------------- |
| Glove, BiLSTM-CRF            | 85.51         | 85.51          | morph, char, pos |
| BERT(dha), BiLSTM-CRF        | 81.25         | 81.39          | morph, pos, BERT as feature-based |
| ELMo, Glove, BiLSTM-CRF      | 86.75         | 86.75          | morph, character, pos |

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

* evaluation eoj-by-eoj
$ cd data/clova2019_morph ; python to-eoj.py < test.txt.pred > test.txt.pred.eoj ; perl ../../etc/conlleval.pl < test.txt.pred.eoj ; cd ../..
accuracy:  93.37%; precision:  84.83%; recall:  83.76%; FB1:  84.29
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

* evaluation eoj-by-eoj
$ cd data/clova2019_morph ; python to-eoj.py < test.txt.pred > test.txt.pred.eoj ; perl ../../etc/conlleval.pl < test.txt.pred.eoj ; cd ../..
accuracy:  92.96%; precision:  82.86%; recall:  84.13%; FB1:  83.49
```

### emb_class=bert, enc_class=bilstm

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

** evaluation eoj-by-eoj
$ cd data/clova2019_morph ; python to-eoj.py < test.txt.pred > test.txt.pred.eoj ; perl ../../etc/conlleval.pl < test.txt.pred.eoj ; cd ../..
accuracy:  93.47%; precision:  84.26%; recall:  84.01%; FB1:  84.13

* for clova2019

$ python evaluate.py --config=configs/config-bert.json --model_path=pytorch-model-bert-kor-eoj.pt --data_dir data/clova2019 --bert_output_dir=bert-checkpoint --use_crf
$ cd data/clova2019; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..
INFO:__main__:[F1] : 0.8524098438884723, 9000
INFO:__main__:[Elapsed Time] : 396846ms, 44.094ms on average
accuracy:  93.81%; precision:  85.49%; recall:  85.02%; FB1:  85.26
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

* --use_crf --embedding_trainable
INFO:__main__:[F1] : 0.8642865647270933, 9000
INFO:__main__:[Elapsed Time] : 744958ms, 82.7731111111111ms on average
accuracy:  94.63%; precision:  86.36%; recall:  86.38%; FB1:  86.37

* evluation eoj-by-eoj
$ cd data/clova2019_morph ; python to-eoj.py < test.txt.pred > test.txt.pred.eoj ; perl ../../etc/conlleval.pl < test.txt.pred.eoj ; cd ../..
accuracy:  94.26%; precision:  86.37%; recall:  86.38%; FB1:  86.37
```

## references

- [transformers_examples](https://github.com/dsindex/transformers_examples)

