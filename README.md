## ntagger

reference pytorch code for named entity tagging


## requirements

- python >= 3.6

- pip install -r requirements.txt

- pretrained embedding
  - glove
    - [download Glove6B](http://nlp.stanford.edu/data/glove.6B.zip)
  - unzip to 'embeddings' dir
  ```
  $ mkdir embeddings
  $ ls embeddings
  glove.6B.zip
  $ unzip glove.6B.zip 
  ```

- additional requirements for BERT(huggingface's [transformers](https://github.com/huggingface/transformers.git))
```
$ pip install tensorflow-gpu==2.0
$ pip install git+https://github.com/huggingface/transformers.git
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
        는 X-ETM - O
        것 NNB - O
        이 X-JKS - O
        좋다 VA - O
        다고 X-EC - O
        하다 VV - O
        ㄴ다 X-EC - O
        . SF - O
        ```
        - 'X-' prefix is prepending to POS(Part of Speech) tag of inside morphs for distinguishing following morphs.
        - we can evaluate the predicted result morph-by-morph or eojeol by eojeol manner(every lines having 'X-' POS tag are removed).
      - there is no test set. so, set valid.txt as test.txt.
    - Korean BERT and Glove were described [here](https://github.com/dsindex/iclassifier/blob/master/KOR_EXPERIMENTS.md)
      - kor.glove.300k.300d.txt
      - pytorch.all.bpe.4.8m_step
      - pytorch.all.dha.2.5m_step


## CoNLL 2003 (english)

### experiments summary

|                          | F1 (%)                 | features  |
| ------------------------ | ---------------------  | --------- |
| Glove, BiLSTM-CRF        | 88.03                  | word, pos |
| BERT(large), BiLSTM      | **91.13**              | word      |
| Glove, BiLSTM-CRF        | 86.48                  | word, [etagger](https://github.com/dsindex/etagger) |
| Glove, BiLSTM-CRF        | 90.47~90.85            | word, character, pos, chunk, [etagger](https://github.com/dsindex/etagger) |
| BERT(large), BiLSTM-CRF  | 91.87~92.23            | word, [etagger](https://github.com/dsindex/etagger) |
| ELMo, Glove, BiLSTM-CRF  | 92.45(avg), 92.83      | word, character, pos, chunk, [etagger](https://github.com/dsindex/etagger) |

### emb_class=glove

- train
```
* token_emb_dim in config.json == 300 (ex, glove.6B.300d.txt )
$ python preprocess.py
$ python train.py
* --use_crf for adding crf layer, --embedding_trainable for fine-tuning pretrained word embedding
$ python train.py --use_crf

* tensorboardX
$ rm -rf runs
$ tensorboard --logdir runs/ --port ${port} --bind_all
```

- evaluation
```
$ python evaluate.py

* seqeval.metrics supports IOB2(BIO) format, so FB1 from conlleval.pl should be similar value with.
$ cd data/conll2003; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

* --use_crf
$ python evaluate.py --use_crf
INFO:__main__:[F1] : 0.8802560227575785, 3684
INFO:__main__:[Elapsed Time] : 123709ms, 33.580076004343105ms on average

accuracy:  97.48%; precision:  88.39%; recall:  87.66%; FB1:  88.03

```

### emb_class=bert

- train
```
* ignore token_emb_dim in config.json
* n_ctx size should be less than 512
* download 'bert-large-cased' to './'
$ python preprocess.py --emb_class=bert --bert_model_name_or_path=./bert-large-cased

* fine-tuning
$ python train.py --emb_class=bert --bert_model_name_or_path=./bert-large-cased --bert_output_dir=bert-checkpoint --batch_size=16 --lr=1e-5 --epoch=10

* --use_crf for adding crf layer

* --bert_use_feature_based for feature-based

* --bert_disable_lstm for removing lstm layer

* tensorboardX
$ rm -rf runs
$ tensorboard --logdir runs/ --port port-number --bind_all
```

- evaluation
```
$ python evaluate.py --emb_class=bert --data_dir=data/conll2003 --bert_output_dir=bert-checkpoint

$ cd data/conll2003; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

* fine-tuning
  * bert-large-cased
    INFO:__main__:[F1] : 0.9113453192808433, 3684
    INFO:__main__:[Elapsed Time] : 170391ms, 46.251628664495115ms on average
    * --use_crf
      1) lstm_dropout:0.0, lr:1e-5
        INFO:__main__:[F1] : 0.908178536843032, 3684
        INFO:__main__:[Elapsed Time] : 244903ms, 66.47747014115092ms on average
        FB1:  91.07 (by conlleval.pl)
      2) lstm_dropout:0.1, lr:1e-5
        INFO:__main__:[F1] : 0.9071403447062961, 3684
        INFO:__main__:[Elapsed Time] : 246333ms, 66.86563517915309ms on average
        FB1:  90.91
      3) lstm_dropout:0.1, lr:2e-5
        INFO:__main__:[F1] : 0.8962568711281739, 3684
        INFO:__main__:[Elapsed Time] : 237240ms, 64.39739413680782ms on average
        FB1:  89.84
    * --bert_disable_lstm
      INFO:__main__:[F1] : 0.9006085192697768, 3684
      INFO:__main__:[Elapsed Time] : 138880ms, 37.69815418023887ms on average
    * --use_crf --bert_disable_lstm
      INFO:__main__:[F1] : 0.9044752682543836, 3684
      INFO:__main__:[Elapsed Time] : 214022ms, 58.09500542888165ms on average
      FB1:  90.65
```

## Naver NER 2019 (Korean)

### experiments summary

- clova2019(eoj-based)

|                       | F1 (%)        | features |
| --------------------- | ------------- | -------- |
| BERT(bpe), BiLSTM-CRF | 84.71         | eoj      |
| BiLSTM-CRF            | 76.45         | eoj, refer to [HanBert-NER](https://github.com/monologg/HanBert-NER#results) |
| KoBERT                | 84.23         | eoj, refer to [HanBert-NER](https://github.com/monologg/HanBert-NER#results) |
| HanBert               | 84.84         | eoj, refer to [HanBert-NER](https://github.com/monologg/HanBert-NER#results) |

- clova2019_morph(morph-based)

|                             | m-by-m F1 (%) | e-by-e F1 (%)  | features     |
| --------------------------- | ------------- | -------------- | ------------ |
| Glove, BiLSTM-CRF           | 83.76         | 83.76          | morph, pos   |
| BERT(dha), BiLSTM-CRF       | 80.05         | 82.10          | morph        |
| BERT(dha), BiLSTM-CRF       | 83.99         | 84.36          | morph, pos   |
| Glove, BiLSTM-CRF           | 85.51         | 85.51          | morph, character, pos, chunk, [etagger](https://github.com/dsindex/etagger) |
| BERT(dha), BiLSTM-CRF       | 79.70         | 80.03          | morph, pos, [etagger](https://github.com/dsindex/etagger), suspecting something goes wrong. |
| ELMo, Glove, BiLSTM-CRF     | 86.75         | 86.75          | morph, character, pos, chunk, [etagger](https://github.com/dsindex/etagger) |

### emb_class=glove

- train
```
* token_emb_dim in config.json == 300 (ex, kor.glove.300k.300d.txt )
$ python preprocess.py --data_dir data/clova2019_morph --embedding_path embeddings/kor.glove.300k.300d.txt
$ python train.py --data_dir data/clova2019_morph
* --use_crf for adding crf layer, --embedding_trainable for fine-tuning pretrained word embedding.
$ python train.py --data_dir data/clova2019_morph --use_crf --embedding_trainable

* tensorboardX
$ rm -rf runs
$ tensorboard --logdir runs/ --port ${port} --bind_all
```

- evaluation
```
$ python evaluate.py --data_dir data/clova2019_morph
* seqeval.metrics supports IOB2(BIO) format, so FB1 from conlleval.pl should be similar value with.
$ cd data/clova2019_morph; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

* --use_crf --embedding_trainable
$ python evaluate.py --data_dir data/clova2019_morph --use_crf
INFO:__main__:[F1] : 0.8381697194210057, 9000
INFO:__main__:[Elapsed Time] : 340432ms, 37.82577777777778ms on average

accuracy:  93.60%; precision:  84.45%; recall:  83.08%; FB1:  83.76

* evaluation eoj-by-eoj
$ cd data/clova2019_morph ; python to-eoj.py < test.txt.pred > test.txt.pred.eoj ; perl ../../etc/conlleval.pl < test.txt.pred.eoj ; cd ../..

accuracy:  93.15%; precision:  84.46%; recall:  83.08%; FB1:  83.76

```

### emb_class=bert

- train
```
* ignore token_emb_dim in config.json
* n_ctx size should be less than 512

* for clova2019_morph

$ python preprocess.py --emb_class=bert --data_dir data/clova2019_morph --bert_model_name_or_path=./pytorch.all.dha.2.5m_step
$ python train.py --emb_class=bert --bert_model_name_or_path=./pytorch.all.dha.2.5m_step --bert_output_dir=bert-checkpoint --batch_size=32 --lr=5e-5 --epoch=20 --data_dir data/clova2019_morph --use_crf

* for clova2019

$ python preprocess.py --emb_class=bert --data_dir data/clova2019 --bert_model_name_or_path=./pytorch.all.bpe.4.8m_step
$ python train.py --emb_class=bert --bert_model_name_or_path=./pytorch.all.bpe.4.8m_step --bert_output_dir=bert-checkpoint --batch_size=32 --lr=5e-5 --epoch=20 --data_dir data/clova2019 --use_crf

* tensorboardX
$ rm -rf runs
$ tensorboard --logdir runs/ --port port-number --bind_all
```

- evaluation
```
* for clova2019_morph

$ python evaluate.py --emb_class=bert --data_dir=data/clova2019_morph --bert_output_dir=bert-checkpoint --use_crf
INFO:__main__:[F1] : 0.8405158425143053, 9000
INFO:__main__:[Elapsed Time] : 488292ms, 54.254666666666665ms on average

$ cd data/clova2019_morph; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..
accuracy:  94.01%; precision:  84.33%; recall:  83.64%; FB1:  83.99

* evaluation eoj-by-eoj
$ cd data/clova2019_morph ; python to-eoj.py < test.txt.pred > test.txt.pred.eoj ; perl ../../etc/conlleval.pl < test.txt.pred.eoj ; cd ../..
accuracy:  93.50%; precision:  84.90%; recall:  83.83%; FB1:  84.36

* for clova2019

$ python evaluate.py --emb_class=bert --data_dir data/clova2019 --bert_output_dir=bert-checkpoint --use_crf

$ cd data/clova2019; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..
accuracy:  93.53%; precision:  84.27%; recall:  85.16%; FB1:  84.71

```

## references

- [transformers_examples](https://github.com/dsindex/transformers_examples)

