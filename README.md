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
      - data/clova2019: converted to CoNLL data format.
      - data/clova2019_morph: tokenized by morphological analyzer and converted to CoNLL data format.
    - there is no test set. so, set valid.txt as test.txt.
    - Korean BERT and Glove were described [here](https://github.com/dsindex/iclassifier/blob/master/KOR_EXPERIMENTS.md)


## CoNLL 2003 (english)

### experiments summary

|                   | F1 (%)           |          |
| ----------------- | ---------------  | -------- |
| Glove, BiLSTM-CRF | 86.05            | word     |
| BERT, BiLSTM      | **91.13**        | word     |
| Glove, BiLSTM-CRF | 86.48(max)       | word, [etagger](https://github.com/dsindex/etagger) |
| Glove, BiLSTM-CRF | 90.47~90.85(max) | word, character, pos, [etagger](https://github.com/dsindex/etagger) |
| BERT, BiLSTM-CRF  | 91.87~92.23(max) | word, [etagger](https://github.com/dsindex/etagger) |
| ELMo, Glove, BiLSTM-CRF  | 92.45(avg), 92.83(max)| word, character, pos, [etagger](https://github.com/dsindex/etagger) |

### emb_class=glove

- train
```
* token_emb_dim in config.json == 300 (ex, glove.6B.300d.txt )
$ python preprocess.py
$ python train.py
* --use_crf for adding crf layer
$ python train.py --use_crf

* tensorboardX
$ rm -rf runs
$ tensorboard --logdir runs/ --port ${port} --bind_all
```

- evaluation
```
$ python evaluate.py
INFO:__main__:[F1] : 0.8569903948772679, 3684
INFO:__main__:[Elapsed Time] : 57432ms, 15.589576547231271ms on average
* seqeval.metrics supports IOB2(BIO) format, so FB1 from conlleval.pl should be similar value with.
$ cd data/conll2003; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..
accuracy:  96.80%; precision:  86.10%; recall:  85.30%; FB1:  85.70

* --use_crf
$ python evaluate.py --use_crf
INFO:__main__:[F1] : 0.8605357142857142, 3684
INFO:__main__:[Elapsed Time] : 156063ms, 42.362377850162865ms on average
  * token_emb_dim: 100
  INFO:__main__:[F1] : 0.8587449933244327, 3684
  INFO:__main__:[Elapsed Time] : 344884ms, 93.61672095548317ms on average (cpu)
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
$ python evaluate.py --emb_class=bert --bert_output_dir=bert-checkpoint --data_path=data/conll2003/test.txt.fs

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

|                      | F1 (%)        |          |
| -------------------- | ------------- | -------- |
| Glove, BiLSTM-CRF    | 77.97         | morph-based, word, emb_non_trainable=False |
| dha BERT, BiLSTM-CRF | 79.32         | morph-based, word |
| bpe BERT, BiLSTM-CRF | **84.71**     | eoj-based, word |
| Glove, BiLSTM-CRF    | 81.34         | morph-based, word, character, pos, [etagger](https://github.com/dsindex/etagger) |
| dha BERT, BiLSTM-CRF | 73.57(?)      | morph-based, word, [etagger](https://github.com/dsindex/etagger) |
| ELMo, BiLSTM-CRF     | 82.15         | morph-based, word, [etagger](https://github.com/dsindex/etagger) |
| BiLSTM-CRF           | 76.45         | eoj-based, word, refer to [HanBert-NER](https://github.com/monologg/HanBert-NER#results) |
| HanBert              | 84.84         | eoj-based, word, [HanBert-NER](https://github.com/monologg/HanBert-NER#results) |

### emb_class=glove

- train
```
* token_emb_dim in config.json == 300 (ex, kor.glove.300k.300d.txt )
$ python preprocess.py --data_dir data/clova2019_morph --embedding_path embeddings/kor.glove.300k.300d.txt
$ python train.py --data_dir data/clova2019_morph
* --use_crf for adding crf layer
$ python train.py --data_dir data/clova2019_morph --use_crf --batch_size 20 --lr 0.001

* tensorboardX
$ rm -rf runs
$ tensorboard --logdir runs/ --port ${port} --bind_all
```

- evaluation
```
$ python evaluate.py --data_path data/clova2019_morph/test.txt.ids --embedding_path data/clova2019_morph/embedding.npy --label_path data/clova2019_morph/label.txt --test_path data/clova2019_morph/test.txt
INFO:__main__:[F1] : 0.7159682824060252, 9000
INFO:__main__:[Elapsed Time] : 898312ms, 99.81244444444444ms on average

* seqeval.metrics supports IOB2(BIO) format, so FB1 from conlleval.pl should be similar value with.
$ cd data/clova2019_morph; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..
accuracy:  91.50%; precision:  69.86%; recall:  73.73%; FB1:  71.74
...

* --use_crf, set emb_non_trainable=False
$ python evaluate.py --data_path data/clova2019_morph/test.txt.ids --embedding_path data/clova2019_morph/embedding.npy --label_path data/clova2019_morph/label.txt --test_path data/clova2019_morph/test.txt  --use_crf
INFO:__main__:[F1] : 0.7802072557030387, 9000
INFO:__main__:[Elapsed Time] : 307779ms, 34.19766666666667ms on average
accuracy:  92.15%; precision:  80.44%; recall:  75.64%; FB1:  77.97
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

$ python evaluate.py --emb_class=bert --bert_output_dir=bert-checkpoint --data_path=data/clova2019_morph/test.txt.fs --label_path=data/clova2019_morph/label.txt --test_path=data/clova2019_morph/test.txt --use_crf
INFO:__main__:[F1] : 0.7938620132057673, 9000
INFO:__main__:[Elapsed Time] : 457444ms, 50.82711111111111ms on average

$ cd data/clova2019_morph; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..
accuracy:  92.99%; precision:  78.55%; recall:  80.11%; FB1:  79.32

* for clova2019

$ python evaluate.py --emb_class=bert --bert_output_dir=bert-checkpoint --data_path=data/clova2019/test.txt.fs --label_path=data/clova2019/label.txt --test_path=data/clova2019/test.txt --use_crf
INFO:__main__:[F1] : 0.8468742604050444, 9000
INFO:__main__:[Elapsed Time] : 486540ms, 54.06ms on average

$ cd data/clova2019; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..
accuracy:  93.53%; precision:  84.27%; recall:  85.16%; FB1:  84.71

```

## references

- [transformers_examples](https://github.com/dsindex/transformers_examples)

