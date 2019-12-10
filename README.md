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
  - CoNLL 2003 english
    - `data/conll2003`
    - from [etagger](https://github.com/dsindex/etagger)
    - [SOTA on CoNLL 2003 english](https://paperswithcode.com/sota/named-entity-recognition-ner-on-conll-2003)

## CoNLL 2003 english

### emb_class=glove

- train
```
* token_emb_dim in config.json == 300 (ex, glove.6B.300d.txt )
$ python preprocess.py
$ python train.py
* --use_crf
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
* seqeval.metrics supports IOB2(BIO) format, so FB1 from conlleval.pl should be same.
$ cd data/conll2003; paste -d ' ' test.txt pred.txt > test-pred.txt ; perl ../../etc/conlleval.pl < test-pred.txt
accuracy:  96.80%; precision:  86.10%; recall:  85.30%; FB1:  85.70
              LOC: precision:  86.43%; recall:  91.61%; FB1:  88.94  1768
             MISC: precision:  76.11%; recall:  70.80%; FB1:  73.36  653
              ORG: precision:  83.27%; recall:  80.01%; FB1:  81.61  1596
              PER: precision:  92.72%; recall:  90.54%; FB1:  91.61  1579

* --use_crf
$ python evaluate.py --use_crf
INFO:__main__:[F1] : 0.8594463853802741, 3684
INFO:__main__:[Elapsed Time] : 154887ms, 42.04315960912052ms on average
accuracy:  96.79%; precision:  86.98%; recall:  84.93%; FB1:  85.94
              LOC: precision:  88.15%; recall:  91.01%; FB1:  89.56  1722
             MISC: precision:  75.27%; recall:  70.23%; FB1:  72.66  655
              ORG: precision:  85.25%; recall:  79.71%; FB1:  82.39  1553
              PER: precision:  92.24%; recall:  90.41%; FB1:  91.32  1585
```

- best : **85.94%** (test set)

### emb_class=bert

- train
```
* ignore token_emb_dim in config.json
* n_ctx size should be less than 512
* download 'bert-large-cased-whole-word-masking' to './'
$ python preprocess.py --emb_class=bert --bert_model_name_or_path=./bert-large-cased-whole-word-masking

* fine-tuning
$ python train.py --emb_class=bert --bert_model_name_or_path=./bert-large-cased-whole-word-masking --bert_output_dir=bert-checkpoint --batch_size=16 --lr=5e-5 --epoch=3

* feature-based
$ python train.py --emb_class=bert --bert_model_name_or_path=./bert-large-cased-whole-word-masking --bert_output_dir=bert-checkpoint --batch_size=16 --lr=2e-5 --bert_use_feature_based

* tensorboardX
$ rm -rf runs
$ tensorboard --logdir runs/ --port port-number --bind_all
```

- evaluation
```
$ python evaluate.py --emb_class=bert --bert_output_dir=bert-checkpoint --data_path=data/conll2003/test.txt.fs

* fine-tuning

  * --use_crf

* feature-based

  * --use_crf

```

- best : **** (test set)

## references

- [transformers_examples](https://github.com/dsindex/transformers_examples)

