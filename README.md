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
INFO:__main__:[F1] : 0.8508414526129319, 3684
INFO:__main__:[Elapsed Time] : 51865ms, 14.078447339847992ms on average
* seqeval.metrics supports IOB2(BIO) format, so FB1 from conlleval.pl should be same.
$ paste -d ' ' data/conll2003/test.txt data/conll2003/pred.txt > pred.txt ; perl etc/conlleval.pl < pred.txt
accuracy:  96.81%; precision:  85.13%; recall:  85.04%; FB1:  85.08
              LOC: precision:  85.99%; recall:  90.89%; FB1:  88.37  1763
             MISC: precision:  72.88%; recall:  69.66%; FB1:  71.23  671
              ORG: precision:  84.49%; recall:  79.35%; FB1:  81.84  1560
              PER: precision:  89.81%; recall:  91.53%; FB1:  90.66  1648

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
$ python preprocess.py --emb_class=bert --bert_model_name_or_path=bert-base-uncased --bert_do_lower_case

* fine-tuning
$ python train.py --emb_class=bert --bert_model_name_or_path=bert-base-uncased --bert_do_lower_case --bert_output_dir=bert-checkpoint --lr=5e-5 --epoch=3
$ python train.py --emb_class=bert --bert_model_name_or_path=bert-base-uncased --bert_do_lower_case --bert_output_dir=bert-checkpoint --lr=5e-5 --epoch=3 --bert_model_class=TextBertCLS

* feature-based
$ python train.py --emb_class=bert --bert_model_name_or_path=bert-base-uncased --bert_do_lower_case --bert_output_dir=bert-checkpoint --bert_use_feature_based
$ python train.py --emb_class=bert --bert_model_name_or_path=bert-base-uncased --bert_do_lower_case --bert_output_dir=bert-checkpoint --bert_use_feature_based --bert_model_class=TextBertCLS

* tensorboardX
$ rm -rf runs
$ tensorboard --logdir runs/ --port port-number --bind_all
```

- evaluation
```
1) --bert_model_class=TextBertCNN
$ python evaluate.py --emb_class=bert --bert_output_dir=bert-checkpoint --bert_do_lower_case --data_path=data/snips/test.txt.fs

* fine-tuning

  ** --bert_model_name_or_path=bert-large-uncased --lr=2e-5

* feature-based, --epoch=30

2) --bert_model_class=TextBertCLS
$ python evaluate.py --emb_class=bert --bert_output_dir=bert-checkpoint --bert_do_lower_case --data_path=data/snips/test.txt.fs --bert_model_class=TextBertCLS

* fine-tuning

  ** --bert_model_name_or_path=bert-large-uncased --lr=2e-5

* feature-based, --epoch=100
```

- best : **** (test set)

## experiments for Korean

- [KOR_EXPERIMENTS.md](/KOR_EXPERIMENTS.md)

## references

