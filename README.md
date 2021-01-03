# Description

**reference pytorch code for named entity tagging.**

- embedding
  - word : GloVe, BERT, DistilBERT, feature-based BERT using DSA(Dynamic Self Attention) pooling, SpanBERT, ALBERT, RoBERTa, BART, ELECTRA, ELMo
  - character : CNN
  - pos : lookup
- encoding
  - BiLSTM
  - DenseNet
    - [Dynamic Self-Attention: Computing Attention over Words Dynamically for Sentence Embedding](https://arxiv.org/pdf/1808.07383.pdf)
    - a slightly modified DenseNet for longer dependency.
- decoding
  - Softmax, CRF
- related: [reference pytorch code for intent(sentence) classification](https://github.com/dsindex/iclassifier)

<br>



# Requirements

- python >= 3.6

- pip install -r requirements.txt

<br>



# Data

## CoNLL 2003 (English)

#### from [etagger](https://github.com/dsindex/etagger)
  
##### data/conll2003

##### data/conll2003_truecase

- [converting conll2003 data to its truecase](https://github.com/google-research/bert/issues/223#issuecomment-649619302)
<details><summary>details</summary>
<p>

```
$ cd data/conll2003_truecase
$ python to-truecase.py --input_path ../conll2003/train.txt > train.txt
$ python to-truecase.py --input_path ../conll2003/valid.txt > valid.txt
$ python to-truecase.py --input_path ../conll2003/test.txt > test.txt
```

</p>
</details>


## Kaggle NER (English)

### from [entity-annotated-corpus](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus)

#### data/kaggle

- converting to CoNLL data format.
<details><summary>details</summary>
<p>

```
* download : https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus?select=ner_dataset.csv
* remove illegal characters
$ sed -e 's/5 storm/storm/' ner_dataset.csv > t ; mv t ner_dataset.csv
$ iconv -f ISO-8859-1 -t UTF-8 ner_dataset.csv > ner_dataset.csv.utf

$ python to-conll.py
$ cp -rf valid.txt test.txt
```

</p>
</details>

## GUM (English)

### from [entity-recognition-datasets](https://github.com/juand-r/entity-recognition-datasets)

#### data/gum

- converting to CoNLL data format.
<details><summary>details</summary>
<p>

```
* remove '*-object', '*-abstract'
$ python to-conll.py --input_train=gum-train.conll --inpu_test=gum-test.conll --train=train.txt --valid=valid.txt --test=test.txt
```

</p>
</details>


## Naver NER 2019 (Korean)

#### from [HanBert-NER](https://github.com/monologg/HanBert-NER)    

##### data/clova2019

- converted to CoNLL data format.
<details><summary>details</summary>
<p>

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

</p>
</details>

##### data/clova2019_morph

- tokenized by morphological analyzer and converted to CoNLL data format.
<details><summary>details</summary>
<p>

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

</p>
</details>

- 'X-' prefix is prepending to POS(Part of Speech) tag of inside morphs for distinguishing following morphs.

- we can evaluate the predicted result morph-by-morph or eojeol by eojeol manner(every lines having 'X-' POS tag are removed).
<details><summary>details</summary>
<p>

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

</p>
</details>


##### data/clova2019_morph_space

- this data is identical to `data/clova2019_morph` except it treats spaces as tokens.
<details><summary>details</summary>
<p>
  
```
이기범 NNP - B-PER
_ _ - O
한두 NNP - O
_ _ - O
쪽 NNB - O
을 X-JKO - O
_ _ - O
먹다 VV - O
고 X-EC - O
_ _ - O
10 SN - B-TIM
분 X-NNB - I-TIM
_ _ - I-TIM
후 NNG - I-TIM
쯤 X-XSN - I-TIM
_ _ - O
화제 NNG - B-CVL
인 X-NNG - I-CVL
을 X-JKO - I-CVL
_ _ - O
먹다 VV - O
...
```

</p>
</details>


## KMOU NER 2019 (Korean)

#### from [KMOU NER](https://github.com/kmounlp/NER)

##### data/kmou2019
   
- build train.raw, valid.raw
  - data version : https://github.com/kmounlp/NER/commit/0b32e066870bda9f65cc190f5e89c2edc6cf8f6d
  - same as [pytorch-bert-crf-ner](https://github.com/eagle705/pytorch-bert-crf-ner)
  - train.raw : 00002_NER.txt, ..., EXOBRAIN_NE_CORPUS_007.txt (1,425 files)
  - valid.raw : EXOBRAIN_NE_CORPUS_009.txt, EXOBRAIN_NE_CORPUS_010.txt (2 files)
  - apply correction rules and converting to CoNLL data format
<details><summary>details</summary>
<p>
  
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
$ cp -rf valid.txt test.txt
```

</p>
</details>

<br>




# Pretrained models

- English
  - glove
    - [download GloVe6B](http://nlp.stanford.edu/data/glove.6B.zip) and unzip to 'embeddings' dir
    ```
    $ mkdir embeddings
    $ ls embeddings
    glove.6B.zip
    $ unzip glove.6B.zip 
    ```
  - BERT, ALBERT, RoBERTa, BART, ELECTRA(huggingface's [transformers](https://github.com/huggingface/transformers.git))
  - [SpanBERT](https://github.com/facebookresearch/SpanBERT/blob/master/README.md)
    - pretrained SpanBERT models are compatible with huggingface's BERT modele except `'bert.pooler.dense.weight', 'bert.pooler.dense.bias'`.
  - ELMo([allennlp](https://github.com/allenai/allennlp))
  ```
  $ cd embeddings
  $ curl -OL https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5
  $ curl -OL https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json
  ```

- Korean
  - [description of Korean GloVe, BERT, DistilBERT, ELECTRA](https://github.com/dsindex/iclassifier/blob/master/KOR_EXPERIMENTS.md)
    - GloVe : `kor.glove.300k.300d.txt`   (inhouse)  
    - bpe BERT : `kor-bert-base-bpe.v1`, `kor-bert-large-bpe.v1, v3` (inhouse)
    - dha-bpe BERT : `kor-bert-base-dha_bpe.v1, v3`, `kor-bert-large-dha_bpe.v1, v3` (inhouse)
    - dha BERT : `kor-bert-base-dha.v1, v2` (inhouse)
    - KcBERT : `kcbert-base`, `kcbert-large`
    - DistilBERT : `kor-distil-bpe-bert.v1`, `kor-distil-dha-bert.v1`, `kor-distil-wp-bert.v1` (inhouse)
    - KoELECTRA-Base : `koelectra-base-v1-discriminator`, `koelectra-base-v3-discriminator`
    - LM-KOR-ELECTRA : `electra-kor-base`
    - ELECTRA-base : `kor-electra-bpe.v1` (inhouse)
    - RoBERTa-base : `kor-roberta-base-bbpe` (inhouse)
  - [ELMo description](https://github.com/dsindex/bilm-tf)
    - `kor_elmo_2x4096_512_2048cnn_2xhighway_1000k_weights.hdf5`, `kor_elmo_2x4096_512_2048cnn_2xhighway_1000k_options.json` (inhouse)
  
<br>




# CoNLL 2003 (English)

## experiments summary

- ntagger, measured by conlleval.pl (micro F1)

|                                 | F1 (%)       | (truecase) F1 (%) | Features             | GPU / CPU          | CONDA    | ONNX      | Dynamic   | Etc                       |
| ------------------------------- | ------------ | ----------------- | -------------------- | ------------------ | -------- | --------- | --------- | ------------------------- |
| GloVe, BiLSTM                   | 88.23        |                   | word, pos            | 5.6217  / -        | 7.3838   | 3.6969    |           | threads=14                |
| GloVe, BiLSTM                   | 88.94        |                   | word, character, pos | 6.4108  / -        | 9.5858   | 8.5656    |           | threads=14                |
| **GloVe, BiLSTM-CRF**           | 90.14        | 90.26             | word, character, pos | 26.2807 / -        | 21.7474  |           |           | threads=14                |
| ConceptNet, BiLSTM-CRF          | 87.78        | -                 | word, character, pos | 25.8119 / -        | -        |           |           |                           |
| ConceptNet, BiLSTM-CRF          | 88.17        | -                 | word, character, pos | 23.3482 / -        | -        |           |           | optuna                    |
| GloVe, DenseNet-CRF             | 88.23        |                   | word, pos            | 24.7893 / -        | 22.5858  |           |           | threads=14                |
| GloVe, DenseNet-CRF             | 88.89        |                   | word, character, pos | 28.0993 / -        | 25.2929  |           |           | threads=14                |
| BERT-tiny, BiLSTM               | 69.65        |                   | word                 | 20.1376 / -        |          |           |           |                           |
| BERT-mini, BiLSTM               | 81.55        |                   | word                 | 21.4632 / -        |          |           |           |                           |
| BERT-small, BiLSTM              | 86.35        |                   | word                 | 22.6087 / -        |          |           |           |                           |
| BERT-medium, BiLSTM             | 88.29        |                   | word                 | 27.0486 / -        |          |           |           |                           |
| DistilBERT, BiLSTM              | 89.50        |                   | word                 | 13.4564 / -        | 58.9260  | 56.2819   | 47.8320   |                           |
| MiniLM, BiLSTM                  | 90.55        |                   | word                 | 17.7890 / -        | -        | -         | -         |                           |
| BERT-base(uncased), BiLSTM-CRF  | 90.20        |                   | word                 | 42.6464 / -        |          |           |           |                           |
| BERT-base(uncased), BiLSTM      | 90.55        |                   | word                 | 18.2323 / -        | 100.0505 | 79.1914   | 83.9590   |                           |
| BERT-base(uncased), CRF         | 89.98        |                   | word                 | 36.6893 / -        |          |           |           |                           |
| BERT-base(uncased)              | 90.25        |                   | word                 | 16.6877 / -        | 96.9004  | 72.8225   | 75.3025   |                           |
| BERT-base(uncased), BiLSTM      | 89.03        |                   | word                 | 24.9076 / -        |          |           |           | del 8,9,10,11, threads=14 |
| bert-base-NER(cased), BiLSTM    | 91.63        | 92.25             | word                 | 17.6680 / -        |          |           |           |                           |
| BERT-large, BiLSTM              | 91.32        | 91.89             | word                 | 40.3581 / -        |          |           |           |                           |
| BERT-large                      | 91.25        |                   | word                 | 29.5740 / -        |          |           |           |                           |
| BERT-large, BiLSTM              | 89.10        |                   | word                 | 33.1376 / -        |          |           |           | del 12 ~ 23               |
| BERT-large, BiLSTM              | 86.11        |                   | word                 | 49.3103 / -        |          |           |           | BERT as feature-based, initial embedding             |
| BERT-large, BiLSTM-CRF          | 86.43        |                   | word                 | 63.1376 / -        |          |           |           | BERT as feature-based, initial embedding             |
| BERT-large, BiLSTM              | 89.72        |                   | word                 | 47.9704 / -        |          |           |           | BERT as feature-based, initial+first+last embedding  |
| BERT-large, BiLSTM-CRF          | 89.96        |                   | word                 | 67.2041 / -        |          |           |           | BERT as feature-based, initial+first+last embedding  |
| BERT-large, BiLSTM-CRF          | 89.67        |                   | word                 | 68.7548 / -        |          |           |           | BERT as feature-based, last embedding                |
| BERT-large, BiLSTM-CRF          | 90.64        |                   | word                 | 63.9397 / -        |          |           |           | BERT as feature-based, [-4:] embedding               |
| BERT-large, BiLSTM-CRF          | 90.52        |                   | word                 | 70.8322 / -        |          |           |           | BERT as feature-based, mean([0:3] + [-4:]) embedding |
| BERT-large, BiLSTM-CRF          | 90.81        |                   | word                 | 68.6139 / -        |          |           |           | BERT as feature-based, mean([0:17]) embedding        |
| BERT-large, BiLSTM-CRF          | 90.76        |                   | word                 | 60.8039 / -        |          |           |           | BERT as feature-based, max([0:17]) embedding         |
| BERT-large, BiLSTM-CRF          | 90.98        |                   | word                 | 58.9112 / -        |          |           |           | BERT as feature-based, mean([0:]) embedding          |
| BERT-large, BiLSTM-CRF          | 90.62        |                   | word                 | 66.6576 / -        |          |           |           | BERT as feature-based, DSA(4, 300)                   |
| BERT-large-squad, BiLSTM        | 91.75        | 92.17             | word                 | 35.6619 / -        |          |           |           |                           |
| SpanBERT-base, BiLSTM           | 90.46        |                   | word                 | 30.0991 / -        |          |           |           |                           |
| SpanBERT-large, BiLSTM          | 91.39        | 92.01             | word                 | 42.5959 / -        |          |           |           |                           |
| ALBERT-base, BiLSTM             | 88.19        |                   | word                 | 31.0868 / -        |          |           |           |                           |
| ALBERT-xxlarge, BiLSTM          | 90.39        |                   | word                 | 107.778 / -        |          |           |           |                           |
| RoBERTa-base                    | 90.03        |                   | word                 | 19.2503 / -        |          |           |           |                           |
| RoBERTa-large                   | 91.83        | 91.90             | word                 | 28.5525 / -        |          |           |           |                           |
| BART-large, BiLSTM              | 90.43        |                   | word                 | 53.3657 / -        |          |           |           |                           |
| ELECTRA-base, BiLSTM            | 90.98        |                   | word                 | 22.4132 / -        |          |           |           |                           |
| ELECTRA-large                   | 91.39        |                   | word                 | 29.5734 / -        |          |           |           |                           |
| ELMo, BiLSTM-CRF                | 91.78        |                   | word, pos            | 74.1001 / -        |          |           |           |                           |
| ELMo, BiLSTM-CRF                | 91.93        |                   | word, character, pos | 67.6931 / -        |          |           |           |                           |
| ELMo, GloVe, BiLSTM-CRF         | **92.63**    | 92.51             | word, pos            | 74.6521 / -        |          |           |           |                           |
| ELMo, GloVe, BiLSTM-CRF         | 92.03        |                   | word, character, pos | 60.4667 / -        | 182.595  |           |           | threads=14                |

```
* GPU / CPU : Elapsed time/example(ms), GPU / CPU(pip 1.2.0), [Tesla V100 1 GPU, Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz, 2 CPU, 14CORES/1CPU, HyperThreading]
* CONDA     : conda pytorch=1.5.0
* ONNX      : conda pytorch=1.5.0, onnxruntime 1.3.0
* Dynamic   : conda pytorch=1.5.0, --enable_dqm
* default batch size, learning rate, n_ctx(max_seq_length) : 32, 1e-3, 180
```

- [etagger](https://github.com/dsindex/etagger), measured by conlleval (micro F1)

|                                     | F1 (%)            | (truecase) F1 (%) | Features                     | GPU / CPU        | Etc                               |
| ----------------------------------- | ----------------- | ----------------- | ---------------------------- | ---------------- | --------------------------------- |
| GloVe, BiLSTM-CRF                   | 87.91             |                   | word                         | - / -            |                                   |
| GloVe, BiLSTM-CRF                   | 89.20             |                   | word, pos                    | 14.9682 / 5.0336 | LSTMBlockFusedCell(), threads=14  |
| GloVe, BiLSTM-CRF                   | 90.06             |                   | word, character, pos         | 15.8913 / 5.7952 | LSTMBlockFusedCell(), threads=14  |
| GloVe, BiLSTM-CRF                   | 90.57             |                   | word, character, pos         | 24.6356 / 7.0887 | LSTMCell(), threads=14            |
| GloVe, BiLSTM-CRF                   | 90.85             |                   | word, character, pos, chunk  | - / -            |                                   |
| GloVe, Transformer                  | 88.62             |                   | word, character, pos, chunk  | 8.6126  / -      | layers=4, heads=4, units=64       |
| GloVe, Transformer-CRF              | 89.26             |                   | word, character, pos, chunk  | 16.528  / -      | layers=4, heads=4, units=64       |
| GloVe, Transformer-CRF              | 89.42             |                   | word, character, pos, chunk  | 19.157  / -      | layers=6, heads=4, units=64       |
| GloVe, Transformer-CRF              | 88.45             |                   | word, character, pos, chunk  | 30.534  / -      | layers=6, heads=6, units=64       |
| GloVe, Transformer-CRF              | 88.93             |                   | word, character, pos, chunk  | 20.257  / -      | layers=6, heads=6, units=96       |
| BERT-large, BiLSTM-CRF              | 90.22             |                   | word                         | - / -            | BERT as feature-based             |
| BERT-large, GloVe, BiLSTM-CRF       | 91.83             |                   | word                         | - / -            | BERT as feature-based             |
| ELMo, GloVe, BiLSTM-CRF             | 91.78             |                   | word, pos                    | - / -            |                                   |
| ELMo, GloVe, BiLSTM-CRF             | 92.38             |                   | word, character, pos         | 46.4205 / 295.28 | threads=14                        |
| ELMo, GloVe, BiLSTM-CRF             | 92.43             |                   | word, character, pos, chunk  | - / -            |                                   |
| ELMo, GloVe, BiLSTM-CRF             | **92.83**         | 92.10             | word, character, pos, chunk  | - / -            | GloVe-100d                        |
| BERT-large, ELMo, GloVe, BiLSTM-CRF | 92.54             |                   | word, character, pos         | - / -            | BERT as feature-based, GloVe-100d |

- [CoNLL 2003(English) leaderboard](https://paperswithcode.com/sota/named-entity-recognition-ner-on-conll-2003), measured by span-level F1 (micro F1, same result by conlleval? Unknown!)

|                                 | F1 (%)   | Etc                 |
| ------------------------------- | -------- | ------------------- |
| LUKE                            | **94.3** |                     |
| ACE                             | 93.63    |                     |
| CNN Large + fine-tune           | 93.5     |                     |
| biaffine-ner                    | 93.5     |                     |
| GCDT + BERT-L                   | 93.47    |                     |
| LSTM-CRF+ELMo+BERT+Flair        | 93.38    |                     |
| Hierarchical + BERT             | 93.37    |                     |
| Flair embeddings + Pooling      | 93.18    |                     |
| BERT Large                      | 92.8     |                     |
| BERT Base                       | 92.4     |                     |
| BiLSTM-CRF+ELMo                 | 92.22    |                     |


<details><summary><b>emb_class=glove, enc_class=bilstm</b></summary>
<p>

- train
```
* token_emb_dim in configs/config-glove.json == 300 (ex, glove.6B.300d.txt )
$ python preprocess.py --data_dir=data/conll2003
* --use_crf for adding crf layer, --embedding_trainable for fine-tuning pretrained word embedding
$ python train.py --data_dir=data/conll2003 --use_crf

* tensorboardX
$ rm -rf runs
$ tensorboard --logdir runs/ --port ${port} --bind_all
```

- evaluation
```
$ python evaluate.py --data_dir=data/conll2003 --use_crf
* seqeval.metrics supports IOB2(BIO) format, so FB1 from conlleval.pl should be similar value with.
$ cd data/conll2003; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.8864423333037378, 3684
INFO:__main__:[Elapsed Time] : 97879ms, 26.54547922888949ms on average
accuracy:  97.60%; precision:  88.90%; recall:  88.39%; FB1:  88.64

* --use_char_cnn 
INFO:__main__:[F1] : 0.9013611454834718, 3684
INFO:__main__:[Elapsed Time] : 96906ms, 26.280749389084985ms on average
accuracy:  97.93%; precision:  89.99%; recall:  90.28%; FB1:  90.14

INFO:__main__:[F1] : 0.8981972428419936, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 95621.84238433838ms, 25.92995391408613ms on average
accuracy:  97.82%; precision:  89.66%; recall:  89.98%; FB1:  89.82

* --data_dir=data/conll2003_truecase --use_char_cnn 
INFO:__main__:[F1] : 0.902609464838567, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 95198.03929328918ms, 25.81831880021024ms on average
accuracy:  97.81%; precision:  90.19%; recall:  90.33%; FB1:  90.26

* --use_char_cnn --embedding_path=./embeddings/numberbatch-en-19.08.txt (from https://github.com/commonsense/conceptnet-numberbatch)
INFO:__main__:[F1] : 0.877781702578594, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 95196.46453857422ms, 25.811972386352142ms on average
accuracy:  97.47%; precision:  87.56%; recall:  88.00%; FB1:  87.78

* --use_char_cnn --lr=0.0025131520181464126 --batch_size=32  --seed=40   --embedding_path=./embeddings/numberbatch-en-19.08.txt (by optuna)
INFO:__main__:[F1] : 0.8817204301075269, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 86105.42750358582ms, 23.34824522455005ms on average
accuracy:  97.55%; precision:  87.79%; recall:  88.56%; FB1:  88.17

```

</p>
</details>


<details><summary><b>emb_class=glove, enc_class=densenet</b></summary>
<p>

- train
```
* token_emb_dim in configs/config-glove.json == 300 (ex, glove.6B.300d.txt )
$ python preprocess.py --config=configs/config-densenet.json --data_dir=data/conll2003
* --use_crf for adding crf layer, --embedding_trainable for fine-tuning pretrained word embedding
$ python train.py --config=configs/config-densenet.json --data_dir=data/conll2003 --save_path=pytorch-model-densenet.pt --use_crf --epoch=64
```

- evaluation
```
$ python evaluate.py --config=configs/config-densenet.json --data_dir=data/conll2003 --model_path=pytorch-model-densenet.pt --use_crf
$ cd data/conll2003; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.8823112371499469, 3684
INFO:__main__:[Elapsed Time] : 91415ms, 24.789302199294053ms on average
accuracy:  97.53%; precision:  88.33%; recall:  88.14%; FB1:  88.23

INFO:__main__:[F1] : 0.8736429969745506, 3684
INFO:__main__:[Elapsed Time] : 96335ms, 26.12136844963345ms on average
accuracy:  97.36%; precision:  87.82%; recall:  86.92%; FB1:  87.36

* --use_char_cnn
INFO:__main__:[F1] : 0.8847577092511013, 3684
INFO:__main__:[Elapsed Time] : 105399ms, 28.585120825414066ms on average
accuracy:  97.60%; precision:  88.06%; recall:  88.90%; FB1:  88.48

INFO:__main__:[F1] : 0.8889086226800461, 3684
INFO:__main__:[Elapsed Time] : 103641ms, 28.099375509095847ms on average
accuracy:  97.75%; precision:  89.17%; recall:  88.62%; FB1:  88.89

```

</p>
</details>


<details><summary><b>emb_class=bert, enc_class=bilstm</b></summary>
<p>

- train
```
* n_ctx size should be less than 512
$ python preprocess.py --config=configs/config-bert.json --data_dir=data/conll2003 --bert_model_name_or_path=./embeddings/bert-large-cased
* --use_crf for adding crf layer
* --bert_use_pos for adding Part-Of-Speech features
* --bert_use_feature_based for feature-based
* --bert_disable_lstm for removing lstm layer
$ python train.py --config=configs/config-bert.json --data_dir=data/conll2003 --save_path=pytorch-model-bert.pt --bert_model_name_or_path=./embeddings/bert-large-cased --bert_output_dir=bert-checkpoint --batch_size=16 --lr=1e-5 --epoch=10
```

- evaluation
```
$ python evaluate.py --config=configs/config-bert.json --data_dir=data/conll2003 --model_path=pytorch-model-bert.pt --bert_output_dir=bert-checkpoint
$ cd data/conll2003; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.9131544214694237, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 148789ms, 40.358131957643224ms on average
accuracy:  98.27%; precision:  90.76%; recall:  91.87%; FB1:  91.32

* --batch_size=32
INFO:__main__:[F1] : 0.9118733509234828, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 141879ms, 38.48330165625848ms on average
accuracy:  98.24%; precision:  90.60%; recall:  91.78%; FB1:  91.19

* --bert_use_pos
INFO:__main__:[F1] : 0.9111325554873234, 3684
INFO:__main__:[Elapsed Time] : 141093ms, 38.29885993485342ms on average
accuracy:  98.24%; precision:  90.30%; recall:  91.94%; FB1:  91.11

* --use_crf
INFO:__main__:[F1] : 0.9058430130235833, 3684
INFO:__main__:[Elapsed Time] : 218823ms, 59.398208469055376ms on average
accuracy:  98.12%; precision:  90.44%; recall:  91.13%; FB1:  90.78

* --data_dir=data/conll2003_truecase 
INFO:__main__:[F1] : 0.9188571428571428, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 120905.13277053833ms, 32.785276922556875ms on average
accuracy:  98.36%; precision:  91.25%; recall:  92.53%; FB1:  91.89

* --bert_disable_lstm --batch_size=32
INFO:__main__:[F1] : 0.9124989051414557, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 109073.38047027588ms, 29.574078060898998ms on average
accuracy:  98.27%; precision:  90.29%; recall:  92.23%; FB1:  91.25

* --bert_model_name_or_path=./embedings/bert-base-uncased --use_crf (BERT-base BiLSTM-CRF)
INFO:__main__:[F1] : 0.8993429697766097, 3684
INFO:__main__:[Elapsed Time] : 100 examples, 4368ms, 42.64646464646464ms on average
accuracy:  97.87%; precision:  89.52%; recall:  90.88%; FB1:  90.20

* --bert_model_name_or_path=./embedings/bert-base-uncased (BERT-base BiLSTM)
INFO:__main__:[F1] : 0.9054532577903682, 3684
INFO:__main__:[Elapsed Time] : 100 examples, 1922ms, 18.232323232323232ms on average
accuracy:  98.00%; precision:  90.55%; recall:  90.55%; FB1:  90.55

* --bert_model_name_or_path=./embedings/bert-base-uncased   --warmup_epoch=0 --weight_decay=0.0 --epoch=20 (BERT-base BiLSTM)
INFO:__main__:[F1] : 0.9049717912552891, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 68218ms, 18.403203909856096ms on average
accuracy:  97.97%; precision:  90.12%; recall:  90.88%; FB1:  90.50

* --bert_model_name_or_path=./embedings/bert-base-uncased  --bert_disable_lstm --use_crf (BERT-base CRF)
INFO:__main__:[F1] : 0.8961356880573526, 3684
INFO:__main__:[Elapsed Time] : 135607ms, 36.68938365462938ms on average
accuracy:  97.78%; precision:  89.24%; recall:  90.74%; FB1:  89.98

* --bert_model_name_or_path=./embedings/bert-base-uncased  --bert_disable_lstm (BERT-base)
INFO:__main__:[F1] : 0.9024668598015978, 3684
INFO:__main__:[Elapsed Time] : 61914ms, 16.68775454792289ms on average
accuracy:  98.01%; precision:  89.50%; recall:  91.01%; FB1:  90.25

* https://huggingface.co/dslim/bert-base-NER
* --bert_model_name_or_path=./embeddings/bert-base-NER  --warmup_epoch=0 --weight_decay=0.0 --epoch=20
INFO:__main__:[F1] : 0.9163073132095397, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 65167.21606254578ms, 17.668077811708017ms on average
accuracy:  98.34%; precision:  91.09%; recall:  92.17%; FB1:  91.63

* --data_dir=data/conll2003_truecase --bert_model_name_or_path=./embeddings/bert-base-NER  --warmup_epoch=0 --weight_decay=0.0 --epoch=20
INFO:__main__:[F1] : 0.9224547212941797, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 66638.26131820679ms, 18.064501684967215ms on average
accuracy:  98.38%; precision:  91.62%; recall:  92.88%; FB1:  92.25

* --bert_model_name_or_path=./embedings/pytorch.uncased_L-8_H-512_A-8 
INFO:__main__:[F1] : 0.8829307140960977, 3684
INFO:__main__:[Elapsed Time] : 99730ms, 27.048601683410265ms on average
accuracy:  97.61%; precision:  88.25%; recall:  88.33%; FB1:  88.29

* --bert_model_name_or_path=./embedings/pytorch.uncased_L-4_H-512_A-8 
INFO:__main__:[F1] : 0.8634692805881324, 3684
INFO:__main__:[Elapsed Time] : 83385ms, 22.60874287265816ms on average
accuracy:  97.23%; precision:  85.38%; recall:  87.34%; FB1:  86.35

* --bert_model_name_or_path=./embedings/pytorch.uncased_L-4_H-256_A-4 
INFO:__main__:[F1] : 0.8155101324677603, 3684
INFO:__main__:[Elapsed Time] : 79173ms, 21.463209340211783ms on average
accuracy:  96.32%; precision:  80.82%; recall:  82.29%; FB1:  81.55

* --bert_model_name_or_path=./embedings/pytorch.uncased_L-2_H-128_A-2 
INFO:__main__:[F1] : 0.6965218958370878, 3684
INFO:__main__:[Elapsed Time] : 74261ms, 20.137659516698342ms on average
accuracy:  94.12%; precision:  70.92%; recall:  68.43%; FB1:  69.65

* --config=configs/config-distilbert.json --bert_model_name_or_path=./embeddings/distilbert-base-uncased 
INFO:__main__:[F1] : 0.894963522897073, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 49652ms, 13.456421395601412ms on average
accuracy:  97.80%; precision:  88.86%; recall:  90.14%; FB1:  89.50

* --config=configs/config-bert.json --bert_model_name_or_path=./embeddings/MiniLM-L12-H384-uncased
INFO:__main__:[F1] : 0.900193627882415, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 67914.6056175232ms, 18.416685349844798ms on average
accuracy:  97.92%; precision:  89.50%; recall:  90.55%; FB1:  90.02

* --config=configs/config-bert.json --bert_model_name_or_path=./embeddings/MiniLM-L12-H384-uncased --warmup_epoch=0 --weight_decay=0.0 --epoch=30
INFO:__main__:[F1] : 0.9055271057743245, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 65600.49557685852ms, 17.789026203927158ms on average
accuracy:  98.05%; precision:  90.31%; recall:  90.79%; FB1:  90.55

* --bert_model_name_or_path=./embeddings/bert-large-cased-whole-word-masking-finetuned-squad
INFO:__main__:[F1] : 0.9130013221683562, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 130683ms, 35.44013032853652ms on average
accuracy:  98.29%; precision:  90.91%; recall:  91.70%; FB1:  91.30

* --bert_model_name_or_path=./embeddings/bert-large-cased-whole-word-masking-finetuned-squad  --warmup_epoch=0 --weight_decay=0.0 --epoch=30
INFO:__main__:[F1] : 0.9175393822054034, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 131497ms, 35.661960358403476ms on average
accuracy:  98.33%; precision:  91.22%; recall:  92.30%; FB1:  91.75

* --data_dir=data/conll2003_truecase --bert_model_name_or_path=./embeddings/bert-large-cased-whole-word-masking-finetuned-squad  --warmup_epoch=0 --weight_decay=0.0 --epoch=30
INFO:__main__:[F1] : 0.9217421785995968, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 121090.73448181152ms, 32.83806317143229ms on average
accuracy:  98.32%; precision:  91.25%; recall:  93.11%; FB1:  92.17

* for using SpanBERT embedding, just replace pretrained BERT model to SpanBERT.
* --bert_model_name_or_path=./embeddings/spanbert_hf_base
INFO:__main__:[F1] : 0.9046450482033305, 3684
INFO:__main__:[Elapsed Time] : 110977ms, 30.09910399131143ms on average
accuracy:  98.02%; precision:  89.57%; recall:  91.38%; FB1:  90.46

* --bert_model_name_or_path=./embeddings/spanbert_hf_large
INFO:__main__:[F1] : 0.9139340659340659, 3684
INFO:__main__:[Elapsed Time] : 157069ms, 42.59598153679066ms on average
accuracy:  98.23%; precision:  90.76%; recall:  92.03%; FB1:  91.39

* --data_dir=data/conll2003_truecase --bert_model_name_or_path=./embeddings/spanbert_hf_large  --warmup_epoch=0 --weight_decay=0.0 --epoch=30
INFO:__main__:[F1] : 0.9201266713581985, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 116231.20188713074ms, 31.509835921387747ms on average
accuracy:  98.35%; precision:  91.43%; recall:  92.60%; FB1:  92.01

* --bert_use_feature_based --epoch=64 --lr=1e-4 , modify model.py to use initial embedding
INFO:__main__:[F1] : 0.8610818405338954, 3684
INFO:__main__:[Elapsed Time] : 181867ms, 49.310344827586206ms on average
accuracy:  97.43%; precision:  85.42%; recall:  86.81%; FB1:  86.11

* --bert_use_feature_based --use_crf --epoch=64 --lr=3e-4 , modify model.py to use initial embedding
INFO:__main__:[F1] : 0.8575426322693486, 3684
INFO:__main__:[Elapsed Time] : 232758ms, 63.13765951669834ms on average
accuracy:  97.06%; precision:  86.06%; recall:  86.81%; FB1:  86.43

* --bert_use_feature_based --epoch=64 --lr=3e-4 , modify model.py to use initial+first+last embedding
INFO:__main__:[F1] : 0.8971681415929202, 3684
INFO:__main__:[Elapsed Time] : 176851ms, 47.970404561498775ms on average
accuracy:  98.00%; precision:  89.69%; recall:  89.75%; FB1:  89.72

* --bert_use_feature_based --use_crf --epoch=64 --lr=3e-4 , modify model.py to use initial+first+last embedding
INFO:__main__:[F1] : 0.8946078431372549, 3684
INFO:__main__:[Elapsed Time] : 247716ms, 67.20418137387999ms on average
accuracy:  97.81%; precision:  89.46%; recall:  90.47%; FB1:  89.96

* --bert_use_feature_based --use_crf --epoch=64 --lr=3e-4 , modify model.py to use last embedding
INFO:__main__:[F1] : 0.8912471496228732, 3684
INFO:__main__:[Elapsed Time] : 253449ms, 68.75481944067336ms on average
accuracy:  97.75%; precision:  89.38%; recall:  89.96%; FB1:  89.67

* --bert_use_feature_based --use_crf --epoch=64 --lr=3e-4 , modify model.py to use [-4:] embedding
INFO:__main__:[F1] : 0.90263319044703, 3684
INFO:__main__:[Elapsed Time] : 235691ms, 63.9397230518599ms on average
accuracy:  97.99%; precision:  89.94%; recall:  91.34%; FB1:  90.64

* --bert_use_feature_based --use_crf --epoch=64 --lr=3e-4 , modify model.py to use mean([0:3] + [-4:]) embedding
INFO:__main__:[F1] : 0.9026813186813187, 3684
INFO:__main__:[Elapsed Time] : 261161ms, 70.83220200923161ms on average
accuracy:  98.02%; precision:  90.15%; recall:  90.90%; FB1:  90.52

* --bert_use_feature_based --use_crf --epoch=64 --lr=3e-4 , modify model.py to use mean([0:17]) embedding
INFO:__main__:[F1] : 0.9050141242937852, 3684
INFO:__main__:[Elapsed Time] : 253013ms, 68.61390171056205ms on average
accuracy:  98.06%; precision:  90.87%; recall:  90.76%; FB1:  90.81

* --bert_use_feature_based --use_crf --epoch=64 --lr=3e-4 , modify model.py to use max([0:17]) embedding
INFO:__main__:[F1] : 0.9054972205064854, 3684
INFO:__main__:[Elapsed Time] : 224124ms, 60.80396415965246ms on average
accuracy:  98.09%; precision:  90.67%; recall:  90.85%; FB1:  90.76

* --bert_use_feature_based --use_crf --epoch=64 --lr=3e-4 , modify model.py to use mean([0:]) embedding
INFO:__main__:[F1] : 0.9071163694155041, 3684
INFO:__main__:[Elapsed Time] : 217202ms, 58.91121368449633ms on average
accuracy:  98.13%; precision:  91.00%; recall:  90.95%; FB1:  90.98

* --bert_use_feature_based --use_crf --epoch=64 --lr=3e-4 , modify model.py to use DSA(4, 300)
INFO:__main__:[F1] : 0.9036879808967896, 3684
INFO:__main__:[Elapsed Time] : 245729ms, 66.65761607385284ms on average
accuracy:  98.08%; precision:  90.78%; recall:  90.46%; FB1:  90.62

* --bert_use_feature_based --use_crf --epoch=64 --lr=1e-3 , modify model.py to use DSA(2, 1024)
INFO:__main__:[F1] : 0.8953335090957026, 3684
INFO:__main__:[Elapsed Time] : 219016ms, 59.40619060548466ms on average
accuracy:  97.94%; precision:  89.37%; recall:  90.19%; FB1:  89.78

* --bert_model_name_or_path=./embedings/bert-base-uncased  --bert_remove_layers=8,9,10,11
INFO:__main__:[F1] : 0.8902760682257781, 3684
INFO:__main__:[Elapsed Time] : 91854ms, 24.907683953298942ms on average
accuracy:  97.75%; precision:  88.42%; recall:  89.64%; FB1:  89.03

* --bert_remove_layers=12,13,14,15,16,17,18,19,20,21,22,23
INFO:__main__:[F1] : 0.8910284463894966, 3684
INFO:__main__:[Elapsed Time] : 122182ms, 33.13765951669834ms on average
accuracy:  97.80%; precision:  88.11%; recall:  90.12%; FB1:  89.10

```

</p>
</details>


<details><summary><b>emb_class=albert, enc_class=bilstm</b></summary>
<p>

- train
```
* share config-bert.json
* n_ctx size should be less than 512
$ python preprocess.py --config=configs/config-bert.json --data_dir=data/conll2003 --bert_model_name_or_path=./embeddings/albert-base-v2 
$ python train.py --config=configs/config-bert.json --data_dir=data/conll2003 --save_path=pytorch-model-albert.pt --bert_model_name_or_path=./embeddings/albert-base-v2 --bert_output_dir=bert-checkpoint-albert --batch_size=32 --lr=1e-5 --epoch=64 
```

- evaluation
```
$ python evaluate.py --config=configs/config-bert.json --data_dir=data/conll2003 --model_path=pytorch-model-albert.pt --bert_output_dir=bert-checkpoint-albert
$ cd data/conll2003; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.8821014050091632, 3684
INFO:__main__:[Elapsed Time] : 114627ms, 31.08688569101276ms on average
accuracy:  97.44%; precision:  86.93%; recall:  89.48%; FB1:  88.19

* --bert_model_name_or_path=./embeddings/albert-xxlarge-v2 --batch_size=16
INFO:__main__:[F1] : 0.9041648399824638, 3684
INFO:__main__:[Elapsed Time] : 397151ms, 107.77871300570187ms on average
accuracy:  98.06%; precision:  89.51%; recall:  91.29%; FB1:  90.39

```

</p>
</details>


<details><summary><b>emb_class=roberta, enc_class=bilstm</b></summary>
<p>

- train
```
* n_ctx size should be less than 512
$ python preprocess.py --config=configs/config-roberta.json --data_dir=data/conll2003 --bert_model_name_or_path=./embeddings/roberta-large
$ python train.py --config=configs/config-roberta.json --data_dir=data/conll2003 --save_path=pytorch-model-roberta.pt --bert_model_name_or_path=./embeddings/roberta-large --bert_output_dir=bert-checkpoint-roberta --batch_size=16 --lr=1e-5 --epoch=10
```

- evaluation
```
$ python evaluate.py --config=configs/config-roberta.json --data_dir=data/conll2003 --model_path=pytorch-model-roberta.pt --bert_output_dir=bert-checkpoint-roberta
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

* --bert_use_pos
INFO:__main__:[F1] : 0.914000175330937, 3684
INFO:__main__:[Elapsed Time] : 153930ms, 41.748574531631824ms on average

* --data_dir=data/conll2003_truecase --bert_disable_lstm  --warmup_epoch=0 --weight_decay=0.0 --epoch=30
INFO:__main__:[F1] : 0.9190283400809718, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 112529.90531921387ms, 30.516853094295158ms on average
accuracy:  98.30%; precision:  91.37%; recall:  92.44%; FB1:  91.90

* --bert_model_name_or_path=./embeddings/roberta-base --bert_disable_lstm
INFO:__main__:[F1] : 0.9002973587545915, 3684
INFO:__main__:[Elapsed Time] : 71015ms, 19.25033939723052ms on average

```

</p>
</details>


<details><summary><b>emb_class=bart, enc_class=bilstm</b></summary>
<p>

- train
```
* n_ctx size should be less than 512
$ python preprocess.py --config=configs/config-bart.json --data_dir=data/conll2003 --bert_model_name_or_path=./embeddings/bart-large
$ python train.py --config=configs/config-bart.json --data_dir=data/conll2003 --save_path=pytorch-model-bart.pt --bert_model_name_or_path=./embeddings/bart-large --bert_output_dir=bert-checkpoint-bart --batch_size=32 --lr=1e-5 --epoch=10
```

- evaluation
```
$ python evaluate.py --config=configs/config-bart.json --data_dir=data/conll2003 --model_path=pytorch-model-bart.pt --bert_output_dir=bert-checkpoint-bart
$ cd data/conll2003; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.9042961928047503, 3684
INFO:__main__:[Elapsed Time] : 196742ms, 53.36573445560684ms on average
accuracy:  98.04%; precision:  89.21%; recall:  91.68%; FB1:  90.43

```

</p>
</details>


<details><summary><b>emb_class=electra, enc_class=bilstm</b></summary>
<p>

- train
```
* share config-bert.json
* n_ctx size should be less than 512
$ python preprocess.py --config=configs/config-bert.json --data_dir=data/conll2003 --bert_model_name_or_path=./embeddings/electra-base-discriminator 
$ python train.py --config=configs/config-bert.json --data_dir=data/conll2003 --save_path=pytorch-model-electra.pt --bert_model_name_or_path=./embeddings/electra-base-discriminator --bert_output_dir=bert-checkpoint-electra --batch_size=32 --lr=1e-5 --epoch=20 
```

- evaluation
```
$ python evaluate.py --config=configs/config-bert.json --data_dir=data/conll2003 --model_path=pytorch-model-electra.pt --bert_output_dir=bert-checkpoint-electra
$ cd data/conll2003; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.9072818526019194, 3684
INFO:__main__:[Elapsed Time] : 106594ms, 28.80966603312517ms on average
accuracy:  98.08%; precision:  90.24%; recall:  91.22%; FB1:  90.73

* --batch_size=64 --gradient_accumulation_steps=2 --lr=8e-5
INFO:__main__:[F1] : 0.9097631833788185, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 82651ms, 22.413250067879446ms on average
accuracy:  97.98%; precision:  90.47%; recall:  91.48%; FB1:  90.98

* --bert_model_name_or_path=./embeddings/electra-large-discriminator --lr=1e-6 --bert_disable_lstm --epoch=40
INFO:__main__:[F1] : 0.91392938696645, 3684
INFO:__main__:[Elapsed Time] : 109367ms, 29.573445560684224ms on average

* --bert_model_name_or_path=./embeddings/electra-large-discriminator --bert_disable_lstm  --warmup_epoch=0 --weight_decay=0.0 --gradient_accumulation_steps=2 --lr=8e-5 --epoch=40
INFO:__main__:[F1] : 0.9042280872098155, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 116181ms, 31.50203638338311ms on average
accuracy:  98.04%; precision:  90.16%; recall:  90.69%; FB1:  90.42

```

</p>
</details>


<details><summary><b>emb_class=elmo, enc_class=bilstm</b></summary>
<p>

- train
```
* token_emb_dim in configs/config-elmo.json == 300 (ex, glove.6B.300d.txt )
* elmo_emb_dim  in configs/config-elmo.json == 1024 (ex, elmo_2x4096_512_2048cnn_2xhighway_5.5B_* )
$ python preprocess.py --config=configs/config-elmo.json --data_dir=data/conll2003 --embedding_path=embeddings/glove.6B.300d.txt
* --use_crf for adding crf layer, --embedding_trainable for fine-tuning pretrained word embedding
$ python train.py --config=configs/config-elmo.json --data_dir=data/conll2003 --save_path=pytorch-model-elmo.pt --use_crf
```

- evaluation
```
$ python evaluate.py --config=configs/config-elmo.json --data_dir=data/conll2003 --model_path=pytorch-model-elmo.pt --use_crf
$ cd data/conll2003; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.9219494967331803, 3684
INFO:__main__:[Elapsed Time] : 239919ms, 65.12459283387622ms on average
accuracy:  98.29%; precision:  91.95%; recall:  92.44%; FB1:  92.19

* --batch_size=64
INFO:__main__:[F1] : 0.926332565964229, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 275097ms, 74.65218571816455ms on average
accuracy:  98.45%; precision:  92.65%; recall:  92.62%; FB1:  92.63

* --data_dir=data/conll2003_truecase --batch_size=64  --warmup_epoch=0 --weight_decay=0.0
INFO:__main__:[F1] : 0.9251124636147129, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 236348.47974777222ms, 64.13688945796157ms on average
accuracy:  98.40%; precision:  92.18%; recall:  92.85%; FB1:  92.51

* --use_char_cnn
INFO:__main__:[F1] : 0.9137136782423814, 3684
INFO:__main__:[Elapsed Time] : 280253ms, 76.05566114580505ms on average
accuracy:  98.15%; precision:  91.44%; recall:  91.31%; FB1:  91.37

* --use_char_cnn 
INFO:__main__:[F1] : 0.9196578181497487, 3684
INFO:__main__:[Elapsed Time] : 291372ms, 79.06489275047515ms on average
accuracy:  98.26%; precision:  91.62%; recall:  92.32%; FB1:  91.97

* --use_char_cnn --batch_size=64
INFO:__main__:[F1] : 0.9202508169213106, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 222863ms, 60.46673907140918ms on average
accuracy:  98.32%; precision:  91.81%; recall:  92.25%; FB1:  92.03

* modify model.py for disabling glove
INFO:__main__:[F1] : 0.917751217352811, 3684
INFO:__main__:[Elapsed Time] : 273089ms, 74.10019006244909ms on average
accuracy:  98.27%; precision:  91.78%; recall:  91.77%; FB1:  91.78

* --use_char_cnn , modify model.py for disabling glove
INFO:__main__:[F1] : 0.9193262411347518, 3684
INFO:__main__:[Elapsed Time] : 249467ms, 67.69318490361118ms on average
accuracy:  98.31%; precision:  92.06%; recall:  91.80%; FB1:  91.93
```

</p>
</details>

<br>




# Kaggle NER  (English)

## experiments summary

- ntagger, measured by conlleval.pl (micro F1)

|                                 | F1 (%)       | Features             | GPU / CPU          | CONDA    | ONNX      | Dynamic   | Etc                       |
| ------------------------------- | ------------ | -------------------- | ------------------ | -------- | --------- | --------- | ------------------------- |
| GloVe, BiLSTM-CRF               | 85.67        | word, pos            | 23.7084 / -        |          |           |           |                           |
| GloVe, BiLSTM-CRF               | 85.78        | word, character, pos | 24.2101 / -        |          |           |           |                           |
| BERT-base(cased), BiLSTM        | 84.43        | word                 | 39.0914 / -        | -        | -         | -         |                           |
| BERT-base(cased), BiLSTM-CRF    | 84.62        | word, pos            | 37.7312 / -        | -        | -         | -         |                           |
| BERT-large-squad, BiLSTM-CRF    | 84.78        | word, pos            | 53.3669 / -        |          |           |           |                           |
| ELMo, GloVe, BiLSTM-CRF         | 85.48        | word, pos            | 80.5333 / -        | -        |           |           |                           |

<details><summary><b>emb_class=glove, enc_class=bilstm</b></summary>
<p>

- train
```
* token_emb_dim in configs/config-glove.json == 300 (ex, glove.6B.300d.txt )
$ python preprocess.py --data_dir=data/kaggle
* --use_crf for adding crf layer, --embedding_trainable for fine-tuning pretrained word embedding
$ python train.py --data_dir=data/kaggle --use_crf
```

- evaluation
```
$ python evaluate.py --data_dir=data/kaggle --use_crf
$ cd data/kaggle; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..
INFO:__main__:[F1] : 0.8567375886524823, 4795
INFO:__main__:[Elapsed Time] : 4795 examples, 113756.6728591919ms, 23.708498821091442ms on average
accuracy:  97.38%; precision:  85.79%; recall:  85.56%; FB1:  85.67

* --use_char_cnn 
INFO:__main__:[F1] : 0.8578012141622723, 4795
INFO:__main__:[Elapsed Time] : 4795 examples, 116158.74361991882ms, 24.210153393910534ms on average
accuracy:  97.38%; precision:  85.87%; recall:  85.69%; FB1:  85.78

```

</p>
</details>

<details><summary><b>emb_class=bert, enc_class=bilstm</b></summary>
<p>

- train
```
* n_ctx size should be less than 512
$ python preprocess.py --config=configs/config-bert.json --data_dir=data/kaggle --bert_model_name_or_path=./embeddings/bert-base-cased
$ python train.py --config=configs/config-bert.json --data_dir=data/kaggle --save_path=pytorch-model-bert.pt --bert_model_name_or_path=./embeddings/bert-base-cased --bert_output_dir=bert-checkpoint --batch_size=32 --lr=5e-5 --epoch=20 --warmup_epoch=0 --weight_decay=0.0 --patience=4
```

- evaluation
```
$ python evaluate.py --config=configs/config-bert.json --data_dir=data/kaggle --model_path=pytorch-model-bert.pt --bert_output_dir=bert-checkpoint
$ cd data/kaggle; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.8443369254453487, 4795
INFO:__main__:[Elapsed Time] : 4795 examples, 187503.9415359497ms, 39.09142493803799ms on average

* {'lr': 5.2929204434436896e-05, 'batch_size': 32, 'seed': 27, 'epochs': 15} by optuna
INFO:__main__:[F1] : 0.840452206044077, 4795
INFO:__main__:[Elapsed Time] : 4795 examples, 185833.47272872925ms, 38.74066833858944ms on average
accuracy:  97.17%; precision:  83.52%; recall:  84.58%; FB1:  84.05

* --bert_use_pos --use_crf
INFO:__main__:[F1] : 0.8459812590735118, 4795
INFO:__main__:[Elapsed Time] : 4795 examples, 181019.96636390686ms, 37.73124167259703ms on average
accuracy:  97.31%; precision:  84.11%; recall:  85.13%; FB1:  84.62

* --bert_model_name_or_path=./embeddings/bert-large-cased-whole-word-masking-finetuned-squad --bert_use_pos --use_crf
INFO:__main__:[F1] : 0.8475636619966518, 4795
INFO:__main__:[Elapsed Time] : 4795 examples, 256007.39908218384ms, 53.3669921921152ms on average
accuracy:  97.30%; precision:  84.40%; recall:  85.16%; FB1:  84.78

```

</p>
</details>

<details><summary><b>emb_class=elmo, enc_class=bilstm</b></summary>
<p>

- train
```
* token_emb_dim in configs/config-elmo.json == 300 (ex, glove.6B.300d.txt )
* elmo_emb_dim  in configs/config-elmo.json == 1024 (ex, elmo_2x4096_512_2048cnn_2xhighway_5.5B_* )
$ python preprocess.py --config=configs/config-elmo.json --data_dir=data/kaggle --embedding_path=embeddings/glove.6B.300d.txt
* --use_crf for adding crf layer, --embedding_trainable for fine-tuning pretrained word embedding
$ python train.py --config=configs/config-elmo.json --data_dir=data/kaggle --save_path=pytorch-model-elmo.pt --batch_size=64 --use_crf
```

- evaluation
```
$ python evaluate.py --config=configs/config-elmo.json --data_dir=data/kaggle --model_path=pytorch-model-elmo.pt --use_crf
$ cd data/kaggle; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..
INFO:__main__:[F1] : 0.8547622213358151, 4795
INFO:__main__:[Elapsed Time] : 4795 examples, 386233.2103252411ms, 80.53337483292677ms on average
accuracy:  97.35%; precision:  85.58%; recall:  85.37%; FB1:  85.48

```

</p>
</details>

<br>




# GUM (English)

## experiments summary

- ntagger, measured by conlleval.pl (micro F1)

|                                 | F1 (%)       | Features             | GPU / CPU          | CONDA    | ONNX      | Dynamic   | Etc                       |
| ------------------------------- | ------------ | -------------------- | ------------------ | -------- | --------- | --------- | ------------------------- |
| GloVe, BiLSTM-CRF               | 53.70        | word, character      | 24.3435 / -        |          |           |           |                           |
| BERT-base(cased), BiLSTM-CRF    | 63.13        | word                 | 40.3382 / -        | -        | -         | -         |                           |
| BERT-large-squad, BiLSTM-CRF    | 62.76        | word                 | 63.1012 / -        |          |           |           |                           |
| ELMo, GloVe, BiLSTM-CRF         | 61.46        | word                 | 79.0402 / -        | -        |           |           |                           |

<details><summary><b>emb_class=glove, enc_class=bilstm</b></summary>
<p>

- train
```
* token_emb_dim in configs/config-glove.json == 300 (ex, glove.6B.300d.txt )
$ python preprocess.py --data_dir=data/gum
* --use_crf for adding crf layer, --embedding_trainable for fine-tuning pretrained word embedding
$ python train.py --data_dir=data/gum --use_crf --use_char_cnn
```

- evaluation
```
$ python evaluate.py --data_dir=data/gum --use_crf --use_char_cnn
$ cd data/gum; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..
INFO:__main__:[F1] : 0.5370499419279907, 1000
INFO:__main__:[Elapsed Time] : 1000 examples, 24429.93712425232ms, 24.343522103341133ms on average
accuracy:  79.30%; precision:  55.47%; recall:  52.05%; FB1:  53.70

```

</p>
</details>

<details><summary><b>emb_class=bert, enc_class=bilstm</b></summary>
<p>

- train
```
* n_ctx size should be less than 512
$ python preprocess.py --config=configs/config-bert.json --data_dir=data/gum --bert_model_name_or_path=./embeddings/bert-base-cased
$ python train.py --config=configs/config-bert.json --data_dir=data/gum --save_path=pytorch-model-bert.pt --bert_model_name_or_path=./embeddings/bert-base-cased --bert_output_dir=bert-checkpoint --batch_size=32 --lr=5e-5 --epoch=20 --warmup_epoch=0 --weight_decay=0.0 --patience=4 --use_crf

```

- evaluation
```
$ python evaluate.py --config=configs/config-bert.json --data_dir=data/gum --model_path=pytorch-model-bert.pt --bert_output_dir=bert-checkpoint --use_crf
$ cd data/gum; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

* {'lr': 5.8243506260824845e-05, 'batch_size': 32, 'seed': 42, 'epochs': 18} by optuna
INFO:__main__:[F1] : 0.6304728546409808, 1000
INFO:__main__:[Elapsed Time] : 1000 examples, 40466.28999710083ms, 40.33822578949494ms on average
accuracy:  83.86%; precision:  61.51%; recall:  64.84%; FB1:  63.13

* --bert_model_name_or_path=./embeddings/bert-large-cased-whole-word-masking-finetuned-squad
INFO:__main__:[F1] : 0.6267972494269639, 1000
INFO:__main__:[Elapsed Time] : 1000 examples, 63216.368436813354ms, 63.10122626441139ms on average
accuracy:  82.62%; precision:  58.48%; recall:  67.72%; FB1:  62.76

```

</p>
</details>

<details><summary><b>emb_class=elmo, enc_class=bilstm</b></summary>
<p>

- train
```
* token_emb_dim in configs/config-elmo.json == 300 (ex, glove.6B.300d.txt )
* elmo_emb_dim  in configs/config-elmo.json == 1024 (ex, elmo_2x4096_512_2048cnn_2xhighway_5.5B_* )
$ python preprocess.py --config=configs/config-elmo.json --data_dir=data/gum --embedding_path=embeddings/glove.6B.300d.txt
* --use_crf for adding crf layer, --embedding_trainable for fine-tuning pretrained word embedding
$ python train.py --config=configs/config-elmo.json --data_dir=data/gum --save_path=pytorch-model-elmo.pt --batch_size=64 --use_crf
```

- evaluation
```
$ python evaluate.py --config=configs/config-elmo.json --data_dir=data/gum --model_path=pytorch-model-elmo.pt --use_crf
$ cd data/gum; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..
INFO:__main__:[F1] : 0.6145741878841089, 1000
INFO:__main__:[Elapsed Time] : 1000 examples, 79157.54914283752ms, 79.0402540812144ms on average
accuracy:  83.04%; precision:  59.96%; recall:  63.03%; FB1:  61.46

```

</p>
</details>

<br>




# Naver NER 2019 (Korean)

## experiments summary

#### clova2019(eoj-based)

- ntagger, measured by conlleval.pl (micro F1)

|                              | F1 (%)      | Features | GPU / CPU      | CONDA    | Dynamic   | Etc    |
| ---------------------------- | ------------| -------- | -------------- | -------- | --------- | ------ |    
| bpe DistilBERT(v1)           | 85.30       | eoj      | 9.0702  / -    |          |           |        |
| wp  DistilBERT(v1)           | 84.45       | eoj      | 8.9646  / -    |          |           |        |
| bpe BERT(v1), BiLSTM-CRF     | 86.11       | eoj      | 53.1818 / -    |          |           |        |
| bpe BERT(v1), BiLSTM         | 86.37       | eoj      | 21.3232 / -    |          |           |        |
| bpe BERT(v1), CRF            | 86.42       | eoj      | 35.2222 / -    |          |           |        |
| bpe BERT(v1)                 | 87.13       | eoj      | 16.2121 / -    |          |           |        |
| bpe BERT-large(v1)           | 85.99       | eoj      | 30.7513 / -    |          |           |        |
| bpe BERT-large(v3)           | 85.89       | eoj      | 27.4264 / -    |          |           |        |
| KcBERT-base, BiLSTM          | 84.76       | eoj      | 15.0553 / -    |          |           |        |
| KcBERT-base, CRF             | 83.32       | eoj      | 31.8019 / -    |          |           |        |
| KcBERT-base                  | 84.72       | eoj      | 13.3129 / -    |          |           |        |
| KcBERT-large                 | 86.34       | eoj      | 26.9639 / -    |          |           |        |
| KoELECTRA-Base-v1            | 86.64       | eoj      | 15.1616 / -    |          |           |        |
| KoELECTRA-Base-v3            | 87.31       | eoj      | 14.8115 / -    |          |           |        |
| LM-KOR-ELECTRA               | **87.39**   | eoj      | 17.1545 / -    |          |           |        |
| bpe ELECTRA-base(v1)         | 86.46       | eoj      | 18.0449 / -    |          |           |        |
| RoBERTa-base                 | 85.45       | eoj      | 15.6986 / -    |          |           |        |


- [HanBert-NER](https://github.com/monologg/HanBert-NER#results), [KoELECTRA](https://github.com/monologg/KoELECTRA), [LM-kor](https://github.com/kiyoungkim1/LM-kor) measured by seqeval (micro F1)

| (update) max_seq_len=50  | F1 (%)        | Features |
| ------------------------ | ------------- | -------- |
| BiLSTM-CRF               | 74.57         | eoj      |
| Bert-multilingual        | 84.20         | eoj      |
| DistilKoBERT             | 84.13         | eoj      |
| KoBERT                   | 87.92         | eoj      |
| HanBert                  | 87.70         | eoj      |
| KoELECTRA-Base           | 87.18         | eoj      |
| KoELECTRA-Base-v2        | 87.16         | eoj      |
| KoELECTRA-Base-v3        | **88.11**     | eoj      |
| LM-KOR-ELECTRA           | 87.14         | eoj      |

```
* note that F1 score from the 'seqeval' package for 'max_seq_len=50' might be similar with that for 'max_seq_len=180'. 
  however, the final evaluation using 'conlleval.pl' should be different.

  for example, with n_ctx=50. 
    the F1 score from 'seqeval' is 0.8602. 
    the F1 score from 'conlleval.pl is 0.8438.
  --------------------------------------------------------------------------
  '--bert_disable_lstm ,  n_ctx=50'
  INFO:__main__:[F1] : 0.8602524268436113, 9000
  INFO:__main__:[Elapsed Time] : 192653ms, 21.39648849872208ms on average
  accuracy:  93.22%; precision:  85.55%; recall:  83.25%; FB1:  84.38

  '--bert_disable_lstm , without --use_crf (bpe BERT), n_ctx=180'
  INFO:__main__:[F1] : 0.8677214324767633, 9000
  INFO:__main__:[Elapsed Time] : 868094ms, 96.45471719079897ms on average
  accuracy:  94.47%; precision:  87.02%; recall:  86.33%; FB1:  86.68
  --------------------------------------------------------------------------

  this is due to the test.txt data has longer sequences than the 'n_ctx' size.
  therefore, the evaluation results using 'seqeval' are not the final F1 score.
  we recommend to use 'conlleval.pl' script for NER results.
```

#### clova2019_morph(morph-based)

- ntagger, measured by conlleval.pl (micro F1)

|                                    | m-by-m F1 (%) | e-by-e F1 (%)  | Features              | GPU / CPU   | Etc           |
| ---------------------------------- | ------------- | -------------- | --------------------- | ----------- | ------------- |  
| GloVe, BiLSTM-CRF                  | 84.29         | 84.29          | morph, pos            | 30.0968 / - |               |
| **GloVe, BiLSTM-CRF**              | 85.82         | 85.82          | morph, character, pos | 25.9623 / - |               |
| GloVe, DenseNet-CRF                | 83.44         | 83.49          | morph, pos            | 25.8059 / - |               |
| GloVe, DenseNet-CRF                | 83.96         | 83.98          | morph, character, pos | 28.4051 / - |               |
| dha DistilBERT(v1), CRF            | 79.88         | 82.27          | morph, pos            | 40.2669 / - |               |
| dha DistilBERT(v1), LSTM           | 82.79         | 83.71          | morph, pos            | 19.8174 / - |               |
| dha BERT(v1), BiLSTM-CRF           | 84.95         | 85.25          | morph, pos            | 42.1063 / - |               |
| dha BERT(v1), BiLSTM               | 84.51         | 85.55          | morph, pos            | 18.9292 / - |               |
| dha BERT(v1), CRF                  | 82.94         | 84.99          | morph, pos            | 46.2323 / - |               |
| dha BERT(v1)                       | 81.15         | 84.26          | morph, pos            | 15.1717 / - |               |
| dha BERT(v1), BiLSTM-CRF           | 83.55         | 83.85          | morph, pos            | 46.0254 / - | del 8,9,10,11 |
| dha BERT(v2), BiLSTM-CRF           | 83.29         | 83.57          | morph, pos            | 44.4813 / - |               |
| dha-bpe BERT(v1), BiLSTM-CRF       | 82.83         | 83.83          | morph, pos            | 42.4347 / - |               |
| dha-bpe BERT(v3), BiLSTM-CRF       | 85.14         | 85.94          | morph, pos            | 40.1359 / - |               |
| dha-bpe BERT(v3), BiLSTM-CRF       | 85.09         | 85.90          | morph, pos            | 39.4648 / - | kor-bert-base-dha_bpe.v3.KMOU (fine-tuned) |
| dha-bpe BERT-large(v1), BiLSTM-CRF | 82.86         | 84.91          | morph, pos            | 53.6760 / - |               |
| dha-bpe BERT-large(v3), BiLSTM-CRF | 85.87         | 86.33          | morph, pos            | 50.7508 / - |               |
| ELMo, BiLSTM-CRF                   | 85.64         | 85.66          | morph, pos            | 95.9868 / - |               |
| ELMo, BiLSTM-CRF                   | 85.81         | 85.82          | morph, character, pos | 95.6196 / - |               |
| ELMo, GloVe, BiLSTM-CRF            | 86.37         | 86.37          | morph, pos            | 82.7731 / - |               |
| ELMo, GloVe, BiLSTM-CRF            | 86.62         | **86.63**      | morph, character, pos | 105.739 / - |               |

- [etagger](https://github.com/dsindex/etagger), measured by conlleval (micro F1)

|                              | m-by-m F1 (%) | e-by-e F1 (%)  | Features              | Etc        |
| ---------------------------- | ------------- | -------------- | ----------------------| ---------- |
| GloVe, BiLSTM-CRF            | 85.51         | 85.51          | morph, character, pos |            |
| dha BERT(v1), BiLSTM-C  RF   | 81.25         | 81.39          | morph, pos            | BERT as feature-based |
| ELMo, GloVe, BiLSTM-CRF      | 86.75         | **86.75**          | morph, character, pos |            |

#### clova2019_morph_space(morph-based + space as token)

- ntagger, measured by conlleval.pl (micro F1)

|                                | m-by-m F1 (%) | e-by-e F1 (%)  | Features              | GPU / CPU   | Etc           |
| ------------------------------ | ------------- | -------------- | --------------------- | ----------- | ------------- |  
| GloVe, BiLSTM-CRF              | 85.59         | 85.72          | morph, character, pos | 29.0723 / - |               |
| dha BERT(v1), BiLSTM-CRF       | 85.17         | 85.61          | morph, pos            | 43.7969 / - |               |
| ELMo, GloVe, BiLSTM-CRF        | 85.95         | **86.06**      | morph, character, pos | 113.177 / - |               |


<details><summary><b>emb_class=glove, enc_class=bilstm</b></summary>
<p>

- train
```
* for clova2019_morph

** token_emb_dim in configs/config-glove.json == 300 (ex, kor.glove.300k.300d.txt )
$ python preprocess.py --data_dir data/clova2019_morph --embedding_path embeddings/kor.glove.300k.300d.txt
** --use_crf for adding crf layer, --embedding_trainable for fine-tuning pretrained word embedding.
$ python train.py --save_path=pytorch-model-glove-kor-clova-morph.pt --data_dir data/clova2019_morph --use_crf --embedding_trainable

* for clova2019_morph_space

$ python preprocess.py --data_dir data/clova2019_morph_space --embedding_path embeddings/kor.glove.300k.300d.txt
$ python train.py --save_path=pytorch-model-glove-kor-clova-morph-space.pt --data_dir data/clova2019_morph_space --use_crf --embedding_trainable --use_char_cnn

```

- evaluation
```
* for clova2019_morph

$ python evaluate.py --model_path=pytorch-model-glove-kor-clova-morph.pt --data_dir data/clova2019_morph --use_crf
** seqeval.metrics supports IOB2(BIO) format, so FB1 from conlleval.pl should be similar value with.
$ cd data/clova2019_morph; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.8434531044045398, 9000
INFO:__main__:[Elapsed Time] : 270872ms, 30.096888888888888ms on average
accuracy:  93.80%; precision:  84.82%; recall:  83.76%; FB1:  84.29
  *** evaluation eoj-by-eoj
  $ cd data/clova2019_morph ; python to-eoj.py < test.txt.pred > test.txt.pred.eoj ; perl ../../etc/conlleval.pl < test.txt.pred.eoj ; cd ../..
  accuracy:  93.37%; precision:  84.83%; recall:  83.76%; FB1:  84.29

** --use_char_cnn
INFO:__main__:[F1] : 0.856091088091773, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 225223ms, 25.014668296477385ms on average
accuracy:  94.23%; precision:  86.11%; recall:  84.99%; FB1:  85.55
  *** evaluation eoj-by-eoj
  accuracy:  93.88%; precision:  86.11%; recall:  85.00%; FB1:  85.55
  
**  --warmup_epoch=0 --weight_decay=0.0 --use_char_cnn
INFO:__main__:[F1] : 0.8588128417407997, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 233760ms, 25.962329147683075ms on average
accuracy:  94.36%; precision:  86.48%; recall:  85.17%; FB1:  85.82
  *** evaluation eoj-by-eoj
  accuracy:  94.01%; precision:  86.48%; recall:  85.17%; FB1:  85.82

* for clova2019_morph_space

$ python evaluate.py --model_path=pytorch-model-glove-kor-clova-morph-space.pt --data_dir data/clova2019_morph_space --use_crf --use_char_cnn
$ cd data/clova2019_morph_space; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.8573393391328268, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 261744ms, 29.072341371263473ms on average
accuracy:  95.38%; precision:  86.43%; recall:  84.77%; FB1:  85.59
  *** evaluation eoj-by-eoj
  $ cd data/clova2019_morph_space ; python to-eoj.py < test.txt.pred > test.txt.pred.eoj ; perl ../../etc/conlleval.pl < test.txt.pred.eoj ; cd ../..
  accuracy:  93.89%; precision:  86.56%; recall:  84.89%; FB1:  85.72

```

</p>
</details>


<details><summary><b>emb_class=glove, enc_class=densenet</b></summary>
<p>

- train
```
* token_emb_dim in configs/config-glove.json == 300 (ex, kor.glove.300k.300d.txt )
$ python preprocess.py --config=configs/config-densenet.json --data_dir data/clova2019_morph --embedding_path embeddings/kor.glove.300k.300d.txt
* --use_crf for adding crf layer, --embedding_trainable for fine-tuning pretrained word embedding.
$ python train.py --config=configs/config-densenet.json --save_path=pytorch-model-densenet-kor-clova-morph.pt --data_dir data/clova2019_morph --use_crf --embedding_trainable
```

- evaluation
```
$ python evaluate.py --config=configs/config-densenet.json --model_path=pytorch-model-densenet-kor-clova-morph.pt --data_dir data/clova2019_morph --use_crf
* seqeval.metrics supports IOB2(BIO) format, so FB1 from conlleval.pl should be similar value with.
$ cd data/clova2019_morph; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.8350127432612621, 9000
INFO:__main__:[Elapsed Time] : 232331ms, 25.805978442049117ms on average
accuracy:  93.42%; precision:  82.80%; recall:  84.10%; FB1:  83.44
  ** evaluation eoj-by-eoj
  $ cd data/clova2019_morph ; python to-eoj.py < test.txt.pred > test.txt.pred.eoj ; perl ../../etc/conlleval.pl < test.txt.pred.eoj ; cd ../..
  accuracy:  92.96%; precision:  82.86%; recall:  84.13%; FB1:  83.49

* --user_char_cnn 
INFO:__main__:[F1] : 0.8402205267380136, 9000
INFO:__main__:[Elapsed Time] : 255785ms, 28.405156128458717ms on average
accuracy:  93.66%; precision:  84.25%; recall:  83.68%; FB1:  83.96
  ** evaluation eoj-by-eoj
  accuracy:  93.24%; precision:  84.28%; recall:  83.69%; FB1:  83.98
```

</p>
</details>


<details><summary><b>emb_class=bert, enc_class=bilstm, bpe BERT, bpe BERT-large, KcBERT-base, KcBERT-large</b></summary>
<p>

- train
```
* n_ctx size should be less than 512

* for clova2019

$ python preprocess.py --config=configs/config-bert.json --data_dir data/clova2019 --bert_model_name_or_path=./embeddings/kor-bert-base-bpe.v1
$ python train.py --config=configs/config-bert.json --save_path=pytorch-model-bert-kor-eoj.pt --bert_model_name_or_path=./embeddings/kor-bert-base-bpe.v1 --bert_output_dir=bert-checkpoint-kor-eoj --batch_size=32 --lr=5e-5 --epoch=20 --data_dir data/clova2019 --use_crf

```

- evaluation
```

* for clova2019

$ python evaluate.py --config=configs/config-bert.json --model_path=pytorch-model-bert-kor-eoj.pt --data_dir data/clova2019 --bert_output_dir=bert-checkpoint-kor-eoj --use_crf
$ cd data/clova2019; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

(bpe BERT BiLSTM-CRF)
INFO:__main__:[F1] : 0.8613367390378885, 9000
INFO:__main__:[Elapsed Time] : 476814ms, 52.96955217246361ms on average
accuracy:  94.15%; precision:  86.24%; recall:  85.99%; FB1:  86.11

** without --use_crf (bpe BERT BiLSTM)
INFO:__main__:[F1] : 0.8646059046587216, 9000
INFO:__main__:[Elapsed Time] : 100 examples, 2224ms, 21.32323232323232ms on average
accuracy:  94.31%; precision:  85.92%; recall:  86.82%; FB1:  86.37 

** --bert_disable_lstm (bpe BERT CRF)
INFO:__main__:[F1] : 0.8643569376373161, 9000
INFO:__main__:[Elapsed Time] : 342154ms, 38.00722302478053ms on average
accuracy:  94.35%; precision:  85.90%; recall:  86.94%; FB1:  86.42

** --bert_disable_lstm , without --use_crf (bpe BERT)
INFO:__main__:[F1] : 0.8677214324767633, 9000
INFO:__main__:[Elapsed Time] : 868094ms, 96.45471719079897ms on average
accuracy:  94.47%; precision:  87.02%; recall:  86.33%; FB1:  86.68

** --bert_disable_lstm ,  n_ctx=50
INFO:__main__:[F1] : 0.8602524268436113, 9000
INFO:__main__:[Elapsed Time] : 192653ms, 21.39648849872208ms on average
accuracy:  93.22%; precision:  85.55%; recall:  83.25%; FB1:  84.38

** --bert_disable_lstm  --warmup_epoch=0 --weight_decay=0.0 --lr=8e-5 --epoch=30 , without --use_crf (bpe BERT)
INFO:__main__:[F1] : 0.863227606609181, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 889043ms, 98.78186465162796ms on average
accuracy:  94.20%; precision:  85.18%; recall:  87.30%; FB1:  86.23

** --bert_disable_lstm  --warmup_epoch=0 --weight_decay=0.0 --lr=8e-5 --gradient_accumulation_steps=2 --epoch=30 , without --use_crf (bpe BERT)
INFO:__main__:[F1] : 0.8722265771446098, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 952261ms, 105.80508945438382ms on average
INFO:__main__:[Elapsed Time] : 100 examples, 1714ms, 16.21212121212121ms on average
accuracy:  94.63%; precision:  87.25%; recall:  87.01%; FB1:  87.13

** --bert_model_name_or_path=./embeddings/kor-bert-large-bpe.v1 --bert_disable_lstm  --warmup_epoch=0 --weight_decay=0.0 --lr=8e-5 --gradient_accumulation_steps=2 --epoch=30 , without --use_crf (bpe BERT-large) 
INFO:__main__:[F1] : 0.8608467232968307, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 1040116.376876831ms, 115.56598331838438ms on average
INFO:__main__:[Elapsed Time] : 100 examples, 3212.3892307281494ms, 30.75131984672161ms on average
accuracy:  94.13%; precision:  86.19%; recall:  85.79%; FB1:  85.99

** --bert_model_name_or_path=./embeddings/kor-bert-large-bpe.v3 --bert_disable_lstm  --warmup_epoch=0 --weight_decay=0.0 --lr=8e-5 --gradient_accumulation_steps=2 --epoch=20 --patience=4 , without --use_crf (bpe BERT-large) 
INFO:__main__:[F1] : 0.8601462833815275, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 1008693.4926509857ms, 112.0714406355684ms on average
INFO:__main__:[Elapsed Time] : 100 examples, 2885.901689529419ms, 27.426498104827573ms on average
accuracy:  94.15%; precision:  85.83%; recall:  85.96%; FB1:  85.89

** --config=configs/config-distilbert.json --bert_model_name_or_path=./embeddings/kor-distil-bpe-bert.v1 --bert_disable_lstm  --warmup_epoch=0 --weight_decay=0.0 --lr=8e-5 --gradient_accumulation_steps=2 --epoch=30 , without --use_crf
INFO:__main__:[F1] : 0.852160598843144, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 820104.8340797424ms, 91.12164719606297ms on average
INFO:__main__:[Elapsed Time] : 100 examples, 1019.8376178741455ms, 9.317352314188023ms on average
accuracy:  93.87%; precision:  85.10%; recall:  85.15%; FB1:  85.12

INFO:__main__:[F1] : 0.85393906493859, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 816257.2820186615ms, 90.68703071211878ms on average
INFO:__main__:[Elapsed Time] : 100 examples, 988.6960983276367ms, 9.070220619741113ms on average
accuracy:  93.97%; precision:  85.03%; recall:  85.57%; FB1:  85.30

** --config=configs/config-distilbert.json --bert_model_name_or_path=./embeddings/kor-distil-wp-bert.v1 --bert_disable_lstm  --warmup_epoch=0 --weight_decay=0.0 --lr=8e-5 --gradient_accumulation_steps=2 --epoch=30 , without --use_crf
INFO:__main__:[F1] : 0.8458054118245821, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 783889.6675109863ms, 87.09866535082064ms on average
INFO:__main__:[Elapsed Time] : 100 examples, 975.9259223937988ms, 8.9646927034012ms on average
accuracy:  93.55%; precision:  84.16%; recall:  84.75%; FB1:  84.45

** --bert_model_name_or_path=./embeddings/kcbert-base  --warmup_epoch=0 --weight_decay=0.0 --lr=8e-5 --gradient_accumulation_steps=2 , --without --use_crf (KcBERT-base, BiLSTM) 
INFO:__main__:[F1] : 0.8491746129396084, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 496907.4342250824ms, 55.207844985460014ms on average
INFO:__main__:[Elapsed Time] : 100 examples, 1583.3139419555664ms, 15.055304825908006ms on average
accuracy:  93.68%; precision:  85.05%; recall:  84.47%; FB1:  84.76

** --bert_model_name_or_path=./embeddings/kcbert-base --bert_disable_lstm  --warmup_epoch=0 --weight_decay=0.0 --lr=8e-5 --gradient_accumulation_steps=2 , (KcBERT-base, CRF) 
INFO:__main__:[F1] : 0.8333361605401095, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 305795.4549789429ms, 33.96846946311906ms on average
INFO:__main__:[Elapsed Time] : 100 examples, 3287.224531173706ms, 31.801955868499448ms on average
accuracy:  93.10%; precision:  83.13%; recall:  83.51%; FB1:  83.32

** --bert_model_name_or_path=./embeddings/kcbert-base --bert_disable_lstm  --warmup_epoch=0 --weight_decay=0.0 --lr=8e-5 --gradient_accumulation_steps=2 , --without --use_crf (KcBERT-base) 
INFO:__main__:[F1] : 0.848736197156657, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 463540.66610336304ms, 51.50017095600344ms on average
INFO:__main__:[Elapsed Time] : 100 examples, 1403.444528579712ms, 13.31293462502836ms on average
accuracy:  93.75%; precision:  84.90%; recall:  84.54%; FB1:  84.72

** --bert_model_name_or_path=./embeddings/kcbert-large --bert_disable_lstm  --warmup_epoch=0 --weight_decay=0.0 --lr=8e-5 --gradient_accumulation_steps=2 , --without --use_crf (KcBERT-large) 
INFO:__main__:[F1] : 0.8650133979621444, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 986830.237865448ms, 109.64403990732721ms on average
INFO:__main__:[Elapsed Time] : 100 examples, 2811.2776279449463ms, 26.963908262927124ms on average
accuracy:  94.31%; precision:  86.53%; recall:  86.16%; FB1:  86.34

```

</p>
</details>


<details><summary><b>emb_class=bert, enc_class=bilstm, dha-bpe BERT, dha-bpe BERT-large, dha BERT</b></summary>
<p>

- train
```
* n_ctx size should be less than 512

* for clova2019_morph

** dha-bpe
$ python preprocess.py --config=configs/config-bert.json --data_dir data/clova2019_morph --bert_model_name_or_path=./embeddings/kor-bert-base-dha_bpe.v1
$ python train.py --config=configs/config-bert.json --save_path=pytorch-model-bert-kor-clova-morph.pt --bert_model_name_or_path=./embeddings/kor-bert-base-dha_bpe.v1 --bert_output_dir=bert-checkpoint-kor-clova-morph --batch_size=32 --lr=5e-5 --epoch=20 --data_dir data/clova2019_morph --use_crf --bert_use_pos

** dha
$ python preprocess.py --config=configs/config-bert.json --data_dir data/clova2019_morph --bert_model_name_or_path=./embeddings/kor-bert-base-dha.v2
$ python train.py --config=configs/config-bert.json --save_path=pytorch-model-bert-kor-clova-morph.pt --bert_model_name_or_path=./embeddings/kor-bert-base-dha.v2 --bert_output_dir=bert-checkpoint-kor-clova-morph --batch_size=32 --lr=5e-5 --epoch=20 --data_dir data/clova2019_morph --use_crf --bert_use_pos

$ python preprocess.py --config=configs/config-bert.json --data_dir data/clova2019_morph --bert_model_name_or_path=./embeddings/kor-bert-base-dha.v1
$ python train.py --config=configs/config-bert.json --save_path=pytorch-model-bert-kor-clova-morph.pt --bert_model_name_or_path=./embeddings/kor-bert-base-dha.v1 --bert_output_dir=bert-checkpoint-kor-clova-morph --batch_size=32 --lr=5e-5 --epoch=20 --data_dir data/clova2019_morph --use_crf --bert_use_pos

* for clova2019_morph_space

$ python preprocess.py --config=configs/config-bert.json --data_dir data/clova2019_morph_space --bert_model_name_or_path=./embeddings/kor-bert-base-dha.v1
$ python train.py --config=configs/config-bert.json --save_path=pytorch-model-bert-kor-clova-morph-space.pt --bert_model_name_or_path=./embeddings/kor-bert-base-dha.v1 --bert_output_dir=bert-checkpoint-kor-clova-morph-space --batch_size=32 --lr=5e-5 --epoch=20 --data_dir data/clova2019_morph_space --use_crf --bert_use_pos

```

- evaluation
```
* for clova2019_morph

** dha-bpe
$ python evaluate.py --config=configs/config-bert.json --model_path=pytorch-model-bert-kor-clova-morph.pt --data_dir=data/clova2019_morph --bert_output_dir=bert-checkpoint-kor-clova-morph --use_crf --bert_use_pos
$ cd data/clova2019_morph; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.8295019157088124, 9000
INFO:__main__:[Elapsed Time] : 382042ms, 42.434714968329814ms on average
accuracy:  93.77%; precision:  81.78%; recall:  83.91%; FB1:  82.83
  **** evaluation eoj-by-eoj
  $ cd data/clova2019_morph ; python to-eoj.py < test.txt.pred > test.txt.pred.eoj ; perl ../../etc/conlleval.pl < test.txt.pred.eoj ; cd ../..
  accuracy:  93.37%; precision:  83.34%; recall:  84.33%; FB1:  83.83

*** --bert_model_name_or_path=./embeddings/kor-bert-base-dha_bpe.v3 --batch_size=64 --warmup_epoch=0 --weight_decay=0.0 --patience=4
INFO:__main__:[F1] : 0.852370233723154, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 361344.5198535919ms, 40.135946203859824ms on average
accuracy:  94.60%; precision:  84.67%; recall:  85.61%; FB1:  85.14
  **** evaluation eoj-by-eoj
  accuracy:  94.22%; precision:  85.91%; recall:  85.98%; FB1:  85.94

*** --bert_model_name_or_path=./bert-checkpoint-kor-kmou-morph --batch_size=64 --warmup_epoch=0 --weight_decay=0.0 --patience=4 , fine-tuned kor-bert-base-dha_bpe.v3 on KMOU data
INFO:__main__:[F1] : 0.8519647696476965, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 355305.17292022705ms, 39.464807812406406ms on average
accuracy:  94.58%; precision:  84.69%; recall:  85.50%; FB1:  85.09
  **** evaluation eoj-by-eoj
  accuracy:  94.21%; precision:  85.95%; recall:  85.86%; FB1:  85.90

*** --bert_model_name_or_path=./embeddings/kor-bert-large-dha_bpe.v1  --warmup_epoch=0 --weight_decay=0.0 --lr=1e-5
INFO:__main__:[F1] : 0.8298886586824331, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 483195.63341140747ms, 53.676061569842936ms on average
accuracy:  94.17%; precision:  81.99%; recall:  83.75%; FB1:  82.86
  **** evaluation eoj-by-eoj
  accuracy:  93.85%; precision:  84.97%; recall:  84.85%; FB1:  84.91

*** --bert_model_name_or_path=./embeddings/kor-bert-large-dha_bpe.v3  --warmup_epoch=0 --weight_decay=0.0 --lr=5e-5 --patience=4
INFO:__main__:[F1] : 0.8597318852876293, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 456936.72800064087ms, 50.750845153406736ms on average
accuracy:  94.66%; precision:  85.63%; recall:  86.11%; FB1:  85.87
  **** evaluation eoj-by-eoj
  accuracy:  94.30%; precision:  86.37%; recall:  86.30%; FB1:  86.33

** dha
$ python evaluate.py --config=configs/config-bert.json --model_path=pytorch-model-bert-kor-clova-morph.pt --data_dir=data/clova2019_morph --bert_output_dir=bert-checkpoint-kor-clova-morph --use_crf --bert_use_pos
$ cd data/clova2019_morph; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.8336304521299173, 9000
INFO:__main__:[Elapsed Time] : 400446ms, 44.48138682075786ms on average
accuracy:  93.58%; precision:  83.12%; recall:  83.46%; FB1:  83.29
  **** evaluation eoj-by-eoj
  $ cd data/clova2019_morph ; python to-eoj.py < test.txt.pred > test.txt.pred.eoj ; perl ../../etc/conlleval.pl < test.txt.pred.eoj ; cd ../..
  accuracy:  93.06%; precision:  83.55%; recall:  83.59%; FB1:  83.57

$ python evaluate.py --config=configs/config-bert.json --model_path=pytorch-model-bert-kor-clova-morph.pt --data_dir=data/clova2019_morph --bert_output_dir=bert-checkpoint-kor-clova-morph --use_crf --bert_use_pos
$ cd data/clova2019_morph; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

(dha BERT BiLSTM-CRF)
INFO:__main__:[F1] : 0.850251256281407, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 379036ms, 42.10634514946105ms on average
accuracy:  94.30%; precision:  85.06%; recall:  84.84%; FB1:  84.95
  *** evaluation eoj-by-eoj
  $ cd data/clova2019_morph ; python to-eoj.py < test.txt.pred > test.txt.pred.eoj ; perl ../../etc/conlleval.pl < test.txt.pred.eoj ; cd ../..
  accuracy:  93.86%; precision:  85.50%; recall:  84.99%; FB1:  85.25

** without --use_crf (dha BERT BiLSTM)
INFO:__main__:[F1] : 0.8453688900983027, 9000
INFO:__main__:[Elapsed Time] : 100 examples, 2197ms, 21.11111111111111ms on average
accuracy:  94.39%; precision:  84.43%; recall:  84.49%; FB1:  84.46
  *** evaluation eoj-by-eoj
  accuracy:  93.94%; precision:  85.90%; recall:  84.93%; FB1:  85.41 

** --bert_disable_lstm (dha BERT CRF)
INFO:__main__:[F1] : 0.8300803673938002, 9000
INFO:__main__:[Elapsed Time] : 100 examples, 4709ms, 46.23232323232323ms on average
accuracy:  94.18%; precision:  82.33%; recall:  83.57%; FB1:  82.94
  *** evaluation eoj-by-eoj
  accuracy:  93.84%; precision:  85.15%; recall:  84.83%; FB1:  84.99

** --bert_disable_lstm , without --use_crf (dha BERT)
INFO:__main__:[F1] : 0.8122244286627849, 9000
INFO:__main__:[Elapsed Time] : 100 examples, 1604ms, 15.171717171717171ms on average
accuracy:  93.85%; precision:  80.28%; recall:  82.04%; FB1:  81.15
  *** evaluation eoj-by-eoj
  accuracy:  93.48%; precision:  84.92%; recall:  83.61%; FB1:  84.26

** bert_outputs[2][-7] 
INFO:__main__:[F1] : 0.8296454550078846, 9000
INFO:__main__:[Elapsed Time] : 376186ms, 41.786642960328926ms on average
accuracy:  93.73%; precision:  82.62%; recall:  83.17%; FB1:  82.90
  *** evaluation eoj-by-eoj
  accuracy:  93.19%; precision:  83.25%; recall:  83.34%; FB1:  83.29

** --bert_remove_layers=8,9,10,11
  INFO:__main__:[F1] : 0.8361804271488914, 9000
  INFO:__main__:[Elapsed Time] : 414339ms, 46.025447271919106ms on average
  accuracy:  93.83%; precision:  83.31%; recall:  83.79%; FB1:  83.55
  *** evaluation eoj-by-eoj
  accuracy:  93.31%; precision:  83.78%; recall:  83.93%; FB1:  83.85 

**  --warmup_epoch=0 --weight_decay=0.0 --lr=5e-5 --gradient_accumulation_steps=2 --epoch=30 , without --use_crf (dha BERT BiLSTM)
INFO:__main__:[F1] : 0.8459056275447281, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 902977ms, 100.3300366707412ms on average
INFO:__main__:[Elapsed Time] : 100 examples, 1982ms, 18.92929292929293ms on average
accuracy:  94.48%; precision:  83.93%; recall:  85.11%; FB1:  84.51
  *** evaluation eoj-by-eoj
  accuracy:  94.03%; precision:  85.55%; recall:  85.55%; FB1:  85.55

**  --warmup_epoch=0 --weight_decay=0.0 --lr=2e-5 --epoch=30 --bert_disable_lstm , without --use_crf (dha BERT)
INFO:__main__:[F1] : 0.8135219179456316, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 887293ms, 98.58706522946994ms on average
accuracy:  94.03%; precision:  79.90%; recall:  82.72%; FB1:  81.28
  *** evaluation eoj-by-eoj
  accuracy:  93.69%; precision:  84.50%; recall:  84.63%; FB1:  84.56

** --config=configs/config-distilbert.json --bert_model_name_or_path=./embeddings/kor-distil-dha-bert.v1  --warmup_epoch=0 --weight_decay=0.0 --epoch=30 --bert_disable_lstm --lr=8e-5
(1) epoch_0
INFO:__main__:[F1] : 0.7994317797470066, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 331695.18399238586ms, 36.84263667049296ms on average
INFO:__main__:[Elapsed Time] : 100 examples, 4152.508020401001ms, 40.266978620278714ms on average
accuracy:  93.10%; precision:  79.41%; recall:  80.35%; FB1:  79.88
  *** evaluation eoj-by-eoj
  accuracy:  92.69%; precision:  82.73%; recall:  81.82%; FB1:  82.27

** --config=configs/config-distilbert.json --bert_model_name_or_path=./embeddings/kor-distil-dha-bert.v1  --warmup_epoch=0 --weight_decay=0.0 --epoch=30 --lr=8e-5  , without --use_crf
(1) epoch_0
INFO:__main__:[F1] : 0.8286240267526895, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 915342.8752422333ms, 101.70127953644766ms on average
INFO:__main__:[Elapsed Time] : 100 examples, 2091.184377670288ms, 19.81741250163377ms on average
accuracy:  93.82%; precision:  82.19%; recall:  83.40%; FB1:  82.79
  *** evaluation eoj-by-eoj
  accuracy:  93.26%; precision:  83.69%; recall:  83.73%; FB1:  83.71

* for clova2019_morph_space

$ python evaluate.py --config=configs/config-bert.json --model_path=pytorch-model-bert-kor-clova-morph-space.pt --data_dir=data/clova2019_morph_space --bert_output_dir=bert-checkpoint-kor-clova-morph-space --use_crf --bert_use_pos
$ cd data/clova2019_morph_space; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.853110511030723, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 394271ms, 43.79697744193799ms on average
accuracy:  95.51%; precision:  84.96%; recall:  85.38%; FB1:  85.17
  *** evaluation eoj-by-eoj
  $ cd data/clova2019_morph_space ; python to-eoj.py < test.txt.pred > test.txt.pred.eoj ; perl ../../etc/conlleval.pl < test.txt.pred.eoj ; cd ../..
  accuracy:  93.93%; precision:  85.52%; recall:  85.70%; FB1:  85.61

```

</p>
</details>


<details><summary><b>emb_class=elmo, enc_class=bilstm</b></summary>
<p>

- train
```
* for clova2019_morph

** token_emb_dim in configs/config-elmo.json == 300 (ex, kor.glove.300k.300d.txt )
** elmo_emb_dim  in configs/config-elmo.json == 1024 (ex, kor_elmo_2x4096_512_2048cnn_2xhighway_1000k* )
$ python preprocess.py --config=configs/config-elmo.json --data_dir=data/clova2019_morph --embedding_path=embeddings/kor.glove.300k.300d.txt
$ python train.py --config=configs/config-elmo.json --save_path=pytorch-model-elmo-kor-clova-morph.pt --data_dir=data/clova2019_morph --elmo_options_file=embeddings/kor_elmo_2x4096_512_2048cnn_2xhighway_1000k_options.json --elmo_weights_file=embeddings/kor_elmo_2x4096_512_2048cnn_2xhighway_1000k_weights.hdf5 --use_crf


* for clova2019_morph_space

$ python preprocess.py --config=configs/config-elmo.json --data_dir=data/clova2019_morph_space --embedding_path=embeddings/kor.glove.300k.300d.txt
$ python train.py --config=configs/config-elmo.json --save_path=pytorch-model-elmo-kor-clova-morph-space.pt --data_dir=data/clova2019_morph_space --elmo_options_file=embeddings/kor_elmo_2x4096_512_2048cnn_2xhighway_1000k_options.json --elmo_weights_file=embeddings/kor_elmo_2x4096_512_2048cnn_2xhighway_1000k_weights.hdf5 --use_crf --use_char_cnn

```

- evaluation
```
* for clova2019_morph

$ python evaluate.py --config=configs/config-elmo.json --model_path=pytorch-model-elmo-kor-clova-morph.pt --data_dir=data/clova2019_morph --elmo_options_file=embeddings/kor_elmo_2x4096_512_2048cnn_2xhighway_1000k_options.json --elmo_weights_file=embeddings/kor_elmo_2x4096_512_2048cnn_2xhighway_1000k_weights.hdf5 --use_crf
$ cd data/clova2019_morph; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

** --embedding_trainable
INFO:__main__:[F1] : 0.8642865647270933, 9000
INFO:__main__:[Elapsed Time] : 744958ms, 82.7731111111111ms on average
accuracy:  94.63%; precision:  86.36%; recall:  86.38%; FB1:  86.37
  *** evluation eoj-by-eoj
  $ cd data/clova2019_morph ; python to-eoj.py < test.txt.pred > test.txt.pred.eoj ; perl ../../etc/conlleval.pl < test.txt.pred.eoj ; cd ../..
  accuracy:  94.26%; precision:  86.37%; recall:  86.38%; FB1:  86.37

** --use_char_cnn --embedding_trainable
INFO:__main__:[F1] : 0.866860266484455, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 951810ms, 105.7396377375264ms on average
accuracy:  94.65%; precision:  86.99%; recall:  86.26%; FB1:  86.62
  *** evaluation eoj-by-eoj
  accuracy:  94.33%; precision:  87.00%; recall:  86.26%; FB1:  86.63
    
** modify model.py for disabling glove
INFO:__main__:[F1] : 0.856990797967312, 9000
INFO:__main__:[Elapsed Time] : 863968ms, 95.98688743193688ms on average
accuracy:  94.30%; precision:  86.44%; recall:  84.85%; FB1:  85.64
  *** evaluation eoj-by-eoj
  accuracy:  93.95%; precision:  86.46%; recall:  84.87%; FB1:  85.66 

** --use_char_cnn , modify model.py for disabling glove
INFO:__main__:[F1] : 0.8587286088699316, 9000
INFO:__main__:[Elapsed Time] : 860675ms, 95.61962440271141ms on average
accuracy:  94.29%; precision:  86.42%; recall:  85.21%; FB1:  85.81
  *** evaluation eoj-by-eoj
  accuracy:  94.01%; precision:  86.42%; recall:  85.22%; FB1:  85.82


* for clova2019_morph_space

$ python evaluate.py --config=configs/config-elmo.json --model_path=pytorch-model-elmo-kor-clova-morph-space.pt --data_dir=data/clova2019_morph_space --elmo_options_file=embeddings/kor_elmo_2x4096_512_2048cnn_2xhighway_1000k_options.json --elmo_weights_file=embeddings/kor_elmo_2x4096_512_2048cnn_2xhighway_1000k_weights.hdf5 --use_crf --use_char_cnn
$ cd data/clova2019_morph_space; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.8609660351595835, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 1018694ms, 113.17768640960107ms on average
accuracy:  95.51%; precision:  86.16%; recall:  85.74%; FB1:  85.95
  *** evaluation eoj-by-eoj
  $ cd data/clova2019_morph_space ; python to-eoj.py < test.txt.pred > test.txt.pred.eoj ; perl ../../etc/conlleval.pl < test.txt.pred.eoj ; cd ../..
  accuracy:  94.08%; precision:  86.28%; recall:  85.85%; FB1:  86.06 

```

</p>
</details>


<details><summary><b>emb_class=electra, enc_class=bilstm, KoELECTRA-Base, LM-KOR-ELECTRA, bpe ELECTRA-base, RoBERTa-base </b></summary>
<p>

- train
```
* share config-bert.json
* n_ctx size should be less than 512

* for clova2019

** KoELECTRA-Base
$ python preprocess.py --config=configs/config-bert.json --data_dir data/clova2019 --bert_model_name_or_path=./embeddings/koelectra-base-v1-discriminator
$ python train.py --config=configs/config-bert.json --save_path=pytorch-model-bert-kor-eoj.pt --bert_model_name_or_path=./embeddings/koelectra-base-v1-discriminator --bert_output_dir=bert-checkpoint-kor-eoj --batch_size=32 --lr=5e-5 --epoch=20 --data_dir data/clova2019 --bert_disable_lstm

** bpe ELECTRA-base(v1)
$ python preprocess.py --config=configs/config-bert.json --data_dir data/clova2019 --bert_model_name_or_path=./embeddings/kor-electra-base-bpe.v1
$ python train.py --config=configs/config-bert.json --save_path=pytorch-model-bert-kor-eoj.pt --bert_model_name_or_path=./embeddings/kor-electra-base-bpe.v1 --bert_output_dir=bert-checkpoint-kor-eoj --batch_size=32 --lr=8e-5 --epoch=30 --data_dir data/clova2019 --bert_disable_lstm  --warmup_epoch=0 --weight_decay=0.0 --gradient_accumulation_steps=2 

** LM-KOR-ELECTRA
$ python preprocess.py --config=configs/config-bert.json --data_dir data/clova2019 --bert_model_name_or_path=./embeddings/electra-kor-base
$ python train.py --config=configs/config-bert.json --save_path=pytorch-model-bert-kor-eoj.pt --bert_model_name_or_path=./embeddings/electra-kor-base --bert_output_dir=bert-checkpoint-kor-eoj --batch_size=32 --lr=8e-5 --epoch=30 --data_dir data/clova2019 --bert_disable_lstm  --warmup_epoch=0 --weight_decay=0.0 --gradient_accumulation_steps=2 

** RoBERTa-base
$ python preprocess.py --config=configs/config-roberta.json --data_dir data/clova2019 --bert_model_name_or_path=./embeddings/kor-roberta-base-bbpe
$ python train.py --config=configs/config-roberta.json --save_path=pytorch-model-bert-kor-eoj.pt --bert_model_name_or_path=./embeddings/kor-roberta-base-bbpe --bert_output_dir=bert-checkpoint-kor-eoj --batch_size=32 --lr=5e-5 --epoch=30 --data_dir data/clova2019 --bert_disable_lstm  --warmup_epoch=0 --weight_decay=0.0 


```

- evaluation
```

* for clova2019

** KoELECTRA-Base

$ python evaluate.py --config=configs/config-bert.json --model_path=pytorch-model-bert-kor-eoj.pt --data_dir data/clova2019 --bert_output_dir=bert-checkpoint-kor-eoj --bert_disable_lstm
$ cd data/clova2019; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.8601332716652106, 9000
INFO:__main__:[Elapsed Time] : 100 examples, 1667ms, 15.737373737373737ms on average
accuracy:  94.17%; precision:  86.46%; recall:  85.36%; FB1:  85.90

***  --warmup_epoch=0 --weight_decay=0.0 --gradient_accumulation_steps=2 --lr=8e-5 , n_ctx=50
INFO:__main__:[F1] : 0.8647250807012538, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 286224ms, 31.79564396044005ms on average
accuracy:  93.01%; precision:  85.94%; recall:  82.36%; FB1:  84.12

***  --warmup_epoch=0 --weight_decay=0.0 --gradient_accumulation_steps=2 --lr=8e-5 --epoch=30
INFO:__main__:[F1] : 0.8674485806561278, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 976734ms, 108.52672519168796ms on average
INFO:__main__:[Elapsed Time] : 100 examples, 1597ms, 15.16161616161616ms on average
accuracy:  94.48%; precision:  86.52%; recall:  86.75%; FB1:  86.64

*** --bert_model_name_or_path=./embeddings/koelectra-base-v3-discriminator  --warmup_epoch=0 --weight_decay=0.0 --gradient_accumulation_steps=2 --lr=8e-5 --epoch=30
INFO:__main__:[F1] : 0.8743705005576774, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 878765.2859687805ms, 97.64226169927423ms on average
INFO:__main__:[Elapsed Time] : 100 examples, 1558.605432510376ms, 14.81159528096517ms on average
accuracy:  94.70%; precision:  86.67%; recall:  87.95%; FB1:  87.31

** LM-KOR-ELECTRA

*** --bert_model_name_or_path=./embeddings/electra-kor-base  --warmup_epoch=0 --weight_decay=0.0 --gradient_accumulation_steps=2 --lr=8e-5 --epoch=30
INFO:__main__:[F1] : 0.8750042550294449, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 848208.4937095642ms, 94.24447772211201ms on average
INFO:__main__:[Elapsed Time] : 100 examples, 1798.86794090271ms, 17.15457800662879ms on average
accuracy:  94.75%; precision:  87.38%; recall:  87.39%; FB1:  87.39


** bpe ELECTRA-base(v1)

$ python evaluate.py --config=configs/config-bert.json --model_path=pytorch-model-bert-kor-eoj.pt --data_dir data/clova2019 --bert_output_dir=bert-checkpoint-kor-eoj --bert_disable_lstm
$ cd data/clova2019; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.8657965765827326, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 1025293.7350273132ms, 113.92257348446465ms on average
INFO:__main__:[Elapsed Time] : 100 examples, 1881.6356658935547ms, 18.044946169612384ms on average
accuracy:  94.43%; precision:  86.76%; recall:  86.16%; FB1:  86.46


** RoBERTa-base

$ python evaluate.py --config=configs/config-roberta.json --model_path=pytorch-model-bert-kor-eoj.pt --data_dir data/clova2019 --bert_output_dir=bert-checkpoint-kor-eoj --bert_disable_lstm
$ cd data/clova2019; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.8556474298866252, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 892125.4014968872ms, 99.12149309250948ms on average
INFO:__main__:[Elapsed Time] : 100 examples, 1671.3252067565918ms, 15.69863280864677ms on average
accuracy:  94.02%; precision:  85.59%; recall:  85.32%; FB1:  85.45

```

</p>
</details>

<br>




# KMOU NER 2019 (Korean)

## experiments summary

- ntagger, measured by conlleval.pl / token_eval.py (micro F1)

|                                | span / token F1 (%)    | Features              | GPU / CPU   | Etc           |
| ------------------------------ | ---------------------- | --------------------- | ----------- | ------------- |    
| GloVe, BiLSTM-CRF              | 85.26 / 86.54          | morph, pos            | 23.0755 / - |               |
| **GloVe, BiLSTM-CRF**          | 85.93 / 86.41          | morph, character, pos | 27.7451 / - |               |
| GloVe, DenseNet-CRF            | 85.30 / 86.89          | morph, pos            | 24.0280 / - |               |
| GloVe, DenseNet-CRF            | 85.91 / 86.38          | morph, character, pos | 22.7710 / - |               |
| dha DistilBERT(v1), BiLSTM-CRF | 84.20 / 86.94          | morph, pos            | 32.1762 / - |               |
| dha DistilBERT(v1), CRF        | 84.85 / 87.34          | morph, pos            | 27.4700 / - |               |
| dha BERT(v1), BiLSTM-CRF       | 87.56 / 90.47          | morph, pos            | 40.0766 / - |               |
| dha BERT(v1), BiLSTM           | 88.00 / 90.24          | morph, pos            | 23.0388 / - |               |
| dha BERT(v1), CRF              | 88.46 / 90.56          | morph, pos            | 34.1522 / - |               |
| dha BERT(v1)                   | 88.04 / 90.64          | morph, pos            | 17.8542 / - |               |
| dha BERT(v1), BiLSTM-CRF       | 83.99 / 87.54          | morph, pos            | 40.5205 / - | del 8,9,10,11 |
| dha BERT(v2), BiLSTM-CRF       | 85.24 / 87.35          | morph, pos            | 37.7829 / - |               |
| dha-bpe BERT(v1), BiLSTM-CRF   | 85.18 / 88.01          | morph, pos            | 39.0183 / - |               |
| dha-bpe BERT(v3), BiLSTM-CRF   | 88.71 / 91.16          | morph, pos            | 40.5316 / - |               |
| dha-bpe BERT-large(v1), CRF    | **89.02** / 91.07      | morph, pos            | 45.1637 / - |               |
| dha-bpe BERT-large(v3), CRF    | 88.62 / 91.17          | morph, pos            | 47.9219 / - |               |
| ELMo, BiLSTM-CRF               | 88.22 / 89.05          | morph, pos            | 128.029 / - |               |
| ELMo, BiLSTM-CRF               | 88.25 / 89.26          | morph, character, pos | 127.514 / - |               |
| ELMo, GloVe, BiLSTM-CRF        | 88.10 / 88.71          | morph, pos            | 127.989 / - |               |
| ELMo, GloVe, BiLSTM-CRF        | 88.00 / 89.20          | morph, character, pos | 116.965 / - |               |

- [etagger](https://github.com/dsindex/etagger), measured by conlleval (micro F1)

|                              | span / token F1 (%) | Features              |
| ---------------------------- | ------------------- | --------------------- |
| ELMo, GloVe, BiLSTM-CRF      | **89.09** / 89.90   | morph, character, pos |

- [Pytorch-BERT-CRF-NER](https://github.com/eagle705/pytorch-bert-crf-ner), measured by sklearn.metrics (token-level F1)

|                       | token-level macro / micro F1 (%)   | Features |
| --------------------- | ---------------------------------- | -------- | 
| KoBERT+CRF            | 87.56 / 89.70                      | morph    |



<details><summary><b>emb_class=glove, enc_class=bilstm</b></summary>
<p>

- train
```
* token_emb_dim in configs/config-glove.json == 300 (ex, kor.glove.300k.300d.txt )
$ python preprocess.py --data_dir data/kmou2019 --embedding_path embeddings/kor.glove.300k.300d.txt
* --use_crf for adding crf layer, --embedding_trainable for fine-tuning pretrained word embedding.
$ python train.py --save_path=pytorch-model-glove-kor-kmou-morph.pt --data_dir data/kmou2019 --use_crf --embedding_trainable

```

- evaluation
```
$ python evaluate.py --model_path=pytorch-model-glove-kor-kmou-morph.pt --data_dir data/kmou2019 --use_crf
* seqeval.metrics supports IOB2(BIO) format, so FB1 from conlleval.pl should be similar value with.
$ cd data/kmou2019; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..
* token-level micro F1
$ cd data/kmou2019; python ../../etc/token_eval.py < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.8526331317303437, 927
INFO:__main__:[Elapsed Time] : 927 examples, 21481ms, 23.0755939524838ms on average
accuracy:  96.79%; precision:  85.66%; recall:  84.88%; FB1:  85.26
token_eval micro F1: 0.865492518703242

* --use_char_cnn
INFO:__main__:[F1] : 0.8593403342700785, 927
INFO:__main__:[Elapsed Time] : 927 examples, 25819ms, 27.7451403887689ms on average
accuracy:  96.74%; precision:  86.56%; recall:  85.32%; FB1:  85.93
token_eval micro F1: 0.8641512381845168

*  --warmup_epoch=0 --weight_decay=0.0 --use_char_cnn
INFO:__main__:[F1] : 0.8589818607372732, 927
INFO:__main__:[Elapsed Time] : 927 examples, 23254ms, 24.571274298056156ms on average
accuracy:  96.87%; precision:  85.57%; recall:  86.23%; FB1:  85.90
token_eval micro F1: 0.8703660344962149

```

</p>
</details>


<details><summary><b>emb_class=glove, enc_class=densenet</b></summary>
<p>

- train
```
* token_emb_dim in configs/config-glove.json == 300 (ex, kor.glove.300k.300d.txt )
$ python preprocess.py --config=configs/config-densenet.json --data_dir data/kmou2019 --embedding_path embeddings/kor.glove.300k.300d.txt
* --use_crf for adding crf layer, --embedding_trainable for fine-tuning pretrained word embedding.
$ python train.py --config=configs/config-densenet.json --save_path=pytorch-model-densenet-kor-kmou-morph.pt --data_dir data/kmou2019 --use_crf --embedding_trainable
```

- evaluation
```
$ python evaluate.py --config=configs/config-densenet.json --model_path=pytorch-model-densenet-kor-kmou-morph.pt --data_dir data/kmou2019 --use_crf
* seqeval.metrics supports IOB2(BIO) format, so FB1 from conlleval.pl should be similar value with.
$ cd data/kmou2019; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..
* token-level micro F1
$ cd data/kmou2019; python ../../etc/token_eval.py < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.8530318602261048, 927
INFO:__main__:[Elapsed Time] : 927 examples, 22368ms, 24.028077753779698ms on average
accuracy:  96.85%; precision:  85.29%; recall:  85.32%; FB1:  85.30
token_eval micro F1: 0.8689248895434463

* --use_char_cnn
INFO:__main__:[F1] : 0.8591299200947027, 927
INFO:__main__:[Elapsed Time] : 927 examples, 21204ms, 22.771058315334773ms on average
accuracy:  96.77%; precision:  86.58%; recall:  85.26%; FB1:  85.91
token_eval micro F1: 0.8638321196460731
```

</p>
</details>


<details><summary><b>emb_class=bert, enc_class=bilstm, dha BERT</b></summary>
<p>

- train
```
* n_ctx size should be less than 512

* dha (v1)
$ python preprocess.py --config=configs/config-bert.json --data_dir data/kmou2019 --bert_model_name_or_path=./embeddings/kor-bert-base-dha.v1
$ python train.py --config=configs/config-bert.json --save_path=pytorch-model-bert-kor-kmou-morph.pt --bert_model_name_or_path=./embeddings/kor-bert-base-dha.v1 --bert_output_dir=bert-checkpoint-kor-kmou-morph --batch_size=32 --lr=5e-5 --epoch=20 --data_dir data/kmou2019 --use_crf --bert_use_pos

* dha (v2)
$ python preprocess.py --config=configs/config-bert.json --data_dir data/kmou2019 --bert_model_name_or_path=./embeddings/kor-bert-base-dha.v2
$ python train.py --config=configs/config-bert.json --save_path=pytorch-model-bert-kor-kmou-morph.pt --bert_model_name_or_path=./embeddings/kor-bert-base-dha.v2 --bert_output_dir=bert-checkpoint-kor-kmou-morph --batch_size=32 --lr=5e-5 --epoch=20 --data_dir data/kmou2019 --use_crf --bert_use_pos

```

- evaluation
```
* dha (v1)
$ python evaluate.py --config=configs/config-bert.json --model_path=pytorch-model-bert-kor-kmou-morph.pt --data_dir=data/kmou2019 --bert_output_dir=bert-checkpoint-kor-kmou-morph --use_crf --bert_use_pos
$ cd data/kmou2019; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..
$ cd data/kmou2019; python ../../etc/token_eval.py < test.txt.pred ; cd ../..

(dha BERT BiLSTM-CRF)
INFO:__main__:[F1] : 0.8751814223512336, 927
INFO:__main__:[Elapsed Time] : 927 examples, 37240ms, 40.076673866090715ms on average
accuracy:  97.58%; precision:  86.59%; recall:  88.55%; FB1:  87.56
token_eval micro F1: 0.9047362341162879

** without --use_crf (dha BERT BiLSTM)
INFO:__main__:[F1] : 0.8800116635078, 927
INFO:__main__:[Elapsed Time] : 927 examples, 21444ms, 23.03887688984881ms on average
accuracy:  97.56%; precision:  87.38%; recall:  88.63%; FB1:  88.00
token_eval micro F1: 0.9024978600887091

** --bert_disable_lstm (dha BERT CRF)
INFO:__main__:[F1] : 0.8844425112367696, 927
INFO:__main__:[Elapsed Time] : 927 examples, 31752ms, 34.152267818574515ms on average
accuracy:  97.64%; precision:  87.37%; recall:  89.57%; FB1:  88.46
token_eval micro F1: 0.9056049478160032

**  --warmup_epoch=0 --weight_decay=0.0 --bert_disable_lstm --epoch=30 (dha BERT CRF)
INFO:__main__:[F1] : 0.8810802962102512, 927
INFO:__main__:[Elapsed Time] : 927 examples, 30447ms, 32.355291576673864ms on average
accuracy:  97.54%; precision:  87.16%; recall:  89.10%; FB1:  88.12
token_eval micro F1: 0.8997748622001397

** --bert_disable_lstm --lr_deacy_rate=0.9 , without --use_crf (dha BERT)
INFO:__main__:[F1] : 0.880439496891716, 927
INFO:__main__:[Elapsed Time] : 927 examples, 16635ms, 17.854211663066955ms on average
accuracy:  97.65%; precision:  86.70%; recall:  89.43%; FB1:  88.04
token_eval micro F1: 0.9064737125702409

** --bert_remove_layers=8,9,10,11
INFO:__main__:[F1] : 0.8392018779342724, 927
INFO:__main__:[Elapsed Time] : 37666ms, 40.52051835853132ms on average
accuracy:  96.88%; precision:  83.99%; recall:  83.99%; FB1:  83.99
token_eval micro F1: 0.8754817902934005

** --config=configs/config-distilbert.json --bert_model_name_or_path=./embeddings/kor-distil-dha-bert.v1  --warmup_epoch=0 --weight_decay=0.0 --bert_disable_lstm --epoch=30 --lr=8e-5
(1) epoch_0
INFO:__main__:[F1] : 0.8477473562219324, 927
INFO:__main__:[Elapsed Time] : 927 examples, 25547.332048416138ms, 27.470011175579952ms on average
accuracy:  96.96%; precision:  83.79%; recall:  85.93%; FB1:  84.85
token_eval micro F1: 0.8734618063617366

** --config=configs/config-distilbert.json --bert_model_name_or_path=./embeddings/kor-distil-dha-bert.v1  --warmup_epoch=0 --weight_decay=0.0 --epoch=30 --lr=8e-5
(1) epoch_2
INFO:__main__:[F1] : 0.8417381194593809, 927
INFO:__main__:[Elapsed Time] : 927 examples, 29921.59104347229ms, 32.176273945340846ms on average
accuracy:  96.85%; precision:  83.36%; recall:  85.05%; FB1:  84.20
token_eval micro F1: 0.8694366635543106

* dha(v2)
$ python evaluate.py --config=configs/config-bert.json --model_path=pytorch-model-bert-kor-kmou-morph.pt --data_dir=data/kmou2019 --bert_output_dir=bert-checkpoint-kor-kmou-morph --use_crf --bert_use_pos
$ cd data/kmou2019; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..
$ cd data/kmou2019; python ../../etc/token_eval.py < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.8519003931847969, 927
INFO:__main__:[Elapsed Time] : 35123ms, 37.7829373650108ms on average
accuracy:  96.83%; precision:  84.59%; recall:  85.90%; FB1:  85.24
token_eval micro F1: 0.8735865242143024

```

</p>
</details>


<details><summary><b>emb_class=bert, enc_class=bilstm, dha-bpe BERT, dha-bpe BERT-large</b></summary>
<p>

- train
```
* n_ctx size should be less than 512

$ python preprocess.py --config=configs/config-bert.json --data_dir data/kmou2019 --bert_model_name_or_path=./embeddings/kor-bert-base-dha_bpe.v1
$ python train.py --config=configs/config-bert.json --save_path=pytorch-model-bert-kor-kmou-morph.pt --bert_model_name_or_path=./embeddings/kor-bert-base-dha_bpe.v1 --bert_output_dir=bert-checkpoint-kor-kmou-morph --batch_size=32 --lr=5e-5 --epoch=20 --data_dir data/kmou2019 --use_crf --bert_use_pos

```

- evaluation
```
$ python evaluate.py --config=configs/config-bert.json --model_path=pytorch-model-bert-kor-kmou-morph.pt --data_dir=data/kmou2019 --bert_output_dir=bert-checkpoint-kor-kmou-morph --use_crf --bert_use_pos
$ cd data/kmou2019; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..
$ cd data/kmou2019; python ../../etc/token_eval.py < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.8517251211861989, 927
INFO:__main__:[Elapsed Time] : 36267ms, 39.018358531317496ms on average
accuracy:  97.14%; precision:  82.79%; recall:  87.72%; FB1:  85.18
token_eval micro F1: 0.8801729462631254

* --bert_model_name_or_path=./embeddings/kor-bert-base-dha_bpe.v3 --batch_size=64 --warmup_epoch=0 --weight_decay=0.0 --patience=4
INFO:__main__:[F1] : 0.886564996368918, 927
INFO:__main__:[Elapsed Time] : 927 examples, 37696.48361206055ms, 40.53165665455565ms on average
accuracy:  97.80%; precision:  87.80%; recall:  89.63%; FB1:  88.71
token_eval micro F1: 0.9116004962779156

* --bert_model_name_or_path=./embeddings/kor-bert-large-dha_bpe.v1 --bert_disable_lstm --lr=1e-5 
INFO:__main__:[F1] : 0.8902403706921518, 927
INFO:__main__:[Elapsed Time] : 927 examples, 41991.15180969238ms, 45.16370275880554ms on average
accuracy:  97.79%; precision:  87.80%; recall:  90.28%; FB1:  89.02
token_eval micro F1: 0.9107763615295481

* --bert_model_name_or_path=./embeddings/kor-bert-large-dha_bpe.v3 --bert_disable_lstm --lr=1e-5 --warmup_epoch=0 --weight_decay=0.0 --patience=4
INFO:__main__:[F1] : 0.8860723030390322, 927
INFO:__main__:[Elapsed Time] : 927 examples, 44546.09036445618ms, 47.92198120389077ms on average
accuracy:  97.81%; precision:  86.97%; recall:  90.34%; FB1:  88.62
token_eval micro F1: 0.9117692189657818

```

</p>
</details>


<details><summary><b>emb_class=elmo, enc_class=bilstm</b></summary>
<p>

- train
```
* token_emb_dim in configs/config-elmo.json == 300 (ex, kor.glove.300k.300d.txt )
* elmo_emb_dim  in configs/config-elmo.json == 1024 (ex, kor_elmo_2x4096_512_2048cnn_2xhighway_1000k* )
$ python preprocess.py --config=configs/config-elmo.json --data_dir=data/kmou2019 --embedding_path=embeddings/kor.glove.300k.300d.txt
$ python train.py --config=configs/config-elmo.json --save_path=pytorch-model-elmo-kor-kmou-morph.pt --data_dir=data/kmou2019 --elmo_options_file=embeddings/kor_elmo_2x4096_512_2048cnn_2xhighway_1000k_options.json --elmo_weights_file=embeddings/kor_elmo_2x4096_512_2048cnn_2xhighway_1000k_weights.hdf5 --use_crf
```

- evaluation
```
$ python evaluate.py --config=configs/config-elmo.json --model_path=pytorch-model-elmo-kor-kmou-morph.pt --data_dir=data/kmou2019 --elmo_options_file=embeddings/kor_elmo_2x4096_512_2048cnn_2xhighway_1000k_options.json --elmo_weights_file=embeddings/kor_elmo_2x4096_512_2048cnn_2xhighway_1000k_weights.hdf5 --use_crf
$ cd data/kmou2019; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..
$ cd data/kmou2019; python ../../etc/token_eval.py < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.8803757522383678, 927
INFO:__main__:[Elapsed Time] : 126180ms, 135.92548596112312ms on average
accuracy:  97.30%; precision:  88.00%; recall:  88.08%; FB1:  88.04
token_eval micro F1: 0.8880736809241336

* --batch_size=64
INFO:__main__:[F1] : 0.8809558502112778, 927
INFO:__main__:[Elapsed Time] : 927 examples, 118820ms, 127.98920086393089ms on average
accuracy:  97.29%; precision:  87.42%; recall:  88.78%; FB1:  88.10
token_eval micro F1: 0.8871589332511176

* --embedding_trainable
INFO:__main__:[F1] : 0.8755125951962508, 927
INFO:__main__:[Elapsed Time] : 125665ms, 135.366090712743ms on average
accuracy:  97.33%; precision:  87.32%; recall:  87.78%; FB1:  87.55
token_eval micro F1: 0.8897515527950312

* --use_char_cnn --batch_size=64 
INFO:__main__:[F1] : 0.8799648248571009, 927
INFO:__main__:[Elapsed Time] : 927 examples, 108596ms, 116.96544276457884ms on average
accuracy:  97.37%; precision:  87.83%; recall:  88.16%; FB1:  88.00
token_eval micro F1: 0.8920632495607668

* --batch_size=64 , modify model.py for disabling glove
INFO:__main__:[F1] : 0.8822326125073057, 927
INFO:__main__:[Elapsed Time] : 118884ms, 128.0291576673866ms on average
accuracy:  97.35%; precision:  87.79%; recall:  88.66%; FB1:  88.22
token_eval micro F1: 0.8905289052890529

* --use_char_cnn --batch_size=64 , modify model.py for disabling glove
INFO:__main__:[F1] : 0.8825072886297376, 927
INFO:__main__:[Elapsed Time] : 118407ms, 127.51403887688984ms on average
accuracy:  97.43%; precision:  87.61%; recall:  88.90%; FB1:  88.25
token_eval micro F1: 0.8926606215608773
```

</p>
</details>

<br>



# Citation

```
@misc{ntagger,
  author = {dsindex},
  title = {ntagger},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/dsindex/ntagger}},
}
```

<br>



# References

- [Named Entity Recoginition](https://paperswithcode.com/task/named-entity-recognition-ner)
- [transformers_examples](https://github.com/dsindex/transformers_examples)
- [macro and micro precision/recall/f1 score](https://datascience.stackexchange.com/a/24051)
- [RoBERTa](https://github.com/pytorch/fairseq/tree/master/examples/roberta)
  - [RoBERTa GLUE task setting](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.glue.md)
- [BERT Miniatures](https://huggingface.co/google/bert_uncased_L-12_H-128_A-2)
  - search range of hyperparameters
    - batch sizes: 8, 16, 32, 64, 128
    - learning rates: 3e-4, 1e-4, 5e-5, 3e-5
- scalar mixtures of BERT all layers
  - [ScalarMixWithDropout](https://github.com/Hyperparticle/udify/blob/master/udify/modules/scalar_mix.py)
  - [ScalarMix](https://github.com/allenai/allennlp/blob/master/allennlp/modules/scalar_mix.py)
- [Poor Man’s BERT: Smaller and Faster Transformer Models](https://arxiv.org/pdf/2004.03844v1.pdf)
  - https://github.com/hsajjad/transformers/blob/master/examples/run_glue.py
- [(pytorch) advanced_tutorial](https://tutorials.pytorch.kr/beginner/nlp/advanced_tutorial.html)
