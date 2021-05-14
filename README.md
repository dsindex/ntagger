# Description

**reference pytorch code for named entity tagging.**

- embedding
  - word : GloVe, BERT, DistilBERT, mDistilBERT, MiniLM, feature-based BERT using DSA(Dynamic Self Attention) pooling, SpanBERT, ALBERT, RoBERTa, XLM-RoBERTa, BART, ELECTRA, DeBERTa, ELMo
  - character : CNN
  - pos : lookup

- encoding
  - BiLSTM
  - DenseNet
    - [Dynamic Self-Attention: Computing Attention over Words Dynamically for Sentence Embedding](https://arxiv.org/pdf/1808.07383.pdf)
    - a slightly modified DenseNet for longer dependency.
  - Multi-Head Attention

- decoding
  - Softmax, CRF

- related: [reference pytorch code for intent(sentence) classification](https://github.com/dsindex/iclassifier)

- document context for BERT paper reproduction
  - see : https://github.com/dsindex/ntagger/issues/4#issuecomment-810304253

- [multi-task learning](/MULTI-TASK.md)
  - joint learning of sequence and token classification

<br>



# Requirements

- python >= 3.6

- pip install -r requirements.txt

<br>



# Data

## CoNLL 2003 (English)

#### from [etagger](https://github.com/dsindex/etagger), [CrossWeigh](https://github.com/ZihanWangKi/CrossWeigh)
  
##### data/conll2003

##### data/conll++

- since CoNLL++/test.txt has incorrect chunk tags, combine it with original CoNLL2003/test.txt
```
$ python combine.py --conll2003 ../conll2003/test.txt --conllpp test.txt > t
$ mv t test.txt
```

##### data/conll2003_truecase, data/conll++_truecase

- [converting conll2003, conll++ data to its truecase](https://github.com/google-research/bert/issues/223#issuecomment-649619302)
<details><summary>details</summary>
<p>

```
$ cd data/conll2003_truecase
$ python to-truecase.py --input_path ../conll2003/train.txt > train.txt
$ python to-truecase.py --input_path ../conll2003/valid.txt > valid.txt
$ python to-truecase.py --input_path ../conll2003/test.txt > test.txt

* same work for data/conll++
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
  - BERT-like models(huggingface's [transformers](https://github.com/huggingface/transformers.git))
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
    - mDistilBERT : `distilbert-base-multilingual-cased`
    - KoELECTRA-Base : `koelectra-base-v1-discriminator`, `koelectra-base-v3-discriminator`
    - LM-KOR-ELECTRA : `electra-kor-base`
    - ELECTRA-base : `kor-electra-bpe.v1`, `kor-electra-base-dhaToken1.large`, `kor-electra-base-dhaSyllable` (inhouse)
    - RoBERTa-base : `kor-roberta-base-bbpe` (inhouse)
    - XLM-RoBERTa : `xlm-roberta-base`, `xlm-roberta-large`
    - Funnel-base : `funnel-kor-base`
  - [ELMo description](https://github.com/dsindex/bilm-tf)
    - `kor_elmo_2x4096_512_2048cnn_2xhighway_1000k_weights.hdf5`, `kor_elmo_2x4096_512_2048cnn_2xhighway_1000k_options.json` (inhouse)
  
<br>




# CoNLL 2003 (English)

## experiments summary

- ntagger, measured by conlleval.pl (micro F1)

|                                 | F1 (%)            | (truecase) F1 (%) | Features             | GPU / CPU          | ONNX      | Dynamic   | Etc                       |
| ------------------------------- | ----------------- | ----------------- | -------------------- | ------------------ | --------- | --------- | ------------------------- |
| GloVe, BiLSTM                   | 88.23             |                   | word, pos            | 5.6217  / -        | 3.6969    |           | threads=14                |
| GloVe, BiLSTM                   | 88.18             |                   | word, pos            | 29.9140 / -        | -         |           | NCRF                      |
| GloVe, BiLSTM                   | 88.94             |                   | word, character, pos | 6.4108  / -        | 8.5656    |           | threads=14                |
| GloVe, BiLSTM-MHA               | 89.99             |                   | word, character, pos | 7.7513  / -        |           |           |                           |
| GloVe, BiLSTM-MHA-CRF           | 90.48             |                   | word, character, pos | 25.8200 / -        |           |           |                           |
| GloVe, BiLSTM-MHA-CRF           | 90.42             |                   | word, character, pos | 30.1932 / -        |           |           | LabelSmoothingCrossEntropy |
| **GloVe, BiLSTM-CRF**           | 90.14 / 90.92     | 90.26 / 90.76     | word, character, pos | 26.2807 / -        | FAIL      |           |                           |
| ConceptNet, BiLSTM-CRF          | 87.78             |                   | word, character, pos | 25.8119 / -        |           |           |                           |
| ConceptNet, BiLSTM-CRF          | 88.17             |                   | word, character, pos | 23.3482 / -        |           |           | optuna                    |
| GloVe, DenseNet-CRF             | 88.23             |                   | word, pos            | 24.7893 / -        |           |           |                           |
| GloVe, DenseNet-CRF             | 88.89             |                   | word, character, pos | 28.0993 / -        |           |           |                           |
| GloVe, DenseNet-MHA             | 88.47             |                   | word, character, pos | 7.9668  / -        |           |           |                           |
| GloVe, DenseNet-MHA-CRF         | 88.58             |                   | word, character, pos | 26.2287 / -        |           |           |                           |
| BERT-tiny, BiLSTM               | 69.65             |                   | word                 | 20.1376 / -        |           |           |                           |
| BERT-mini, BiLSTM               | 81.55             |                   | word                 | 21.4632 / -        |           |           |                           |
| BERT-small, BiLSTM              | 86.35             |                   | word                 | 22.6087 / -        |           |           |                           |
| BERT-medium, BiLSTM             | 88.29             |                   | word                 | 27.0486 / -        |           |           |                           |
| DistilBERT, BiLSTM              | 89.50             |                   | word                 | 13.4564 / -        | 56.2819   | 47.8320   | threads=14                |
| mDistilBERT, BiLSTM             | 90.21             |                   | word                 | 13.4910 / -        |           |           |                           |
| MiniLM, BiLSTM                  | 90.55             |                   | word                 | 17.7890 / -        |           |           |                           |
| BERT-base(uncased), BiLSTM-CRF  | 90.20             |                   | word                 | 42.6464 / -        |           |           | epoch=10                  |
| BERT-base(uncased), BiLSTM      | 90.55             |                   | word                 | 18.2323 / -        | 79.1914   | 83.9590   | threads=14, epoch=10      |
| BERT-base(uncased), BiLSTM      | 90.60             |                   | word                 | 22.7757 / -        |           |           | freezing BERT during some epochs, epoch=10 |
| BERT-base(uncased), CRF         | 89.98             |                   | word                 | 36.6893 / -        |           |           | epoch=10                  |
| BERT-base(uncased)              | 90.25             |                   | word                 | 16.6877 / -        | 72.8225   | 75.3025   | threads=14, epoch=10      |
| BERT-base(uncased), BiLSTM      | 89.03             |                   | word                 | 24.9076 / -        |           |           | del 8,9,10,11, threads=14, epoch=10                                     |
| bert-base-NER(cased), BiLSTM    | 91.63 / 92.71     | 92.25 / 93.16     | word                 | 17.6680 / -        |           |           | freezing BERT during some epochs for conll++,conll++_truecase, epoch=10 |
| BERT-base(cased)                | 91.45             |                   | word                 | 45.5024 / -        |           |           | epoch=30                                               |
| BERT-base(cased)                | 91.37             |                   | word                 | 46.2147 / -        |           |           | subword pooling, epoch=30                              |
| BERT-base(cased)                | 91.20             |                   | word                 | 33.3276 / -        |           |           | subword pooling, word embedding, epoch=30              |
| BERT-base(cased)                | 92.35             |                   | word                 | 32.6524 / -        |           |           | document context, epoch=30, n_ctx=512                  |
| BERT-base(cased)                | 92.23             |                   | word                 | 46.7166 / -        |           |           | document context, subword pooling, epoch=30, n_ctx=512 |
| BERT-base(cased)                | 92.15             |                   | word                 | 46.5353 / -        |           |           | document context, subword pooling, word embedding, epoch=30, n_ctx=512 |
| BERT-base(cased), BiLSTM        | 90.20             |                   | word                 | 21.5844 / -        |           |           | epoch=10                                   |
| BERT-base(cased), BiLSTM        | 90.99             |                   | word                 | 21.7328 / -        |           |           | freezing BERT during some epochs, epoch=10 |
| BERT-base(cased), BiLSTM-MHA    | 90.95             |                   | word                 | 21.9845 / -        |           |           | freezing BERT during some epochs, epoch=10 |
| BERT-base(cased), BiLSTM-CRF    | 90.17             |                   | word                 | 43.4804 / -        |           |           | epoch=10                                                                |
| BERT-base(cased), BiLSTM-CRF    | 91.55             |                   | word                 | 42.2709 / -        |           |           | freezing BERT during some epochs, epoch=10                              |
| BERT-base(cased), BiLSTM-CRF    | 91.60             |                   | word                 | 39.6135 / -        |           |           | using sub token label, freezing BERT during some epochs, epoch=10       |
| BERT-base(cased), BiLSTM-CRF    | 91.33             |                   | word                 | 41.1204 / -        |           |           | slicing logits, freezing BERT during some epochs, epoch=10, https://github.com/dsindex/ntagger/releases/tag/v1.0        |
| BERT-base(cased), BiLSTM-CRF    | 91.37             |                   | word, character, pos | 40.2822 / -        |           |           | using sub token label, freezing BERT during some epochs, epoch=10, https://github.com/dsindex/ntagger/releases/tag/v1.0 |
| BERT-base(cased), BiLSTM-CRF    | 91.66             |                   | word, character, pos | 39.6657 / -        |           |           | using sub token label, freezing BERT during some epochs, epoch=30, https://github.com/dsindex/ntagger/releases/tag/v1.0 |
| BERT-base(cased), BiLSTM-CRF    | 91.33             |                   | word                 | 40.1493 / -        |           |           | subword pooling, freezing BERT during some epochs, epoch=30                              |
| BERT-base(cased), BiLSTM-CRF    | 92.31             |                   | word                 | 40.7793 / -        |           |           | subword pooling, word embedding, freezing BERT during some epochs, epoch=30              |
| BERT-base(cased), BiLSTM-CRF    | 92.08             |                   | word, character, pos | 41.1466 / -        |           |           | subword pooling, word embedding, freezing BERT during some epochs, epoch=30              |
| BERT-base(cased), BiLSTM-CRF    | 92.04             |                   | word                 | 92.8827 / -        |           |           | document context, freezing BERT during some epochs, epoch=30, n_ctx=512                  |
| BERT-base(cased), BiLSTM-CRF    | 92.08             |                   | word                 | 81.3581 / -        |           |           | document context, subword pooling, freezing BERT during some epochs, epoch=30, n_ctx=512 |
| BERT-base(cased), BiLSTM-CRF    | 92.85             |                   | word                 | 80.1447 / -        |           |           | document context, subword pooling, word embedding, freezing BERT during some epochs, epoch=30, n_ctx=512 |
| BERT-large                      | 91.13             |                   | word                 | 31.2675 / -        |           |           | epoch=30                                                 |
| BERT-large                      | 92.27             |                   | word                 | 68.4826 / -        |           |           | document context, epoch=30, n_ctx=512                    |
| BERT-large, BiLSTM              | 91.32             | 91.89             | word                 | 40.3581 / -        |           |           | epoch=10                                   |
| BERT-large, BiLSTM              | 91.57             |                   | word                 | 35.2808 / -        |           |           | freezing BERT during some epochs, epoch=10 |
| BERT-large, BiLSTM+CRF          | 90.78             |                   | word                 | 59.3982 / -        |           |           | epoch=10                                   |
| BERT-large, BiLSTM+CRF          | 92.02             | 91.96             | word                 | 54.4254 / -        |           |           | freezing BERT during some epochs, epoch=10 |
| BERT-large, BiLSTM+CRF          | 92.83             |                   | word                 | 103.930 / -        |           |           | document context, subword pooling, word embedding, freezing BERT during some epochs, n_ctx=512, epoch=30 |
| BERT-large, BiLSTM              | 89.10             |                   | word                 | 33.1376 / -        |           |           | del 12 ~ 23, epoch=10                      |
| BERT-large, BiLSTM-CRF          | 90.64             |                   | word                 | 63.9397 / -        |           |           | BERT as feature-based, [-4:] embedding, epoch=64               |
| BERT-large, BiLSTM-CRF          | 90.52             |                   | word                 | 70.8322 / -        |           |           | BERT as feature-based, mean([0:3] + [-4:]) embedding, epoch=64 |
| BERT-large, BiLSTM-CRF          | 90.81             |                   | word                 | 68.6139 / -        |           |           | BERT as feature-based, mean([0:17]) embedding, epoch=64        |
| BERT-large, BiLSTM-CRF          | 90.76             |                   | word                 | 60.8039 / -        |           |           | BERT as feature-based, max([0:17]) embedding, epoch=64         |
| BERT-large, BiLSTM-CRF          | 90.98             |                   | word                 | 58.9112 / -        |           |           | BERT as feature-based, mean([0:]) embedding, epoch=64          |
| BERT-large, BiLSTM-CRF          | 90.62             |                   | word                 | 66.6576 / -        |           |           | BERT as feature-based, DSA(4, 300), epoch=64                   |
| BERT-large-squad, BiLSTM        | 91.75             | 92.17             | word                 | 35.6619 / -        |           |           | epoch=10                  |
| BERT-large-conll03, BiLSTM      | 91.87 / 92.62     | 92.40 / 93.36     | word                 | 32.2211 / -        |           |           | freezing BERT during some epochs for conll++,conll++_truecase, epoch=10 |
| BERT-large-conll03              | 91.63             | 92.24             | word                 | 29.8476 / -        |           |           | epoch=10                  |
| SpanBERT-base, BiLSTM           | 90.46             |                   | word                 | 30.0991 / -        |           |           |                           |
| SpanBERT-large, BiLSTM          | 91.39             | 92.01             | word                 | 42.5959 / -        |           |           |                           |
| ALBERT-base, BiLSTM             | 88.19             |                   | word                 | 31.0868 / -        |           |           |                           |
| ALBERT-xxlarge, BiLSTM          | 90.39             |                   | word                 | 107.778 / -        |           |           |                           |
| RoBERTa-base                    | 90.03             |                   | word                 | 19.2503 / -        |           |           | epoch=10                  |
| RoBERTa-large                   | 91.83             | 91.90             | word                 | 28.5525 / -        |           |           | epoch=10                  |
| XLM-RoBERTa-base                | 91.20             |                   | word                 | 18.9604 / -        |           |           | epoch=30                  |
| XLM-RoBERTa-base, BiLSTM        | 90.81             |                   | word                 | 21.4667 / -        |           |           | freezing BERT during some epochs, epoch=30                                                                       |
| XLM-RoBERTa-base, BiLSTM-CRF    | 91.12             |                   | word                 | 39.4418 / -        |           |           | freezing BERT during some epochs, epoch=30                                                                       |
| XLM-RoBERTa-base, BiLSTM-CRF    | 91.79             |                   | word                 | 43.0662 / -        |           |           | using sub token label, freezing BERT during some epochs, epoch=30                                                |
| XLM-RoBERTa-base, BiLSTM-CRF    | 91.16             |                   | word                 | 39.3642 / -        |           |           | slicing logits, freezing BERT during some epochs, epoch=30, https://github.com/dsindex/ntagger/releases/tag/v1.0 |
| XLM-RoBERTa-large               | 92.75 / 93.95     | 92.89 / 94.11     | word                 | 27.9144 / -        |           |           | epoch=30                                                                                                         |
| XLM-RoBERTa-large               | 93.86             |                   | word                 | 50.2684 / -        |           |           | document context, epoch=30, n_ctx=512                                                                            |
| XLM-RoBERTa-large, BiLSTM       | -     / 93.75     | -         / 93.81 | word                 | 34.4894 / -        |           |           | freezing BERT during some epochs, epoch=30                                                                       |
| XLM-RoBERTa-large, BiLSTM-CRF   | 92.97             |                   | word                 | 52.8133 / -        |           |           | subword pooling, word embedding, freezing BERT during some epochs, epoch=30                                      |
| XLM-RoBERTa-large, BiLSTM-CRF   | 93.59             |                   | word                 | 105.188 / -        |           |           | document context, subword pooling, word embedding, freezing BERT during some epochs, n_ctx=512, epoch=30         |
| BART-large, BiLSTM              | 90.43             |                   | word                 | 53.3657 / -        |           |           |                           |
| ELECTRA-base, BiLSTM            | 90.98             |                   | word                 | 22.4132 / -        |           |           |                           |
| ELECTRA-large                   | 91.39             |                   | word                 | 29.5734 / -        |           |           |                           |
| DeBERTa-base                    | 90.41             |                   | word                 | 28.6874 / -        |           |           | epoch=20                  |
| DeBERTa-large                   | 91.45             |                   | word                 | 53.9249 / -        |           |           | epoch=20                  |
| DeBERTa-v2-xlarge               | 93.12             |                   | word                 | 62.9722 / -        |           |           | --use_sharded_ddp, epoch=20, https://github.com/dsindex/ntagger/releases/tag/v1.1                                |
| DeBERTa-v2-xlarge               | **94.00**         |                   | word                 | 157.075 / -        |           |           | --use_sharded_ddp, document context, epoch=20, n_ctx=512, https://github.com/dsindex/ntagger/releases/tag/v1.1   |
| ELMo, BiLSTM-CRF                | 91.78             |                   | word, pos            | 74.1001 / -        |           |           |                           |
| ELMo, BiLSTM-CRF                | 91.93             |                   | word, character, pos | 67.6931 / -        |           |           |                           |
| ELMo, GloVe, BiLSTM-CRF         | 92.63 / 93.49     | 92.51 / 93.68     | word, pos            | 74.6521 / -        |           |           |                           |
| ELMo, GloVe, BiLSTM-CRF         | 92.03             |                   | word, character, pos | 60.4667 / -        |           |           | threads=14                |

```
* GPU / CPU     : Elapsed time/example(ms), GPU / CPU, [Tesla V100 1 GPU, Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz, 2 CPU, 14CORES/1CPU, HyperThreading]
* F1            : conll2003 / conll++
* (truecase) F1 : conll2003_truecase / conll++_truecase
* ONNX          : --enable_ort 
* Dynamic       : --enable_dqm
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
| ACE + document-context          | 94.14    |                     |
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

- [CoNLL++(English) leaderboard](https://paperswithcode.com/sota/named-entity-recognition-on-conll), measured by span-level F1(micro F1, same result by conlleval? unknown!)

|                                 | F1 (%)    | Etc                 |
| ------------------------------- | --------- | ------------------- |
| CrossWeigh + Pooled Flair       | **94.28** |                     |
| ELMo                            | 93.42     |                     |
| BLSTM-CNN-CRF                   | 91.87     |                     |


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

* --data_dir=data/conll++ --use_char_cnn
INFO:__main__:[F1] : 0.9092189838865897, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 91787.76407241821ms, 24.89247232748379ms on average
accuracy:  98.04%; precision:  91.30%; recall:  90.55%; FB1:  90.92

* --data_dir=data/conll++_truecase --use_char_cnn
INFO:__main__:[F1] : 0.9076029567053854, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 91913.47312927246ms, 24.927138764347486ms on average
accuracy:  98.01%; precision:  91.08%; recall:  90.44%; FB1:  90.76

* --use_char_cnn --use_mha , without --use_crf
INFO:__main__:[F1] : 0.899894254494184, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 28661.02647781372ms, 7.751333820344053ms on average
accuracy:  97.96%; precision:  89.58%; recall:  90.40%; FB1:  89.99

* --use_char_cnn --use_mha
INFO:__main__:[F1] : 0.9047997878546804, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 95233.61730575562ms, 25.82004158968774ms on average
accuracy:  97.97%; precision:  90.34%; recall:  90.62%; FB1:  90.48

* --use_char_cnn --use_mha --criterion=LabelSmoothingCrossEntropy
INFO:__main__:[F1] : 0.9042186256301407, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 111375.56338310242ms, 30.19321803129985ms on average
accuracy:  97.96%; precision:  90.33%; recall:  90.51%; FB1:  90.42

* --use_ncrf (use NCRF instead of pytorch-crf)
$ python preprocess.py --data_dir=data/conll2003 --use_ncrf
$ python train.py --data_dir=data/conll2003 --use_crf --use_ncrf
$ python evaluate.py --data_dir=data/conll2003 --use_crf --use_ncrf
INFO:__main__:[token classification F1] : 0.8818254461511142, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 110281.77309036255ms, 29.91403479502313ms on average
accuracy:  97.40%; precision:  88.44%; recall:  87.92%; FB1:  88.18

$ python evaluate.py --data_dir=data/conll2003 --use_crf --use_ncrf --convert_onnx
...
File "/usr/local/lib/python3.6/dist-packages/torch/onnx/symbolic_opset9.py", line 441, in transpose
    axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
    IndexError: list index out of range

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

* --use_char_cnn --use_mha , without --use_crf
INFO:__main__:[F1] : 0.8846794703148294, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 29449.145555496216ms, 7.9668397460655775ms on average
accuracy:  97.75%; precision:  87.65%; recall:  89.31%; FB1:  88.47

* --use_char_cnn --use_mha
INFO:__main__:[F1] : 0.8857545839210157, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 96725.40020942688ms, 26.228771176054657ms on average
accuracy:  97.62%; precision:  88.20%; recall:  88.95%; FB1:  88.58

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

* --batch_size=32 --bert_freezing_epoch=4 --bert_lr_during_freezing=1e-3
INFO:__main__:[F1] : 0.9156838584756204, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 130123.88586997986ms, 35.28089417150219ms on average
accuracy:  98.33%; precision:  91.04%; recall:  92.10%; FB1:  91.57

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

* --batch_size=32 --bert_freezing_epoch=4 --bert_lr_during_freezing=1e-3 --use_crf
INFO:__main__:[F1] : 0.9178914019185074, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 194960.72125434875ms, 52.88945214113333ms on average
accuracy:  98.31%; precision:  91.70%; recall:  92.33%; FB1:  92.02

* --data_dir=data/conll2003_truecase --batch_size=32 --bert_freezing_epoch=4 --bert_lr_during_freezing=1e-3 --use_crf
INFO:__main__:[F1] : 0.9170551499692144, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 200677.77919769287ms, 54.42541078675742ms on average
accuracy:  98.23%; precision:  91.63%; recall:  92.30%; FB1:  91.96

* --data_dir=data/conll2003_truecase 
INFO:__main__:[F1] : 0.9188571428571428, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 120905.13277053833ms, 32.785276922556875ms on average
accuracy:  98.36%; precision:  91.25%; recall:  92.53%; FB1:  91.89

* --bert_disable_lstm --batch_size=32
INFO:__main__:[F1] : 0.9124989051414557, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 109073.38047027588ms, 29.574078060898998ms on average
accuracy:  98.27%; precision:  90.29%; recall:  92.23%; FB1:  91.25

* --bert_disable_lstm --epoch=30
INFO:__main__:[F1] : 0.9112654999560285, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 115351.17220878601ms, 31.267560567345726ms on average
accuracy:  98.30%; precision:  90.53%; recall:  91.73%; FB1:  91.13

* document context, --bert_disable_lstm --batch_size=8, n_ctx: 512, --epoch=30
# preprocessing
$ python preprocess.py --config=configs/config-bert.json --data_dir=data/conll2003 --bert_model_name_or_path=./embeddings/bert-large-cased --bert_use_doc_context
# train
$ python train.py --config=configs/config-bert.json --data_dir=data/conll2003 --save_path=pytorch-model-bert.pt --bert_model_name_or_path=./embeddings/bert-large-cased --bert_output_dir=bert-checkpoint --batch_size=8 --lr=1e-5 --epoch=30 --bert_use_doc_context --bert_disable_lstm  --eval_batch_size=32
# evaluate
$ python evaluate.py --config=configs/config-bert.json --data_dir=data/conll2003 --model_path=pytorch-model-bert.pt --bert_output_dir=bert-checkpoint --bert_use_doc_context --bert_disable_lstm
$ cd data/conll2003; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..
INFO:__main__:[F1] : 0.9217056973086334, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 252364.78638648987ms, 68.47469475875883ms on average
accuracy:  98.39%; precision:  90.99%; recall:  93.38%; FB1:  92.17

** --bert_doc_context_option=2
# preprocessing
$ python preprocess.py --config=configs/config-bert.json --data_dir=data/conll2003 --bert_model_name_or_path=./embeddings/bert-large-cased --bert_use_doc_context --bert_doc_context_option=2 
INFO:__main__:[F1] : 0.9226704245199928, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 252395.40195465088ms, 68.4826407068746ms on average
accuracy:  98.47%; precision:  91.80%; recall:  92.74%; FB1:  92.27

* document context, --bert_subword_pooling --bert_word_embedding --use_crf --batch_size=8, n_ctx: 512, --epoch=30
# preprocessing
$ python preprocess.py --config=configs/config-bert.json --data_dir=data/conll2003 --bert_model_name_or_path=./embeddings/bert-large-cased --bert_use_doc_context --bert_use_subword_pooling --bert_use_word_embedding
# train
$ python train.py --config=configs/config-bert.json --data_dir=data/conll2003 --save_path=pytorch-model-bert.pt --bert_model_name_or_path=./embeddings/bert-large-cased --bert_output_dir=bert-checkpoint --batch_size=8 --lr=1e-5 --epoch=30 --bert_freezing_epoch=3 --bert_lr_during_freezing=1e-3 --use_crf --bert_use_doc_context --bert_use_subword_pooling --bert_use_word_embedding --eval_batch_size=32
INFO:__main__:[F1] : 0.9283378987364141, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 382980.3364276886ms, 103.93010136099117ms on average
accuracy:  98.48%; precision:  92.66%; recall:  93.01%; FB1:  92.83

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

* --bert_model_name_or_path=./embedings/bert-base-uncased --batch_size=32 --bert_freezing_epoch=3 --bert_lr_during_freezing=1e-3
INFO:__main__:[F1] : 0.906032584764421, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 84019.82283592224ms, 22.77578745657356ms on average
accuracy:  98.07%; precision:  90.13%; recall:  91.08%; FB1:  90.60

* --bert_model_name_or_path=./embedings/bert-base-uncased --batch_size=32 --epoch=20 --bert_freezing_epoch=7 --bert_lr_during_freezing=1e-3
INFO:__main__:[F1] : 0.9056670497745558, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 79642.83394813538ms, 21.5918773741234ms on average
accuracy:  98.06%; precision:  90.45%; recall:  90.69%; FB1:  90.57

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

* --data_dir=data/conll++ --bert_model_name_or_path=./embeddings/bert-base-NER  --warmup_epoch=0 --weight_decay=0.0 --epoch=20 --bert_freezing_epoch=3 --bert_lr_during_freezing=1e-3
INFO:__main__:[F1] : 0.9271244409365955, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 82041.87273979187ms, 22.243568633776334ms on average
accuracy:  98.53%; precision:  92.72%; recall:  92.70%; FB1:  92.71

* --data_dir=data/conll++_truecase --bert_model_name_or_path=./embeddings/bert-base-NER  --warmup_epoch=0 --weight_decay=0.0 --epoch=20 --bert_freezing_epoch=3 --bert_lr_during_freezing=1e-3
INFO:__main__:[F1] : 0.9315955213435969, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 83032.81283378601ms, 22.508826101729316ms on average
accuracy:  98.59%; precision:  92.93%; recall:  93.39%; FB1:  93.16

* --bert_model_name_or_path=./embedings/bert-base-cased --batch_size=32 --epoch=10 --use_crf
INFO:__main__:[F1] : 0.8986090455778146, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 160266.587972641ms, 43.480443009338465ms on average
accuracy:  97.95%; precision:  89.42%; recall:  90.93%; FB1:  90.17

* --bert_model_name_or_path=./embedings/bert-base-cased --batch_size=32 --epoch=10 --bert_freezing_epoch=3 --bert_lr_during_freezing=1e-3 --use_crf
INFO:__main__:[F1] : 0.9135758963967932, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 155832.32188224792ms, 42.27091953413272ms on average
accuracy:  98.21%; precision:  91.30%; recall:  91.80%; FB1:  91.55

* --bert_model_name_or_path=./embedings/bert-base-cased --bert_disable_lstm --epoch=30
INFO:__main__:[F1] : 0.914522382798731, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 167704.35976982117ms, 45.50245252925677ms on average
accuracy:  98.34%; precision:  91.04%; recall:  91.87%; FB1:  91.45

* using sub token label, --bert_use_sub_label --use_crf
# preprocessing
$ python preprocess.py --config=configs/config-bert.json --data_dir=data/conll2003 --bert_model_name_or_path=./embeddings/bert-base-cased --bert_use_sub_label
# train
$ python train.py --config=configs/config-bert.json --data_dir=data/conll2003 --save_path=pytorch-model-bert.pt --bert_model_name_or_path=./embeddings/bert-base-cased --bert_output_dir=bert-checkpoint --batch_size=32 --lr=1e-5 --epoch=10 --bert_freezing_epoch=3 --bert_lr_during_freezing=1e-3 --use_crf
# evaluate
$ python evaluate.py --config=configs/config-bert.json --data_dir=data/conll2003 --model_path=pytorch-model-bert.pt --bert_output_dir=bert-checkpoint --use_crf
$ cd data/conll2003; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..
INFO:__main__:[F1] : 0.9128322882628279, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 146032.53650665283ms, 39.613566708053355ms on average
accuracy:  98.26%; precision:  91.59%; recall:  91.61%; FB1:  91.60

* using sub token label, --bert_use_sub_label + --bert_use_pos --use_char_cnn --use_crf
# preprocessing
$ python preprocess.py --config=configs/config-bert.json --data_dir=data/conll2003 --bert_model_name_or_path=./embeddings/bert-base-cased --bert_use_sub_label
# train
$ python train.py --config=configs/config-bert.json --data_dir=data/conll2003 --save_path=pytorch-model-bert.pt --bert_model_name_or_path=./embeddings/bert-base-cased --bert_output_dir=bert-checkpoint --batch_size=32 --lr=1e-5 --epoch=10 --bert_freezing_epoch=3 --bert_lr_during_freezing=1e-3 --use_crf --bert_use_pos --use_char_cnn
# evaluate
$ python evaluate.py --config=configs/config-bert.json --data_dir=data/conll2003 --model_path=pytorch-model-bert.pt --bert_output_dir=bert-checkpoint --use_crf --bert_use_pos --use_char_cnn
$ cd data/conll2003; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..
INFO:__main__:[F1] : 0.9113209212035649, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 148480.65185546875ms, 40.282283112823464ms on average
accuracy:  98.27%; precision:  91.23%; recall:  91.52%; FB1:  91.37

** using sub token label, --bert_use_sub_label + --bert_use_pos --use_char_cnn --epoch=30 --use_crf
INFO:__main__:[F1] : 0.9142604856512141, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 146215.95406532288ms, 39.665775003881194ms on average
accuracy:  98.32%; precision:  91.57%; recall:  91.75%; FB1:  91.66

* slicing logits to remain first token's of word's before applying crf, --use_crf --bert_use_crf_slice 
# https://github.com/dsindex/ntagger/releases/tag/v1.0
# preprocessing
$ python preprocess.py --config=configs/config-bert.json --data_dir=data/conll2003 --bert_model_name_or_path=./embeddings/bert-base-cased
# train
$ python train.py --config=configs/config-bert.json --data_dir=data/conll2003 --save_path=pytorch-model-bert.pt --bert_model_name_or_path=./embeddings/bert-base-cased --bert_output_dir=bert-checkpoint --batch_size=32 --lr=1e-5 --epoch=10 --bert_freezing_epoch=3 --bert_lr_during_freezing=1e-3 --use_crf --bert_use_crf_slice
# evaluate
$ python evaluate.py --config=configs/config-bert.json --data_dir=data/conll2003 --model_path=pytorch-model-bert.pt --bert_output_dir=bert-checkpoint --use_crf --bert_use_crf_slice
$ cd data/conll2003; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..
INFO:__main__:[F1] : 0.913277459197177, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 151587.14032173157ms, 41.12043155459907ms on average
accuracy:  98.26%; precision:  91.01%; recall:  91.64%; FB1:  91.33

* subword pooling, --use_crf
# preprocessing
$ python preprocess.py --config=configs/config-bert.json --data_dir=data/conll2003 --bert_model_name_or_path=./embeddings/bert-base-cased --bert_use_subword_pooling
# train
$ python train.py --config=configs/config-bert.json --data_dir=data/conll2003 --save_path=pytorch-model-bert.pt --bert_model_name_or_path=./embeddings/bert-base-cased --bert_output_dir=bert-checkpoint --batch_size=32 --lr=1e-5 --epoch=10 --bert_freezing_epoch=3 --bert_lr_during_freezing=1e-3 --use_crf --bert_use_subword_pooling
# evaluate
$ python evaluate.py --config=configs/config-bert.json --data_dir=data/conll2003 --model_path=pytorch-model-bert.pt --bert_output_dir=bert-checkpoint --use_crf --bert_use_subword_pooling
INFO:__main__:[F1] : 0.9132734003172924, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 148011.45482063293ms, 40.14931384098779ms on average
accuracy:  98.26%; precision:  90.93%; recall:  91.73%; FB1:  91.33

* subword pooling, word embedding, --use_crf
# preprocessing
$ python preprocess.py --config=configs/config-bert.json --data_dir=data/conll2003 --bert_model_name_or_path=./embeddings/bert-base-cased --bert_use_subword_pooling --bert_use_word_embedding
# train
$ python train.py --config=configs/config-bert.json --data_dir=data/conll2003 --save_path=pytorch-model-bert.pt --bert_model_name_or_path=./embeddings/bert-base-cased --bert_output_dir=bert-checkpoint --batch_size=32 --lr=1e-5 --epoch=10 --bert_freezing_epoch=3 --bert_lr_during_freezing=1e-3 --use_crf --bert_use_subword_pooling --bert_use_word_embedding
# evaluate
$ python evaluate.py --config=configs/config-bert.json --data_dir=data/conll2003 --model_path=pytorch-model-bert.pt --bert_output_dir=bert-checkpoint --use_crf --bert_use_subword_pooling --bert_use_word_embedding
INFO:__main__:[F1] : 0.9231446430143498, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 150338.47284317017ms, 40.779396660975905ms on average
accuracy:  98.33%; precision:  91.81%; recall:  92.83%; FB1:  92.31

* subword pooling, word/pos/char embedding, --use_char_cnn --bert_use_pos --use_crf
INFO:__main__:[F1] : 0.920799929521628, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 151703.0575275421ms, 41.146695921351785ms on average
accuracy:  98.34%; precision:  91.64%; recall:  92.53%; FB1:  92.08

* subword pooling, --bert_disable_lstm, --batch_size=16 --epoch=30
INFO:__main__:[F1] : 0.9137430757056186, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 170345.35002708435ms, 46.214719198010975ms on average
accuracy:  98.27%; precision:  90.76%; recall:  92.00%; FB1:  91.37

* subword pooling, word embedding, --bert_disable_lstm, --batch_size=16 --epoch=30
INFO:__main__:[F1] : 0.9120337790288531, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 122857.85698890686ms, 33.327639701733474ms on average
accuracy:  98.28%; precision:  90.63%; recall:  91.78%; FB1:  91.20

* document context, --batch_size=16, n_ctx: 512, --epoch=30 --use_crf
# preprocessing
$ python preprocess.py --config=configs/config-bert.json --data_dir=data/conll2003 --bert_model_name_or_path=./embeddings/bert-base-cased --bert_use_doc_context
# train
$ python train.py --config=configs/config-bert.json --data_dir=data/conll2003 --save_path=pytorch-model-bert.pt --bert_model_name_or_path=./embeddings/bert-base-cased --bert_output_dir=bert-checkpoint --batch_size=16 --lr=1e-5 --epoch=30 --bert_freezing_epoch=3 --bert_lr_during_freezing=1e-3 --use_crf --bert_use_doc_context
# evaluate
$ python evaluate.py --config=configs/config-bert.json --data_dir=data/conll2003 --model_path=pytorch-model-bert.pt --bert_output_dir=bert-checkpoint --use_crf --bert_use_doc_context
$ cd data/conll2003; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..
INFO:__main__:[F1] : 0.9169225370646549, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 342276.90076828003ms, 92.88276947616009ms on average
accuracy:  98.23%; precision:  91.56%; recall:  92.53%; FB1:  92.04

* document context, subword pooling, --batch_size=16, n_ctx: 512, --epoch=30 --use_crf
$ python train.py --config=configs/config-bert.json --data_dir=data/conll2003 --save_path=pytorch-model-bert.pt --bert_model_name_or_path=./embeddings/bert-base-cased --bert_output_dir=bert-checkpoint --batch_size=8 --lr=1e-5 --epoch=30 --bert_freezing_epoch=3 --bert_lr_during_freezing=1e-3 --use_crf --bert_use_doc_context --bert_use_subword_pooling
INFO:__main__:[F1] : 0.920799929521628, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 299821.12765312195ms, 81.3581446348051ms on average
accuracy:  98.30%; precision:  91.64%; recall:  92.53%; FB1:  92.08

* document context, subword pooling, word embedding, --batch_size=8 --lr=1e-5, n_ctx: 512, --epoch=30 --use_crf
$ python train.py --config=configs/config-bert.json --data_dir=data/conll2003 --save_path=pytorch-model-bert.pt --bert_model_name_or_path=./embeddings/bert-base-cased --bert_output_dir=bert-checkpoint --batch_size=8 --lr=1e-5 --epoch=30 --bert_freezing_epoch=3 --bert_lr_during_freezing=1e-3 --use_crf --bert_use_doc_context --bert_use_subword_pooling --bert_use_word_embedding 
INFO:__main__:[F1] : 0.9285462244177841, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 295369.29631233215ms, 80.1447755837939ms on average
accuracy:  98.47%; precision:  92.53%; recall:  93.18%; FB1:  92.85
              LOC: precision:  93.84%; recall:  94.06%; FB1:  93.95  1672
             MISC: precision:  80.17%; recall:  82.34%; FB1:  81.24  721
              ORG: precision:  91.42%; recall:  92.41%; FB1:  91.92  1679
              PER: precision:  97.83%; recall:  97.77%; FB1:  97.80  1616

* document context, --bert_disable_lstm, --batch_size=16 --lr=2e-5, n_ctx: 512, --epoch=30
$ python train.py --config=configs/config-bert.json --data_dir=data/conll2003 --save_path=pytorch-model-bert.pt --bert_model_name_or_path=./embeddings/bert-base-cased --bert_output_dir=bert-checkpoint --batch_size=16 --lr=2e-5 --epoch=30 --bert_use_doc_context --bert_disable_lstm
INFO:__main__:[F1] : 0.9176470588235294, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 170834.2945575714ms, 46.35193099996301ms on average
accuracy:  98.27%; precision:  91.01%; recall:  92.53%; FB1:  91.76

** --lr=1e-5
INFO:__main__:[F1] : 0.9184963990865976, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 117725.76570510864ms, 31.933620964545387ms on average
accuracy:  98.35%; precision:  91.13%; recall:  92.58%; FB1:  91.85

** --bert_doc_context_option=2
# preprocessing
$ python preprocess.py --config=configs/config-bert.json --data_dir=data/conll2003 --bert_model_name_or_path=./embeddings/bert-base-cased --bert_use_doc_context --bert_doc_context_option=2 
INFO:__main__:[F1] : 0.9234554881799807, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 120362.42485046387ms, 32.65242268461071ms on average
accuracy:  98.40%; precision:  91.68%; recall:  93.02%; FB1:  92.35
              LOC: precision:  92.84%; recall:  94.00%; FB1:  93.42  1689
             MISC: precision:  80.44%; recall:  83.19%; FB1:  81.79  726
              ORG: precision:  89.92%; recall:  92.35%; FB1:  91.12  1706
              PER: precision:  97.39%; recall:  96.97%; FB1:  97.18  1610

** --bert_doc_context_option=2, try again!
INFO:__main__:[F1] : 0.9232660228270413, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 101272.75466918945ms, 27.464767011196564ms on average
accuracy:  98.44%; precision:  91.57%; recall:  93.09%; FB1:  92.33

* document context, subword pooling, --bert_disable_lstm, --batch_size=16 --lr=2e-5, n_ctx: 512, --epoch=30
$ python train.py --config=configs/config-bert.json --data_dir=data/conll2003 --save_path=pytorch-model-bert.pt --bert_model_name_or_path=./embeddings/bert-base-cased --bert_output_dir=bert-checkpoint --batch_size=16 --lr=2e-5 --epoch=30 --bert_use_doc_context --bert_use_subword_pooling --bert_disable_lstm
INFO:__main__:[F1] : 0.9223207232511191, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 172187.91961669922ms, 46.71668637358815ms on average
accuracy:  98.41%; precision:  91.45%; recall:  93.02%; FB1:  92.23

** --bert_doc_context_option=2
# preprocessing
$ python preprocess.py --config=configs/config-bert.json --data_dir=data/conll2003 --bert_model_name_or_path=./embeddings/bert-base-cased --bert_use_doc_context --bert_use_subword_pooling --bert_doc_context_option=2 
INFO:__main__:[F1] : 0.920534458509142, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 134746.63543701172ms, 36.54770698360927ms on average
accuracy:  98.35%; precision:  91.41%; recall:  92.71%; FB1:  92.05

* document context, subword pooling, word embedding, --bert_disable_lstm, --batch_size=16 --lr=1e-5, n_ctx: 512, --epoch=30
$ python train.py --config=configs/config-bert.json --data_dir=data/conll2003 --save_path=pytorch-model-bert.pt --bert_model_name_or_path=./embeddings/bert-base-cased --bert_output_dir=bert-checkpoint --batch_size=16 --lr=2e-5 --epoch=30 --bert_use_doc_context --bert_use_subword_pooling --bert_use_word_embedding --bert_disable_lstm
INFO:__main__:[F1] : 0.9215065117916227, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 171518.04447174072ms, 46.53535477013941ms on average
accuracy:  98.44%; precision:  91.60%; recall:  92.71%; FB1:  92.15

** --bert_doc_context_option=2 --lr=1e-5
INFO:__main__:[F1] : 0.9202453987730062, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 173408.38193893433ms, 47.04242836286476ms on average
accuracy:  98.36%; precision:  91.11%; recall:  92.95%; FB1:  92.02

* --bert_model_name_or_path=./embedings/bert-base-cased --batch_size=32 --epoch=10
INFO:__main__:[F1] : 0.9020433219328247, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 79621.61326408386ms, 21.58440276070066ms on average
accuracy:  98.00%; precision:  89.37%; recall:  91.06%; FB1:  90.20

* --bert_model_name_or_path=./embedings/bert-base-cased --batch_size=32 --epoch=10 --bert_freezing_epoch=3 --bert_lr_during_freezing=1e-3
INFO:__main__:[F1] : 0.9098765432098764, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 80155.51614761353ms, 21.732826838828675ms on average
accuracy:  98.24%; precision:  90.64%; recall:  91.34%; FB1:  90.99

* --bert_model_name_or_path=./embedings/bert-base-cased --batch_size=32 --epoch=10 --bert_freezing_epoch=3 --bert_lr_during_freezing=1e-3 --use_mha
INFO:__main__:[F1] : 0.9095074455899198, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 81080.08480072021ms, 21.98459193850871ms on average
accuracy:  98.24%; precision:  90.53%; recall:  91.38%; FB1:  90.95

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

* --config=configs/config-distilbert.json --bert_model_name_or_path=./embeddings/distilbert-base-multilingual-cased 
INFO:__main__:[F1] : 0.9021052631578947, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 49763.64517211914ms, 13.491093125189254ms on average
accuracy:  98.02%; precision:  89.39%; recall:  91.04%; FB1:  90.21

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

* --bert_model_name_or_path=./embeddings/bert-large-cased-finetuned-conll03-english --warmup_epoch=0 --weight_decay=0.0 --epoch=30
INFO:__main__:[F1] : 0.9187279151943464, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 118809.88335609436ms, 32.22117830546304ms on average
accuracy:  98.44%; precision:  91.68%; recall:  92.07%; FB1:  91.87

* --data_dir=data/conll2003_truecase --bert_model_name_or_path=./embeddings/bert-large-cased-finetuned-conll03-english --warmup_epoch=0 --weight_decay=0.0 --epoch=30
INFO:__main__:[F1] : 0.9239971850809289, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 114439.17107582092ms, 31.03302174406857ms on average
accuracy:  98.44%; precision:  91.82%; recall:  92.99%; FB1:  92.40

* --bert_model_name_or_path=./embeddings/bert-large-cased-finetuned-conll03-english --warmup_epoch=0 --weight_decay=0.0 --epoch=30 --bert_disable_lstm
INFO:__main__:[F1] : 0.9163143058491896, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 109043.2801246643ms, 29.56439605265443ms on average
accuracy:  98.35%; precision:  91.18%; recall:  92.09%; FB1:  91.63

* --data_dir=data/conll2003_truecase --bert_model_name_or_path=./embeddings/bert-large-cased-finetuned-conll03-english --warmup_epoch=0 --weight_decay=0.0 --epoch=30 --bert_disable_lstm
INFO:__main__:[F1] : 0.9224137931034484, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 110067.8985118866ms, 29.847684645827556ms on average
accuracy:  98.41%; precision:  91.66%; recall:  92.83%; FB1:  92.24

* --data_dir=data/conll++ --bert_model_name_or_path=./embeddings/bert-large-cased-finetuned-conll03-english --warmup_epoch=0 --weight_decay=0.0 --epoch=30 --bert_freezing_epoch=3 --bert_lr_during_freezing=1e-3
INFO:__main__:[F1] : 0.926168632590651, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 132705.81126213074ms, 35.98520220410329ms on average
accuracy:  98.52%; precision:  92.29%; recall:  92.95%; FB1:  92.62

* --data_dir=data/conll++_truecase --bert_model_name_or_path=./embeddings/bert-large-cased-finetuned-conll03-english --warmup_epoch=0 --weight_decay=0.0 --epoch=30 --bert_freezing_epoch=3 --bert_lr_during_freezing=1e-3
INFO:__main__:[F1] : 0.9336480461578809, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 132453.9577960968ms, 35.92103879930143ms on average
accuracy:  98.63%; precision:  93.08%; recall:  93.65%; FB1:  93.36

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
F
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

* --bert_model_name_or_path=./embeddings/xlm-roberta-base --bert_disable_lstm --batch_size=32 --epoch=30 --patience=4
INFO:__main__:[F1] : 0.912046393111326, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 69937.09135055542ms, 18.960459090560924ms on average
accuracy:  98.21%; precision:  90.53%; recall:  91.89%; FB1:  91.20

* --bert_model_name_or_path=./embeddings/xlm-roberta-base --bert_freezing_epoch=3 --bert_lr_during_freezing=1e-3 --batch_size=32 --epoch=30 --patience=4
INFO:__main__:[F1] : 0.9081166549543219, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 79163.73658180237ms, 21.466715388281205ms on average
accuracy:  98.15%; precision:  90.12%; recall:  91.52%; FB1:  90.81

* --bert_model_name_or_path=./embeddings/xlm-roberta-base --use_crf --bert_freezing_epoch=3 --bert_lr_during_freezing=1e-3 --batch_size=32 --epoch=30 --patience=4
INFO:__main__:[F1] : 0.9111501316944689, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 145396.85487747192ms, 39.44182609560177ms on average
accuracy:  98.17%; precision:  90.37%; recall:  91.87%; FB1:  91.12

* using sub token label, --bert_use_sub_label
# preprocessing
$ python preprocess.py --config=configs/config-roberta.json --data_dir=data/conll2003 --bert_model_name_or_path=./embeddings/xlm-roberta-base --bert_use_sub_label
# --bert_model_name_or_path=./embeddings/xlm-roberta-base --use_crf --bert_freezing_epoch=3 --bert_lr_during_freezing=1e-3 --batch_size=32 --epoch=30 --patience=4
INFO:__main__:[F1] : 0.9172601776136463, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 158769.8678970337ms, 43.06626947813632ms on average
accuracy:  98.28%; precision:  91.24%; recall:  92.35%; FB1:  91.79

* slicing logits, --bert_use_crf_slice
# https://github.com/dsindex/ntagger/releases/tag/v1.0
# train
$ python train.py --config=configs/config-roberta.json --data_dir=data/conll2003 --save_path=pytorch-model-roberta.pt --bert_model_name_or_path=./embeddings/xlm-roberta-base --bert_output_dir=bert-checkpoint-roberta --batch_size=32 --lr=1e-5  --epoch=30 --patience=4 --bert_freezing_epoch=3 --bert_lr_during_freezing=1e-3 --use_crf --bert_use_crf_slice
INFO:__main__:[F1] : 0.911604155661208, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 145113.0769252777ms, 39.364287298463005ms on average
accuracy:  98.20%; precision:  90.67%; recall:  91.66%; FB1:  91.16

* --bert_model_name_or_path=./embeddings/xlm-roberta-large --bert_disable_lstm
INFO:__main__:[F1] : 0.927465220054248, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 111304.85033988953ms, 30.18471685208091ms on average
INFO:__main__:[Elapsed Time] : 100 examples, 2893.0163383483887ms, 27.91449758741591ms on average
accuracy:  98.47%; precision:  91.68%; recall:  93.84%; FB1:  92.75

* --bert_model_name_or_path=./embeddings/xlm-roberta-large --data_dir=data/conll2003_truecase --bert_disable_lstm
INFO:__main__:[F1] : 0.928909952606635, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 112505.32031059265ms, 30.506469942518073ms on average
INFO:__main__:[Elapsed Time] : 100 examples, 2932.1699142456055ms, 28.066835018119427ms on average
accuracy:  98.48%; precision:  92.10%; recall:  93.70%; FB1:  92.89

* subword pooling, word embedding
# preprocessing
$ python preprocess.py --config=configs/config-roberta.json --data_dir=data/conll2003 --bert_model_name_or_path=./embeddings/xlm-roberta-large --bert_use_subword_pooling --bert_use_word_embedding
# train
$ python train.py --config=configs/config-roberta.json --data_dir=data/conll2003 --save_path=pytorch-model-roberta.pt --bert_model_name_or_path=./embeddings/xlm-roberta-large --bert_output_dir=bert-checkpoint-roberta --batch_size=32 --lr=1e-5  --epoch=30 --patience=4 --bert_freezing_epoch=3 --bert_lr_during_freezing=1e-3 --use_crf --bert_use_subword_pooling --bert_use_word_embedding
# evaluate
$ python evaluate.py --config=configs/config-roberta.json --data_dir=data/conll2003 --model_path=pytorch-model-roberta.pt --bert_output_dir=bert-checkpoint-roberta --use_crf --bert_use_subword_pooling --bert_use_word_embedding
INFO:__main__:[F1] : 0.9296765119549929, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 194712.32175827026ms, 52.813371497264804ms on average
accuracy:  98.44%; precision:  92.32%; recall:  93.63%; FB1:  92.97

* --data_dir=data/conll++ --bert_model_name_or_path=./embeddings/xlm-roberta-large --bert_disable_lstm
INFO:__main__:[F1] : 0.9394968224949942, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 123247.82752990723ms, 33.42233563841862ms on average
accuracy:  98.69%; precision:  93.28%; recall:  94.63%; FB1:  93.95

* --data_dir=data/conll++_truecase --bert_model_name_or_path=./embeddings/xlm-roberta-large --bert_disable_lstm
INFO:__main__:[F1] : 0.941063308849248, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 121127.03013420105ms, 32.838388593695974ms on average
accuracy:  98.75%; precision:  93.84%; recall:  94.37%; FB1:  94.11

* --data_dir=data/conll++ --bert_model_name_or_path=./embeddings/xlm-roberta-large --bert_freezing_epoch=3 --bert_lr_during_freezing=1e-3
INFO:__main__:[F1] : 0.9374890867819102, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 127180.99355697632ms, 34.48944825890464ms on average
accuracy:  98.69%; precision:  93.34%; recall:  94.16%; FB1:  93.75

* --data_dir=data/conll++_truecase --bert_model_name_or_path=./embeddings/xlm-roberta-large --bert_freezing_epoch=3 --bert_lr_during_freezing=1e-3
INFO:__main__:[F1] : 0.9381488266596877, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 127575.44207572937ms, 34.58970893571922ms on average
accuracy:  98.65%; precision:  93.33%; recall:  94.30%; FB1:  93.81

* document context, --bert_disable_lstm --batch_size=8, n_ctx: 512, --epoch=30
# preprocessing
$ python preprocess.py --config=configs/config-roberta.json --data_dir=data/conll2003 --bert_model_name_or_path=./embeddings/xlm-roberta-large --bert_use_doc_context --bert_doc_context_option=2
# train
$ python train.py --config=configs/config-roberta.json --data_dir=data/conll2003 --save_path=pytorch-model-roberta.pt --bert_model_name_or_path=./embeddings/xlm-roberta-large --bert_output_dir=bert-checkpoint-roberta --batch_size=8 --lr=1e-5 --epoch=30 --bert_use_doc_context --bert_disable_lstm  --eval_batch_size=32
# evaluate
$ python evaluate.py --config=configs/config-roberta.json --data_dir=data/conll2003 --model_path=pytorch-model-roberta.pt --bert_output_dir=bert-checkpoint-roberta --bert_use_doc_context --bert_disable_lstm
$ cd data/conll2003; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..
INFO:__main__:[F1] : 0.938635161347205, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 185295.64094543457ms, 50.268435588259756ms on average
accuracy:  98.73%; precision:  93.48%; recall:  94.25%; FB1:  93.86
              LOC: precision:  95.44%; recall:  94.12%; FB1:  94.78  1645
             MISC: precision:  82.48%; recall:  87.18%; FB1:  84.76  742
              ORG: precision:  92.06%; recall:  93.56%; FB1:  92.80  1688
              PER: precision:  98.02%; recall:  98.14%; FB1:  98.08  1619

* document context, subword pooling, word embedding, --batch_size=8, n_ctx: 512, --epoch=30 --use_crf
# preprocessing
$ python preprocess.py --config=configs/config-roberta.json --data_dir=data/conll2003 --bert_model_name_or_path=./embeddings/xlm-roberta-large --bert_use_doc_context --bert_doc_context_option=2 --bert_use_subword_pooling --bert_use_word_embedding
# train
$ python train.py --config=configs/config-roberta.json --data_dir=data/conll2003 --save_path=pytorch-model-roberta.pt --bert_model_name_or_path=./embeddings/xlm-roberta-large --bert_output_dir=bert-checkpoint-roberta --batch_size=8 --lr=1e-5 --epoch=30 --bert_use_doc_context --bert_use_subword_pooling --bert_use_word_embedding --use_crf --eval_batch_size=32 --patience=4 --bert_freezing_epoch=1 --bert_lr_during_freezing=1e-3 --eval_and_save_steps=1000
# evaluate
$ python evaluate.py --config=configs/config-roberta.json --data_dir=data/conll2003 --model_path=pytorch-model-roberta.pt --bert_output_dir=bert-checkpoint-roberta --bert_use_doc_context --bert_use_subword_pooling --bert_use_word_embedding --use_crf
$ cd data/conll2003; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..
INFO:__main__:[F1] : 0.9358929200422684, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 387618.5405254364ms, 105.18818713386779ms on average
accuracy:  98.61%; precision:  93.10%; recall:  94.09%; FB1:  93.59

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

<details><summary><b>emb_class=deberta, enc_class=bilstm</b></summary>
<p>

- train
```
* share config-bert.json
* n_ctx size should be less than 512
$ python preprocess.py --config=configs/config-bert.json --data_dir=data/conll2003 --bert_model_name_or_path=./embeddings/deberta-base
$ python train.py --config=configs/config-bert.json --data_dir=data/conll2003 --save_path=pytorch-model-deberta.pt --bert_model_name_or_path=./embeddings/deberta-base --bert_output_dir=bert-checkpoint-deberta --batch_size=32 --lr=1e-5 --epoch=20 --bert_disable_lstm
```

- evaluation
```
$ python evaluate.py --config=configs/config-bert.json --data_dir=data/conll2003 --model_path=pytorch-model-deberta.pt --bert_output_dir=bert-checkpoint-deberta --bert_disable_lstm
$ cd data/conll2003; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.9040613161835961, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 114359.83681678772ms, 31.012554386210383ms on average
accuracy:  98.10%; precision:  89.97%; recall:  90.85%; FB1:  90.41
INFO:__main__:[Elapsed Time] : 100 examples, 2987.0100021362305ms, 28.687477111816406ms on average

* --bert_model_name_or_path=./embeddings/deberta-large --batch_size=16 --gradient_accumulation_steps=2
INFO:__main__:[F1] : 0.9144771645212484, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 206926.0606765747ms, 56.128300501864594ms on average
INFO:__main__:[Elapsed Time] : 100 examples, 5544.259309768677ms, 53.92497717732131ms on average
accuracy:  98.32%; precision:  91.08%; recall:  91.82%; FB1:  91.45

* --use_sharded_ddp
# https://github.com/dsindex/ntagger/releases/tag/v1.1
$ export NCCL_DEBUG=INFO
$ python train.py --config=configs/config-bert.json --data_dir=data/conll2003 --save_path=pytorch-model-deberta.pt --bert_model_name_or_path=./embeddings/deberta-v2-xlarge/ --bert_output_dir=bert-checkpoint-deberta --batch_size=16 --lr=1e-5 --epoch=20 --bert_disable_lstm --gradient_accumulation_steps=2 --use_sharded_ddp --world_size=2 --master_port=8666
INFO:__main__:[F1] : 0.931234611326064, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 232110.55159568787ms, 62.972230753301766ms on average
accuracy:  98.57%; precision:  92.51%; recall:  93.75%; FB1:  93.12

* --use_sharded_ddp --use_fsdp
# https://github.com/dsindex/ntagger/releases/tag/v1.1
$ export NCCL_DEBUG=INFO
$ python train.py --config=configs/config-bert.json --data_dir=data/conll2003 --save_path=pytorch-model-deberta.pt --bert_model_name_or_path=./embeddings/deberta-v2-xlarge/ --bert_output_dir=bert-checkpoint-deberta --batch_size=16 --lr=1e-5 --epoch=20 --bert_disable_lstm --gradient_accumulation_steps=2 --use_sharded_ddp --use_fsdp --world_size=2 --master_port=5176
# 0 accuracy! 
how to load the saved model from FSDP.state_dict()?

* --use_sharded_ddp, document context, --bert_disable_lstm --batch_size=8, n_ctx: 512, --epoch=20
# https://github.com/dsindex/ntagger/releases/tag/v1.1
# preprocessing
$ python preprocess.py --config=configs/config-bert.json --data_dir=data/conll2003 --bert_model_name_or_path=./embeddings/deberta-v2-xlarge --bert_use_doc_context --bert_doc_context_option=2
# train
$ python train.py --config=configs/config-bert.json --data_dir=data/conll2003 --save_path=pytorch-model-deberta.pt --bert_model_name_or_path=./embeddings/deberta-v2-xlarge --bert_output_dir=bert-checkpoint-deberta --batch_size=4 --gradient_accumulation_steps=4 --lr=1e-5 --epoch=20 --bert_use_doc_context --bert_disable_lstm  --eval_batch_size=16 --use_sharded_ddp --world_size=2 --master_port=20630
# evaluate
$ python evaluate.py --config=configs/config-bert.json --data_dir=data/conll2003 --model_path=pytorch-model-deberta.pt --bert_output_dir=bert-checkpoint-deberta --bert_use_doc_context --bert_disable_lstm
$ cd data/conll2003; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..
INFO:__main__:[F1] : 0.9400386847195358, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 578826.6246318817ms, 157.07525959682957ms on average
accuracy:  98.73%; precision:  93.36%; recall:  94.65%; FB1:  94.00
              LOC: precision:  94.47%; recall:  95.26%; FB1:  94.87  1682
             MISC: precision:  80.48%; recall:  86.32%; FB1:  83.30  753
              ORG: precision:  92.81%; recall:  94.10%; FB1:  93.45  1684
              PER: precision:  98.82%; recall:  98.21%; FB1:  98.51  1607

* accelerate & torch.distributed.launch
# 1 node, 2 gpu
$ export NCCL_DEBUG=INFO
$ python -m torch.distributed.launch --nnodes 1 --nproc_per_node 2 --use_env --node_rank 0 --master_addr 10.55.14.139 --master_port 3667 train.py --config=configs/config-bert.json --data_dir=data/conll2003 --save_path=pytorch-model-deberta.pt --bert_model_name_or_path=./embeddings/deberta-v2-xlarge/ --bert_output_dir=bert-checkpoint-deberta --batch_size=8 --lr=1e-5 --epoch=20 --bert_disable_lstm --gradient_accumulation_steps=4 --eval_batch_size=4
INFO:__main__:[F1] : 0.9230634266163698, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 233286.6771221161ms, 63.29245888406804ms on average

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

* --data_dir=data/conll++ --batch_size=64
INFO:__main__:[F1] : 0.9349278930706999, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 240186.9957447052ms, 65.17468371405538ms on average
accuracy:  98.59%; precision:  93.76%; recall:  93.23%; FB1:  93.49

* --data_dir=data/conll++_truecase --batch_size=64
INFO:__main__:[F1] : 0.9368402533427164, 3684
INFO:__main__:[Elapsed Time] : 3684 examples, 245365.75412750244ms, 66.58202281892866ms on average
accuracy:  98.64%; precision:  93.98%; recall:  93.39%; FB1:  93.68

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

|                                          | F1 (%)      | Features | GPU / CPU      | CONDA    | Dynamic   | Etc    |
| ---------------------------------------- | ------------| -------- | -------------- | -------- | --------- | ------ |    
| bpe DistilBERT(v1)                       | 85.30       | eoj      | 9.0702  / -    |          |           |        |
| wp  DistilBERT(v1)                       | 84.45       | eoj      | 8.9646  / -    |          |           |        |
| mDistilBERT                              | 83.89       | eoj      | 9.2205  / -    |          |           |        |
| bpe BERT(v1), BiLSTM-CRF                 | 86.34       | eoj      | 46.9389 / -    |          |           |        |
| bpe BERT(v1), BiLSTM-CRF                 | 87.17       | eoj      | 39.1787 / -    |          |           | freezing BERT during some epochs |
| bpe BERT(v1), BiLSTM-CRF                 | 86.99       | eoj      | 39.0575 / -    |          |           | using sub token label, freezing BERT during some epochs |
| bpe BERT(v1), BiLSTM-CRF                 | 87.46       | eoj      | 39.6914 / -    |          |           | slicing logits, freezing BERT during some epochs, https://github.com/dsindex/ntagger/releases/tag/v1.0 |
| bpe BERT(v1), BiLSTM                     | 86.37       | eoj      | 21.3232 / -    |          |           |        |
| bpe BERT(v1), CRF                        | 86.42       | eoj      | 35.2222 / -    |          |           |        |
| bpe BERT(v1)                             | 87.13       | eoj      | 16.2121 / -    |          |           |        |
| bpe BERT-large(v1)                       | 85.99       | eoj      | 30.7513 / -    |          |           |        |
| bpe BERT-large(v1), BiLSTM               | 85.82       | eoj      | 32.0083 / -    |          |           | freezing BERT during some epochs |
| bpe BERT-large(v3)                       | 85.89       | eoj      | 27.4264 / -    |          |           |        |
| KcBERT-base, BiLSTM                      | 84.76       | eoj      | 15.0553 / -    |          |           |        |
| KcBERT-base, CRF                         | 83.32       | eoj      | 31.8019 / -    |          |           |        |
| KcBERT-base                              | 84.72       | eoj      | 13.3129 / -    |          |           |        |
| KcBERT-large                             | 86.34       | eoj      | 26.9639 / -    |          |           |        |
| KoELECTRA-Base-v1                        | 86.64       | eoj      | 15.1616 / -    |          |           |        |
| KoELECTRA-Base-v3                        | 87.31       | eoj      | 14.8115 / -    |          |           |        |
| KoELECTRA-Base-v3, BiLSTM-CRF            | 87.76       | eoj      | 40.4698 / -    |          |           | freezing BERT during some epochs |
| KoELECTRA-Base-v3, BiLSTM-CRF            | 87.32       | eoj      | 39.8039 / -    |          |           | using sub token label, freezing BERT during some epochs |
| KoELECTRA-Base-v3, BiLSTM-CRF            | **88.13**   | eoj      | 40.0855 / -    |          |           | slicing logits, freezing BERT during some epochs, https://github.com/dsindex/ntagger/releases/tag/v1.0 |
| KoELECTRA-Base-v3, BiLSTM-CRF            | 87.89       | eoj      | 47.0637 / -    |          |           | subword pooling                  |
| LM-KOR-ELECTRA                           | 87.39       | eoj      | 17.1545 / -    |          |           |        |
| LM-KOR-ELECTRA, BiLSTM-CRF               | 87.49       | eoj      | 39.7247 / -    |          |           | slicing logits, freezing BERT during some epochs, https://github.com/dsindex/ntagger/releases/tag/v1.0 |
| bpe ELECTRA-base(v1)                     | 86.46       | eoj      | 18.0449 / -    |          |           |        |
| dhaToken1.large ELECTRA-base, BiLSTM-CRF | 86.90       | eoj      | 44.3714 / -    |          |           | slicing logits, freezing BERT during some epochs, https://github.com/dsindex/ntagger/releases/tag/v1.0 |
| dhaSyllable ELECTRA-base, BiLSTM-CRF     | 86.31       | eoj      | 41.0562 / -    |          |           | subword pooling                  |
| RoBERTa-base                             | 85.45       | eoj      | 15.6986 / -    |          |           |        |
| XLM-RoBERTa-base                         | 86.84       | eoj      | 18.1326 / -    |          |           |        |
| XLM-RoBERTa-large                        | 87.01       | eoj      | 35.9521 / -    |          |           |        |
| Funnel-base                              | 87.97       | eoj      | 42.9287 / -    |          |           |        |
| Funnel-base, BiLSTM-CRF                  | 87.92       | eoj      | 83.9707 / -    |          |           | slicing logits, freezing BERT during some epochs, https://github.com/dsindex/ntagger/releases/tag/v1.0 |


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
| electra-kor-base         | 87.14         | eoj      |
| funnel-kor-base          | 88.02         | eoj      |

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


<details><summary><b>emb_class=bert, enc_class=bilstm, eoj-based</b></summary>
<p>

- train
```
* n_ctx size should be less than 512

* for clova2019

$ python preprocess.py --config=configs/config-bert.json --data_dir data/clova2019 --bert_model_name_or_path=./embeddings/kor-bert-base-bpe.v1
$ python train.py --config=configs/config-bert.json --save_path=pytorch-model-bert-kor-eoj.pt --bert_model_name_or_path=./embeddings/kor-bert-base-bpe.v1 --bert_output_dir=bert-checkpoint-kor-eoj --batch_size=32 --lr=5e-5 --epoch=20 --data_dir data/clova2019 --use_crf --eval_and_save_steps=1000

```

- evaluation
```

* for clova2019

$ python evaluate.py --config=configs/config-bert.json --model_path=pytorch-model-bert-kor-eoj.pt --data_dir data/clova2019 --bert_output_dir=bert-checkpoint-kor-eoj --use_crf
$ cd data/clova2019; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..
(bpe BERT BiLSTM-CRF)
INFO:__main__:[F1] : 0.863551528725101, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 409696.9916820526ms, 45.51278335383924ms on average
INFO:__main__:[Elapsed Time] : 100 examples, 4773.615837097168ms, 46.938994918206724ms on average
accuracy:  94.33%; precision:  86.17%; recall:  86.52%; FB1:  86.34

** --bert_freezing_epoch=4 --bert_lr_during_freezing=1e-3 --eval_and_save_steps=1000
INFO:__main__:[F1] : 0.872042113153546, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 368655.66992759705ms, 40.952059067226884ms on average
INFO:__main__:[Elapsed Time] : 100 examples, 4002.8138160705566ms, 39.17879769296357ms on average
accuracy:  94.64%; precision:  87.33%; recall:  87.02%; FB1:  87.17

** using sub token label, --bert_use_sub_label
*** --bert_freezing_epoch=4 --bert_lr_during_freezing=1e-3 --eval_and_save_steps=1000
INFO:__main__:[F1] : 0.867761275378117, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 371622.62415885925ms, 41.28106700962074ms on average
NFO:__main__:[Elapsed Time] : 100 examples, 3990.586280822754ms, 39.05752692559753ms on average
accuracy:  94.63%; precision:  87.15%; recall:  86.84%; FB1:  86.99

** slicing logits
*** --bert_use_crf_slice --bert_freezing_epoch=4 --bert_lr_during_freezing=1e-3 --eval_and_save_steps=1000
# https://github.com/dsindex/ntagger/releases/tag/v1.0
INFO:__main__:[F1] : 0.8755258093632384, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 378671.01287841797ms, 42.06319689313522ms on average
INFO:__main__:[Elapsed Time] : 100 examples, 4061.053991317749ms, 39.69146506954925ms on average
accuracy:  94.73%; precision:  87.50%; recall:  87.39%; FB1:  87.45

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

**  --bert_model_name_or_path=./embeddings/kor-bert-large-bpe.v1 --warmup_epoch=0 --weight_decay=0.0 --lr=1e-6 --epoch=30 --bert_freezing_epoch=4 --bert_lr_during_freezing=1e-3 , without --use_crf
INFO:__main__:[F1] : 0.8591666808561358, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 984067.9261684418ms, 109.33762731148357ms on average
INFO:__main__:[Elapsed Time] : 100 examples, 3314.924716949463ms, 32.00836855955798ms on average
accuracy:  94.17%; precision:  85.87%; recall:  85.77%; FB1:  85.82

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

** --config=configs/config-distilbert.json --bert_model_name_or_path=./embeddings/distilbert-base-multilingual-cased --bert_disable_lstm  --warmup_epoch=0 --weight_decay=0.0 --lr=8e-5 --gradient_accumulation_steps=2 --epoch=30 , without --use_crf
INFO:__main__:[F1] : 0.8407898796667697, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 998458.5220813751ms, 110.94100147581773ms on average
accuracy:  93.35%; precision:  84.41%; recall:  83.38%; FB1:  83.89
INFO:__main__:[Elapsed Time] : 100 examples, 1001.7592906951904ms, 9.220564004146691ms on average

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


<details><summary><b>emb_class=bert, enc_class=bilstm, morph-based</b></summary>
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


<details><summary><b>emb_class=electra/roberta, enc_class=bilstm </b></summary>
<p>

- train
```
* share config-bert.json if possible
* n_ctx size should be less than 512

* for clova2019

** KoELECTRA-Base
$ python preprocess.py --config=configs/config-bert.json --data_dir data/clova2019 --bert_model_name_or_path=./embeddings/koelectra-base-v1-discriminator
$ python train.py --config=configs/config-bert.json --save_path=pytorch-model-bert-kor-eoj.pt --bert_model_name_or_path=./embeddings/koelectra-base-v1-discriminator --bert_output_dir=bert-checkpoint-kor-eoj --batch_size=32 --lr=5e-5 --epoch=20 --data_dir data/clova2019 --bert_disable_lstm

** bpe ELECTRA-base(v1)
$ python preprocess.py --config=configs/config-bert.json --data_dir data/clova2019 --bert_model_name_or_path=./embeddings/kor-electra-base-bpe.v1
$ python train.py --config=configs/config-bert.json --save_path=pytorch-model-bert-kor-eoj.pt --bert_model_name_or_path=./embeddings/kor-electra-base-bpe.v1 --bert_output_dir=bert-checkpoint-kor-eoj --batch_size=32 --lr=8e-5 --epoch=30 --data_dir data/clova2019 --bert_disable_lstm  --warmup_epoch=0 --weight_decay=0.0 --gradient_accumulation_steps=2 

** dhaToken1.large ELECTRA-base
# https://github.com/dsindex/ntagger/releases/tag/v1.0
$ python preprocess.py --config=configs/config-bert.json --data_dir data/clova2019 --bert_model_name_or_path=./embeddings/kor-electra-base-dhaToken1.large
$ python train.py --config=configs/config-bert.json --save_path=pytorch-model-bert-kor-eoj.pt --bert_model_name_or_path=./embeddings/kor-electra-base-dhaToken1.large --bert_output_dir=bert-checkpoint-kor-eoj --batch_size=32 --lr=8e-5 --epoch=30 --data_dir data/clova2019 --warmup_epoch=0 --weight_decay=0.0 --use_crf --bert_use_crf_slice --bert_freezing_epoch=4 --bert_lr_during_freezing=1e-3 --eval_and_save_steps=1000 

** dhaSyllable ELECTRA-base
$ python preprocess.py --config=configs/config-bert.json --data_dir data/clova2019 --bert_model_name_or_path=./embeddings/kor-electra-base-dhaSyllable --bert_use_subword_pooling
$ python train.py --config=configs/config-bert.json --save_path=pytorch-model-bert-kor-eoj.pt --bert_model_name_or_path=./embeddings/kor-electra-base-dhaSyllable --bert_output_dir=bert-checkpoint-kor-eoj --batch_size=32 --lr=8e-5 --epoch=30 --data_dir data/clova2019 --warmup_epoch=0 --weight_decay=0.0 --use_crf --bert_use_subword_pooling --eval_and_save_steps=1000 

** LM-KOR-ELECTRA
$ python preprocess.py --config=configs/config-bert.json --data_dir data/clova2019 --bert_model_name_or_path=./embeddings/electra-kor-base
$ python train.py --config=configs/config-bert.json --save_path=pytorch-model-bert-kor-eoj.pt --bert_model_name_or_path=./embeddings/electra-kor-base --bert_output_dir=bert-checkpoint-kor-eoj --batch_size=32 --lr=8e-5 --epoch=30 --data_dir data/clova2019 --bert_disable_lstm  --warmup_epoch=0 --weight_decay=0.0 --gradient_accumulation_steps=2 

** RoBERTa-base
$ python preprocess.py --config=configs/config-roberta.json --data_dir data/clova2019 --bert_model_name_or_path=./embeddings/kor-roberta-base-bbpe
$ python train.py --config=configs/config-roberta.json --save_path=pytorch-model-bert-kor-eoj.pt --bert_model_name_or_path=./embeddings/kor-roberta-base-bbpe --bert_output_dir=bert-checkpoint-kor-eoj --batch_size=32 --lr=5e-5 --epoch=30 --data_dir data/clova2019 --bert_disable_lstm  --warmup_epoch=0 --weight_decay=0.0 

** XLM-RoBERTa-base, XLM-RoBERTa-large
$ python preprocess.py --config=configs/config-roberta.json --data_dir data/clova2019 --bert_model_name_or_path=./embeddings/xlm-roberta-base
$ python train.py --config=configs/config-roberta.json --save_path=pytorch-model-bert-kor-eoj.pt --bert_model_name_or_path=./embeddings/xlm-roberta-base --bert_output_dir=bert-checkpoint-kor-eoj --batch_size=32 --lr=5e-5 --epoch=30 --data_dir data/clova2019 --bert_disable_lstm  --warmup_epoch=0 --weight_decay=0.0 

** Funnel-base
$ python preprocess.py --config=configs/config-bert.json --data_dir data/clova2019 --bert_model_name_or_path=./embeddings/funnel-kor-base
$ python train.py --config=configs/config-bert.json --save_path=pytorch-model-bert-kor-eoj.pt --bert_model_name_or_path=./embeddings/funnel-kor-base --bert_output_dir=bert-checkpoint-kor-eoj --batch_size=32 --lr=5e-5 --epoch=30 --data_dir data/clova2019 --bert_disable_lstm  --warmup_epoch=0 --weight_decay=0.0 

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

*** --bert_model_name_or_path=./embeddings/koelectra-base-v3-discriminator --lr=8e-5 --epoch=30 --use_crf --bert_freezing_epoch=4 --bert_lr_during_freezing=1e-3 --eval_and_save_steps=1000 , without --bert_disable_lstm
INFO:__main__:[F1] : 0.8784531327126347, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 377629.4982433319ms, 41.94557212408231ms on average
INFO:__main__:[Elapsed Time] : 100 examples, 4139.477491378784ms, 40.46985838148329ms on average
accuracy:  94.84%; precision:  87.78%; recall:  87.73%; FB1:  87.76

*** using sub token label, --bert_use_sub_label
**** --bert_model_name_or_path=./embeddings/koelectra-base-v3-discriminator --lr=8e-5 --epoch=30 --use_crf --bert_freezing_epoch=4 --bert_lr_during_freezing=1e-3 --eval_and_save_steps=1000 , without --bert_disable_lstm
INFO:__main__:[F1] : 0.874318247875669, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 375233.0663204193ms, 41.67938974250991ms on average
INFO:__main__:[Elapsed Time] : 100 examples, 4068.805932998657ms, 39.803909532951586ms on average
accuracy:  94.70%; precision:  87.71%; recall:  86.93%; FB1:  87.32

*** slicing logits
**** --bert_model_name_or_path=./embeddings/koelectra-base-v3-discriminator --lr=8e-5 --epoch=30 --use_crf --bert_use_crf_slice --bert_freezing_epoch=4 --bert_lr_during_freezing=1e-3 --eval_and_save_steps=1000 , without --bert_disable_lstm
# https://github.com/dsindex/ntagger/releases/tag/v1.0
INFO:__main__:[F1] : 0.8827849438546868, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 398727.7433872223ms, 44.294040061882015ms on average
INFO:__main__:[Elapsed Time] : 100 examples, 4092.3564434051514ms, 40.08559265522042ms on average
accuracy:  94.92%; precision:  88.19%; recall:  88.07%; FB1:  88.13

*** subword pooling
**** --bert_model_name_or_path=./embeddings/koelectra-base-v3-discriminator --lr=8e-5 --epoch=30 --use_crf --bert_use_subword_pooling --eval_and_save_steps=1000 , without --bert_disable_lstm
INFO:__main__:[F1] : 0.8789345427469033, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 423700.3529071808ms, 47.06374085311135ms on average
accuracy:  94.94%; precision:  87.50%; recall:  88.29%; FB1:  87.89

** LM-KOR-ELECTRA

*** --bert_model_name_or_path=./embeddings/electra-kor-base  --warmup_epoch=0 --weight_decay=0.0 --gradient_accumulation_steps=2 --lr=8e-5 --epoch=30
INFO:__main__:[F1] : 0.8750042550294449, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 848208.4937095642ms, 94.24447772211201ms on average
INFO:__main__:[Elapsed Time] : 100 examples, 1798.86794090271ms, 17.15457800662879ms on average
accuracy:  94.75%; precision:  87.38%; recall:  87.39%; FB1:  87.39

*** slicing logits
*** --bert_model_name_or_path=./embeddings/electra-kor-base  --lr=8e-5 --epoch=30 --use_crf --bert_use_crf_slice --bert_freezing_epoch=4 --bert_lr_during_freezing=1e-3 --eval_and_save_steps=1000 , without --bert_disable_lstm
# https://github.com/dsindex/ntagger/releases/tag/v1.0
INFO:__main__:[F1] : 0.8761777503683147, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 372020.70689201355ms, 41.32575471608662ms on average
INFO:__main__:[Elapsed Time] : 100 examples, 4059.163808822632ms, 39.72473770681054ms on average
accuracy:  94.72%; precision:  88.04%; recall:  86.94%; FB1:  87.49

** bpe ELECTRA-base(v1)

$ python evaluate.py --config=configs/config-bert.json --model_path=pytorch-model-bert-kor-eoj.pt --data_dir data/clova2019 --bert_output_dir=bert-checkpoint-kor-eoj --bert_disable_lstm
$ cd data/clova2019; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.8657965765827326, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 1025293.7350273132ms, 113.92257348446465ms on average
INFO:__main__:[Elapsed Time] : 100 examples, 1881.6356658935547ms, 18.044946169612384ms on average
accuracy:  94.43%; precision:  86.76%; recall:  86.16%; FB1:  86.46

** dhaToken1.large ELECTRA-base
# https://github.com/dsindex/ntagger/releases/tag/v1.0
$ python evaluate.py --config=configs/config-bert.json --model_path=pytorch-model-bert-kor-eoj.pt --data_dir data/clova2019 --bert_output_dir=bert-checkpoint-kor-eoj --use_crf --bert_use_crf_slice
$ cd data/clova2019; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..
INFO:__main__:[F1] : 0.8705621049034683, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 407867.568731308ms, 45.30506216694162ms on average
INFO:__main__:[Elapsed Time] : 100 examples, 4550.331354141235ms, 44.371419482760956ms on average
accuracy:  94.50%; precision:  87.18%; recall:  86.62%; FB1:  86.90

** dhaSyllable ELECTRA-base
$ python evaluate.py --config=configs/config-bert.json --model_path=pytorch-model-bert-kor-eoj.pt --data_dir data/clova2019 --bert_output_dir=bert-checkpoint-kor-eoj --use_crf --bert_use_subword_pooling
$ cd data/clova2019; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..
INFO:__main__:[F1] : 0.8630221340357839, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 369628.701210022ms, 41.056287965584836ms on average
accuracy:  94.25%; precision:  86.86%; recall:  85.77%; FB1:  86.31

** RoBERTa-base

$ python evaluate.py --config=configs/config-roberta.json --model_path=pytorch-model-bert-kor-eoj.pt --data_dir data/clova2019 --bert_output_dir=bert-checkpoint-kor-eoj --bert_disable_lstm
$ cd data/clova2019; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..

INFO:__main__:[F1] : 0.8556474298866252, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 892125.4014968872ms, 99.12149309250948ms on average
INFO:__main__:[Elapsed Time] : 100 examples, 1671.3252067565918ms, 15.69863280864677ms on average
accuracy:  94.02%; precision:  85.59%; recall:  85.32%; FB1:  85.45

** XLM-RoBERTa-base, XLM-RoBERTa-large

$ python evaluate.py --config=configs/config-roberta.json --model_path=pytorch-model-bert-kor-eoj.pt --data_dir data/clova2019 --bert_output_dir=bert-checkpoint-kor-eoj --bert_disable_lstm
$ cd data/clova2019; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..
INFO:__main__:[F1] : 0.8699418515423837, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 847419.8191165924ms, 94.15775130147178ms on average
INFO:__main__:[Elapsed Time] : 100 examples, 1889.4524574279785ms, 18.13263122481529ms on average
accuracy:  94.51%; precision:  86.95%; recall:  86.72%; FB1:  86.84

*** --bert_model_name_or_path=./embeddings/xlm-roberta-large
INFO:__main__:[F1] : 0.8716396592474196, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 958549.7736930847ms, 106.50296955191304ms on average
INFO:__main__:[Elapsed Time] : 100 examples, 3725.3706455230713ms, 35.952108074920346ms on average
accuracy:  94.48%; precision:  86.88%; recall:  87.14%; FB1:  87.01


** Funnel-base

$ python evaluate.py --config=configs/config-bert.json --model_path=pytorch-model-bert-kor-eoj.pt --data_dir data/clova2019 --bert_output_dir=bert-checkpoint-kor-eoj --bert_disable_lstm
$ cd data/clova2019; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..
INFO:__main__:[F1] : 0.8808256677945353, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 1106301.5718460083ms, 122.92021880588051ms on average
INFO:__main__:[Elapsed Time] : 100 examples, 4384.497880935669ms, 42.92876551849673ms on average
accuracy:  94.98%; precision:  87.87%; recall:  88.06%; FB1:  87.97

*** slicing logits
*** --use_crf --bert_use_crf_slice --bert_freezing_epoch=4 --bert_lr_during_freezing=1e-3 --eval_and_save_steps=1000 , without --bert_disable_lstm
# https://github.com/dsindex/ntagger/releases/tag/v1.0
INFO:__main__:[F1] : 0.8805391055748417, 9000
INFO:__main__:[Elapsed Time] : 9000 examples, 605362.9877567291ms, 67.25383387736339ms on average
INFO:__main__:[Elapsed Time] : 100 examples, 8475.262641906738ms, 83.97073697562169ms on average
accuracy:  94.80%; precision:  87.89%; recall:  87.96%; FB1:  87.92

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


<details><summary><b>emb_class=bert, enc_class=bilstm, morph-based [1]</b></summary>
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


<details><summary><b>emb_class=bert, enc_class=bilstm, morph-based [2]</b></summary>
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



