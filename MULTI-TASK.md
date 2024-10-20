## multi-task learning

### data
- https://github.com/monologg/JointBERT/tree/master/data/atis
```
$ ls data/atis
```

### preprocess
```
$ python preprocess.py --config=configs/config-bert.json --data_dir=data/atis --bert_model_name_or_path=./embeddings/bert-base-cased --bert_use_mtl

* xlm-roberta-large
$ python preprocess.py --config=configs/config-roberta.json --data_dir=data/atis --bert_model_name_or_path=./embeddings/xlm-roberta-large --bert_use_mtl

```

### train
```
# hyper parameter search
$ python train.py --config=configs/config-bert.json --data_dir=data/atis --save_path=pytorch-model-bert.pt --bert_model_name_or_path=./embeddings/bert-base-cased --bert_output_dir=bert-checkpoint --batch_size=32 --lr=5e-5 --epoch=10 --bert_use_mtl --hp_search_optuna --hp_trials 24 --patience 4
INFO:__main__:[study.best_params] : {'lr': 0.00015433586280916495, 'batch_size': 32, 'seed': 23, 'epochs': 7}
INFO:__main__:[study.best_value] : 0.9175167589624017

$ python train.py --config=configs/config-bert.json --data_dir=data/atis --save_path=pytorch-model-bert.pt --bert_model_name_or_path=./embeddings/bert-base-cased --bert_output_dir=bert-checkpoint --batch_size=32 --lr=0.00015433586280916495 --epoch=30 --seed 23 --bert_use_mtl

* xlm-roberta-large
# hyper parameter search
$ python train.py --config=configs/config-roberta.json --data_dir=data/atis --save_path=pytorch-model-bert.pt --bert_model_name_or_path=./embeddings/xlm-roberta-base --bert_output_dir=bert-checkpoint --batch_size=32 --lr=0.00015433586280916495 --epoch=10 --bert_use_mtl --hp_search_optuna --hp_trials 24 --patience 4 --eval_batch_size=32
INFO:__main__:[study.best_params] : {'lr': 6.586651781916603e-05, 'batch_size': 16, 'seed': 19, 'epochs': 7}
INFO:__main__:[study.best_value] : 0.8430804863925884

$ python train.py --config=configs/config-roberta.json --data_dir=data/atis --save_path=pytorch-model-bert.pt --bert_model_name_or_path=./embeddings/xlm-roberta-base --bert_output_dir=bert-checkpoint --batch_size=16 --lr=6.586651781916603e-05 --epoch=30 --seed 19 --bert_use_mtl 

```

### evaluate
```
$ python evaluate.py --config=configs/config-bert.json --data_dir=data/atis --model_path=pytorch-model-bert.pt --bert_output_dir=bert-checkpoint --bert_use_mtl
$ cd data/atis; perl ../../etc/conlleval.pl < test.txt.pred ; cd ../..
INFO:__main__:[sequence classification F1] : 0.9776035834266518, 893
INFO:__main__:[token classification F1] : 0.9570422535211267, 893
INFO:__main__:[Elapsed Time] : 893 examples, 45115.190505981445ms, 50.43707888222596ms on average
accuracy:  98.01%; precision:  95.30%; recall:  95.77%; FB1:  95.53

* entity tagging only
INFO:__main__:[F1] : 0.9562764456981664, 893
INFO:__main__:[Elapsed Time] : 893 examples, 44129.048109054565ms, 49.33518411867287ms on average
accuracy:  97.98%; precision:  95.36%; recall:  95.56%; FB1:  95.46

* xlm-roberta-large
$ python evaluate.py --config=configs/config-roberta.json --data_dir=data/atis --model_path=pytorch-model-bert.pt --bert_output_dir=bert-checkpoint --bert_use_mtl
INFO:__main__:[sequence classification F1] : 0.9798432250839866, 893
INFO:__main__:[token classification F1] : 0.9532019704433498, 893
INFO:__main__:[Elapsed Time] : 893 examples, 33223.02174568176ms, 37.11147613054968ms on average
accuracy:  97.95%; precision:  94.88%; recall:  95.45%; FB1:  95.17

```

### conversion to onnx
```
$ python evaluate.py --config=configs/config-bert.json --data_dir=data/atis --model_path=pytorch-model-bert.pt --bert_output_dir=bert-checkpoint --bert_use_mtl --convert_onnx

$ python evaluate.py --config=configs/config-bert.json --data_dir=data/atis --model_path=pytorch-model-bert.pt --bert_output_dir=bert-checkpoint --bert_use_mtl --enable_ort


```



