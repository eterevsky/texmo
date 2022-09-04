# TexMo — training simple models for text prediction.

## Train model

```
python3 train.py -d data -c rec.128.relu-gru.512.tanh-dense.128 -t 3600 -o models
```

## Evaluate trained model

```
python3 eval.py -d data -m models/some-model.json --prefix="..."
```

(Either `-d` or `--prefix` is enough.)

## Searching the configurations and metaparameters

```
python3 search.py -d data -c dense.128 -t 8 --vary=struc,batch,lr
```

## Results

Baseline: predicting each symbol with its prior probability results in loss
4.9449.

### 1 Layer

| Model                        | Params   | Steps          | Loss   | Time |
| ---------------------------- | -------- | -------------- | ------ | ---- |
| suffix.1                     |    65792 | 10000          | 3.7006 | 607  |
| suffix.2                     |   131328 | 10000          | 3.3875 | 663  |
| suffix.5                     |   327936 | 10000          | 3.2485 | 640  |

### 2 Layers

| Model                                        | Params   | Steps   | Loss   | Time | LR  |
| -------------------------------------------- | -------- | ------- | ------ | ---- | --- |
| suffix.2-dense.128.relu                      |    98688 |   10000 | 3.0116 |  758 |     |
| suffix.2-dense.512.relu                      |   393984 |   10000 | 2.9369 |  787 |
| suffix.3-conv.2.128.relu                     |   131456 |   10000 | 2.7876 | 1067 |
| suffix.5-conv.2.128.relu                     |   196992 |   10000 | 2.7268 |  987 |
| rec.128.sigmoid                              |    82304 |   10000 | 2.6419 |  774 |
| rec.128.relu.init                            |    82432 |   10000 | 2.5738 |  766 |
| rec.128.relu                                 |    82304 |   10000 | 2.5628 |  766 |
| suffix.5-dense.128.tanh                      |   196992 |   10000 | 2.5417 |  813 | 0.1 |
| rec.128.tanh                                 |    82304 |   10000 | 2.4358 |  764 |
| rec.128.relu                                 |    82304 |   10000 | 2.4615 |  877 |
| suffix.2-rec.128.relu                        |   115072 |   10000 | 2.4443 |  793 |
| gru.128.relu                                 |   180864 |   10000 | 2.2995 | 1025 |
| rec.256.relu                                 |   197120 |   20000 | 2.2341 | 1683 |
| gru.128.tanh                                 |   180864 |   27375 | 2.1509 | 3000 |

### 3 Layers

| Model                           | Params   | Steps   | Loss   | Time  | LR   |
| ------------------------------- | -------- | ------- | ------ | ----- | ---- |
| rec.256.tanh-dense.128.tanh     |   197248 |   10000 | 2.3890 |   907 | 0.05 |
| rec.256.relu-dense.128.relu     |   197248 |   10000 | 2.3717 |   887 | 0.05 |
| gru.128.tanh-dense.128.tanh     |   197376 |   20000 | 2.2652 |  2437 | 0.05 |
| rec.128.tanh-gru.512.tanh       |  1165184 |   10000 | 1.9288 |  1612 | 0.05 |
| gru.128.tanh-gru.256.tanh       |   509312 |   87881 | 1.8272 | 14400 | 0.05 |
| gru.512.tanh-gru.512.tanh       |  2887k   |   16019 | 1.7119 |  3000 | 0.05 | B128
| gru.128.tanh-gru.512.tanh       |  1263744 |  161044 | 1.6548 | 30000 | 0.05 |
| gru.128.tanh-gru.1025.tanh      |  3959046 |   35788 | 1.6040 | 24000 | 0.06 |

### 4 Layers

| Model                           | Params   | Steps   | Loss   | Time  | LR   |
| ------------------------------- | -------- | ------- | ------ | ----- | ---- |
| gru.128-gru.512-gru.512         |  2838144 |  108648 | 1.5724 | 30000 | 0.05 |
| gru.128-gru.256-gru.1024        |  4641152 |   41622 | 1.5079 | 30000 | 0.05 |