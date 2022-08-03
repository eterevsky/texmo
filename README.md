## Train model

```
python3 train.py -d data -s 100000 -o models -t temp
```

## Evaluate trained model

```
python3 train.py -d data -m models/some-model.json
```

## Results

| Model                        | Params   | Steps          | Loss   | Time |
| ---------------------------- | -------- | -------------- | ------ | ---- |
| equal                        |        0 |                | 8.0000 | 0    |
| freq                         |      256 | 10000          | 4.9449 |      |
| freq                         |      256 | 5000 L256 B256 | 4.9211 |      |
| markov1                      |    65792 | 20000          | 3.7621 |      |
| markov(2)                    |   131328 | 10000          | 3.4896 | 527  |
| forward(2, 128)              |    98688 | 10000          | 3.4629 | 588  |
| markov(5)                    |   327936 | 20000          | 3.2909 |      |
| forward(5, 128)              |   196992 | 10000          | 3.2766 | 606  |
| recurrent1(128)              |    82304 | 10000          | 2.7206 | 640  |
| recurrent1(256)              |   197120 | 10000          | 2.5017 | 687  |
| recurrent2(256, 128)         |   197248 | 10000          | 2.4773 | 682  | Mac: 1500
| recurrent3(3, 128, 512, 128) |   525312 | 100000         | 2.2153 | 8707 |
| recurrent3(2, 128, 512, 128) |   492544 | 100000         | 2.1947 | 9030 |
| rec-gru-128-256-512          |  1476736 | 200000         | 1.9856 | 20577|
| rec-gru-                     |          | 100000         | 2.0689 | 10305|
| recurrent-conv2(128, 512)    |          | 10000          | 2.6631 |
| recurrent-l1(256)            |          | 20000          | 2.4929 |
| recurrent-l1(512)            |          | 50000          | 2.2196 |
| recurrent-conv2(128, 512)    |          | 100000         | 2.0748 |
| conv-gru2-128-512            |  1378176 |  73900         | 2.0436 |
| lstm2-512                    |  2230528 | 100000         | 1.9256 | | L0.02
| conv3-gru(128, 256)          |   624000 | 100000         | 1.8907 |
| gru2-512                     |  1705728 | 100000         | 1.8571 | 8436 |
| gru-gru-512-512              |  3280128 | 100000         | 1.8658 | 11474 |
| recurrent-gru(256)           |   722176 | 85500          | 1.8215 |
| conv-gru(128, 256)           |   591232 | 100000         | 1.7893 |
| gru-gru-512s-512             |  3280128 | 100000         | 1.7708 | 11549 | L0.02
| conv-gru(128, 512)           |  1640832 | 100000         | 1.7256 |
| gru-gru-512-512              |  3280128 | 200000         | 1.7049 | 23165 | L0.02

## Notes

According to https://arxiv.org/abs/2203.15556, the amount of training tokens for a model with X params should be roughly 20X. Consdiring that a token is on average 4 bytes long and we are training on bytes, this translates into 80X bytes.
