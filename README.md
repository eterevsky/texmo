## Train model

```
python3 train.py -d data -s 100000 -o models -t temp
```

## Evaluate trained model

```
python3 train.py -d data -m models/some-model.json
```

## Results

| Model                     | Params   | Steps          | Loss   | Time |
| ------------------------- | -------- | -------------- | ------ | ---- |
| equal                     | 0        |                | 8.0000 | 0    |
| freq                      | 256      | 10000          | 4.9449 |      |
| freq                      | 256      | 5000 L256 B256 | 4.9211 |      |
| markov1                   | 65792    | 20000          | 3.7621 |      |
| markov(2)                 | 131328   | 10000          | 3.4896 | 527  |
| forward(2, 128)           | 98688    | 10000          | 3.4629 | 588  |
| markov(5)                 | 327936   | 20000          | 3.2909 |      |
| forward(5, 128)           | 196992   | 10000          | 3.2766 | 606  |
| recurrent-conv2(128, 512) |          | 10000          | 2.6631 |
| recurrent-l1(256)         |          | 20000          | 2.4929 |
| recurrent-l1(512)         |          | 50000          | 2.2196 |
| recurrent-conv2(128, 512) |          | 100000         | 2.0748 |
| recurrent-gru(256)        | 722176   | 85500          | 1.8215 |
| conv-gru(128, 256)        | 591232   | 100000         | 1.7893 |
| conv3-gru(128, 256)       | 624000   | 100000         | 1.8907 |
| conv-gru(128, 512)        | 1640832  | 100000         | 1.7256 |
| conv-conv-gru(128, 512)   |          |                |        |

## Notes

According to https://arxiv.org/abs/2203.15556, the amount of training tokens for a model with X params should be roughly 20X. Consdiring that a token is on average 4 bytes long and we are training on bytes, this translates into 80X bytes.