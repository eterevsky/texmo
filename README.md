## Train model

```
python3 train.py -d data -s 100000 -o models -t temp
```

## Evaluate trained model

```
python3 train.py -d data -m models/some-model.json
```

## Results

| Model                     | Steps          | Loss   |
| ------------------------- | -------------- | ------ |
| equal                     |     0          | 8.0000 |
| freq                      | 10000          | 4.9449 |
| freq                      | 5000 L256 B256 | 4.9211 |
| markov1                   | 10000          | 3.7713 |
| markov1                   | 10001 R0.01    | 3.7408 |
| markov2                   | 10000          | 3.1147 |
| markov2true               | 7600           | 3.1193 |
| recurrent-l1(256)         | 20000          | 2.4929 |
| recurrent-l1(512)         | 50000          | 2.2196 |
| recurrent-conv2(128, 512) | 10000          | 2.6631 |
| recurrent-gru(256)        |

## Notes

According to https://arxiv.org/abs/2203.15556, the amount of training tokens for a model with X params should be roughly 20*X. Consdiring that a token is on average 4 bytes long and we are training on bytes, this translates into 80*X bytes.