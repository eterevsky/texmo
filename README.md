## Train model

```
python3 train.py -d data

## Evaluate trained model

```
python3 train.py -d data -s 100000 -o models -t temp
```

## Results

| Model             | Steps          | Loss   |
| ----------------- | -------------- | ------ |
| equal             |     0          | 8.0000 |
| freq              | 10000          | 4.9449 |
| freq              | 5000 L256 B256 | 4.9211 |
| markov1           | 10000          | 3.7713 |
| markov1           | 10001 R0.01    | 3.7408 |
| markov2           | 10000          | 3.1147 |
| markon2true       | 7600           | 3.1193 |
| recurrent-l1(256) | 20000          | 2.4929 |