## Training on Mac for 10 seconds

Full log in [search1-mac-10.csv](search1-mac-10.csv) (2900 configurations).

Top:

```
suffix.4-dense.256.relu-dense.256.tanh-dense.128.tanh         LR   0.1  LEN  16  B 256  2.7583
suffix.4-dense.512.relu-dense.64.tanh-dense.128.relu          LR   0.1  LEN  16  B 512  2.7587
suffix.4-dense.256.relu-dense.128.relu-dense.128.tanh         LR   0.1  LEN  16  B 256  2.7589
suffix.5-dense.256.relu-dense.256.tanh                        LR   0.1  LEN  16  B 256  2.7686
suffix.4-dense.512.relu-dense.64.tanh-dense.256.tanh          LR   0.1  LEN  16  B 256  2.7687
suffix.4-dense.512.relu-dense.64.tanh-dense.128.tanh          LR   0.1  LEN  16  B 256  2.7706
suffix.5-dense.256.relu-dense.256.tanh-dense.128.tanh         LR   0.1  LEN  16  B 256  2.7719
suffix.4-dense.512.relu-dense.128.tanh-dense.128.tanh         LR   0.1  LEN  16  B 256  2.7729
suffix.5-dense.512.relu-dense.64.tanh-dense.128.tanh          LR   0.1  LEN  16  B 256  2.7752
suffix.4-dense.256.relu-dense.128.tanh-dense.128.tanh         LR   0.1  LEN  16  B 256  2.7767
suffix.4-dense.256.relu-dense.64.tanh-dense.128.relu          LR   0.1  LEN  16  B 256  2.7770
suffix.4-dense.512.relu-dense.32.tanh-dense.256.tanh          LR   0.1  LEN  16  B 256  2.7780
suffix.4-dense.256.relu-dense.128.tanh-dense.128.relu         LR   0.1  LEN  16  B 256  2.7785
suffix.4-dense.256.relu-dense.64.tanh-dense.256.tanh          LR   0.1  LEN  16  B 256  2.7800
suffix.4-dense.128.relu-dense.256.relu-dense.128.tanh         LR   0.1  LEN  16  B 256  2.7801
suffix.4-dense.256.relu-dense.128.tanh                        LR   0.1  LEN   8  B 512  2.7811
suffix.2-rec.128.relu-dense.256.relu-dense.128.tanh           LR   0.1  LEN   8  B 512  2.7828
suffix.4-dense.256.relu-dense.64.tanh-dense.64.relu           LR   0.1  LEN  16  B 256  2.7833
suffix.4-dense.256.relu-dense.128.tanh                        LR   0.1  LEN  16  B 256  2.7841
suffix.4-dense.512.relu-dense.64.tanh-dense.128.relu          LR   0.1  LEN  16  B 256  2.7845
```

Training the winner for more time:

```
Model: suffix.4-dense.256.relu-dense.256.tanh-dense.128.tanh, 394k weights
Loss: loss 2.3331
Training: 101779 steps, 6000 s
LR: 0.1, R0.1
Training data: 834M / 3836M
```

Alternative:

```
Model: suffix.2-rec.128.relu-dense.256.relu-dense.128.tanh, 181k weights
Loss: loss 2.4527
Training: 2141 steps, 120 s
LR: 0.1, R0.1
Training data: 18M / 3836M
```


## Training on GPU for 20 seconds

Full log in [search2-gpu-20.csv](search2-gpu-20.csv) (616 configurations).

```
suffix.3-gru.512.relu-dense.256.tanh                          LR   0.1  LEN  64  B1024  2.1516
suffix.3-gru.512.relu-dense.256.tanh                          LR   0.1  LEN  64  B 512  2.1537
suffix.3-gru.512.relu-dense.128.tanh                          LR   0.1  LEN  32  B1024  2.1614
suffix.3-gru.512.relu-dense.512.relu                          LR   0.1  LEN  64  B 512  2.1644
suffix.2-gru.512.tanh-dense.256.tanh                          LR   0.1  LEN  64  B 512  2.1660
suffix.2-gru.512.relu-dense.256.tanh                          LR   0.1  LEN  64  B 512  2.1691
suffix.3-gru.512.tanh-dense.512.relu                          LR   0.1  LEN  64  B 512  2.1693
gru.512.relu-dense.128.tanh                                   LR   0.1  LEN  32  B1024  2.1712
suffix.3-gru.512.relu-dense.128.tanh                          LR   0.1  LEN  64  B 512  2.1719
suffix.3-gru.512.tanh-dense.128.tanh                          LR   0.1  LEN  64  B 512  2.1719
gru.512.relu-dense.256.tanh                                   LR   0.1  LEN  64  B 512  2.1779
suffix.2-gru.512.tanh-dense.128.tanh                          LR   0.1  LEN  32  B1024  2.1779
gru.512.tanh-dense.256.tanh                                   LR   0.1  LEN  32  B1024  2.1799
suffix.2-gru.512.tanh-dense.256.tanh                          LR   0.1  LEN  32  B 512  2.1825
suffix.3-gru.512.relu-dense.256.tanh                          LR   0.1  LEN  32  B 512  2.1850
suffix.2-gru.512.relu-dense.256.tanh                          LR   0.1  LEN  32  B 512  2.1893
suffix.2-gru.512.tanh-dense.256.tanh                          LR   0.1  LEN  32  B1024  2.1896
gru.512.relu-dense.128.tanh                                   LR   0.1  LEN  64  B 512  2.1899
suffix.3-gru.256.tanh-dense.128.tanh                          LR   0.1  LEN  32  B1024  2.1913
suffix.2-gru.512.relu-dense.256.relu                          LR   0.1  LEN  32  B 512  2.1914
```

Top model:

```
Model: suffix.3-gru.512.relu-dense.256.tanh, 2165k weights
Loss: loss 1.8567
Training: 45200 steps, 6000 s
LR: 0.1, R0.1
Training data: 2962M / 3836M
```

## Best configurations per max_weights / time_limit (Mac)

Sample length = 128

| Max. weights | Time (s) | Best model                                        | Loss   | Batch | LR  |
| ------------ | -------- | ------------------------------------------------- | ------ | ----- | --- |
| 1024         |  1       | rec.1.tanh                                        | 4.3997 |   16  | 0.5 |
|              |  2       | rec.1.tanh                                        | 4.3583 |   64  | 0.5 |
|              |  4       | rec.1.relu                                        | 4.3271 |   64  | 0.5 |
| -------------| -------- | ------------------------------------------------- | ------ | ----- | --- |
| 2048         |  1       | suffix.2-dense.2.tanh                             | 4.1527 |   16  | 0.5 |
|              |  2       | rec.2.tanh-gru.4.tanh                             | 3.9460 |   16  | 0.5 |
|              |  4       | rec.2.tanh-dense.8.relu-rec.8.tanh-dense.4.relu   | 3.7828 |   32  | 0.2 |
|              |  8       | rec.2.tanh-dense.8.relu-rec.8.tanh-dense.4.relu   | 3.7165 |   64  | 0.2 |
|              | 16       | rec.2.relu-gru.4.tanh-rec.8.tanh-dense.4.relu     | 3.6714 |  128  | 0.2 |
| ------------ | -------- | ------------------------------------------------- | ------ | ----- | --- |
|              |  1       | suffix.2-dense.4.tanh                             | 3.8501 |   32  | 0.5 |
|              |  2       | rec.4.tanh-dense.16.tanh-rec.16.tanh-dense.8.relu | 3.6166 |   16  | 0.2 |
|              |  4       | rec.8.tanh-dense.32.relu-rec.4.tanh-dense.4.relu  | 3.5701 |   32  | 0.2 |
| 4096         |  8       | rec.4.tanh-dense.16.relu-rec.16.tanh-dense.8.relu | 3.4323 |   32  | 0.1 |
|              | 16       | rec.4.tanh-gru.8.tanh-rec.8.relu                  | 3.4027 |   64  | 0.2 |
| ------------ | -------- | ------------------------------------------------- | ------ | ----- | --- |
| 8192         |  1       | suffix.2-dense.8.tanh                             | 3.6739 |   32  | 0.5 |
|              |  2       | rec.8.tanh-dense.32.relu-dense.16.tanh            | 3.4602 |   32  | 0.2 |

## Best configurations per max_weights / time_limit (GPU)

Sample length = 128

| Max. weights | Time (s) | Best model                            | Loss   | Batch | LR  |
| ------------ | -------- | ------------------------------------- | ------ | ----- | --- |
| 1024         |  1       | rec.1.relu                            | 4.3871 |  128  | 0.5 |
|              |  2       | rec.1.relu                            | 4.3302 |  256  | 0.5 |
|              |  4       | rec.1.relu                            | 4.2969 |  512  | 0.5 |
| -------------| -------- | ------------------------------------- | ------ | ----- | --- |
| 2048         |  1       | suffix.2-rec.2.tanh                   | 4.0870 |  256  | 0.5 |
|              |  2       | suffix.2-dense.2.relu                 | 3.9936 |  128  | 0.5 |
|              |  4       | rec.2.tanh-gru.4.tanh                 | 3.8589 |  128  | 0.5 |
|              |  8       | rec.2.tanh-gru.4.tanh                 | 3.8135 |  256  | 0.5 |
|              | 16       | rec.2.relu-dense.8.tanh-lstm.4        | 3.7201 |  512  | 0.2 |
| ------------ | -------- | ------------------------------------- | ------ | ----- | --- |
|              |  1       | suffix.2-dense.4.tanh                 | 3.7851 |   64  | 0.5 |
|              |  2       | suffix.2-rec.4.tanh                   | 3.6814 |  128  | 0.5 |
| 4096         |  4       | rec.4.tanh-dense.32.relu-dense.8.relu | 3.5916 |  256  | 0.2 |
|              |  8       | rec.8.tanh-dense.16.relu-dense.4.relu | 3.4916 | 2048  | 0.2 |
|              | 16       | rec.8.tanh-gru.8.tanh-dense.16.relu-dense.4.relu | 3.4034 |  512  | 0.2 |
