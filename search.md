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