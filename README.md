# TexMo — training simple language models

This is a repository in which I attempt to re-implement various language models and related algorithms. All of the ML code is written using JAX, which is basically numpy with JIT compilation and auto differentiation.

Features:

* Selection of a "good" token set based on a given text corpus. (In progress)
* Training a language model with any combination of the following layers:
  * Dense
  * Recurrent
  * LSTM
  * (m)GRU
  * Suffix (stack the last few positions)
  * Attention (with trained relative position encoding)
* Search over a space of metaparameters and model architectures for an optimal configuration given constraints for training time and number of weights.
* Model, predicting the loss of a model with given metaparameters etc.
* Training of pretrained models.
  * Also adding extra layers to a pretrained model making it possible to incrementally train a deep model.

The project is very much in an "alpha" state.

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
