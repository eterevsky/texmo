# Plans

* Fix DataSet to be able to produce samples of arbitrary length.

* Once we have position embedding make a variant of `suffix` that adds together values from positions, rather than stacking them. There will be two variants: `suffix.X.add` and `suffix.X.stack`

* Same for Attention.

* Improve loss predictor to use some sort of RNN or even make a "recurrent random forest".

* Improve `train` to automatically optimize metaparameters and the model within given constraints

* Add new strategies to Search, including finding a local maximum starting from one of the winning configurations