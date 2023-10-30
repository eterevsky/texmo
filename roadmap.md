# Plans

* Refactor Model to include tokenization, embedding and position encoding.

* Refactor Configuration and database to use the new Model.

* Once we have position embedding make a variant of `suffix` that adds together values from positions, rather than stacking them. There will be two variants: `suffix.X.add` and `suffix.X.stack`

* Same for Attention.

* Refactor the database to use the same instance for several machines. (Instead of adding extra DBs in the prediction model).

* Improve loss predictor to use some sort of RNN or even make a "recurrent random forest".

* Improve `train` to automatically optimize metaparameters and the model within given constraints

* Add new strategies to Search, including finding a local maximum starting from one of the winning configurations