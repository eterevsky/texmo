# Plans

* Try to migrate to PyTorch

* Separate a server process which selects configurations to run an client processes that just fetch the configuration and run it

* Move more stuff from ResultSet to the DB so that it doesn't need to be recomputed on each run.

* Improve loss predictor to use some sort of RNN or even make a "recurrent random forest".

* Improve `train` to automatically optimize metaparameters and the model within given constraints

* Add new strategies to Search, including finding a local maximum starting from one of the winning configurations