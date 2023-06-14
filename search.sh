#!/bin/sh

python3 texmo.py search -d data/books3.txt -s '[^-]+' -t 1-4 --ntokens=2-1024 --token-type=all,bits1,bits2,bits4 --db=results/db-4090-3.sqlite --tokens-dir=tokens --min-max-weights=10 --token-type=bits1,bits2,bits4,all