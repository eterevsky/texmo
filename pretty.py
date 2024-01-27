import json
import sys

from texmo import pjson

with open(sys.argv[1], "r", encoding="utf-8") as f: 
    d = json.load(f)

with open(sys.argv[1], "w", encoding="utf-8") as f:
    pjson.save_json(d, f)
