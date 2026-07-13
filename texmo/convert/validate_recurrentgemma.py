"""End-to-end validation of a converted RecurrentGemma model.

Loads the manifest, greedily continues a fixed prompt through the
step path (the reference-faithful one: no synthetic pre-BOS padding;
conv buffers, RG-LRU state and attention positions all boot from
zero exactly like the upstream implementation), and compares the
token ids against the reference continuation captured from HF
transformers 5.13.1 (fp32, scratch-generated via
AutoModelForCausalLM ... .generate(do_sample=False)).

Deliberately a script, not a *_test.py: it needs the ~10 GB
checkpoint on disk and a few minutes of CPU, so it runs on demand:

    uv run python -m texmo.convert.validate_recurrentgemma
"""
import argparse
import sys
import time

import jax
import numpy as np

from ..model_store import load_model
from ..precision import Precision
from ..tokens import get_tokenizer, set_tokens_dir

PROMPT = "The capital of France is"
# <bos>The capital of France is| a city of art, fashion, and culture.
REF_IDS = [2, 651, 6037, 576, 6081, 603,
           476, 3413, 576, 3096, 235269, 4281, 235269, 578, 6981, 235265]
N_PROMPT = 6


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='models/recurrentgemma-2b.json')
    p.add_argument('--tokens-dir', default='tokens')
    args = p.parse_args()

    set_tokens_dir(args.tokens_dir)
    t0 = time.perf_counter()
    md, weights = load_model(args.model, precision=Precision.FP32)
    print(f'loaded {md.num_weights:,} weights '
          f'in {time.perf_counter() - t0:.1f}s', flush=True)

    tokenizer = get_tokenizer(md.codec.tokens_name)
    ids = tokenizer.encode(PROMPT, add_bos=True)
    assert ids == REF_IDS[:N_PROMPT], f'tokenization drift: {ids}'

    model = md.build_jax()
    step = jax.jit(model.step)
    # Fresh states, NOT initial_step: the training-shape padding is
    # texmo's convention, not the reference's.
    states = [model.codec.init_state(), model.layer_seq.init_state()]
    logits = None
    for t in ids:
        states, logits = step(weights, states, t)

    out = list(ids)
    for _ in range(len(REF_IDS) - N_PROMPT):
        nxt = int(np.asarray(logits.argmax()))
        out.append(nxt)
        states, logits = step(weights, states, nxt)

    text = tokenizer.untokenize(out).decode('utf-8', errors='replace')
    print(f'generated: {text}')
    if out == REF_IDS:
        print('PASS: greedy continuation matches the HF reference')
        return 0
    print(f'FAIL:\n  ref:  {REF_IDS}\n  ours: {out}')
    return 1


if __name__ == '__main__':
    sys.exit(main())
