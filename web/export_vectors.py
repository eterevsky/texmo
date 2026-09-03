"""Export reference vectors that pin `web/texmo.js` to the engine.

Runs the real Python/JAX path -- the same `load_model` +
`Model2Jax.initial_step` / `step` that `texmo.py chat` uses -- over a
handful of prompts and writes token ids, per-position logits and a
greedy continuation to `web/test/vectors.json`. `web/texmo_test.mjs`
replays the same prompts through the JavaScript port and asserts they
match.

    uv run python web/export_vectors.py

Prompts deliberately cover what the capswords2 preprocessing and the
hexbpe byte fallback can disagree about: a multi-turn dialog, running
lowercase text with no sentence punctuation, dense punctuation,
mixed-case and ALL-CAPS words, and non-ASCII characters (which the
tokenset spells as nibble pairs).

Between them the three exported models cover every layer the JS port
implements: `mg12k-s5` is the only one exercising `gru`, `mingru`,
`split.mul` (both the gated and the `pass`-value self-gate form), a
`tanh` dense and `rglru` with a single full-width block.

CPU only: the models are 8k-12k weights, and a GPU would only add the
XLA differences we are not trying to measure.
"""
import argparse
import datetime
import functools
import os
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from texmo import pjson
from texmo.chat import collect_reply
from texmo.generate import jit_step
from texmo.model_store import load_model
from texmo.precision import Precision
from texmo.tokens import get_tokenizer, set_tokens_dir

MODELS = [
    'models/rl32-8k-s5.json',
    'models/hb32-8k-s5.json',
    'models/mg-12k-s5.json',
]

PROMPTS = [
    ('dialog',
     'User: Hi there!\n\nBot: Hello! How are you today?\n\n'
     'User: I am good, thanks. What do you like to do?\n\nBot: '),
    ('lowercase-no-period',
     'user: hey do you want to go to the park with me and the dog\n\n'
     'bot: '),
    ('punctuation',
     'User: Wow!! Really?? That\'s... odd -- isn\'t it? (I think so.)'
     '\n\nBot: '),
    ('mixed-case',
     'User: My iPhone AND my McDonald\'s app are BOTH broken.\n\nBot: '),
    ('non-ascii',
     'User: I drank a café latte — naïve, maybe. '
     '你好! ☕\n\nBot: '),
    ('raw-prose',
     'The quick brown fox jumps over the lazy dog. '
     'It was a bright cold day in April, and the clocks were striking '
     'thirteen.'),
]

# Enough positions to exercise the recurrence well past the point
# where any state drift would show, without a huge fixture.
MAX_POSITIONS = 64
GREEDY_TOKENS = 32


def _round(row, places):
    return [round(float(x), places) for x in row]


def _decode(tokenizer, ids) -> str:
    """chat._decode: a partial reply can end mid character."""
    return tokenizer.untokenize(list(ids)).decode('utf-8', errors='replace')


def _argmax(logits) -> int:
    best = 0
    for i in range(1, len(logits)):
        if logits[i] > logits[best]:
            best = i
    return best


def run_case(model, weights, tokenizer, prompt: str, places: int) -> dict:
    """Tokenize, prefill and greedily continue one prompt."""
    ids = [int(c) for c in tokenizer.tokenize(prompt.encode())]
    step = jit_step(model)

    states, logits0 = model.initial_step(weights)
    logits = []
    row = None
    for i, c in enumerate(ids):
        states, out = step(weights, states, c)
        row = [float(x) for x in out]
        if i < MAX_POSITIONS:
            logits.append(_round(row, places))

    # Greedy continuation from the state the prompt leaves behind.
    greedy = []
    cur = _argmax(row)
    for _ in range(GREEDY_TOKENS):
        greedy.append(cur)
        states, out = step(weights, states, cur)
        cur = _argmax([float(x) for x in out])

    # The streaming path, replayed over the greedy tokens so it is
    # deterministic: every partial decode (a reply can stop mid
    # character) and what `collect_reply` would have streamed.
    decode = functools.partial(_decode, tokenizer)
    deltas = []
    collected, ncollected = collect_reply(
        iter(greedy), decode, GREEDY_TOKENS, deltas.append)

    return {
        'prompt': prompt,
        'tokens': ids,
        'untokenized': decode(ids),
        'logits0': _round([float(x) for x in logits0], places),
        'logits': logits,
        'greedy': greedy,
        'greedy_text': decode(greedy),
        'greedy_partials': [decode(greedy[:k])
                            for k in range(1, len(greedy) + 1)],
        'collected_text': collected,
        'collected_tokens': ncollected,
        'streamed': ''.join(deltas),
    }


def export(args) -> dict:
    set_tokens_dir(args.tokens_dir)
    models = []
    for path in args.models:
        md, weights = load_model(path, precision=args.precision)
        model = md.build_jax()
        tokenizer = get_tokenizer(md.codec.tokens_name)
        print(f'{path}: {md.num_weights:,} weights, {md.spec}')
        cases = []
        for name, prompt in PROMPTS:
            case = run_case(
                model, weights, tokenizer, prompt, args.places)
            case['name'] = name
            cases.append(case)
            print(f'  {name}: {len(case["tokens"])} tokens')
        models.append({
            'manifest': path,
            'spec': md.spec,
            'precision': str(md.precision),
            'num_weights': md.num_weights,
            'ntokens': md.ntokens,
            'tokens_name': md.codec.tokens_name,
            'cases': cases,
        })
    return {
        'generated': datetime.date.today().isoformat(),
        'max_positions': MAX_POSITIONS,
        'greedy_tokens': GREEDY_TOKENS,
        'decimals': args.places,
        'models': models,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-m', '--models', nargs='+', default=MODELS,
        help='model manifests to export (default: all chat models)')
    parser.add_argument(
        '-o', '--out', default='web/test/vectors.json',
        help='output file (default: web/test/vectors.json)')
    parser.add_argument(
        '--precision', type=Precision, default=None,
        choices=list(Precision),
        help="override the manifest's precision. Only for diagnosis: "
             "exporting at fp64 (with JAX_ENABLE_X64=1) and diffing "
             "against the fp32 export measures how much of a JS/Python "
             "gap is the reference's own rounding rather than a port "
             "bug -- see per_model_tolerance in texmo_test.mjs")
    parser.add_argument(
        '--tokens-dir', default='tokens',
        help="directory with token sets (default: 'tokens')")
    parser.add_argument(
        '--places', type=int, default=5,
        help='decimals kept per logit (default: 5; the JS test '
             'tolerance is 1e-4 relative)')
    args = parser.parse_args()

    doc = export(args)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        # maxlen high enough to keep one position's 32 logits on one
        # line; pjson still expands the structure around them.
        for line in pjson.serialize(doc, maxlen=4000):
            f.write(line + '\n')
    print(f'wrote {args.out} ({os.path.getsize(args.out):,} bytes)')


if __name__ == '__main__':
    main()
