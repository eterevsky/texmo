"""Convert a HuggingFace `tokenizer.json` (SentencePiece-BPE style, as
shipped with Gemma / RecurrentGemma) into texmo's extended tokenset
JSON.

    uv run python -m texmo.tokens.convert_hf <tokenizer.json> <out.json>

Conversion choices (see docs/tokens.md for the discussion):
  - Pieces are rewritten with the SentencePiece whitespace marker
    U+2581 replaced by a literal space, so tokens carry their spaces
    directly and decode is plain concatenation. Safe because no piece
    contains a real space. The runtime keeps one normalization quirk:
    input U+2581 must be replaced by " " before encoding to reproduce
    the reference's space/U+2581 conflation (processing "gemma").
  - `<0xXX>` byte-fallback pieces become raw single-byte tokens
    (stored as one-element int lists).
  - Control tokens (HF added tokens with special=true: <bos>, <eos>,
    <unused..>, ...) become ext (string-less) tokens, listed in the
    `specials` table with a private-use char (U+F0000 + id) so token
    sequences round-trip through plain strings.
  - Non-special added tokens (newline runs, space runs, HTML tags) keep
    their strings and are listed in `priority_tokens`: the encoder must
    match them with a longest-match scan BEFORE running BPE (some have
    no merge path, so BPE alone cannot produce them).
  - `merges` is stored as [left_id, right_id, merged_id] triples --
    ids, not strings, because converted pieces may contain spaces and
    several distinct ids can share one byte string (e.g. converted
    "▁" vs <0x20>).

The `stats` block is synthetic (~4 bytes/token, typical for Gemma on
English); it only seeds the sampler's chunk sizing, which self-corrects.
"""
import json
import sys

from ..pjson import save_json

_SP_SPACE = '▁'
_PUA_BASE = 0xF0000


def _is_byte_piece(piece: str) -> bool:
    return (
        len(piece) == 6 and piece.startswith('<0x') and piece.endswith('>')
    )


def convert(hf: dict) -> dict:
    model = hf['model']
    assert model['type'] == 'BPE', model['type']
    assert model.get('byte_fallback'), "expected byte_fallback"
    # The whole normalizer must be the single " " -> U+2581 replace;
    # anything else (NFKC, dummy prefix, ...) would need porting.
    norm = hf.get('normalizer')
    assert norm == {
        'type': 'Replace', 'pattern': {'String': ' '}, 'content': _SP_SPACE,
    }, f"unexpected normalizer: {norm}"
    assert hf.get('pre_tokenizer') is None, "expected no pre-tokenizer"

    vocab: dict[str, int] = model['vocab']
    n = len(vocab)
    pieces = [None] * n
    for piece, i in vocab.items():
        assert pieces[i] is None, f"duplicate id {i}"
        pieces[i] = piece
    assert all(p is not None for p in pieces), "vocab ids not dense"

    added = hf.get('added_tokens', [])
    special_ids = {t['id']: t['content'] for t in added if t['special']}
    priority_ids = sorted(t['id'] for t in added if not t['special'])
    for t in added:
        assert vocab.get(t['content']) == t['id'], t

    tokens: list = []
    for i, piece in enumerate(pieces):
        if i in special_ids:
            tokens.append(i)  # ext token: no string form of its own
        elif _is_byte_piece(piece):
            tokens.append([int(piece[3:5], 16)])
        else:
            tokens.append(piece.replace(_SP_SPACE, ' '))

    merges = []
    for m in model['merges']:
        if isinstance(m, str):
            # Original-form pieces never contain a real space, so the
            # single separator space splits unambiguously.
            left, _, right = m.partition(' ')
        else:
            left, right = m
        li, ri = vocab[left], vocab[right]
        merged = vocab[left + right]
        merges.append([li, ri, merged])

    specials = [
        {'id': i, 'name': name, 'char': chr(_PUA_BASE + i)}
        for i, name in sorted(special_ids.items())
    ]
    by_name = {s['name']: s['id'] for s in specials}

    return {
        'type': 'bytes',
        'processing': 'gemma',
        'algorithm': 'bpe',
        'split_paragraphs': False,
        'add_bos': True,
        'bos_id': by_name['<bos>'],
        'eos_id': by_name['<eos>'],
        'pad_id': by_name['<pad>'],
        'unk_id': by_name['<unk>'],
        'stats': {
            'bytes_per_token': 4.0,
            'scanned_bytes': 4,
            'total_tokens': 1,
            'ntokens': n,
            'synthetic': True,
        },
        'specials': specials,
        'priority_tokens': priority_ids,
        'tokens': tokens,
        'merges': merges,
    }


def main():
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, encoding='utf-8') as f:
        hf = json.load(f)
    out = convert(hf)
    with open(dst, 'w', encoding='utf-8') as f:
        save_json(out, f)
    print(f"{dst}: {len(out['tokens'])} tokens, "
          f"{len(out['merges'])} merges, "
          f"{len(out['specials'])} specials, "
          f"{len(out['priority_tokens'])} priority tokens")


if __name__ == '__main__':
    main()
