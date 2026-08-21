"""Tests for the dialog turn logic.

The prompt/boundary functions are tested with a stub token stream;
`generate_reply` gets one end-to-end pass over a tiny untrained model
(the reply is noise, so only the contract is asserted).
"""

import jax

from texmo.chat import (
    TURN_SEPARATOR,
    append_reply,
    build_prompt,
    collect_reply,
    generate_reply,
)
from texmo.precision import Precision
from texmo.spec_parser import parse_model2

# A one-byte-per-token set, like the dialog models': any prefix of a
# reply decodes, which is what the streaming assertions check.
SPEC = 'tokens.32.fold.oh|dense.4.gelu'


def _make_model(spec: str = SPEC):
    md = parse_model2(spec, precision=Precision.FP32)
    model = md.build_jax()
    weights = model.init_weights(jax.random.PRNGKey(0))
    return model, weights, md.codec.tokens_name


def _join(ids: list[str]) -> str:
    """Decoder for stub streams whose 'tokens' are single characters."""
    return ''.join(ids)


def test_build_prompt_opens_the_bot_turn():
    assert build_prompt('', 'hi', 'User', 'Bot') == 'User: hi\n\nBot:'


def test_build_prompt_extends_history():
    dialog = 'User: hi\n\nBot: hello.\n\n'
    assert build_prompt(dialog, 'bye', 'User', 'Bot') == (
        'User: hi\n\nBot: hello.\n\nUser: bye\n\nBot:')


def test_build_prompt_uses_custom_names():
    assert build_prompt('', 'hi', 'Ann', 'Bob') == 'Ann: hi\n\nBob:'


def test_append_reply_normalizes_the_boundary():
    prompt = 'User: hi\n\nBot:'
    assert append_reply(prompt, ' hello.\n\n') == (
        'User: hi\n\nBot: hello.\n\n')


def test_append_reply_adds_a_missing_boundary():
    """A reply cut off by the token cap still closes its turn."""
    prompt = 'User: hi\n\nBot:'
    assert append_reply(prompt, ' hello and') == (
        'User: hi\n\nBot: hello and\n\n')


def test_collect_reply_stops_at_the_boundary():
    stream = list(' hi.\n\nUser: more')
    text, n = collect_reply(stream, _join, max_tokens=100)
    assert text == ' hi.\n\n'
    assert n == 6
    assert TURN_SEPARATOR in text


def test_collect_reply_stops_at_the_cap():
    stream = list('abcdefgh')
    text, n = collect_reply(stream, _join, max_tokens=3)
    assert text == 'abc'
    assert n == 3


def test_collect_reply_consumes_lazily():
    """The stream is a generator in real use; stopping must not have
    pulled more tokens than reported."""
    pulled = []

    def gen():
        for c in ' hi\n\nxxxx':
            pulled.append(c)
            yield c

    text, n = collect_reply(gen(), _join, max_tokens=100)
    assert text == ' hi\n\n'
    assert len(pulled) == n == 5


def test_collect_reply_streams_deltas_without_the_boundary():
    deltas = []
    stream = list(' hi.\n\n')
    collect_reply(
        stream, _join, max_tokens=100, on_delta=deltas.append)
    assert ''.join(deltas) == ' hi.'
    assert deltas == [' ', 'h', 'i', '.']


def test_collect_reply_holds_back_a_rewritten_character():
    """A multi-byte character decodes as U+FFFD until its last byte
    arrives; the replacement char must not be streamed and then
    contradicted."""
    texts = ['a', 'a�', 'aé', 'aé!']
    deltas = []
    collect_reply(
        range(len(texts)),
        lambda ids: texts[len(ids) - 1],
        max_tokens=100,
        on_delta=deltas.append,
    )
    assert ''.join(deltas) == 'aé!'


def test_collect_reply_flushes_the_tail_after_the_cap():
    texts = ['a', 'ab�']
    deltas = []
    collect_reply(
        range(len(texts)),
        lambda ids: texts[len(ids) - 1],
        max_tokens=2,
        on_delta=deltas.append,
    )
    assert ''.join(deltas) == 'ab�'


def test_generate_reply_over_a_tiny_model():
    model, weights, tokens_name = _make_model()
    prompt = build_prompt('', 'hi', 'User', 'Bot')
    reply, n = generate_reply(
        model, weights, tokens_name, prompt,
        max_tokens=32, temperature=1.0)
    assert isinstance(reply, str)
    assert 0 < n <= 32
    # The prompt is not echoed, and the reply either ends at a
    # boundary or ran to the cap.
    assert not reply.startswith(prompt)
    assert TURN_SEPARATOR in reply or n == 32


def test_generate_reply_streams_the_same_text():
    model, weights, tokens_name = _make_model()
    prompt = build_prompt('', 'hi', 'User', 'Bot')
    deltas = []
    reply, _ = generate_reply(
        model, weights, tokens_name, prompt,
        max_tokens=32, temperature=1.0, on_delta=deltas.append)
    assert ''.join(deltas) == reply.split(TURN_SEPARATOR, 1)[0]
