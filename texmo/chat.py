"""Multi-turn dialog over the per-token step path.

The model-facing half of `texmo.py chat`: assemble a prompt in the
turn format of the dialog corpora (`Name: utterance` separated by
blank lines), then sample a reply and stop at the next turn boundary
instead of at a fixed token count. `texmo/cli/chat.py` is the
interactive shell around it.

Every turn re-tokenizes the whole dialog rather than feeding the
appended bytes into a kept state: tokenization is context-dependent
(the DP tokenizer segments a whole chunk at once), so an incremental
feed can split a token across the append boundary and desynchronize
the model from the segmentation it was trained on. Re-prefilling
costs one step per token of history -- fine at these model sizes.

Lossy tokensets (fold) map the prompt onto their own alphabet, so a
reply comes back in the folded form ('bot. hi' for 'Bot: Hi'). That
is the model's own view of the dialog and is what the history keeps.
"""
import functools

from .generate import sample_tokens
from .model2_jax import Model2Jax
from .tokens import get_tokenizer

TURN_SEPARATOR = '\n\n'
# What a byte that is not (yet) a character decodes to.
REPLACEMENT = '�'


def build_prompt(
    dialog: str,
    utterance: str,
    user_name: str,
    bot_name: str,
) -> str:
    """The dialog with the user's turn appended and the bot's opened.

    `dialog` is empty or ends with a turn separator (see
    `append_reply`), and the result ends with the bot's name, exactly
    where the corpus has the model produce a reply.
    """
    return (f'{dialog}{user_name}: {utterance}{TURN_SEPARATOR}'
            f'{bot_name}: ')


def append_reply(prompt: str, reply: str) -> str:
    """The dialog history after the bot's reply.

    Normalized back to the `build_prompt` invariant: exactly one turn
    separator at the end, whether the reply stopped at a boundary or
    ran into the token cap. Leading whitespace goes too -- the prompt
    already ends with the format's colon-space, so a model that emits
    its own space would double it in the history.
    """
    return prompt + reply.strip() + TURN_SEPARATOR


def collect_reply(
    tokens,
    decode,
    max_tokens: int,
    on_delta=None,
) -> tuple[str, int]:
    """Consume sampled tokens up to the turn boundary or the cap.

    `decode(ids) -> str` renders the reply so far; it is called per
    token because only the decoded text can say where the boundary
    is. Returns the raw reply text (boundary included, if reached)
    and the number of tokens consumed.

    `on_delta(text)` streams the reply as it is sampled, truncated at
    the boundary. Streamed text can never be taken back, so anything
    the next token could still change is held back: a trailing
    replacement character (the head of a multi-byte character), a
    trailing newline (which may turn out to be the boundary's first
    half), and any delta that does not extend what was already
    streamed. Whatever is held back is flushed at the end.
    """
    ids = []
    text = ''
    shown = ''
    for token in tokens:
        ids.append(token)
        text = _decode_partial(decode, ids, text)
        if on_delta is not None:
            visible = text.split(
                TURN_SEPARATOR, 1)[0].rstrip(REPLACEMENT + '\n')
            if visible != shown and visible.startswith(shown):
                on_delta(visible[len(shown):])
                shown = visible
        if TURN_SEPARATOR in text or len(ids) >= max_tokens:
            break

    if on_delta is not None:
        visible = text.split(TURN_SEPARATOR, 1)[0]
        tail = visible[_common_prefix_len(shown, visible):]
        if tail:
            on_delta(tail)
    return text, len(ids)


def generate_reply(
    model: Model2Jax,
    weights,
    tokens_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    on_delta=None,
) -> tuple[str, int]:
    """Sample one reply to `prompt`: (text, tokens generated)."""
    tokenizer = get_tokenizer(tokens_name)
    return collect_reply(
        sample_tokens(model, weights, tokenizer, prompt, temperature),
        functools.partial(_decode, tokenizer),
        max_tokens,
        on_delta,
    )


def _decode(tokenizer, ids: list[int]) -> str:
    # A partial reply can end mid character; replacing beats failing.
    return tokenizer.untokenize(ids).decode('utf-8', errors='replace')


def _decode_partial(decode, ids: list[int], previous: str) -> str:
    """Decode a reply that may stop mid token group.

    The bits.N tokenizers withhold a trailing partial group, so their
    partial decodes are always clean prefixes. The guard stays for
    any tokenizer that raises mid group instead; until the group
    closes, the text of the previous token stands.
    """
    try:
        return decode(ids)
    except ValueError:
        return previous


def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    while n < len(a) and n < len(b) and a[n] == b[n]:
        n += 1
    return n
