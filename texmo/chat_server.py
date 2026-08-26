"""OpenAI-compatible chat-completions endpoint over a stored model.

Seats a texmo model as a participant in the dialog harness
(`scripts/dialog_harness.py`) without the harness knowing anything
about texmo: it POSTs `/v1/chat/completions` and reads
`choices[0].message.content`, exactly as it does for llama-server or
vLLM. This is the student side of the scripted-examiner eval
(milestone 2 in `docs/roadmap.md`).

The bridge is a format translation, not a protocol: chat-completions
`messages` become the flat `Name: utterance` dialog the models are
trained on (see `texmo.chat`), the reply is sampled up to the next
turn boundary, and only the utterance comes back -- no bot-name
prefix, no turn separator.

What is deliberately NOT supported:

- **system messages**: texmo models have no system channel. A system
  message is dropped rather than rejected, so an unchanged harness
  config (every seat carries a `system_prompt`) still runs.
- **streaming**: `stream` is accepted and ignored; the response is
  always a whole completion.
- everything else in the request body -- `top_p`, `logit_bias`,
  `seed`, the harness's `extra` passthrough (`chat_template_kwargs`
  and friends) -- is accepted and ignored too. An endpoint that 400s
  on an unknown field would force harness changes, which is the one
  thing this module exists to avoid.

`prompt_tokens` is reported as 0: the prompt is re-tokenized inside
`generate_reply` and the count never leaves it. `completion_tokens`
is real.

One model is loaded per process and generation is serialized by a
lock -- the harness is a single-threaded client, and the WSGI
listener stays threaded only so `/health` answers while a reply is
being sampled.
"""
import logging
import threading
import time
import uuid

from flask import Flask, jsonify, request
from werkzeug.serving import make_server

from .chat import TURN_SEPARATOR, build_prompt, generate_reply
from .model2_jax import Model2Jax

# Roles this endpoint maps onto dialog turns. `system` is handled
# separately (dropped); anything else is a request error.
_USER = 'user'
_ASSISTANT = 'assistant'
_SYSTEM = 'system'


def parse_messages(messages) -> list[tuple[str, str]]:
    """Validate `messages` and return the (role, text) turns.

    System messages are dropped here -- see the module docstring.
    Raises ValueError on anything that cannot become a dialog turn;
    the caller turns that into a 400.
    """
    if not isinstance(messages, list) or not messages:
        raise ValueError('`messages` must be a non-empty list')
    turns: list[tuple[str, str]] = []
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f'message {i} must be an object')
        role = message.get('role')
        content = message.get('content')
        if content is None:
            content = ''
        if not isinstance(content, str):
            raise ValueError(f'message {i}: `content` must be a string')
        if role == _SYSTEM:
            continue
        if role not in (_USER, _ASSISTANT):
            raise ValueError(
                f'message {i}: unsupported role {role!r} '
                f'(expected {_USER!r}, {_ASSISTANT!r} or {_SYSTEM!r})')
        turns.append((role, content.strip()))
    return turns


def messages_to_prompt(messages, user_name: str, bot_name: str) -> str:
    """Chat-completions `messages` -> a dialog prompt for the model.

    Every turn renders as `Name: utterance` and is closed by one turn
    separator, which is the invariant `texmo.chat.append_reply`
    maintains for the REPL's history; the final user turn plus the
    bot's opened turn come from `build_prompt` itself, so the two
    paths cannot drift.

    The last message must be the user's: the endpoint's whole job is
    to answer it. A trailing assistant message would mean "continue
    your own utterance", which the turn format cannot express.
    """
    turns = parse_messages(messages)
    if not turns:
        raise ValueError(
            '`messages` has no user or assistant turn (system messages '
            'are ignored -- this model has no system channel)')
    role, utterance = turns[-1]
    if role != _USER:
        raise ValueError(
            f'the last message must have role {_USER!r}, got {role!r}')
    names = {_USER: user_name, _ASSISTANT: bot_name}
    dialog = ''.join(
        f'{names[r]}: {text}{TURN_SEPARATOR}' for r, text in turns[:-1])
    return build_prompt(dialog, utterance, user_name, bot_name)


def finish_reason(reply: str) -> str:
    """'stop' if the reply reached a turn boundary, else 'length'.

    `generate_reply` stops on the boundary or on the token cap, and
    the boundary being in the text is exactly the difference.
    """
    return 'stop' if TURN_SEPARATOR in reply else 'length'


def reply_content(reply: str, bot_name: str) -> str:
    """The utterance alone: no boundary, no name prefix, stripped.

    The prompt already ends with `Bot: `, so a model has no reason to
    repeat it -- but a lossy tokenset can fold the prompt onto its own
    alphabet and the model may re-emit the label anyway, and the
    harness would record it as part of the utterance.
    """
    text = reply.split(TURN_SEPARATOR, 1)[0].strip()
    prefix = f'{bot_name}:'
    if text.startswith(prefix):
        text = text[len(prefix):].strip()
    return text


def completion_response(
    content: str,
    tokens: int,
    reason: str,
    model_name: str,
) -> dict:
    """The minimal chat-completion body the harness reads."""
    return {
        'id': f'chatcmpl-{uuid.uuid4().hex}',
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': model_name,
        'choices': [{
            'index': 0,
            'message': {'role': _ASSISTANT, 'content': content},
            'finish_reason': reason,
        }],
        # No prompt accounting: see the module docstring.
        'usage': {
            'prompt_tokens': 0,
            'completion_tokens': tokens,
            'total_tokens': tokens,
        },
    }


def error_body(message: str) -> dict:
    """OpenAI's error envelope, which clients know how to print."""
    return {'error': {'message': message, 'type': 'invalid_request_error'}}


class ChatServer:
    """One loaded model, answering chat-completions requests."""

    def __init__(
        self,
        model: Model2Jax,
        weights,
        tokens_name: str,
        model_name: str,
        temperature: float = 0.7,
        max_reply_tokens: int = 512,
        user_name: str = 'User',
        bot_name: str = 'Bot',
    ):
        self.model = model
        self.weights = weights
        self.tokens_name = tokens_name
        # Echoed back as `model`; the manifest path is what identifies
        # a texmo model, and the request's own `model` field is a
        # label the harness sends for llama-server's benefit.
        self.model_name = model_name
        self.temperature = temperature
        self.max_reply_tokens = max_reply_tokens
        self.user_name = user_name
        self.bot_name = bot_name
        # Generation is single-threaded: the JAX step path is shared
        # mutable-by-XLA state and the harness never overlaps requests
        # anyway. The lock is what makes a stray parallel client wait
        # instead of interleave.
        self._lock = threading.Lock()

    def sampling(self, body: dict) -> tuple[float, int]:
        """Per-request `temperature` / `max_tokens`, or the defaults."""
        temperature = body.get('temperature')
        if temperature is None:
            temperature = self.temperature
        elif isinstance(temperature, bool) or not isinstance(
                temperature, (int, float)):
            raise ValueError('`temperature` must be a number')
        elif temperature < 0:
            raise ValueError('`temperature` must be >= 0')
        max_tokens = body.get('max_tokens')
        if max_tokens is None:
            max_tokens = self.max_reply_tokens
        elif (isinstance(max_tokens, bool) or not isinstance(max_tokens, int)
                or max_tokens <= 0):
            raise ValueError('`max_tokens` must be a positive integer')
        return float(temperature), int(max_tokens)

    def complete(self, body) -> dict:
        """Answer one chat-completions request.

        Raises ValueError for anything the client got wrong.
        """
        if not isinstance(body, dict):
            raise ValueError('request body must be a JSON object')
        prompt = messages_to_prompt(
            body.get('messages'), self.user_name, self.bot_name)
        temperature, max_tokens = self.sampling(body)

        t0 = time.perf_counter()
        with self._lock:
            reply, tokens = generate_reply(
                self.model, self.weights, self.tokens_name, prompt,
                max_tokens, temperature)
        elapsed = time.perf_counter() - t0

        reason = finish_reason(reply)
        content = reply_content(reply, self.bot_name)
        logging.info(
            f'chat-completion: {len(body["messages"])} turns in, '
            f'{tokens} tokens out ({reason}), T={temperature}, '
            f'{elapsed:.2f}s')
        return completion_response(
            content, tokens, reason, self.model_name)

    def build_app(self) -> Flask:
        app = Flask('texmo-chat')

        @app.route('/v1/chat/completions', methods=['POST'])
        def _completions():
            # silent=True: a malformed body is a 400 from us with a
            # readable message, not a werkzeug HTML error page.
            try:
                return jsonify(self.complete(request.get_json(silent=True)))
            except ValueError as e:
                logging.warning(f'Rejecting completion request: {e}')
                return jsonify(error_body(str(e))), 400

        @app.route('/health', methods=['GET'])
        def _health():
            # The harness polls this before the first turn; the model
            # is loaded before the listener binds, so 200 here really
            # does mean ready.
            return jsonify({'status': 'ok'}), 200

        return app

    def serve(self, host: str, port: int) -> None:
        """Bind and serve until Ctrl-C."""
        server = make_server(host, port, self.build_app(), threaded=True)
        logging.info(
            f'Serving {self.model_name} on http://{host}:{port} '
            f'(POST /v1/chat/completions)')
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logging.info('Interrupted; shutting down')
        finally:
            server.shutdown()
