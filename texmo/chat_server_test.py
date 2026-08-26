"""Tests for the OpenAI-compatible chat endpoint.

The mapping functions are tested against `texmo.chat`'s own prompt
builders, so the two cannot drift; the HTTP layer runs on Flask's
test client (no sockets) with either a stubbed `generate_reply` or,
for the end-to-end shape, one tiny untrained model.
"""
import jax

from texmo import chat_server
from texmo.chat import TURN_SEPARATOR, append_reply, build_prompt
from texmo.chat_server import (
    ChatServer,
    finish_reason,
    messages_to_prompt,
    reply_content,
)
from texmo.precision import Precision
from texmo.spec_parser import parse_model2

# One byte per token, as the dialog models use.
SPEC = 'tokens.32.fold.oh|dense.4.gelu'


def _make_server(**kwargs) -> ChatServer:
    """A server over a tiny untrained model (replies are noise)."""
    md = parse_model2(SPEC, precision=Precision.FP32)
    model = md.build_jax()
    weights = model.init_weights(jax.random.PRNGKey(0))
    args = {
        'model_name': 'models/test.json',
        'temperature': 0.7,
        'max_reply_tokens': 32,
    }
    args.update(kwargs)
    return ChatServer(model, weights, md.codec.tokens_name, **args)


def _stub_server(monkeypatch, reply: str = ' hello.\n\n', **kwargs):
    """A server whose generation is a recorded stub.

    Returns (server, calls) where each call is the (prompt,
    max_tokens, temperature) `generate_reply` was handed.
    """
    calls = []

    def fake_generate_reply(model, weights, tokens_name, prompt,
                            max_tokens, temperature, on_delta=None):
        calls.append((prompt, max_tokens, temperature))
        return reply, len(reply)

    monkeypatch.setattr(chat_server, 'generate_reply', fake_generate_reply)
    server = ChatServer(None, None, 'tokens.32.fold', **{
        'model_name': 'models/test.json', **kwargs})
    return server, calls


def _msg(role: str, content: str) -> dict:
    return {'role': role, 'content': content}


def test_single_user_message_matches_build_prompt():
    messages = [_msg('user', 'hi')]
    assert messages_to_prompt(messages, 'User', 'Bot') == build_prompt(
        '', 'hi', 'User', 'Bot')


def test_history_matches_the_repl_prompt_chain():
    """The same dialog assembled turn by turn, as `texmo.py chat` does."""
    dialog = append_reply(build_prompt('', 'hi', 'User', 'Bot'), 'hello.')
    expected = build_prompt(dialog, 'who are you?', 'User', 'Bot')
    messages = [
        _msg('user', 'hi'),
        _msg('assistant', 'hello.'),
        _msg('user', 'who are you?'),
    ]
    assert messages_to_prompt(messages, 'User', 'Bot') == expected


def test_system_message_is_ignored():
    messages = [
        _msg('system', 'You are a patient teacher.'),
        _msg('user', 'hi'),
    ]
    prompt = messages_to_prompt(messages, 'User', 'Bot')
    assert prompt == 'User: hi\n\nBot: '
    assert 'teacher' not in prompt


def test_custom_names_are_used_for_both_roles():
    messages = [
        _msg('user', 'hi'),
        _msg('assistant', 'hello.'),
        _msg('user', 'bye'),
    ]
    assert messages_to_prompt(messages, 'Ann', 'Bob') == (
        'Ann: hi\n\nBob: hello.\n\nAnn: bye\n\nBob: ')


def test_trailing_assistant_message_is_rejected():
    messages = [_msg('user', 'hi'), _msg('assistant', 'hello.')]
    try:
        messages_to_prompt(messages, 'User', 'Bot')
    except ValueError as e:
        assert 'user' in str(e)
    else:
        raise AssertionError('expected ValueError')


def test_reply_content_strips_the_boundary_and_the_name():
    assert reply_content(' hello.\n\nUser: more', 'Bot') == 'hello.'
    assert reply_content('Bot: hello.\n\n', 'Bot') == 'hello.'
    assert reply_content('hello and', 'Bot') == 'hello and'


def test_finish_reason():
    assert finish_reason(' hi.\n\n') == 'stop'
    assert finish_reason(' hi and') == 'length'


def test_health():
    server = ChatServer(None, None, 'tokens.32.fold',
                        model_name='models/test.json')
    response = server.build_app().test_client().get('/health')
    assert response.status_code == 200


def test_completion_shape_over_a_tiny_model():
    server = _make_server()
    client = server.build_app().test_client()
    response = client.post('/v1/chat/completions', json={
        'model': 'texmo',
        'messages': [_msg('user', 'hi')],
        'stream': False,
    })
    assert response.status_code == 200
    body = response.get_json()
    assert body['object'] == 'chat.completion'
    assert body['model'] == 'models/test.json'
    assert body['id'].startswith('chatcmpl-')
    choice = body['choices'][0]
    assert choice['index'] == 0
    assert choice['message']['role'] == 'assistant'
    content = choice['message']['content']
    assert isinstance(content, str)
    assert TURN_SEPARATOR not in content
    assert content == content.strip()
    assert choice['finish_reason'] in ('stop', 'length')
    n = body['usage']['completion_tokens']
    assert 0 < n <= 32
    assert body['usage']['total_tokens'] == n


def test_max_tokens_one_cannot_reach_a_boundary():
    """A one-byte token can't hold '\\n\\n', so the cap must fire."""
    server = _make_server()
    client = server.build_app().test_client()
    response = client.post('/v1/chat/completions', json={
        'messages': [_msg('user', 'hi')],
        'max_tokens': 1,
    })
    body = response.get_json()
    assert body['choices'][0]['finish_reason'] == 'length'
    assert body['usage']['completion_tokens'] == 1


def test_request_overrides_the_defaults(monkeypatch):
    server, calls = _stub_server(
        monkeypatch, temperature=0.7, max_reply_tokens=512)
    client = server.build_app().test_client()
    client.post('/v1/chat/completions',
                json={'messages': [_msg('user', 'hi')]})
    client.post('/v1/chat/completions', json={
        'messages': [_msg('user', 'hi')],
        'temperature': 0.1,
        'max_tokens': 8,
    })
    assert [(t, n) for _, n, t in calls] == [(0.7, 512), (0.1, 8)]


def test_harness_request_shape_is_accepted(monkeypatch):
    """The exact body `dialog_harness.build_request` produces, `extra`
    passthrough included -- unknown fields must not 400."""
    server, calls = _stub_server(monkeypatch)
    client = server.build_app().test_client()
    response = client.post('/v1/chat/completions', json={
        'model': 'texmo-fold-8k',
        'messages': [
            _msg('system', 'You are a curious child.'),
            _msg('user', 'Hi there!'),
        ],
        'stream': False,
        'temperature': 0.8,
        'top_p': 0.95,
        'max_tokens': 96,
        'chat_template_kwargs': {'enable_thinking': False},
    })
    assert response.status_code == 200
    assert response.get_json()['choices'][0]['message']['content'] == 'hello.'
    # The system prompt never reached the model.
    assert calls[0][0] == 'User: Hi there!\n\nBot: '


def test_bad_requests_are_400(monkeypatch):
    server, _ = _stub_server(monkeypatch)
    client = server.build_app().test_client()
    bad = [
        {'messages': [_msg('user', 'hi'), _msg('assistant', 'hello.')]},
        {'messages': []},
        {'messages': [_msg('system', 'only a system prompt')]},
        {'messages': [_msg('tool', 'nope')]},
        {'messages': 'hi'},
        {},
        {'messages': [_msg('user', 'hi')], 'temperature': 'warm'},
        {'messages': [_msg('user', 'hi')], 'max_tokens': 0},
    ]
    for body in bad:
        response = client.post('/v1/chat/completions', json=body)
        assert response.status_code == 400, body
        assert response.get_json()['error']['message']
