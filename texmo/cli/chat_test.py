import argparse
import datetime
import io
import json

from texmo.cli.chat import (
    exchange_record,
    normalize_dialog,
    read_preamble,
    session_record,
    transcript_path,
    write_record,
)


def _args(**kwargs) -> argparse.Namespace:
    values = {
        'model': 'models/fold.32-8k-1.json',
        'temperature': 0.7,
        'max_reply_tokens': 512,
        'user_name': 'User',
        'bot_name': 'Bot',
        'preamble_file': None,
        'log_dir': 'data/chat_logs',
        'tokens_dir': 'tokens',
        'func': lambda a: None,
    }
    values.update(kwargs)
    return argparse.Namespace(**values)


def test_normalize_dialog_ends_with_one_separator():
    assert normalize_dialog('User: hi\n\nBot: hello.\n\n\n') == (
        'User: hi\n\nBot: hello.\n\n')
    assert normalize_dialog('User: hi') == 'User: hi\n\n'


def test_normalize_dialog_empty():
    assert normalize_dialog('') == ''
    assert normalize_dialog('\n\n') == ''


def test_read_preamble_none():
    assert read_preamble(None) == ''


def test_read_preamble_file(tmp_path):
    path = tmp_path / 'preamble.txt'
    path.write_text('User: hi\n\nBot: hello.\n', encoding='utf-8')
    assert read_preamble(str(path)) == 'User: hi\n\nBot: hello.\n\n'


def test_transcript_path():
    started = datetime.datetime(2026, 8, 21, 9, 5, 3)
    assert transcript_path('logs', started).replace('\\', '/') == (
        'logs/chat-20260821-090503.jsonl')


def test_session_record_is_json_serializable():
    started = datetime.datetime(2026, 8, 21, 9, 5, 3)
    record = session_record(_args(), 'tokens.32.fold', started)
    assert record['model'] == 'models/fold.32-8k-1.json'
    assert record['tokens'] == 'tokens.32.fold'
    assert record['temperature'] == 0.7
    assert record['started'] == '2026-08-21T09:05:03'
    # `func` is not serializable and must have been dropped.
    assert 'func' not in record['args']
    assert record['args']['bot_name'] == 'Bot'
    assert json.loads(json.dumps(record)) == record


def test_exchange_record():
    record = exchange_record('hi', 'hello.', 12, 1.23456)
    assert record == {
        'user': 'hi',
        'reply': 'hello.',
        'tokens_generated': 12,
        'elapsed_s': 1.235,
    }


def test_write_record_one_line_each():
    log = io.StringIO()
    write_record(log, {'a': 1})
    write_record(log, {'user': 'café'})
    lines = log.getvalue().splitlines()
    assert [json.loads(line) for line in lines] == [
        {'a': 1}, {'user': 'café'}]
    # Non-ASCII stays readable rather than being \u-escaped.
    assert 'café' in lines[1]
