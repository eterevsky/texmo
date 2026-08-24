import numpy as np

from .bits_tokenizer import (
    BitsTokenizer1,
    BitsTokenizer2,
    BitsTokenizer4,
    BytesTokenizer,
)


def test_bits4_partial_decode_withholds_the_open_group():
    tk = BitsTokenizer4()
    tokens = list(tk.tokenize(b' I'))
    assert tokens == [0, 2, 9, 4]  # low nibble first

    # The length-3 regression: numpy broadcast used to pair the lone
    # high nibble with both low nibbles, rendering " I" as " )".
    assert tk.untokenize(tokens[:0]) == b''
    assert tk.untokenize(tokens[:1]) == b''
    assert tk.untokenize(tokens[:2]) == b' '
    assert tk.untokenize(tokens[:3]) == b' '
    assert tk.untokenize(tokens[:4]) == b' I'


def test_bits1_partial_decode_never_zero_pads():
    tk = BitsTokenizer1()
    tokens = list(tk.tokenize(b'AB'))
    assert len(tokens) == 16

    assert tk.untokenize(tokens[:7]) == b''
    assert tk.untokenize(tokens[:8]) == b'A'
    assert tk.untokenize(tokens[:15]) == b'A'
    assert tk.untokenize(tokens) == b'AB'


def test_bits2_partial_decode_withholds_the_open_group():
    tk = BitsTokenizer2()
    tokens = list(tk.tokenize(b'AB'))
    assert len(tokens) == 8

    assert tk.untokenize(tokens[:3]) == b''
    assert tk.untokenize(tokens[:4]) == b'A'
    assert tk.untokenize(tokens[:7]) == b'A'
    assert tk.untokenize(tokens) == b'AB'


def test_round_trip_all_byte_values():
    data = bytes(range(256))
    for tk in (BitsTokenizer1(), BitsTokenizer2(), BitsTokenizer4(),
               BytesTokenizer()):
        assert tk.untokenize(list(tk.tokenize(data))) == data


def test_empty_decodes_to_empty():
    for tk in (BitsTokenizer1(), BitsTokenizer2(), BitsTokenizer4(),
               BytesTokenizer()):
        assert tk.untokenize([]) == b''


def test_untokenize_accepts_numpy_arrays():
    tk = BitsTokenizer4()
    assert tk.untokenize(np.array([0, 2, 9, 4])) == b' I'
    assert tk.untokenize(np.array([0, 2, 9])) == b' '
