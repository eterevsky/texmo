from copy import deepcopy

from .search2 import _generate_limits, _run_limit_sequences


def test_run_limit_sequences():
    seqs = []
    for i, s in enumerate(_run_limit_sequences()):
        seqs.append(list(s))
        if i == 3:
            break
    assert seqs == [
        [1],
        [2, 1, 1],
        [3, 2, 2, 1, 1, 1, 1, 1, 1],
        [4, 3, 3] + [2] * 6 + [1] * 18,
    ]


def test_generate_limits():
    limits = []
    for seq in _generate_limits():
        limits.append(deepcopy(seq))
        if len(limits) == 16: break

    assert limits == [
        [[1, 0]],
        [[2, 0]],
        [[2, 1]],
        [[2, 1], [1, 0]],
        [[3, 1], [1, 0]],
        [[3, 2], [1, 0]],
        [[3, 2], [2, 0]],
        [[3, 2], [2, 1]],
        [[3, 2], [2, 1], [1, 0]],
        [[4, 2], [2, 1], [1, 0]],
        [[4, 3], [2, 1], [1, 0]],
        [[4, 3], [3, 1], [1, 0]],
        [[4, 3], [3, 2], [1, 0]],
        [[4, 3], [3, 2], [2, 0]],
        [[4, 3], [3, 2], [2, 1]],
        [[4, 3], [3, 2], [2, 1], [1, 0]],
    ]