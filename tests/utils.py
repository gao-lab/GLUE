r"""
Utilities for tests
"""

# pylint: disable=missing-function-docstring

import numpy as np


def cmp_arrays(a, b, squeeze=False):
    a, b = np.asarray(a), np.asarray(b)
    if squeeze:
        a, b = a.squeeze(), b.squeeze()
    assert np.array_equal(a, b)


def cmp_graphs(a, b):
    set_a = set(
        (e, tuple(sorted(p for p in a.get_edge_data(*e).items())))
        for e in a.edges
    )
    set_b = set(
        (e, tuple(sorted(p for p in b.get_edge_data(*e).items())))
        for e in b.edges
    )
    assert set_a == set_b
