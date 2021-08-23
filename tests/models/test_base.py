r"""
Tests for the :mod:`scglue.models.base` module
"""

import pytest
import torch

import scglue


def test_base():
    model = scglue.models.Model()
    with pytest.raises(RuntimeError):
        _ = model.trainer
    model.compile()
    with pytest.raises(NotImplementedError):
        model.fit([torch.randn(128, 10)])
