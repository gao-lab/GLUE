r"""
Tests for the :mod:`scglue.plot` module
"""

# pylint: disable=redefined-outer-name, wildcard-import, unused-wildcard-import

import scglue

from .fixtures import *


def test_set_publication_params():
    scglue.plot.set_publication_params()


def test_sankey(rna):
    _ = scglue.plot.sankey(rna.obs["ct"], rna.obs["ct"], show=False)


def test_roc(ewt):
    _ = scglue.plot.roc(ewt > 0.5, ewt)


def test_prc(ewt):
    _ = scglue.plot.prc(ewt > 0.5, ewt)
