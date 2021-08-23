r"""
Integration models
"""

import os

from .base import Model
from .scglue import AUTO, configure_dataset, SCGLUEModel


def load_model(fname: os.PathLike) -> Model:
    r"""
    Load model from file

    Parameters
    ----------
    fname
        Specifies path to the file
    """
    return Model.load(fname)
