r"""
Type hint definitions
"""

import numbers
from typing import Any, Mapping, Optional, TypeVar, Union

import numpy as np
import scipy.sparse

Array = Union[np.ndarray, scipy.sparse.spmatrix]
ArrayOrScalar = Union[np.ndarray, numbers.Number]
Kws = Optional[Mapping[str, Any]]
RandomState = Optional[Union[np.random.RandomState, int]]

T = TypeVar("T")  # Generic type var
