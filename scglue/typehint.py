r"""
Type hint definitions
"""

import numbers
from typing import Any, Mapping, Optional, TypeVar, Union

import h5py
import numpy as np
import scipy.sparse

try:
    from anndata._core.sparse_dataset import SparseDataset
except ImportError:  # Newer version of anndata
    from anndata._core.sparse_dataset import \
        BaseCompressedSparseDataset as SparseDataset

Array = Union[np.ndarray, scipy.sparse.spmatrix]
BackedArray = Union[h5py.Dataset, SparseDataset]
AnyArray = Union[Array, BackedArray]
ArrayOrScalar = Union[np.ndarray, numbers.Number]
Kws = Optional[Mapping[str, Any]]
RandomState = Optional[Union[np.random.RandomState, int]]

T = TypeVar("T")  # Generic type var
