r"""
Type hint definitions
"""

import numbers
from typing import Any, Mapping, Optional, TypeVar, Union

import anndata as ad
import h5py
import numpy as np
import scipy.sparse

Array = Union[np.ndarray, scipy.sparse.spmatrix]
BackedArray = Union[h5py.Dataset, ad._core.sparse_dataset.SparseDataset]
AnyArray = Union[Array, BackedArray]
ArrayOrScalar = Union[np.ndarray, numbers.Number]
Kws = Optional[Mapping[str, Any]]
RandomState = Optional[Union[np.random.RandomState, int]]

T = TypeVar("T")  # Generic type var
