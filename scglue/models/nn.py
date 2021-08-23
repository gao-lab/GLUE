r"""
Neural network modules, datasets & data loaders, and other utilities
"""

import collections
import functools
import multiprocessing
import operator
import os
import queue
import signal
from math import ceil, sqrt
from typing import Any, Hashable, List, Mapping, Optional, Tuple

import numpy as np
import pynvml
import scipy.sparse
import torch
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _NormBase

from ..num import vertex_degrees
from ..typehint import Array, RandomState
from ..utils import config, get_rs, logged, processes


#-------------------------- Neural network modules -----------------------------

class GraphConv(torch.nn.Module):

    r"""
    Graph convolution (propagation only)
    """

    def forward(
            self, input: torch.Tensor, eidx: torch.Tensor,
            enorm: torch.Tensor, esgn: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Forward propagation

        Parameters
        ----------
        input
            Input data (:math:`n_{vertices} \times n_{features}`)
        eidx
            Vertex indices of edges (:math:`2 \times n_{edges}`)
        enorm
            Normalized weight of edges (:math:`n_{edges}`)
        esgn
            Sign of edges (:math:`n_{edges}`)

        Returns
        -------
        result
            Graph convolution result (:math:`n_{vertices} \times n_{features}`)
        """
        sidx, tidx = eidx  # source index and target index
        message = input[sidx] * (esgn * enorm).unsqueeze(1)  # n_edges * n_features
        res = torch.zeros_like(input)
        tidx = tidx.unsqueeze(1).expand_as(message)  # n_edges * n_features
        res.scatter_add_(0, tidx, message)
        return res


class GraphAttent(torch.nn.Module):  # pragma: no cover

    r"""
    Graph attention

    Parameters
    ----------
    in_features
        Input dimensionality
    out_featres
        Output dimensionality

    Note
    ----
    **EXPERIMENTAL**
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = torch.nn.ParameterDict({
            "pos": torch.nn.Parameter(torch.Tensor(out_features, in_features)),
            "neg": torch.nn.Parameter(torch.Tensor(out_features, in_features))
        })
        self.head = torch.nn.ParameterDict({
            "pos": torch.nn.Parameter(torch.zeros(out_features * 2)),
            "neg": torch.nn.Parameter(torch.zeros(out_features * 2))
        })
        torch.nn.init.kaiming_uniform_(self.weight["pos"], sqrt(5))  # Following torch.nn.Linear
        torch.nn.init.kaiming_uniform_(self.weight["neg"], sqrt(5))  # Following torch.nn.Linear

    def forward(
            self, input: torch.Tensor, eidx: torch.Tensor,
            ewt: torch.Tensor, esgn: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Forward propagation

        Parameters
        ----------
        input
            Input data (:math:`n_{vertices} \times n_{features}`)
        eidx
            Vertex indices of edges (:math:`2 \times n_{edges}`)
        ewt
            Weight of edges (:math:`n_{edges}`)
        esgn
            Sign of edges (:math:`n_{edges}`)

        Returns
        -------
        result
            Graph attention result (:math:`n_{vertices} \times n_{features}`)
        """
        res_dict = {}
        for sgn in ("pos", "neg"):
            mask = esgn == 1 if sgn == "pos" else esgn == -1
            sidx, tidx = eidx[:, mask]
            ptr = input @ self.weight[sgn].T
            alpha = torch.cat([ptr[sidx], ptr[tidx]], dim=1) @ self.head[sgn]
            alpha = F.leaky_relu(alpha, negative_slope=0.2).exp() * ewt[mask]
            normalizer = torch.zeros(ptr.shape[0], device=ptr.device)
            normalizer.scatter_add_(0, tidx, alpha)
            alpha = alpha / normalizer[tidx]  # Only entries with non-zero denominators will be used
            message = ptr[sidx] * alpha.unsqueeze(1)
            res = torch.zeros_like(ptr)
            tidx = tidx.unsqueeze(1).expand_as(message)
            res.scatter_add_(0, tidx, message)
            res_dict[sgn] = res
        return res_dict["pos"] + res_dict["neg"]


#------------------------------- Data handlers ---------------------------------

@logged
class Dataset(torch.utils.data.Dataset):

    r"""
    Abstract dataset interface extending that of :class:`torch.utils.data.Dataset`

    Parameters
    ----------
    getitem_size
        Unitary fetch size for each __getitem__ call
    """

    def __init__(self, getitem_size: int = 1) -> None:
        super().__init__()
        self.getitem_size = getitem_size
        self.shuffle_seed: Optional[int] = None
        self.seed_queue: Optional[multiprocessing.Queue] = None
        self.propose_queue: Optional[multiprocessing.Queue] = None
        self.propose_cache: Mapping[int, Any] = {}

    @property
    def has_workers(self) -> bool:
        r"""
        Whether background shuffling workers have been registered
        """
        self_processes = processes[id(self)]
        pl = bool(self_processes)
        sq = self.seed_queue is not None
        pq = self.propose_queue is not None
        if not pl == sq == pq:
            raise RuntimeError("Background shuffling seems broken!")
        return pl and sq and pq

    def prepare_shuffle(self, num_workers: int = 1, random_seed: int = 0) -> None:
        r"""
        Prepare dataset for custom shuffling

        Parameters
        ----------
        num_workers
            Number of background workers for data shuffling
        random_seed
            Initial random seed (will increase by 1 with every shuffle call)
        """
        if self.has_workers:
            self.clean()
        self_processes = processes[id(self)]
        self.shuffle_seed = random_seed
        if num_workers:
            self.seed_queue = multiprocessing.Queue()
            self.propose_queue = multiprocessing.Queue()
            for i in range(num_workers):
                p = multiprocessing.Process(target=self.shuffle_worker)
                p.start()
                self.logger.debug("Started background process: %d", p.pid)
                self_processes[p.pid] = p
                self.seed_queue.put(self.shuffle_seed + i)

    def shuffle(self) -> None:
        r"""
        Custom shuffling
        """
        if self.has_workers:
            self_processes = processes[id(self)]
            self.seed_queue.put(self.shuffle_seed + len(self_processes))  # Look ahead
            while self.shuffle_seed not in self.propose_cache:
                shuffle_seed, shuffled = self.propose_queue.get()
                self.propose_cache[shuffle_seed] = shuffled
            self.accept_shuffle(self.propose_cache.pop(self.shuffle_seed))
        else:
            self.accept_shuffle(self.propose_shuffle(self.shuffle_seed))
        self.shuffle_seed += 1

    def shuffle_worker(self) -> None:
        r"""
        Background shuffle worker
        """
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        while True:
            seed = self.seed_queue.get()
            if seed is None:
                self.propose_queue.put((None, os.getpid()))
                break
            self.propose_queue.put((seed, self.propose_shuffle(seed)))

    def propose_shuffle(self, seed: int) -> Any:
        r"""
        Propose shuffling using a given random seed

        Parameters
        ----------
        seed
            Random seed

        Returns
        -------
        shuffled
            Shuffled result
        """
        raise NotImplementedError  # pragma: no cover

    def accept_shuffle(self, shuffled: Any) -> None:
        r"""
        Accept shuffling result

        Parameters
        ----------
        shuffled
            Shuffled result
        """
        raise NotImplementedError  # pragma: no cover

    def clean(self) -> None:
        r"""
        Clean up multi-process resources used in custom shuffling
        """
        self_processes = processes[id(self)]
        if not self.has_workers:
            return
        for _ in self_processes:
            self.seed_queue.put(None)
        self.propose_cache.clear()
        while self_processes:
            try:
                first, second = self.propose_queue.get(
                    timeout=config.FORCE_TERMINATE_WORKER_PATIENCE
                )
            except queue.Empty:
                break
            if first is not None:
                continue
            pid = second
            self_processes[pid].join()
            self.logger.debug("Joined background process: %d", pid)
            del self_processes[pid]
        for pid in list(self_processes.keys()):  # If some background processes failed to exit gracefully
            self_processes[pid].terminate()
            self_processes[pid].join()
            self.logger.debug("Terminated background process: %d", pid)
            del self_processes[pid]
        self.propose_queue = None
        self.seed_queue = None

    def __del__(self) -> None:
        self.clean()


@logged
class ArrayDataset(Dataset):

    r"""
    Dataset for :class:`numpy.ndarray` and :class:`scipy.sparse.spmatrix`
    objects with grouping support. Arrays from the same group should have
    the same size in the first dimension, while arrays from different groups
    can have varying sizes (arrays of smaller sizes are cycled). Also, data
    fetched from this dataset are automatically densified.

    Parameters
    ----------
    *arrays
        An arbitrary number of data arrays
    grouping
        Array grouping. Arrays in the same group should have the same number of
        samples. During shuffling and splitting, sample correspondence is
        preserved within the same group, but not across different groups.
        E.g., `grouping=[0, 1, 0, 1, 1]` indicates that array 0, 2 are paired,
        and array 1, 3, 4 are paired. If no grouping pattern is specified,
        it is assumed that all arrays are in the same group.
    getitem_size
        Unitary fetch size for each __getitem__ call

    Note
    ----
    We keep using arrays because sparse tensors do not support slicing.
    Arrays are only converted to tensors after minibatch slicing.
    """

    def __init__(
            self, *arrays: Array, grouping: Optional[List[Hashable]] = None,
            getitem_size: int = 1
    ) -> None:
        super().__init__(getitem_size=getitem_size)
        arrays = [
            array.tocsr() if scipy.sparse.issparse(array) else np.asarray(array)
            for array in arrays
        ]
        self.arrays = arrays
        self.grouping = grouping
        self.size = max(self.sizes.values())

    @property
    def grouping(self) -> List[Hashable]:
        r"""
        Array grouping
        """
        return self._grouping

    @grouping.setter
    def grouping(
            self, grouping: Optional[List[Hashable]] = None
    ) -> None:
        grouping = np.asarray(grouping or [0] * len(self.arrays))
        if len(grouping) != len(self.arrays):
            raise ValueError("Invalid grouping pattern!")
        self._groups = collections.OrderedDict([
            (g, np.where(grouping == g)[0].tolist())
            for g in np.unique(grouping)
        ])
        self._sizes = collections.OrderedDict()
        for g, group in self.groups.items():
            size_set = set(self.arrays[i].shape[0] for i in group)
            if len(size_set) > 1:
                raise ValueError(
                    "Paired arrays do not match in the first dimension!"
                )
            self.sizes[g] = size_set.pop()
            if self.sizes[g] == 0:
                raise ValueError("Zero-sized array is not allowed!")
        self._grouping = grouping.tolist()

    @property
    def groups(self) -> Mapping[Hashable, List[int]]:
        r"""
        Indices of arrays in each group
        """
        return self._groups

    @property
    def sizes(self) -> Mapping[Hashable, int]:
        r"""
        Array sizes in each group
        """
        return self._sizes

    def __len__(self) -> int:
        return ceil(self.size / self.getitem_size)

    def __getitem__(self, index: int) -> List[torch.Tensor]:
        index = np.arange(
            index * self.getitem_size,
            min((index + 1) * self.getitem_size, self.size)
        )
        return [
            torch.as_tensor(array[np.mod(index, array.shape[0])].toarray())
            if scipy.sparse.issparse(array)
            else torch.as_tensor(array[np.mod(index, array.shape[0])])
            for array in self.arrays
        ]

    def propose_shuffle(self, seed: int) -> List[Array]:
        rs = get_rs(seed)
        shuffled = [None] * len(self.arrays)
        for g, group in self.groups.items():
            permutation = rs.permutation(self.sizes[g])
            for i in group:
                shuffled[i] = self.arrays[i][permutation]
        return shuffled

    def accept_shuffle(self, shuffled: List[Array]) -> None:
        self.arrays = shuffled

    def random_split(
            self, fractions: List[float], random_state: RandomState = None
    ) -> Tuple[List["ArrayDataset"], List[Mapping[Hashable, np.ndarray]]]:
        r"""
        Randomly split the dataset into multiple subdatasets according to
        given fractions.

        Parameters
        ----------
        fractions
            Fraction of each split
        random_state
            Random state

        Returns
        -------
        subdatasets
            A list of splitted subdatasets
        split_idx_list
            A list containing group-index mapping for each fraction
        """
        if min(fractions) <= 0:
            raise ValueError("Fractions should be greater than 0!")
        if sum(fractions) != 1:
            raise ValueError("Fractions do not sum to 1!")
        rs = get_rs(random_state)
        cum_frac = np.cumsum(fractions)
        subarrays_list = [[None] * len(self.arrays) for _ in range(len(fractions))]
        for g, group in self.groups.items():
            permutation = rs.permutation(self.sizes[g])
            cum_idx = np.round(cum_frac * self.sizes[g]).astype(int)
            split_idx = np.split(permutation, cum_idx[:-1])  # Last idx produces an extra empty split
            for i, idx in enumerate(split_idx):
                for j in group:
                    subarrays_list[i][j] = self.arrays[j][idx]
        subdatasets = [
            ArrayDataset(
                *subarrays, grouping=self.grouping, getitem_size=self.getitem_size
            ) for subarrays in subarrays_list
        ]
        return subdatasets


@logged
class GraphDataset(Dataset):

    r"""
    Dataset for graphs with support for negative sampling

    Parameters
    ----------
    eidx
        Vertex indices of edges (:math:`2 \times n_{edges}`)
    ewt
        Weight of edges (:math:`n_{edges}`), must be in range ``(0.0, 1.0]``.
    esgn
        Sign of edges (:math:`n_{edges}`)
    neg_samples
        Number of negative samples per edge
    weighted_sampling
        Whether to do negative sampling based on vertex importance
    deemphasize_loops
        Whether to deemphasize self-loops when computing vertex importance
    getitem_size
        Unitary fetch size for each __getitem__ call

    Note
    ----
    Custom shuffling performs negative sampling.
    """

    def __init__(
            self, eidx: np.ndarray, ewt: np.ndarray, esgn: np.ndarray,
            neg_samples: int = 1, weighted_sampling: bool = True,
            deemphasize_loops: bool = True, getitem_size: int = 1
    ) -> None:
        super().__init__(getitem_size=getitem_size)
        if eidx.ndim != 2 or ewt.ndim != 1 or esgn.ndim != 1 or eidx.shape[0] != 2:
            raise ValueError("Invalid data shape!")
        if not eidx.shape[1] == ewt.shape[0] == esgn.shape[0]:
            raise ValueError("Inconsistent edge number!")
        if eidx.min() < 0:
            raise ValueError("Invalid edge index!")
        if np.any(ewt <= 0):
            raise ValueError("Invalid edge weight!")
        if set(esgn).difference({-1, 1}):
            raise ValueError("Invalid edge sign!")

        self.eidx = eidx
        self.ewt = ewt
        self.esgn = esgn
        self.eset = {
            (i, j, s) for (i, j), s in
            zip(self.eidx.T, self.esgn)
        }

        self.vnum = self.eidx.max() + 1
        if weighted_sampling:
            if deemphasize_loops:
                non_loop = self.eidx[0] != self.eidx[1]
                eidx = self.eidx[:, non_loop]
                ewt = self.ewt[non_loop]
            else:
                eidx = self.eidx
                ewt = self.ewt
            degree = vertex_degrees(eidx, ewt, vnum=self.vnum, direction="both")
        else:
            degree = np.ones(self.vnum, dtype=self.ewt.dtype)
        self.vprob = degree / degree.sum()  # Vertex sampling probability

        effective_enum = self.ewt.sum()
        self.eprob = self.ewt / effective_enum  # Edge sampling probability
        self.effective_enum = round(effective_enum)

        self.neg_samples = neg_samples
        self.size = self.effective_enum * (1 + self.neg_samples)
        self.samp_eidx: Optional[np.ndarray] = None
        self.samp_ewt: Optional[np.ndarray] = None
        self.samp_esgn: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return ceil(self.size / self.getitem_size)

    def __getitem__(self, index: int) -> List[torch.Tensor]:
        index = slice(
            index * self.getitem_size,
            min((index + 1) * self.getitem_size, self.size)
        )
        return [
            torch.as_tensor(self.samp_eidx[:, index]),
            torch.as_tensor(self.samp_ewt[index]),
            torch.as_tensor(self.samp_esgn[index])
        ]

    def propose_shuffle(
            self, seed: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        (pi, pj), pw, ps = self.eidx, self.ewt, self.esgn
        rs = get_rs(seed)
        psamp = rs.choice(self.ewt.size, self.effective_enum, replace=True, p=self.eprob)
        pi_, pj_, pw_, ps_ = pi[psamp], pj[psamp], pw[psamp], ps[psamp]
        pw_ = np.ones_like(pw_)
        ni_ = np.tile(pi_, self.neg_samples)
        nw_ = np.zeros(pw_.size * self.neg_samples, dtype=pw_.dtype)
        ns_ = np.tile(ps_, self.neg_samples)
        nj_ = rs.choice(self.vnum, pj_.size * self.neg_samples, replace=True, p=self.vprob)

        remain = np.where([
            item in self.eset
            for item in zip(ni_, nj_, ns_)
        ])[0]
        while remain.size:  # NOTE: Potential infinite loop if graph too dense
            newnj = rs.choice(self.vnum, remain.size, replace=True, p=self.vprob)
            nj_[remain] = newnj
            remain = remain[[
                item in self.eset
                for item in zip(ni_[remain], newnj, ns_[remain])
            ]]
        idx = np.stack([np.concatenate([pi_, ni_]), np.concatenate([pj_, nj_])])
        w = np.concatenate([pw_, nw_])
        s = np.concatenate([ps_, ns_])
        perm = rs.permutation(idx.shape[1])
        return idx[:, perm], w[perm], s[perm]

    def accept_shuffle(
            self, shuffled: Tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        self.samp_eidx, self.samp_ewt, self.samp_esgn = shuffled


class DataLoader(torch.utils.data.DataLoader):

    r"""
    Custom data loader that manually shuffles the internal dataset before each
    round of iteration (see :class:`torch.utils.data.DataLoader` for usage)
    """

    def __init__(self, dataset: Dataset, **kwargs) -> None:
        super().__init__(dataset, **kwargs)
        self.collate_fn = self._collate
        self.shuffle = kwargs["shuffle"] if "shuffle" in kwargs else False

    def __iter__(self) -> "DataLoader":
        if self.shuffle:
            self.dataset.shuffle()  # Customized shuffling
        return super().__iter__()

    @staticmethod
    def _collate(batch):
        return tuple(map(lambda x: torch.cat(x, dim=0), zip(*batch)))


class GraphDataLoader(DataLoader):

    r"""
    Data loader for graph datasets with a special collate function (see
    :class:`torch.utils.data.DataLoader` for usage)
    """

    def __init__(self, dataset: GraphDataset, **kwargs) -> None:
        super().__init__(dataset, **kwargs)

    @staticmethod
    def _collate(batch):
        eidx, ewt, esgn = zip(*batch)
        eidx = torch.cat(eidx, dim=1)
        ewt = torch.cat(ewt, dim=0)
        esgn = torch.cat(esgn, dim=0)
        return eidx, ewt, esgn


class ParallelDataLoader:

    r"""
    Parallel data loader

    Parameters
    ----------
    *data_loaders
        An arbitrary number of data loaders
    cycle_flags
        Whether each data loader should be cycled in case they are of
        different lengths, by default none of them are cycled.
    """

    def __init__(
            self, *data_loaders: DataLoader,
            cycle_flags: Optional[List[bool]] = None
    ) -> None:
        cycle_flags = cycle_flags or [False] * len(data_loaders)
        if len(cycle_flags) != len(data_loaders):
            raise ValueError("Invalid cycle flags!")
        self.cycle_flags = cycle_flags
        self.data_loaders = list(data_loaders)
        self.length = len(self.data_loaders)
        self.iterators = None

    def __iter__(self) -> "ParallelDataLoader":
        self.iterators = [iter(loader) for loader in self.data_loaders]
        return self

    def _next(self, i: int) -> List[torch.Tensor]:
        try:
            return next(self.iterators[i])
        except StopIteration as e:
            if self.cycle_flags[i]:
                self.iterators[i] = iter(self.data_loaders[i])
                return next(self.iterators[i])
            raise e

    def __next__(self) -> List[torch.Tensor]:
        return functools.reduce(
            operator.add, [self._next(i) for i in range(self.length)]
        )


#----------------------------- Utility functions -------------------------------

def freeze_running_stats(m: torch.nn.Module) -> None:
    r"""
    Selectively stops normalization layers from updating running stats

    Parameters
    ----------
    m
        Network module
    """
    if isinstance(m, _NormBase):
        m.eval()


def get_default_numpy_dtype() -> type:
    r"""
    Get numpy dtype matching that of the pytorch default dtype

    Returns
    -------
    dtype
        Default numpy dtype
    """
    return getattr(np, str(torch.get_default_dtype()).replace("torch.", ""))


@logged
@functools.lru_cache(maxsize=1)
def autodevice() -> torch.device:
    r"""
    Get torch computation device automatically
    based on GPU availability and memory usage

    Returns
    -------
    device
        Computation device
    """
    used_device = -1
    if not config.CPU_ONLY:
        try:
            pynvml.nvmlInit()
            free_mems = np.array([
                pynvml.nvmlDeviceGetMemoryInfo(
                    pynvml.nvmlDeviceGetHandleByIndex(i)
                ).free for i in range(pynvml.nvmlDeviceGetCount())
            ])
            for item in config.MASKED_GPUS:
                free_mems[item] = -1
            best_devices = np.where(free_mems == free_mems.max())[0]
            used_device = np.random.choice(best_devices, 1)[0]
            if free_mems[used_device] < 0:
                used_device = -1
        except pynvml.NVMLError:
            pass
    if used_device == -1:
        autodevice.logger.info("Using CPU as computation device.")
        return torch.device("cpu")
    autodevice.logger.info("Using GPU %d as computation device.", used_device)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(used_device)
    return torch.device("cuda")
