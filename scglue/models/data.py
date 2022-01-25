r"""
Data handling utilities
"""

import functools
import multiprocessing
import operator
import os
import queue
import signal
from math import ceil
from typing import Any, List, Mapping, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse
import torch
from anndata._core.sparse_dataset import SparseDataset

from ..num import vertex_degrees
from ..typehint import Array, RandomState
from ..utils import config, get_rs, logged, processes
from .nn import get_default_numpy_dtype


#---------------------------------- Datasets -----------------------------------

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
    Array dataset for :class:`numpy.ndarray` and :class:`scipy.sparse.spmatrix`
    objects. Different arrays are considered as unpaired, and thus do not need
    to have identical sizes in the first dimension. Smaller arrays are recycled.
    Also, data fetched from this dataset are automatically densified.

    Parameters
    ----------
    *arrays
        An arbitrary number of data arrays

    Note
    ----
    We keep using arrays because sparse tensors do not support slicing.
    Arrays are only converted to tensors after minibatch slicing.
    """

    def __init__(self, *arrays: Array, getitem_size: int = 1) -> None:
        super().__init__(getitem_size=getitem_size)
        self.sizes = None
        self.size = None
        self.view_idx = None
        self.shuffle_idx = None
        self.arrays = arrays

    @property
    def arrays(self) -> List[Array]:
        r"""
        Internal array objects
        """
        return self._arrays

    @arrays.setter
    def arrays(self, arrays: List[Array]) -> None:
        self.sizes = [array.shape[0] for array in arrays]
        if min(self.sizes) == 0:
            raise ValueError("Empty array is not allowed!")
        self.size = max(self.sizes)
        self.view_idx = [np.arange(s) for s in self.sizes]
        self.shuffle_idx = self.view_idx
        self._arrays = arrays

    def __len__(self) -> int:
        return ceil(self.size / self.getitem_size)

    def __getitem__(self, index: int) -> List[torch.Tensor]:
        index = np.arange(
            index * self.getitem_size,
            min((index + 1) * self.getitem_size, self.size)
        )
        return [
            torch.as_tensor(a[self.shuffle_idx[i][np.mod(index, self.sizes[i])]].toarray())
            if scipy.sparse.issparse(a) or isinstance(a, SparseDataset)
            else torch.as_tensor(a[self.shuffle_idx[i][np.mod(index, self.sizes[i])]])
            for i, a in enumerate(self.arrays)
        ]

    def propose_shuffle(self, seed: int) -> List[np.ndarray]:
        rs = get_rs(seed)
        return [rs.permutation(view_idx) for view_idx in self.view_idx]

    def accept_shuffle(self, shuffled: List[np.ndarray]) -> None:
        self.shuffle_idx = shuffled

    def random_split(
            self, fractions: List[float], random_state: RandomState = None
    ) -> List["ArrayDataset"]:
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
        """
        if min(fractions) <= 0:
            raise ValueError("Fractions should be greater than 0!")
        if sum(fractions) != 1:
            raise ValueError("Fractions do not sum to 1!")
        rs = get_rs(random_state)
        cum_frac = np.cumsum(fractions)
        subdatasets = [
            ArrayDataset(
                *self.arrays, getitem_size=self.getitem_size
            ) for _ in fractions
        ]
        for j, view_idx in enumerate(self.view_idx):
            view_idx = rs.permutation(view_idx)
            split_pos = np.round(cum_frac * view_idx.size).astype(int)
            split_idx = np.split(view_idx, split_pos[:-1])  # Last pos produces an extra empty split
            for i, idx in enumerate(split_idx):
                subdatasets[i].sizes[j] = len(idx)
                subdatasets[i].view_idx[j] = idx
                subdatasets[i].shuffle_idx[j] = idx
        return subdatasets


@logged
class GraphDataset(Dataset):

    r"""
    Dataset for graphs with support for negative sampling

    Parameters
    ----------
    graph
        Graph object
    vertices
        Indexer of graph vertices
    edge_weight
        Key of edge attribute for edge weight
    edge_sign
        Key of edge attribute for edge sign
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
            self, graph: nx.Graph, vertices: pd.Index,
            edge_weight: str, edge_sign: str,
            neg_samples: int = 1, weighted_sampling: bool = True,
            deemphasize_loops: bool = True, getitem_size: int = 1
    ) -> None:
        super().__init__(getitem_size=getitem_size)
        self.eidx, self.ewt, self.esgn = \
            self.graph2triplet(graph, vertices, edge_weight, edge_sign)
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
        degree_sum = degree.sum()
        if degree_sum:
            self.vprob = degree / degree_sum  # Vertex sampling probability
        else:  # Possible when `deemphasize_loops` is set on a loop-only graph
            self.vprob = np.ones(self.vnum, dtype=self.ewt.dtype) / self.vnum

        effective_enum = self.ewt.sum()
        self.eprob = self.ewt / effective_enum  # Edge sampling probability
        self.effective_enum = round(effective_enum)

        self.neg_samples = neg_samples
        self.size = self.effective_enum * (1 + self.neg_samples)
        self.samp_eidx: Optional[np.ndarray] = None
        self.samp_ewt: Optional[np.ndarray] = None
        self.samp_esgn: Optional[np.ndarray] = None

    def graph2triplet(
            self, graph: nx.Graph, vertices: pd.Index,
            edge_weight: str, edge_sign: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Convert graph object to graph triplet

        Parameters
        ----------
        graph
            Graph object
        vertices
            Graph vertices
        edge_weight
            Key of edge attribute for edge weight
        edge_sign
            Key of edge attribute for edge sign

        Returns
        -------
        eidx
            Vertex indices of edges (:math:`2 \times n_{edges}`)
        ewt
            Weight of edges (:math:`n_{edges}`)
        esgn
            Sign of edges (:math:`n_{edges}`)
        """
        graph = nx.MultiDiGraph(graph)  # Convert undirecitonal to bidirectional, while keeping multi-edges

        default_dtype = get_default_numpy_dtype()
        i, j, w, s = [], [], [], []
        for k, v in dict(graph.edges).items():
            i.append(k[0])
            j.append(k[1])
            w.append(v[edge_weight])
            s.append(v[edge_sign])
        eidx = np.stack([
            vertices.get_indexer(i),
            vertices.get_indexer(j)
        ]).astype(np.int64)
        if eidx.min() < 0:
            raise ValueError("Missing vertices!")
        ewt = np.asarray(w).astype(default_dtype)
        if ewt.min() <= 0 or ewt.max() > 1:
            raise ValueError("Invalid edge weight!")
        esgn = np.asarray(s).astype(default_dtype)
        if set(esgn).difference({-1, 1}):
            raise ValueError("Invalid edge sign!")
        return eidx, ewt, esgn

    def __len__(self) -> int:
        return ceil(self.size / self.getitem_size)

    def __getitem__(self, index: int) -> List[torch.Tensor]:
        s = slice(
            index * self.getitem_size,
            min((index + 1) * self.getitem_size, self.size)
        )
        return [
            torch.as_tensor(self.samp_eidx[:, s]),
            torch.as_tensor(self.samp_ewt[s]),
            torch.as_tensor(self.samp_esgn[s])
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


#-------------------------------- Data loaders ---------------------------------

class DataLoader(torch.utils.data.DataLoader):

    r"""
    Custom data loader that manually shuffles the internal dataset before each
    round of iteration (see :class:`torch.utils.data.DataLoader` for usage)
    """

    def __init__(self, dataset: Dataset, **kwargs) -> None:
        super().__init__(dataset, **kwargs)
        self.collate_fn = self._collate_graph if isinstance(
            dataset, GraphDataset
        ) else self._collate
        self.shuffle = kwargs["shuffle"] if "shuffle" in kwargs else False

    def __iter__(self) -> "DataLoader":
        if self.shuffle:
            self.dataset.shuffle()  # Customized shuffling
        return super().__iter__()

    @staticmethod
    def _collate(batch):
        return tuple(map(lambda x: torch.cat(x, dim=0), zip(*batch)))

    @staticmethod
    def _collate_graph(batch):
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
        self.num_loaders = len(self.data_loaders)
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
            operator.add, [self._next(i) for i in range(self.num_loaders)]
        )
