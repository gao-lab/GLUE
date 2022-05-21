r"""
Neural network modules, datasets & data loaders, and other utilities
"""

import functools
import os
from math import sqrt

import numpy as np
import pynvml
import torch
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _NormBase

from ..utils import config, logged


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
            if free_mems.size:
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
