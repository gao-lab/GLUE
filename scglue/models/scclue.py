r"""
Concat/cross-linked unified embedding (CLUE) for single-cell multi-omics
data integration

**EXPERIMENTAL**
"""

# pylint: disable=missing-class-docstring, missing-function-docstring

import collections
import copy
import os
from abc import abstractmethod
from itertools import chain
from math import ceil
from typing import Any, List, Mapping, NoReturn, Optional, Tuple

import ignite
import numpy as np
import pandas as pd
import torch
import torch.distributions as D
import torch.nn.functional as F
from anndata import AnnData

from ..num import EPS
from ..utils import config, get_chained_attr, logged
from .base import Model, Trainer, TrainingPlugin
from .data import AnnDataset, DataLoader
from .nn import autodevice
from .plugins import EarlyStopping, LRScheduler, Tensorboard
from .prob import MSE, RMSE, ZILN, ZIN
from .sc import Prior

AUTO = -1  # Flag for using automatically determined hyperparameters


#----------------------------- Network components ------------------------------

@logged
class Transferrable(torch.nn.Module):

    r"""
    Mixin class for transferrable weights
    """

    def transfer_weights_from(self, other: "Transferrable") -> None:
        r"""
        Transfer weights from another module

        Parameters
        ----------
        other
            Module to transfer weight from
        """
        for key, target in chain(self.named_parameters(), self.named_buffers()):
            source = get_chained_attr(other, key)
            if isinstance(target, torch.nn.Parameter):
                target = target.data
            if isinstance(source, torch.nn.Parameter):
                source = source.data
            source = source.to(device=target.device, dtype=target.dtype)
            target.copy_(source)
            self.logger.info("Copied: %s", key)


@logged
class BatchedTransferrable(Transferrable):

    r"""
    Mixin class for transferrable weights containing expandable batch-specific
    weights at the end of second dimension
    """

    @property
    def batches(self) -> List[str]:
        r"""
        List of batch labels
        """
        return self._batches

    @batches.setter
    def batches(self, batches: List[str]) -> None:
        self._batches = batches

    @property
    def batched_weights(self) -> List[str]:
        r"""
        List of batch-specific weights
        """
        raise NotImplementedError

    def transfer_weights_from(self, other: "BatchedTransferrable") -> None:
        if (len(self.batches) == 0) != (len(other.batches) == 0):
            raise RuntimeError(
                "Batches of the two discriminators should be either "
                "both empty or both non-empty!"
            )
        for key, target in chain(self.named_parameters(), self.named_buffers()):
            source = get_chained_attr(other, key)
            if isinstance(target, torch.nn.Parameter):
                target = target.data
            if isinstance(source, torch.nn.Parameter):
                source = source.data
            source = source.to(device=target.device, dtype=target.dtype)
            if key in self.batched_weights and len(self.batches):
                target[:, :-len(self.batches)].copy_(source[:, :-len(other.batches)])
                target = target[:, -len(self.batches):]
                source = source[:, -len(other.batches):]
                index = pd.Index(self.batches).get_indexer(other.batches)
                mask = index >= 0
                source = source[:, mask]
                index = torch.as_tensor(
                    index[mask], device=target.device
                ).expand_as(source)
                target.scatter_(dim=1, index=index, src=source)
            else:
                target.copy_(source)
            self.logger.info("Copied: %s", key)


class ElementDataEncoder(Transferrable):

    def __init__(
            self, in_features: int, out_features: int,
            h_depth: int = 2, h_dim: int = 256,
            dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.h_depth = h_depth
        ptr_dim = in_features
        for layer in range(self.h_depth):
            setattr(self, f"linear_{layer}", torch.nn.Linear(ptr_dim, h_dim))
            setattr(self, f"act_{layer}", torch.nn.LeakyReLU(negative_slope=0.2))
            setattr(self, f"bn_{layer}", torch.nn.BatchNorm1d(h_dim))
            setattr(self, f"dropout_{layer}", torch.nn.Dropout(p=dropout))
            ptr_dim = h_dim
        self.loc = torch.nn.Linear(ptr_dim, out_features)
        self.std_lin = torch.nn.Linear(ptr_dim, out_features)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ptr = x
        for layer in range(self.h_depth):
            ptr = getattr(self, f"linear_{layer}")(ptr)
            ptr = getattr(self, f"act_{layer}")(ptr)
            ptr = getattr(self, f"bn_{layer}")(ptr)
            ptr = getattr(self, f"dropout_{layer}")(ptr)
        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + EPS
        return loc, std


class CompositeDataEncoder(Transferrable):

    def __init__(
            self, in_features: int, in_raw_features: int, out_features: int,
            n_modalities: int, h_depth: int = 2, h_dim: int = 256,
            dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.encoders = torch.nn.ModuleList([
            ElementDataEncoder(
                in_features, out_features,
                h_depth=h_depth, h_dim=h_dim, dropout=dropout
            ) for _ in range(n_modalities)
        ])

    @abstractmethod
    def compute_l(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def normalize(
            self, x: torch.Tensor, l: Optional[torch.Tensor]
    ) -> torch.Tensor:
        raise NotImplementedError  # pragma: no cover

    def forward(
            self, x: torch.Tensor, xrep: torch.Tensor,
            lazy_normalizer: bool = True
    ) -> Tuple[D.Normal, Optional[torch.Tensor]]:
        if xrep.numel():
            l = None if lazy_normalizer else self.compute_l(x)
            ptr = xrep
        else:
            l = self.compute_l(x)
            ptr = self.normalize(x, l)
        locs, stds = zip(*(encoder(ptr) for encoder in self.encoders))
        loc = torch.cat(locs, dim=1)
        std = torch.cat(stds, dim=1)
        return D.Normal(loc, std), l


class VanillaCompositeDataEncoder(CompositeDataEncoder):

    def compute_l(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        return None

    def normalize(
            self, x: torch.Tensor, l: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return x


class NBCompositeDataEncoder(CompositeDataEncoder):

    TOTAL_COUNT = 1e4

    def __init__(
            self, in_features: int, in_raw_features: int, out_features: int,
            n_modalities: int, h_depth: int = 2, h_dim: int = 256,
            dropout: float = 0.2
    ) -> None:
        super().__init__(
            in_features, in_raw_features, out_features, n_modalities,
            h_depth=h_depth, h_dim=h_dim, dropout=dropout
        )
        self.est_l = torch.nn.Linear(in_raw_features, n_modalities)

    def compute_l(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        return F.relu(self.est_l(x))

    def normalize(
            self, x: torch.Tensor, l: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return (x * (self.TOTAL_COUNT / x.sum(dim=1, keepdim=True))).log1p()


class DataDecoder(BatchedTransferrable):

    def __init__(
            self, in_features: int, out_features: int,
            batches: Optional[List[str]] = None,
            h_depth: int = 2, h_dim: int = 256, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.batches = batches or []
        self.n_batches = len(self.batches) or 1
        self.h_depth = h_depth
        ptr_dim = in_features + self.n_batches
        for layer in range(self.h_depth):
            setattr(self, f"linear_{layer}", torch.nn.Linear(ptr_dim, h_dim))
            setattr(self, f"act_{layer}", torch.nn.LeakyReLU(negative_slope=0.2))
            setattr(self, f"bn_{layer}", torch.nn.BatchNorm1d(h_dim))
            setattr(self, f"dropout_{layer}", torch.nn.Dropout(p=dropout))
            ptr_dim = h_dim + self.n_batches
        self.loc = torch.nn.Linear(ptr_dim, out_features)

    @property
    def batched_weights(self) -> List[str]:
        return [f"linear_{layer}.weight" for layer in range(self.h_depth)] + ["loc.weight"]

    def compute_dist(
            self, loc: torch.Tensor, b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> D.Distribution:
        raise NotImplementedError

    def forward(
            self, x: torch.Tensor, b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> D.Distribution:
        b_one_hot = F.one_hot(b, num_classes=self.n_batches)
        ptr = torch.cat([x, b_one_hot], dim=1)
        for layer in range(self.h_depth):
            ptr = getattr(self, f"linear_{layer}")(ptr)
            ptr = getattr(self, f"act_{layer}")(ptr)
            ptr = getattr(self, f"bn_{layer}")(ptr)
            ptr = getattr(self, f"dropout_{layer}")(ptr)
            ptr = torch.cat([ptr, b_one_hot], dim=1)
        loc = self.loc(ptr)
        return self.compute_dist(loc, b, l)


class MSEDataDecoder(DataDecoder):

    def compute_dist(
            self, loc: torch.Tensor, b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> MSE:
        return MSE(loc)


class RMSEDataDecoder(DataDecoder):

    def compute_dist(
            self, loc: torch.Tensor, b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> RMSE:
        return RMSE(loc)


class NormalDataDecoder(DataDecoder):

    def __init__(
            self, in_features: int, out_features: int,
            batches: Optional[List[str]] = None,
            h_depth: int = 2, h_dim: int = 256, dropout: float = 0.2
    ) -> None:
        super().__init__(
            in_features, out_features, batches,
            h_depth=h_depth, h_dim=h_dim, dropout=dropout
        )
        self.std_lin = torch.nn.Parameter(torch.zeros(out_features, self.n_batches))

    @property
    def batched_weights(self) -> List[str]:
        return super().batched_weights + ["std_lin"]

    def compute_dist(
            self, loc: torch.Tensor, b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> D.Normal:
        std = F.softplus(self.std_lin.T[b]) + EPS
        return D.Normal(loc, std)


class ZINDataDecoder(NormalDataDecoder):

    def __init__(
            self, in_features: int, out_features: int,
            batches: Optional[List[str]] = None,
            h_depth: int = 2, h_dim: int = 256, dropout: float = 0.2
    ) -> None:
        super().__init__(
            in_features, out_features, batches,
            h_depth=h_depth, h_dim=h_dim, dropout=dropout
        )
        self.zi_logits = torch.nn.Parameter(torch.zeros(out_features, self.n_batches))

    @property
    def batched_weights(self) -> List[str]:
        return super().batched_weights + ["zi_logits"]

    def compute_dist(
            self, loc: torch.Tensor, b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> ZIN:
        std = F.softplus(self.std_lin.T[b]) + EPS
        return ZIN(self.zi_logits.T[b].expand_as(loc), loc, std)


class ZILNDataDecoder(DataDecoder):

    def __init__(
            self, in_features: int, out_features: int,
            batches: Optional[List[str]] = None,
            h_depth: int = 2, h_dim: int = 256, dropout: float = 0.2
    ) -> None:
        super().__init__(
            in_features, out_features, batches,
            h_depth=h_depth, h_dim=h_dim, dropout=dropout
        )
        self.std_lin = torch.nn.Parameter(torch.zeros(out_features, self.n_batches))
        self.zi_logits = torch.nn.Parameter(torch.zeros(out_features, self.n_batches))

    @property
    def batched_weights(self) -> List[str]:
        return super().batched_weights + ["std_lin", "zi_logits"]

    def compute_dist(
            self, loc: torch.Tensor, b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> ZILN:
        std = F.softplus(self.std_lin.T[b]) + EPS
        return ZILN(self.zi_logits.T[b].expand_as(loc), loc, std)


class BernoulliDataDecoder(DataDecoder):

    def __init__(
            self, in_features: int, out_features: int,
            batches: Optional[List[str]] = None,
            h_depth: int = 2, h_dim: int = 256, dropout: float = 0.2
    ) -> None:
        super().__init__(
            in_features, out_features, batches,
            h_depth=h_depth, h_dim=h_dim, dropout=dropout
        )
        self.scale_lin = torch.nn.Parameter(torch.zeros(out_features, self.n_batches))
        self.bias = torch.nn.Parameter(torch.zeros(out_features, self.n_batches))

    @property
    def batched_weights(self) -> List[str]:
        return super().batched_weights + ["scale_lin", "bias"]

    def compute_dist(
            self, loc: torch.Tensor, b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> D.Bernoulli:
        scale = F.softplus(self.scale_lin.T[b])
        return D.Bernoulli(logits=scale * loc + self.bias.T[b])


class NBDataDecoder(DataDecoder):

    def __init__(
            self, in_features: int, out_features: int,
            batches: Optional[List[str]] = None,
            h_depth: int = 2, h_dim: int = 256, dropout: float = 0.2
    ) -> None:
        super().__init__(
            in_features, out_features, batches=batches,
            h_depth=h_depth, h_dim=h_dim, dropout=dropout
        )
        self.scale_lin = torch.nn.Parameter(torch.zeros(out_features, self.n_batches))
        self.bias = torch.nn.Parameter(torch.zeros(out_features, self.n_batches))
        self.log_theta = torch.nn.Parameter(torch.zeros(out_features, self.n_batches))

    @property
    def batched_weights(self) -> List[str]:
        return super().batched_weights + ["scale_lin", "bias", "log_theta"]

    def compute_dist(
            self, loc: torch.Tensor, b: torch.Tensor, l: torch.Tensor
    ) -> D.NegativeBinomial:
        scale = F.softplus(self.scale_lin.T[b])
        logit_mu = scale * loc + self.bias.T[b]
        mu = F.softmax(logit_mu, dim=1) * l.unsqueeze(1)
        log_theta = self.log_theta.T[b]
        return D.NegativeBinomial(
            log_theta.exp(),
            logits=(mu + EPS).log() - log_theta
        )


class Discriminator(torch.nn.Sequential, BatchedTransferrable):

    def __init__(
            self, in_features: int, out_features: int,
            batches: Optional[List[str]] = None,
            h_depth: int = 2, h_dim: Optional[int] = 256,
            dropout: float = 0.2
    ) -> None:
        self.batches = batches or []
        self.n_batches = len(self.batches)  # NOTE: Do not add 1, or we can't feed b properly
        od = collections.OrderedDict()
        ptr_dim = in_features + self.n_batches
        for layer in range(h_depth):
            od[f"linear_{layer}"] = torch.nn.Linear(ptr_dim, h_dim)
            od[f"act_{layer}"] = torch.nn.LeakyReLU(negative_slope=0.2)
            od[f"dropout_{layer}"] = torch.nn.Dropout(p=dropout)
            ptr_dim = h_dim
        od["pred"] = torch.nn.Linear(ptr_dim, out_features)
        super().__init__(od)

    @property
    def batched_weights(self) -> List[str]:
        return ["linear_0.weight" if hasattr(self, "linear_0") else "pred.weight"]

    def forward(self, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        if self.n_batches:
            b_one_hot = F.one_hot(b, num_classes=self.n_batches)
            x = torch.cat([x, b_one_hot], dim=1)
        return super().forward(x)


class Classifier(torch.nn.Linear, Transferrable):

    r"""
    Linear label classifier

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    """


#---------------------------------- Utilities ----------------------------------

_ENCODER_MAP: Mapping[str, type] = {}
_DECODER_MAP: Mapping[str, type] = {}


def register_prob_model(prob_model: str, encoder: type, decoder: type) -> None:
    r"""
    Register probabilistic model

    Parameters
    ----------
    prob_model
        Data probabilistic model
    encoder
        Encoder type of the probabilistic model
    decoder
        Decoder type of the probabilistic model
    """
    _ENCODER_MAP[prob_model] = encoder
    _DECODER_MAP[prob_model] = decoder


register_prob_model("MSE", VanillaCompositeDataEncoder, MSEDataDecoder)
register_prob_model("RMSE", VanillaCompositeDataEncoder, RMSEDataDecoder)
register_prob_model("Normal", VanillaCompositeDataEncoder, NormalDataDecoder)
register_prob_model("ZIN", VanillaCompositeDataEncoder, ZINDataDecoder)
register_prob_model("ZILN", VanillaCompositeDataEncoder, ZILNDataDecoder)
register_prob_model("Bernoulli", VanillaCompositeDataEncoder, BernoulliDataDecoder)
register_prob_model("NB", NBCompositeDataEncoder, NBDataDecoder)


#----------------------------- Network definition ------------------------------

class SCCLUE(torch.nn.Module):

    def __init__(
            self, x2u: Mapping[str, CompositeDataEncoder],
            u2x: Mapping[str, DataDecoder],
            du: Discriminator, prior: Prior,
            u2c: Optional[Classifier] = None
    ) -> None:
        super().__init__()
        if not set(x2u.keys()) == set(u2x.keys()) != set():
            raise ValueError("`x2u` and `u2x` must have the same keys and non-empty!")
        self.keys = list(x2u.keys())

        self.x2u = torch.nn.ModuleDict(x2u)
        self.u2x = torch.nn.ModuleDict(u2x)
        self.du = du
        self.prior = prior
        self.u2c = u2c

        self.device = autodevice()

    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, device: torch.device) -> None:
        self._device = device
        self.to(self._device)

    def forward(self) -> NoReturn:
        raise NotImplementedError("SCCLUE does not support forward operation!")

    def transfer_weights_from(self, source: "SCCLUE") -> None:
        for k in self.keys:
            self.x2u[k].transfer_weights_from(source.x2u[k])
            self.u2x[k].transfer_weights_from(source.u2x[k])
        self.du.transfer_weights_from(source.du)
        if self.u2c and source.u2c:
            self.u2c.transfer_weights_from(source.u2c)


#----------------------------- Trainer definition ------------------------------

PairedDataTensors = Tuple[
    Mapping[str, torch.Tensor],  # x (data)
    Mapping[str, torch.Tensor],  # xrep (alternative input data)
    Mapping[str, torch.Tensor],  # xbch (data batch)
    Mapping[str, torch.Tensor],  # xlbl (data label)
    Mapping[str, torch.Tensor],  # xdwt (modality discriminator sample weight)
    Mapping[str, torch.Tensor],  # xflag (modality indicator)
    torch.Tensor,  # pmsk (paired mask)
]  # Specifies the data format of input to SCCLUETrainer.compute_losses


@logged
class SCCLUETrainer(Trainer):

    def __init__(
            self, net: SCCLUE, lam_data: float = None, lam_kl: float = None,
            lam_align: float = None, lam_sup: float = None,
            lam_joint_cross: float = None, lam_real_cross: float = None,
            lam_cos: float = None, normalize_u: bool = None,
            modality_weight: Mapping[str, float] = None,
            optim: str = None, lr: float = None, **kwargs
    ) -> None:
        required_kwargs = (
            "lam_data", "lam_kl", "lam_align",
            "lam_sup", "lam_joint_cross", "lam_real_cross",
            "lam_cos", "normalize_u", "modality_weight",
            "optim", "lr"
        )
        for required_kwarg in required_kwargs:
            if locals()[required_kwarg] is None:
                raise ValueError(f"`{required_kwarg}` must be specified!")

        super().__init__(net)
        self.required_losses = [
            "dsc_loss", "gen_loss",
            "joint_cross_loss", "real_cross_loss",
            "cos_loss"
        ]
        if self.net.u2c:
            self.required_losses.append("sup_loss")
        for k in self.net.keys:
            self.required_losses += [f"x_{k}_nll", f"x_{k}_kl", f"x_{k}_elbo"]
        self.earlystop_loss = "gen_loss"

        self.lam_data = lam_data
        self.lam_kl = lam_kl
        self.lam_align = lam_align
        self.lam_sup = lam_sup
        self.lam_joint_cross = lam_joint_cross
        self.lam_real_cross = lam_real_cross
        self.lam_cos = lam_cos
        self.normalize_u = normalize_u
        if min(modality_weight.values()) < 0:
            raise ValueError("Modality weights must be non-negative!")
        normalizer = sum(modality_weight.values()) / len(modality_weight)
        self.modality_weight = {k: v / normalizer for k, v in modality_weight.items()}

        self.lr = lr
        self.vae_optim = getattr(torch.optim, optim)(
            chain(
                self.net.x2u.parameters(),
                self.net.u2x.parameters()
            ), lr=self.lr, **kwargs
        )
        self.dsc_optim = getattr(torch.optim, optim)(
            self.net.du.parameters(), lr=self.lr, **kwargs
        )

        self.align_burnin: Optional[int] = None

    def format_data(self, data: List[torch.Tensor]) -> PairedDataTensors:
        device = self.net.device
        keys = self.net.keys
        K = len(keys)
        x, xrep, xbch, xlbl, xdwt, pmsk = \
            data[0:K], data[K:2*K], data[2*K:3*K], \
            data[3*K:4*K], data[4*K:5*K], data[5*K]
        x = {
            k: x[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xrep = {
            k: xrep[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xbch = {
            k: xbch[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xlbl = {
            k: xlbl[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xdwt = {
            k: xdwt[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xflag = {
            k: torch.as_tensor(
                i, dtype=torch.int64, device=device
            ).expand(x[k].shape[0])
            for i, k in enumerate(keys)
        }
        pmsk = pmsk.to(device, non_blocking=True)
        return x, xrep, xbch, xlbl, xdwt, xflag, pmsk

    def compute_losses(
            self, data: PairedDataTensors, epoch: int, dsc_only: bool = False
    ) -> Mapping[str, torch.Tensor]:
        net = self.net
        x, xrep, xbch, xlbl, xdwt, xflag, pmsk = data

        u, l = {}, {}
        for k in net.keys:
            u[k], l[k] = net.x2u[k](x[k], xrep[k], lazy_normalizer=dsc_only)
        usamp = {k: u[k].rsample() for k in net.keys}
        if self.normalize_u:
            usamp = {k: F.normalize(usamp[k], dim=1) for k in net.keys}
        usamp_split = {
            k: torch.stack(torch.tensor_split(
                usamp[k], len(net.keys), dim=1
            ))  # Stacking and indexing on first dim should be faster
            for k in net.keys
        }
        prior = net.prior()

        u_cat = torch.cat([u[k].mean for k in net.keys])
        xbch_cat = torch.cat([xbch[k] for k in net.keys])
        xdwt_cat = torch.cat([xdwt[k] for k in net.keys])
        xflag_cat = torch.cat([xflag[k] for k in net.keys])
        anneal = max(1 - (epoch - 1) / self.align_burnin, 0) \
            if self.align_burnin else 0
        if anneal:
            noise = prior.sample(u_cat.shape)
            u_cat = u_cat + anneal * noise
        dsc_loss = F.cross_entropy(net.du(u_cat, xbch_cat), xflag_cat, reduction="none")
        dsc_loss = (dsc_loss * xdwt_cat).sum() / xdwt_cat.numel()
        if dsc_only:
            return {"dsc_loss": self.lam_align * dsc_loss}

        if net.u2c:
            xlbl_cat = torch.cat([xlbl[k] for k in net.keys])
            lmsk = xlbl_cat >= 0
            sup_loss = F.cross_entropy(
                net.u2c(u_cat[lmsk]), xlbl_cat[lmsk], reduction="none"
            ).sum() / max(lmsk.sum(), 1)
        else:
            sup_loss = torch.tensor(0.0, device=self.net.device)

        x_nll = {
            k: -net.u2x[k](
                usamp_split[k][i], xbch[k],
                None if l[k] is None else l[k][:, i]
            ).log_prob(x[k]).mean()
            for i, k in enumerate(net.keys)
        }
        x_kl = {
            k: D.kl_divergence(
                u[k], prior
            ).sum(dim=1).mean() / x[k].shape[1]
            for k in net.keys
        }
        x_elbo = {
            k: x_nll[k] + self.lam_kl * x_kl[k]
            for k in net.keys
        }
        x_elbo_sum = sum(self.modality_weight[k] * x_elbo[k] for k in net.keys)

        pmsk = pmsk.T
        usamp_stack = torch.stack([usamp[k] for k in net.keys])
        pmsk_stack = pmsk.unsqueeze(2).expand_as(usamp_stack)
        usamp_mean = (usamp_stack * pmsk_stack).sum(dim=0) / pmsk_stack.sum(dim=0)
        if self.normalize_u:
            usamp_mean = F.normalize(usamp_mean, dim=1)
        usamp_mean_split = torch.stack(torch.tensor_split(
            usamp_mean, len(net.keys), dim=1
        ))

        if self.lam_joint_cross:
            x_joint_cross_nll = {
                k: -net.u2x[k](
                    usamp_mean_split[i, m], xbch[k][m],
                    None if l[k] is None else l[k][m, i]
                    # FIXME: This is directly using target modality estimated l.
                    # We are supposed to use source modality estimated l as well.
                    # Maybe take an average like in usamp? But l does not have
                    # adversarial learning as u, so it is unsure whether that's
                    # going to work.
                ).log_prob(x[k][m]).mean()
                for i, (k, m) in enumerate(zip(net.keys, pmsk)) if m.sum()
            }  # Decode the usamp_mean to all modalities
            joint_cross_loss = sum(
                self.modality_weight[k] * nll
                for k, nll in x_joint_cross_nll.items()
            )
        else:
            joint_cross_loss = torch.as_tensor(0.0, device=net.device)

        if self.lam_real_cross:
            x_real_cross_nll = {}
            for i, (k_tgt, m_tgt) in enumerate(zip(net.keys, pmsk)):
                x_tgt_real_cross_nll = torch.as_tensor(0.0, device=net.device)
                for k_src, m_src in zip(net.keys, pmsk):
                    if k_src == k_tgt:
                        continue
                    m = m_src & m_tgt
                    if m.sum():
                        x_tgt_real_cross_nll += -net.u2x[k_tgt](
                            usamp_split[k_src][i, m], xbch[k_tgt][m],
                            None if l[k_src] is None else l[k_src][m, i]
                        ).log_prob(x[k_tgt][m]).mean()
                x_real_cross_nll[k_tgt] = x_tgt_real_cross_nll
            real_cross_loss = sum(
                self.modality_weight[k] * nll
                for k, nll in x_real_cross_nll.items()
            )
        else:
            real_cross_loss = torch.as_tensor(0.0, device=net.device)

        if self.lam_cos:
            cos_loss = sum(
                1 - F.cosine_similarity(
                    usamp_stack[i, m], usamp_mean[m]
                ).mean()
                for i, m in enumerate(pmsk) if m.sum()
            )
        else:
            cos_loss = torch.as_tensor(0.0, device=net.device)

        gen_loss = self.lam_data * x_elbo_sum \
            + self.lam_sup * sup_loss \
            + self.lam_joint_cross * joint_cross_loss \
            + self.lam_real_cross * real_cross_loss \
            + self.lam_cos * cos_loss \
            - self.lam_align * dsc_loss

        losses = {
            "dsc_loss": dsc_loss, "gen_loss": gen_loss,
            "joint_cross_loss": joint_cross_loss,
            "real_cross_loss": real_cross_loss,
            "cos_loss": cos_loss
        }
        for k in net.keys:
            losses.update({
                f"x_{k}_nll": x_nll[k],
                f"x_{k}_kl": x_kl[k],
                f"x_{k}_elbo": x_elbo[k]
            })
        if net.u2c:
            losses["sup_loss"] = sup_loss
        return losses

    def train_step(
            self, engine: ignite.engine.Engine, data: List[torch.Tensor]
    ) -> Mapping[str, torch.Tensor]:  # pragma: no cover
        self.net.train()
        data = self.format_data(data)
        epoch = engine.state.epoch

        # Discriminator step
        losses = self.compute_losses(data, epoch, dsc_only=True)
        self.net.zero_grad(set_to_none=True)
        losses["dsc_loss"].backward()  # Already scaled by lam_align
        self.dsc_optim.step()

        # Generator step
        losses = self.compute_losses(data, epoch)
        self.net.zero_grad(set_to_none=True)
        losses["gen_loss"].backward()
        self.vae_optim.step()

        return losses

    @torch.no_grad()
    def val_step(
            self, engine: ignite.engine.Engine, data: List[torch.Tensor]
    ) -> Mapping[str, torch.Tensor]:
        self.net.eval()
        data = self.format_data(data)
        return self.compute_losses(data, engine.state.epoch)

    def fit(  # pylint: disable=arguments-renamed
            self, data: AnnDataset,
            val_split: float = None, batch_size: int = None,
            align_burnin: int = None, max_epochs: int = None,
            patience: Optional[int] = None,
            reduce_lr_patience: Optional[int] = None,
            random_seed: int = None, directory: Optional[os.PathLike] = None,
            plugins: Optional[List[TrainingPlugin]] = None
    ) -> None:
        required_kwargs = {
            "val_split", "batch_size", "align_burnin",
            "max_epochs", "random_seed"
        }
        for required_kwarg in required_kwargs:
            if locals()[required_kwarg] is None:
                raise ValueError(f"`{required_kwarg}` must be specified!")
        if patience and reduce_lr_patience and reduce_lr_patience >= patience:
            self.logger.warning(
                "Parameter `reduce_lr_patience` should be smaller than `patience`, "
                "otherwise learning rate scheduling would be ineffective."
            )

        data.getitem_size = max(1, round(batch_size / config.DATALOADER_FETCHES_PER_BATCH))
        data_train, data_val = data.random_split([1 - val_split, val_split], random_state=random_seed)
        data_train.prepare_shuffle(num_workers=config.ARRAY_SHUFFLE_NUM_WORKERS, random_seed=random_seed)
        data_val.prepare_shuffle(num_workers=config.ARRAY_SHUFFLE_NUM_WORKERS, random_seed=random_seed)

        train_loader = DataLoader(
            data_train, batch_size=config.DATALOADER_FETCHES_PER_BATCH, shuffle=True,
            num_workers=config.DATALOADER_NUM_WORKERS,
            pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY,
            drop_last=len(data_train) > config.DATALOADER_FETCHES_PER_BATCH,
            generator=torch.Generator().manual_seed(random_seed),
            persistent_workers=False
        )

        val_loader = DataLoader(
            data_val, batch_size=config.DATALOADER_FETCHES_PER_BATCH, shuffle=True,
            num_workers=config.DATALOADER_NUM_WORKERS,
            pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY, drop_last=False,
            generator=torch.Generator().manual_seed(random_seed),
            persistent_workers=False
        )

        self.align_burnin = align_burnin

        default_plugins = [Tensorboard()]
        if patience:
            default_plugins.append(EarlyStopping(
                monitor=self.earlystop_loss, patience=patience,
                burnin=self.align_burnin
            ))
        if reduce_lr_patience:
            default_plugins.append(LRScheduler(
                self.vae_optim, self.dsc_optim,
                monitor=self.earlystop_loss, patience=reduce_lr_patience,
                burnin=self.align_burnin
            ))
        plugins = default_plugins + (plugins or [])
        try:
            super().fit(
                train_loader, val_loader=val_loader,
                max_epochs=max_epochs, random_seed=random_seed,
                directory=directory, plugins=plugins
            )
        finally:
            data.clean()
            data_train.clean()
            data_val.clean()
            self.align_burnin = None

    def state_dict(self) -> Mapping[str, Any]:
        return {
            **super().state_dict(),
            "vae_optim": self.vae_optim.state_dict(),
            "dsc_optim": self.dsc_optim.state_dict()
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.vae_optim.load_state_dict(state_dict.pop("vae_optim"))
        self.dsc_optim.load_state_dict(state_dict.pop("dsc_optim"))
        super().load_state_dict(state_dict)


#--------------------------------- Public API ----------------------------------

@logged
class SCCLUEModel(Model):

    NET_TYPE = SCCLUE
    TRAINER_TYPE = SCCLUETrainer

    ALIGN_BURNIN_PRG: float = 12.0
    MAX_EPOCH_PRG: float = 72.0
    PATIENCE_PRG: float = 9.0
    REDUCE_LR_PATIENCE_PRG: float = 3.0

    def __init__(
            self, adatas: Mapping[str, AnnData], latent_dim: int = 50,
            x2u_h_depth: int = 2, x2u_h_dim: int = 256,
            u2x_h_depth: int = 2, u2x_h_dim: int = 256,
            du_h_depth: int = 2, du_h_dim: int = 256,
            dropout: float = 0.2, shared_batches: bool = False,
            random_seed: int = 0
    ) -> None:
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)

        self.modalities, x2u, u2x, all_ct = {}, {}, {}, set()
        for k, adata in adatas.items():
            if config.ANNDATA_KEY not in adata.uns:
                raise ValueError(
                    f"The '{k}' dataset has not been configured. "
                    f"Please call `configure_dataset` first!"
                )
            data_config = copy.deepcopy(adata.uns[config.ANNDATA_KEY])
            if data_config["rep_dim"] and data_config["rep_dim"] < latent_dim:
                self.logger.warning(
                    "It is recommended that `use_rep` dimensionality "
                    "be equal or larger than `latent_dim`."
                )
            x2u[k] = _ENCODER_MAP[data_config["prob_model"]](
                data_config["rep_dim"] or len(data_config["features"]),
                len(data_config["features"]), latent_dim, len(adatas),
                h_depth=x2u_h_depth, h_dim=x2u_h_dim, dropout=dropout
            )
            data_config["batches"] = pd.Index([]) if data_config["batches"] is None \
                else pd.Index(data_config["batches"])
            u2x[k] = _DECODER_MAP[data_config["prob_model"]](
                latent_dim, len(data_config["features"]),
                batches=data_config["batches"].to_list(),
                h_depth=u2x_h_depth, h_dim=u2x_h_dim, dropout=dropout
            )
            all_ct = all_ct.union(
                set() if data_config["cell_types"] is None
                else data_config["cell_types"]
            )
            self.modalities[k] = data_config
        all_ct = pd.Index(all_ct).sort_values()
        for modality in self.modalities.values():
            modality["cell_types"] = all_ct
        if shared_batches:
            all_batches = [modality["batches"] for modality in self.modalities.values()]
            ref_batch = all_batches[0]
            for batches in all_batches:
                if not np.array_equal(batches, ref_batch):
                    raise RuntimeError("Batches must match when using `shared_batches`!")
            du_batches = ref_batch.to_list()
        else:
            du_batches = []
        du = Discriminator(
            latent_dim * len(self.modalities), len(self.modalities),
            batches=du_batches,
            h_depth=du_h_depth, h_dim=du_h_dim, dropout=dropout
        )
        prior = Prior()
        super().__init__(
            x2u, u2x, du, prior,
            u2c=None if all_ct.empty else Classifier(
                latent_dim * len(self.modalities), all_ct.size
            )
        )

    def adopt_pretrained_model(self, source: "SCCLUEModel") -> None:
        self.net.transfer_weights_from(source.net)

    def compile(
            self, lam_data: float = 1.0,
            lam_kl: float = 1.0,
            lam_align: float = 0.02,
            lam_sup: float = 0.02,
            lam_joint_cross: float = 0.02,
            lam_real_cross: float = 0.02,
            lam_cos: float = 0.02,
            normalize_u: bool = True,
            modality_weight: Optional[Mapping[str, float]] = None,
            lr: float = 2e-3, **kwargs
    ) -> None:
        if modality_weight is None:
            modality_weight = {k: 1.0 for k in self.modalities}
        self.normalize_u = normalize_u
        super().compile(
            lam_data=lam_data, lam_kl=lam_kl,
            lam_align=lam_align, lam_sup=lam_sup,
            lam_joint_cross=lam_joint_cross, lam_real_cross=lam_real_cross,
            lam_cos=lam_cos, normalize_u=normalize_u, modality_weight=modality_weight,
            optim="RMSprop", lr=lr, **kwargs
        )

    def fit(  # pylint: disable=arguments-differ
            self, adatas: Mapping[str, AnnData],
            val_split: float = 0.1, batch_size: int = 128,
            align_burnin: int = AUTO, max_epochs: int = AUTO,
            patience: Optional[int] = AUTO, reduce_lr_patience: Optional[int] = AUTO,
            directory: Optional[os.PathLike] = None
    ) -> None:
        data = AnnDataset(
            [adatas[key] for key in self.net.keys],
            [self.modalities[key] for key in self.net.keys],
            mode="train"
        )
        batch_per_epoch = data.size * (1 - val_split) / batch_size
        if align_burnin == AUTO:
            align_burnin = max(
                ceil(self.ALIGN_BURNIN_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.ALIGN_BURNIN_PRG)
            )
            self.logger.info("Setting `align_burnin` = %d", align_burnin)
        if max_epochs == AUTO:
            max_epochs = max(
                ceil(self.MAX_EPOCH_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.MAX_EPOCH_PRG)
            )
            self.logger.info("Setting `max_epochs` = %d", max_epochs)
        if patience == AUTO:
            patience = max(
                ceil(self.PATIENCE_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.PATIENCE_PRG)
            )
            self.logger.info("Setting `patience` = %d", patience)
        if reduce_lr_patience == AUTO:
            reduce_lr_patience = max(
                ceil(self.REDUCE_LR_PATIENCE_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.REDUCE_LR_PATIENCE_PRG)
            )
            self.logger.info("Setting `reduce_lr_patience` = %d", reduce_lr_patience)

        super().fit(
            data, val_split=val_split,
            batch_size=batch_size, align_burnin=align_burnin,
            max_epochs=max_epochs, patience=patience,
            reduce_lr_patience=reduce_lr_patience,
            random_seed=self.random_seed, directory=directory
        )

    @torch.no_grad()
    def encode_data(
            self, key: str, adata: AnnData, batch_size: int = 128
    ) -> np.ndarray:
        r"""
        Compute data sample (cell) embedding

        Parameters
        ----------
        key
            Modality key
        adata
            Input dataset
        batch_size
            Size of minibatches

        Returns
        -------
        embedding
            Data sample (cell) embedding
        """
        self.net.eval()
        encoder = self.net.x2u[key]
        data = AnnDataset(
            [adata], [self.modalities[key]],
            mode="eval", getitem_size=batch_size
        )
        data_loader = DataLoader(
            data, batch_size=1,
            shuffle=False, drop_last=False
        )
        result = []
        for x, xrep, *_ in data_loader:
            result.append(encoder(
                x.to(self.net.device), xrep.to(self.net.device),
                lazy_normalizer=True
            )[0].mean.detach().cpu())
        return torch.cat(result).numpy()

    @torch.no_grad()
    def cross_predict(
            self, keys: Tuple[str, str], adata: AnnData,
            batch_size: int = 128
    ) -> AnnData:
        self.net.eval()

        source_key, target_key = keys
        if self.modalities[target_key]["prob_model"] == "NB":
            raise RuntimeError("Cannot use NB prob model for cross prediction!")
        if self.modalities[target_key]["use_batch"] is not None:
            raise RuntimeError("Cannot use batch for cross prediction!")
        target_idx = self.net.keys.index(target_key)
        x2u = self.net.x2u[source_key]
        u2x = self.net.u2x[target_key]

        data = AnnDataset(
            [adata], [self.modalities[source_key]],
            mode="eval", getitem_size=batch_size
        )
        data_loader = DataLoader(
            data, batch_size=1,
            shuffle=False, drop_last=False
        )

        result = []
        for x, xrep, *_ in data_loader:
            u, _ = x2u(
                x.to(self.net.device), xrep.to(self.net.device),
                lazy_normalizer=True
            )
            umean = F.normalize(u.mean, dim=1) if self.normalize_u else u.mean
            umean_split = torch.stack(torch.tensor_split(
                umean, len(self.net.keys), dim=1
            ), dim=0)
            result.append(u2x(
                umean_split[target_idx],
                torch.zeros(x.shape[0], dtype=int, device=self.net.device),
                None
            ).mean.detach().cpu())

        X = torch.cat(result).numpy()
        return AnnData(
            X=X, obs=adata.obs,
            var=pd.DataFrame(index=self.modalities[target_key]["features"]),
            dtype=X.dtype
        )

    def __repr__(self) -> str:
        return (
            f"SCCLUE model with the following network and trainer:\n\n"
            f"{repr(self.net)}\n\n"
            f"{repr(self.trainer)}\n"
        )
