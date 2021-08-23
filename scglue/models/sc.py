r"""
GLUE component modules for single-cell omics data
"""

import collections
from abc import abstractmethod
from typing import Optional, Tuple

import torch
import torch.distributions as D
import torch.nn.functional as F

from ..num import EPS
from . import glue, nn, prob


class GraphEncoder(glue.GraphEncoder):

    r"""
    Graph encoder

    Parameters
    ----------
    vnum
        Number of vertices
    out_features
        Output dimensionality
    """

    def __init__(
            self, vnum: int, out_features: int
    ) -> None:
        super().__init__()
        self.vrepr = torch.nn.Parameter(torch.zeros(vnum, out_features))
        self.conv = nn.GraphConv()
        self.loc = torch.nn.Linear(out_features, out_features)
        self.std_lin = torch.nn.Linear(out_features, out_features)

    def forward(
            self, eidx: torch.Tensor, enorm: torch.Tensor, esgn: torch.Tensor
    ) -> D.Normal:
        ptr = self.conv(self.vrepr, eidx, enorm, esgn)
        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + EPS
        return D.Normal(loc, std)


class GraphDecoder(glue.GraphDecoder):

    r"""
    Graph decoder
    """

    def forward(
            self, v: torch.Tensor, eidx: torch.Tensor, esgn: torch.Tensor
    ) -> D.Bernoulli:
        sidx, tidx = eidx  # Source index and target index
        logits = esgn * (v[sidx] * v[tidx]).sum(dim=1)
        return D.Bernoulli(logits=logits)


class DataEncoder(glue.DataEncoder):

    r"""
    Abstract data encoder

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    h_depth
        Hidden layer depth
    h_dim
        Hidden layer dimensionality
    dropout
        Dropout rate
    """

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

    @abstractmethod
    def compute_l(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        r"""
        Compute normalizer

        Parameters
        ----------
        x
            Input data

        Returns
        -------
        l
            Normalizer
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def normalize(
            self, x: torch.Tensor, l: Optional[torch.Tensor]
    ) -> torch.Tensor:
        r"""
        Normalize data

        Parameters
        ----------
        x
            Input data
        l
            Normalizer

        Returns
        -------
        xnorm
            Normalized data
        """
        raise NotImplementedError  # pragma: no cover

    def forward(  # pylint: disable=arguments-differ
            self, x: torch.Tensor, xalt: torch.Tensor,
            lazy_normalizer: bool = True
    ) -> Tuple[D.Normal, Optional[torch.Tensor]]:
        r"""
        Encode data to sample latent distribution

        Parameters
        ----------
        x
            Input data
        xalt
            Alternative input data
        lazy_normalizer
            Whether to skip computing `x` normalizer (just return None)
            if `xalt` is non-empty

        Returns
        -------
        u
            Sample latent distribution
        normalizer
            Data normalizer

        Note
        ----
        Normalization is always computed on `x`.
        If xalt is empty, the normalized `x` will be used as input
        to the encoder neural network, otherwise xalt is used instead.
        """
        if xalt.numel():
            l = None if lazy_normalizer else self.compute_l(x)
            ptr = xalt
        else:
            l = self.compute_l(x)
            ptr = self.normalize(x, l)
        for layer in range(self.h_depth):
            ptr = getattr(self, f"linear_{layer}")(ptr)
            ptr = getattr(self, f"act_{layer}")(ptr)
            ptr = getattr(self, f"bn_{layer}")(ptr)
            ptr = getattr(self, f"dropout_{layer}")(ptr)
        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + EPS
        return D.Normal(loc, std), l


class VanillaDataEncoder(DataEncoder):

    r"""
    Vanilla data encoder

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    h_depth
        Hidden layer depth
    h_dim
        Hidden layer dimensionality
    dropout
        Dropout rate
    """

    def compute_l(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        return None

    def normalize(
            self, x: torch.Tensor, l: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return x


class NBDataEncoder(DataEncoder):

    r"""
    Data encoder for negative binomial data

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    h_depth
        Hidden layer depth
    h_dim
        Hidden layer dimensionality
    dropout
        Dropout rate
    """

    TOTAL_COUNT = 1e4

    def compute_l(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=1, keepdim=True)

    def normalize(
            self, x: torch.Tensor, l: torch.Tensor
    ) -> torch.Tensor:
        return (x * (self.TOTAL_COUNT / l)).log1p()


class DataDecoder(glue.DataDecoder):

    r"""
    Abstract data decoder
    """

    @abstractmethod
    def forward(  # pylint: disable=arguments-differ
            self, u: torch.Tensor, v: torch.Tensor, l: Optional[torch.Tensor]
    ) -> D.Normal:
        r"""
        Decode data from sample and feature latent

        Parameters
        ----------
        u
            Sample latent
        v
            Feature latent
        l
            Optional normalizer

        Returns
        -------
        recon
            Data reconstruction distribution
        """
        raise NotImplementedError  # pragma: no cover


class NormalDataDecoder(DataDecoder):

    r"""
    Normal data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    """

    def __init__(self, out_features: int) -> None:
        super().__init__()
        self.scale_lin = torch.nn.Parameter(torch.zeros(out_features))
        self.bias = torch.nn.Parameter(torch.zeros(out_features))
        self.std_lin = torch.nn.Parameter(torch.Tensor(out_features))

    def forward(
            self, u: torch.Tensor, v: torch.Tensor, l: Optional[torch.Tensor]
    ) -> D.Normal:
        scale = F.softplus(self.scale_lin)
        loc = scale * (u @ v.t()) + self.bias
        std = F.softplus(self.std_lin) + EPS
        return D.Normal(loc, std)


class ZINDataDecoder(NormalDataDecoder):

    r"""
    Zero-inflated normal data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    """

    def __init__(self, out_features: int) -> None:
        super().__init__(out_features)
        self.zi_logits = torch.nn.Parameter(torch.zeros(out_features))

    def forward(
            self, u: torch.Tensor, v: torch.Tensor, l: Optional[torch.Tensor]
    ) -> prob.ZIN:
        scale = F.softplus(self.scale_lin)
        loc = scale * (u @ v.t()) + self.bias
        std = F.softplus(self.std_lin) + EPS
        return prob.ZIN(self.zi_logits.expand_as(loc), loc, std)


class ZILNDataDecoder(DataDecoder):

    r"""
    Zero-inflated log-normal data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    """

    def __init__(self, out_features: int) -> None:
        super().__init__()
        self.scale_lin = torch.nn.Parameter(torch.zeros(out_features))
        self.bias = torch.nn.Parameter(torch.zeros(out_features))
        self.zi_logits = torch.nn.Parameter(torch.zeros(out_features))
        self.std_lin = torch.nn.Parameter(torch.zeros(out_features))

    def forward(
            self, u: torch.Tensor, v: torch.Tensor, l: Optional[torch.Tensor]
    ) -> prob.ZILN:
        scale = F.softplus(self.scale_lin)
        loc = scale * (u @ v.t()) + self.bias
        std = F.softplus(self.std_lin) + EPS
        return prob.ZILN(self.zi_logits.expand_as(loc), loc, std)


class NBDataDecoder(DataDecoder):

    r"""
    Negative binomial data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    """

    def __init__(self, out_features: int) -> None:
        super().__init__()
        self.scale_lin = torch.nn.Parameter(torch.zeros(out_features))
        self.bias = torch.nn.Parameter(torch.zeros(out_features))
        self.log_theta = torch.nn.Parameter(torch.zeros(out_features))

    def forward(
            self, u: torch.Tensor, v: torch.Tensor, l: torch.Tensor
    ) -> D.NegativeBinomial:
        scale = F.softplus(self.scale_lin)
        logit_mu = scale * (u @ v.t()) + self.bias
        mu = F.softmax(logit_mu, dim=1) * l
        return D.NegativeBinomial(
            self.log_theta.exp(),
            logits=(mu + EPS).log() - self.log_theta
        )


class ZINBDataDecoder(NBDataDecoder):

    r"""
    Zero-inflated negative binomial data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    """

    def __init__(self, out_features: int) -> None:
        super().__init__(out_features)
        self.zi_logits = torch.nn.Parameter(torch.zeros(out_features))

    def forward(
            self, u: torch.Tensor, v: torch.Tensor, l: Optional[torch.Tensor]
    ) -> prob.ZINB:
        scale = F.softplus(self.scale_lin)
        logit_mu = scale * (u @ v.t()) + self.bias
        mu = F.softmax(logit_mu, dim=1) * l
        return prob.ZINB(
            self.zi_logits.expand_as(mu),
            self.log_theta.exp(),
            logits=(mu + EPS).log() - self.log_theta
        )


class Discriminator(torch.nn.Sequential, glue.Discriminator):

    r"""
    Domain discriminator

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    h_depth
        Hidden layer depth
    h_dim
        Hidden layer dimensionality
    dropout
        Dropout rate
    """

    def __init__(
            self, in_features: int, out_features: int,
            h_depth: int = 2, h_dim: Optional[int] = 256,
            dropout: float = 0.2
    ) -> None:
        od = collections.OrderedDict()
        ptr_dim = in_features
        for layer in range(h_depth):
            od[f"linear_{layer}"] = torch.nn.Linear(ptr_dim, h_dim)
            od[f"act_{layer}"] = torch.nn.LeakyReLU(negative_slope=0.2)
            od[f"dropout_{layer}"] = torch.nn.Dropout(p=dropout)
            ptr_dim = h_dim
        od["pred"] = torch.nn.Linear(ptr_dim, out_features)
        super().__init__(od)


class Prior(glue.Prior):

    r"""
    Prior distribution

    Parameters
    ----------
    loc
        Mean of the normal distribution
    std
        Standard deviation of the normal distribution
    """

    def __init__(
            self, loc: float = 0.0, std: float = 1.0
    ) -> None:
        super().__init__()
        loc = torch.as_tensor(loc, dtype=torch.get_default_dtype())
        std = torch.as_tensor(std, dtype=torch.get_default_dtype())
        self.register_buffer("loc", loc)
        self.register_buffer("std", std)

    def forward(self) -> D.Normal:
        return D.Normal(self.loc, self.std)
