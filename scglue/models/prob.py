r"""
Probability distributions
"""

import torch
import torch.distributions as D
import torch.nn.functional as F
from pyro.distributions import BetaBinomial as BetaBinomialDistribution

from ..num import EPS
from .nn import zero_nan_grad

# ------------------------------- Distributions --------------------------------


class MSE(D.Distribution):
    r"""
    A "sham" distribution that outputs negative MSE on ``log_prob``

    Parameters
    ----------
    loc
        Mean of the distribution
    """

    def __init__(self, loc: torch.Tensor) -> None:
        super().__init__(validate_args=False)
        self.loc = loc

    def log_prob(self, value: torch.Tensor) -> None:
        return -F.mse_loss(self.loc, value)

    @property
    def mean(self) -> torch.Tensor:
        return self.loc


class RMSE(MSE):
    r"""
    A "sham" distribution that outputs negative RMSE on ``log_prob``

    Parameters
    ----------
    loc
        Mean of the distribution
    """

    def log_prob(self, value: torch.Tensor) -> None:
        return -F.mse_loss(self.loc, value).sqrt()


class ZIN(D.Normal):
    r"""
    Zero-inflated normal distribution with subsetting support

    Parameters
    ----------
    zi_logits
        Zero-inflation logits
    loc
        Location of the normal distribution
    scale
        Scale of the normal distribution
    """

    def __init__(
        self, zi_logits: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor
    ) -> None:
        if zi_logits.requires_grad:
            zi_logits.register_hook(zero_nan_grad)
        if loc.requires_grad:
            loc.register_hook(zero_nan_grad)
        if scale.requires_grad:
            scale.register_hook(zero_nan_grad)
        super().__init__(loc, scale, validate_args=False)
        self.zi_logits = zi_logits

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        raw_log_prob = super().log_prob(value)
        zi_log_prob = torch.empty_like(raw_log_prob)
        z_mask = value.abs() < EPS
        z_zi_logits, nz_zi_logits = self.zi_logits[z_mask], self.zi_logits[~z_mask]
        zi_log_prob[z_mask] = (
            raw_log_prob[z_mask].exp() + z_zi_logits.exp() + EPS
        ).log() - F.softplus(z_zi_logits)
        zi_log_prob[~z_mask] = raw_log_prob[~z_mask] - F.softplus(nz_zi_logits)
        return zi_log_prob


class ZILN(D.LogNormal):
    r"""
    Zero-inflated log-normal distribution with subsetting support

    Parameters
    ----------
    zi_logits
        Zero-inflation logits
    loc
        Location of the log-normal distribution
    scale
        Scale of the log-normal distribution
    """

    def __init__(
        self, zi_logits: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor
    ) -> None:
        super().__init__(loc, scale)
        self.zi_logits = zi_logits

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        zi_log_prob = torch.empty_like(value)
        z_mask = value.abs() < EPS
        z_zi_logits, nz_zi_logits = self.zi_logits[z_mask], self.zi_logits[~z_mask]
        zi_log_prob[z_mask] = z_zi_logits - F.softplus(z_zi_logits)
        zi_log_prob[~z_mask] = D.LogNormal(
            self.loc[~z_mask], self.scale[~z_mask]
        ).log_prob(value[~z_mask]) - F.softplus(nz_zi_logits)
        return zi_log_prob


class ZINB(D.NegativeBinomial):
    r"""
    Zero-inflated negative binomial distribution

    Parameters
    ----------
    zi_logits
        Zero-inflation logits
    total_count
        Total count of the negative binomial distribution
    logits
        Logits of the negative binomial distribution
    """

    def __init__(
        self,
        zi_logits: torch.Tensor,
        total_count: torch.Tensor,
        logits: torch.Tensor = None,
    ) -> None:
        super().__init__(total_count, logits=logits)
        self.zi_logits = zi_logits

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        raw_log_prob = super().log_prob(value)
        zi_log_prob = torch.empty_like(raw_log_prob)
        z_mask = value.abs() < EPS
        z_zi_logits, nz_zi_logits = self.zi_logits[z_mask], self.zi_logits[~z_mask]
        zi_log_prob[z_mask] = (
            raw_log_prob[z_mask].exp() + z_zi_logits.exp() + EPS
        ).log() - F.softplus(z_zi_logits)
        zi_log_prob[~z_mask] = raw_log_prob[~z_mask] - F.softplus(nz_zi_logits)
        return zi_log_prob


class Beta(D.Beta):
    r"""
    Stable beta distribution parameterized by mean and concentration

    Parameters
    ----------
    logit_mu
        Logit of mean of the beta distribution
    size
        Concentration of the beta distribution
    """

    def __init__(self, logit_mu: torch.Tensor, size: torch.Tensor) -> None:
        if logit_mu.requires_grad:
            logit_mu.register_hook(zero_nan_grad)
        if size.requires_grad:
            size.register_hook(zero_nan_grad)
        mu = logit_mu.sigmoid()
        super().__init__(mu * size + EPS, (1 - mu) * size + EPS, validate_args=False)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return super().log_prob(value.clamp(EPS, 1 - EPS))


class BetaBinomial(D.Beta):
    r"""
    Stable beta-binomial distribution parameterized by mean and concentration

    Parameters
    ----------
    logit_mu
        Logit of mean of the beta distribution
    size
        Concentration of the beta distribution
    """

    def __init__(self, logit_mu: torch.Tensor, size: torch.Tensor) -> None:
        mu = logit_mu.sigmoid()
        super().__init__(mu * size + EPS, (1 - mu) * size + EPS)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return BetaBinomialDistribution(
            self.concentration1, self.concentration0, total_count=value.imag
        ).log_prob(value.real)


class Bernoulli(D.Bernoulli):

    def __init__(self, logits: torch.Tensor) -> None:
        if logits.requires_grad:
            logits.register_hook(zero_nan_grad)
        super().__init__(logits=logits, validate_args=False)
