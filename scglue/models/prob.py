r"""
Probability distributions
"""

import torch
import torch.distributions as D
import torch.nn.functional as F
from torch.distributions import constraints

from ..num import EPS



#-------------------------------- Distributions --------------------------------

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
            self, zi_logits: torch.Tensor,
            loc: torch.Tensor, scale: torch.Tensor
    ) -> None:
        super().__init__(loc, scale)
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
            self, zi_logits: torch.Tensor,
            loc: torch.Tensor, scale: torch.Tensor
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
            self, zi_logits: torch.Tensor,
            total_count: torch.Tensor, logits: torch.Tensor = None
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

class BetaBinomial:
    def __init__(self, total_count, alpha, beta):
        self.total_count = total_count
        self.beta_dist = D.Beta(alpha, beta)

    def sample(self, sample_shape= torch.Size()):
        p = self.beta_dist.sample(sample_shape)
        binomial_dist = D.Binomial(self.total_count, probs=p)
        return binomial_dist.sample()
    
    def log_prob(self, counts): 
        """
        Computes the log probability of counts in the Beta-Binomial distribution.
        
        Args:
            counts (torch.Tensor): Observed counts (successes).
        
        Returns:
            torch.Tensor: Log probabilities of the observed counts.
        """
        # Log Beta function
        alpha, beta = self.beta_dist.concentration1, self.beta_dist.concentration0
        total_count = self.total_count
        
        log_beta_ab = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
        log_beta_ab_counts = (torch.lgamma(counts + alpha) +
                              torch.lgamma(total_count - counts + beta) -
                              torch.lgamma(total_count + alpha + beta))
        
        log_binomial_coeff = (torch.lgamma(total_count + 1) -
                              torch.lgamma(counts + 1) -
                              torch.lgamma(total_count - counts + 1))
        
        return log_binomial_coeff + log_beta_ab_counts - log_beta_ab
    

class MuPhiBetaBinomial:
    """
    Beta-Binomial distribution with (mu, phi) parametrization.

    Parameters:
    ----------
    mu : torch.Tensor
        Mean parameter (0 < mu < 1). #from the paper we found
    phi : torch.Tensor
        Dispersion parameter (phi > 0).
    total_count : torch.Tensor
        Total number of trials. (m > 0).
    """

    def __init__(self, mu: torch.Tensor, phi: torch.Tensor, total_count: torch.Tensor):
        self.mu = mu
        self.phi = phi
        self.total_count = total_count

        # Ensure constraints on parameters
        assert torch.all((mu > 0) & (mu < 1)), "Mu must be in the range (0, 1)"
        assert torch.all(phi > 0), "Phi must be positive"
        assert torch.all(total_count > 0), "Total count must be positive"

    def log_prob(self, value):
        """
        Compute the log-probability of the Beta-Binomial distribution.

        Parameters:
        ----------
        value : torch.Tensor
            The observed counts (successes).
        """
        m = self.total_count
        y = value
        alpha = self.mu * self.phi
        beta = (1 - self.mu) * self.phi

        # Log-probability using the beta-binomial density formula
        #Sterling approximation is used (from Mackay textbook)
        log_coeff = (
            torch.lgamma(m + 1)
            - torch.lgamma(y + 1)
            - torch.lgamma(m - y + 1)
        )
        log_beta = (
            torch.lgamma(y + alpha)
            + torch.lgamma(m - y + beta)
            - torch.lgamma(m + alpha + beta)
        )
        log_beta_norm = torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)
        return log_coeff + log_beta - log_beta_norm

    def mean(self):
        """Mean of the distribution."""
        return self.total_count * self.mu

    def variance(self):
        """Variance of the distribution."""
        m = self.total_count
        return (
            m * self.mu * (1 - self.mu)
            * (self.phi + m) / (self.phi + 1)
        )
    
class BetaBinomialReparams(D.Distribution):
    """
    Beta-Binomial distribution parameterized by mean (μ) and dispersion (ϕ).
    
    The probability mass function is defined as:
    
    P(X = x) = (n choose x) * B(x + μ*n, n-x + ϕ(1-μ)*n) / B(μ*n, ϕ(1-μ)*n)
    
    where:
    - n is the total count
    - μ is the mean
    - ϕ is the dispersion parameter
    - B(a, b) is the Beta function
    """
    arg_constraints = {
        'total_count': torch.distributions.constraints.positive_integer,
        'concentration1': torch.distributions.constraints.positive,
        'concentration0': torch.distributions.constraints.positive
    }
    support = torch.distributions.constraints.integer_interval(0, 'total_count')
    
    def __init__(self, total_count, concentration1, concentration0, validate_args=None):
        self.total_count, self.concentration1, self.concentration0 = D.utils.broadcast_all(total_count, concentration1, concentration0)
        super(BetaBinomial, self).__init__(validate_args=validate_args)
    
    def _new(self, *args, **kwargs):
        return self.__class__(*args, **kwargs)

    @property
    def mean(self):
        return self.concentration1 / (self.concentration1 + self.concentration0)

    @property
    def variance(self):
        return self.total_count * self.concentration1 * self.concentration0 / ((self.concentration1 + self.concentration0)**2 * (self.concentration1 + self.concentration0 + 1))

    def sample(self, sample_shape=torch.Size()):
        return torch.distributions.binomial(self.total_count, self.mean).sample(sample_shape)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        
        log_prob_binomial = torch.lgamma(self.total_count + 1) - torch.lgamma(value + 1) - torch.lgamma(self.total_count - value + 1)
        log_prob_beta = torch.lgamma(self.concentration1 + value) + torch.lgamma(self.concentration0 + self.total_count - value) - torch.lgamma(self.concentration1) - torch.lgamma(self.concentration0) - torch.lgamma(self.total_count + 1)
        return log_prob_binomial + log_prob_beta
