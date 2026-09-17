"""Time decay, dynamic skill tracking, and Brownian variance expansion for ChGK Rating System."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TimeDecayConfig:
    """Hyperparameters for temporal skill and synergy decay."""

    grace_period_days: float = 365.0       # 1 year grace period before decay begins
    half_life_factual_days: float = 730.0  # Fast decay for factual knowledge (~2 years)
    half_life_logical_days: float = 2920.0 # Slow decay for logical reasoning (~8 years)
    half_life_synergy_days: float = 540.0  # Synergy decay half-life (~1.5 years)
    alpha_gap: float = 0.5                 # Brownian variance expansion parameter


def _inverse_softplus(y: float) -> float:
    """Compute x such that softplus(x) = y."""
    if y <= 0:
        return -20.0
    return math.log(math.expm1(y)) if y < 20.0 else y


class TimeDecayModule(nn.Module):
    """Dynamic module modeling player skill decay, synergy erosion, and uncertainty expansion over time.

    1. Asymmetric knowledge decay:
       k_i(t) = k_i * exp(-lambda_decay * max(0, Delta_t - G_grace))
       with factual dimensions decaying faster than logical reasoning dimensions.

    2. Pairwise team synergy decay:
       S_im(t) = S_im * exp(-gamma_syn * max(0, Delta_t_joint))

    3. Brownian variance expansion:
       eta_i(t) = sqrt(1.0 + alpha_gap * (Delta_t / 365.0))
    """

    def __init__(
        self,
        dim_k: int = 16,
        config: Optional[TimeDecayConfig] = None,
        trainable_rates: bool = True,
    ) -> None:
        super().__init__()
        self.dim_k = dim_k
        self.config = config or TimeDecayConfig()
        self.grace_period_days = self.config.grace_period_days
        self.alpha_gap = self.config.alpha_gap

        # Compute initial lambda rates from half-lives: lambda = ln(2) / HalfLife
        # First dim_k // 2 dimensions: factual knowledge (fast decay)
        # Remaining dimensions: logical reasoning (slow decay)
        half_dim = dim_k // 2
        lambda_fact = math.log(2.0) / self.config.half_life_factual_days
        lambda_logic = math.log(2.0) / self.config.half_life_logical_days

        init_lambdas = torch.zeros(dim_k)
        init_lambdas[:half_dim] = lambda_fact
        init_lambdas[half_dim:] = lambda_logic

        # Convert to raw parameter space via inverse softplus: softplus(raw) = lambda
        raw_lambdas = torch.tensor([_inverse_softplus(float(v)) for v in init_lambdas])
        if trainable_rates:
            self.raw_lambda_decay = nn.Parameter(raw_lambdas)
        else:
            self.register_buffer("raw_lambda_decay", raw_lambdas)

        # Synergy decay rate: gamma_syn = ln(2) / HalfLife_synergy
        gamma_syn_val = math.log(2.0) / self.config.half_life_synergy_days
        raw_gamma = _inverse_softplus(gamma_syn_val)
        if trainable_rates:
            self.raw_gamma_syn = nn.Parameter(torch.tensor(raw_gamma))
        else:
            self.register_buffer("raw_gamma_syn", torch.tensor(raw_gamma))

    @property
    def lambda_decay(self) -> torch.Tensor:
        """Effective non-negative per-dimension knowledge decay rates."""
        return F.softplus(self.raw_lambda_decay)

    @property
    def gamma_syn(self) -> torch.Tensor:
        """Effective non-negative synergy decay rate."""
        return F.softplus(self.raw_gamma_syn)

    def apply_knowledge_decay(
        self,
        k_embs: torch.Tensor,
        delta_t_days: torch.Tensor,
    ) -> torch.Tensor:
        """Apply exponential inactivity decay to player knowledge embeddings.

        Args:
            k_embs: Tensor of shape (..., D)
            delta_t_days: Tensor of shape (..., 1) or (...) containing elapsed days since last game

        Returns:
            Decayed knowledge tensor of shape (..., D)
        """
        if delta_t_days.dim() < k_embs.dim():
            delta_t_days = delta_t_days.unsqueeze(-1)

        # Effective inactive days past grace period
        effective_days = F.relu(delta_t_days - self.grace_period_days)  # (..., 1)

        # Per-dimension decay factor: exp(-lambda_decay * effective_days)
        # lambda_decay: (D,) -> broadcasts with (..., 1)
        decay_factor = torch.exp(-self.lambda_decay * effective_days)
        return k_embs * decay_factor

    def apply_synergy_decay(
        self,
        s_mat: torch.Tensor,
        pairwise_delta_t: torch.Tensor,
    ) -> torch.Tensor:
        """Apply joint inactivity decay to pairwise synergy matrix.

        Args:
            s_mat: Tensor of shape (..., K, K)
            pairwise_delta_t: Tensor of shape (..., K, K) containing elapsed days since last joint game

        Returns:
            Decayed pairwise synergy matrix of shape (..., K, K)
        """
        effective_days = F.relu(pairwise_delta_t)
        decay_factor = torch.exp(-self.gamma_syn * effective_days)
        return s_mat * decay_factor

    def compute_training_gap_scale(self, delta_t_days: torch.Tensor) -> torch.Tensor:
        """Compute Brownian variance expansion factor: sqrt(1.0 + alpha_gap * (delta_t / 365.0)).

        Args:
            delta_t_days: Tensor of shape (..., 1) or (...)

        Returns:
            Scale tensor of same shape as delta_t_days
        """
        norm_years = F.relu(delta_t_days) / 365.0
        return torch.sqrt(1.0 + self.alpha_gap * norm_years)
