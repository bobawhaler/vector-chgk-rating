"""Time decay, dynamic skill tracking, and Brownian variance expansion for ChGK Rating System."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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

        # Grace period parameter: G_grace (days)
        raw_grace = _inverse_softplus(self.config.grace_period_days)
        if trainable_rates:
            self.raw_grace_period = nn.Parameter(torch.tensor(raw_grace))
        else:
            self.register_buffer("raw_grace_period", torch.tensor(raw_grace))

        # Brownian variance expansion parameter: alpha_gap
        raw_alpha = _inverse_softplus(self.config.alpha_gap)
        if trainable_rates:
            self.raw_alpha_gap = nn.Parameter(torch.tensor(raw_alpha))
        else:
            self.register_buffer("raw_alpha_gap", torch.tensor(raw_alpha))

    @property
    def lambda_decay(self) -> torch.Tensor:
        """Effective non-negative per-dimension knowledge decay rates."""
        return F.softplus(self.raw_lambda_decay)

    @property
    def gamma_syn(self) -> torch.Tensor:
        """Effective non-negative synergy decay rate."""
        return F.softplus(self.raw_gamma_syn)

    @property
    def grace_period_days(self) -> torch.Tensor:
        """Effective learned grace period in days."""
        return F.softplus(self.raw_grace_period)

    @property
    def alpha_gap(self) -> torch.Tensor:
        """Effective learned Brownian variance factor."""
        return F.softplus(self.raw_alpha_gap)

    def get_calibration_summary(self) -> Dict[str, Any]:
        """Compute human-interpretable calibration values (half-lives and grace period in days)."""
        with torch.no_grad():
            lambdas = self.lambda_decay.cpu().numpy()
            gamma = float(self.gamma_syn.item())
            grace = float(self.grace_period_days.item())
            alpha = float(self.alpha_gap.item())

            ln2 = math.log(2.0)
            # Clip small lambdas to avoid division by zero
            half_lives = [float(ln2 / max(lam, 1e-7)) for lam in lambdas]
            half_dim = len(half_lives) // 2
            avg_fact_hl = float(np.mean(half_lives[:half_dim]))
            avg_logic_hl = float(np.mean(half_lives[half_dim:]))
            syn_hl = float(ln2 / max(gamma, 1e-7))

            return {
                "learned_grace_period_days": round(grace, 1),
                "factual_half_life_days": round(avg_fact_hl, 1),
                "logical_half_life_days": round(avg_logic_hl, 1),
                "synergy_half_life_days": round(syn_hl, 1),
                "alpha_gap": round(alpha, 4),
            }

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
