"""PyTorch implementation of the Dual Latent Embedding Neural Rating Model for ChGK."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from chgk_rating.time_decay import TimeDecayConfig, TimeDecayModule


class DualEmbeddingChGKModel(nn.Module):
    """Dual Latent Embedding Neural Rating Model for ChGK.

    Represents players with knowledge (D) and communication/format (M) embeddings,
    questions with requirements (D), bias, and discrimination,
    and teams through max-pooled knowledge and pairwise communication synergy,
    with dynamic time-decay tracking for inactivity and synergy erosion.
    """

    def __init__(
        self,
        num_players: int,
        num_questions: int,
        dim_k: int = 16,
        dim_c: int = 8,
        num_formats: int = 3,
        padding_player_idx: int = 0,
        time_decay_config: Optional[TimeDecayConfig] = None,
    ) -> None:
        super().__init__()
        self.num_players = num_players
        self.num_questions = num_questions
        self.dim_k = dim_k
        self.dim_c = dim_c
        self.num_formats = num_formats
        self.padding_player_idx = padding_player_idx

        # Player Embeddings
        # Knowledge Latent Embedding: k_i in R^D ~ Normal(0, 0.1)
        self.player_k_emb = nn.Embedding(
            num_players, dim_k, padding_idx=padding_player_idx
        )
        nn.init.normal_(self.player_k_emb.weight, mean=0.0, std=0.1)

        # Communication Base Embedding: c_base_i in R^M ~ Normal(0, 0.1)
        self.player_c_base = nn.Embedding(
            num_players, dim_c, padding_idx=padding_player_idx
        )
        nn.init.normal_(self.player_c_base.weight, mean=0.0, std=0.1)

        # Format Sensitivity Embedding: c_delta_i in R^M ~ Zeros
        self.player_c_delta = nn.Embedding(
            num_players, dim_c, padding_idx=padding_player_idx
        )
        nn.init.zeros_(self.player_c_delta.weight)

        # Reset padding embeddings to strictly 0
        with torch.no_grad():
            self.player_k_emb.weight[padding_player_idx].zero_()
            self.player_c_base.weight[padding_player_idx].zero_()
            self.player_c_delta.weight[padding_player_idx].zero_()

        # Question Embeddings
        # Requirement Latent Embedding: q_j in R^D ~ Normal(0, 0.1)
        self.question_q_emb = nn.Embedding(num_questions, dim_k)
        nn.init.normal_(self.question_q_emb.weight, mean=0.0, std=0.1)

        # Difficulty Scalar Bias: b_j in R^1 ~ Constant(1.0)
        self.question_b = nn.Embedding(num_questions, 1)
        nn.init.constant_(self.question_b.weight, 1.0)

        # Discrimination Factor: a_j in R^1 (constrained via Softplus)
        self.question_a_raw = nn.Embedding(num_questions, 1)
        # Initialize raw so softplus(raw) is ~ 1.0 (raw ~ 0.5413)
        nn.init.constant_(self.question_a_raw.weight, 0.5413248)

        # Format Constraints (offline, sync, online)
        self.format_emb = nn.Embedding(num_formats, dim_c)
        nn.init.normal_(self.format_emb.weight, mean=0.0, std=0.1)

        # Synergy Interaction Tensor: W_synergy in R^(M x M)
        self.W_synergy = nn.Parameter(torch.empty(dim_c, dim_c))
        nn.init.xavier_normal_(self.W_synergy, gain=0.1)

        # Dynamic Time Decay & Skill Tracking Module
        self.time_decay = TimeDecayModule(dim_k=dim_k, config=time_decay_config)

        # Online Anomaly Adjustment parameter/bias
        self.online_anomaly_weight = nn.Parameter(torch.zeros(1))

    @property
    def question_a(self) -> torch.Tensor:
        """Constrained discrimination factor via Softplus."""
        return F.softplus(self.question_a_raw.weight)

    def get_effective_communication(
        self, player_ids: torch.Tensor, format_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute c_i(F) = c_base_i + e_F * c_delta_i for given players and formats.

        Args:
            player_ids: Tensor of shape (..., K)
            format_ids: Tensor of shape (...) or (..., 1)

        Returns:
            Tensor of shape (..., K, M)
        """
        # player_ids: (B, K)
        c_base = self.player_c_base(player_ids)    # (B, K, M)
        c_delta = self.player_c_delta(player_ids)  # (B, K, M)

        # format_ids: (B) -> format_emb: (B, 1, M)
        if format_ids.dim() == 1:
            e_f = self.format_emb(format_ids).unsqueeze(1)  # (B, 1, M)
        else:
            e_f = self.format_emb(format_ids)

        c_eff = c_base + e_f * c_delta  # (B, K, M)

        # Mask padding players
        mask = (player_ids != self.padding_player_idx).unsqueeze(-1)  # (B, K, 1)
        return c_eff * mask

    def compute_team_synergy_matrix(
        self,
        c_eff: torch.Tensor,
        player_mask: Optional[torch.Tensor] = None,
        pairwise_delta_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute pairwise synergy matrix for team players: S_im = c_i^T * W * c_m.

        Args:
            c_eff: Tensor of shape (B, K, M)
            player_mask: Optional boolean Tensor of shape (B, K) where True = valid player
            pairwise_delta_t: Optional Tensor of shape (B, K, K) containing days since joint game

        Returns:
            Tensor of shape (B, K, K) containing pairwise synergy scores.
        """
        # C: (B, K, M), W: (M, M) -> C_W: (B, K, M)
        c_w = torch.matmul(c_eff, self.W_synergy)  # (B, K, M)
        # S_mat: (B, K, K) = C_W * C^T
        s_mat = torch.matmul(c_w, c_eff.transpose(-1, -2))  # (B, K, K)

        # Apply pairwise time decay if elapsed time provided
        if pairwise_delta_t is not None:
            s_mat = self.time_decay.apply_synergy_decay(s_mat, pairwise_delta_t)

        if player_mask is not None:
            # Mask out invalid pairs
            valid_2d = player_mask.unsqueeze(1) & player_mask.unsqueeze(2)  # (B, K, K)
            s_mat = s_mat * valid_2d

        return s_mat

    def compute_team_synergy(
        self,
        c_eff: torch.Tensor,
        player_mask: Optional[torch.Tensor] = None,
        pairwise_delta_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute total team synergy: S_team = sum_{i < m} S_im.

        Args:
            c_eff: Tensor of shape (B, K, M)
            player_mask: Optional boolean Tensor of shape (B, K)
            pairwise_delta_t: Optional Tensor of shape (B, K, K)

        Returns:
            Tensor of shape (B, 1)
        """
        s_mat = self.compute_team_synergy_matrix(c_eff, player_mask, pairwise_delta_t)  # (B, K, K)
        # Extract upper triangle strictly above diagonal (i < m)
        k = s_mat.size(-1)
        triu_indices = torch.triu_indices(k, k, offset=1, device=s_mat.device)
        s_team = s_mat[:, triu_indices[0], triu_indices[1]].sum(dim=-1, keepdim=True)
        return s_team  # (B, 1)

    def forward(
        self,
        player_ids: torch.Tensor,
        question_ids: torch.Tensor,
        format_ids: torch.Tensor,
        is_factual: Optional[torch.Tensor] = None,
        anomaly_delta: Optional[torch.Tensor] = None,
        player_delta_t: Optional[torch.Tensor] = None,
        pairwise_delta_t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for a batch of team-question observations with time decay.

        Args:
            player_ids: LongTensor of shape (B, K) - roster player indices
            question_ids: LongTensor of shape (B,) - question indices
            format_ids: LongTensor of shape (B,) - format indices (0=offline, 1=sync, 2=online)
            is_factual: Optional LongTensor/FloatTensor of shape (B, 1) - 1 if factual question
            anomaly_delta: Optional FloatTensor of shape (B, 1) - explicit cheat adjustment
            player_delta_t: Optional FloatTensor of shape (B, K) - elapsed days since last game per player
            pairwise_delta_t: Optional FloatTensor of shape (B, K, K) - elapsed days since last joint game per pair

        Returns:
            Dict containing:
                - 'logits': z_final (B, 1)
                - 'probs': P_final (B, 1)
                - 'p_gen': P_gen (B, 1)
                - 'z_know': z_know (B, 1)
                - 's_team': S_team (B, 1)
                - 'z_anomaly': z_anomaly (B, 1)
                - 'k_team': K_team (B, D)
        """
        batch_size, k_players = player_ids.shape
        player_mask = player_ids != self.padding_player_idx  # (B, K)

        # 1. Player Knowledge Embeddings & Temporal Decay
        k_player_embs = self.player_k_emb(player_ids)  # (B, K, D)

        if player_delta_t is not None:
            # Apply asymmetric inactivity decay:
            # k_i(t) = k_i * exp(-lambda_decay * max(0, Delta_t - G_grace))
            k_player_embs = self.time_decay.apply_knowledge_decay(k_player_embs, player_delta_t)

            # In training mode, apply Brownian variance scaling:
            # eta_i(t) = sqrt(1.0 + alpha_gap * (Delta_t / 365.0))
            if self.training:
                gap_scale = self.time_decay.compute_training_gap_scale(player_delta_t).unsqueeze(-1)
                k_player_embs = k_player_embs * gap_scale

        # Team Knowledge Pooling: K_team(t) = max_pooling({k_i(t) for i in T})
        # For masked players, replace with large negative number before max pooling
        masked_k = k_player_embs.masked_fill(~player_mask.unsqueeze(-1), -1e9)
        # Handle edge case where entire team is padding
        has_players = player_mask.any(dim=-1, keepdim=True).unsqueeze(-1)  # (B, 1, 1)
        k_team = torch.max(masked_k, dim=1).values  # (B, D)
        k_team = torch.where(has_players.squeeze(1), k_team, torch.zeros_like(k_team))

        # 2. Raw Generation Probability:
        # z_know = a_j * (dot(K_team, q_j) - b_j)
        q_j = self.question_q_emb(question_ids)  # (B, D)
        b_j = self.question_b(question_ids)      # (B, 1)
        a_j = F.softplus(self.question_a_raw(question_ids))  # (B, 1)

        dot_kq = (k_team * q_j).sum(dim=-1, keepdim=True)  # (B, 1)
        z_know = a_j * (dot_kq - b_j)                      # (B, 1)
        p_gen = torch.sigmoid(z_know)

        # 3. Team Synergy Matrix & S_team with Pairwise Joint Decay
        c_eff = self.get_effective_communication(player_ids, format_ids)  # (B, K, M)
        s_team = self.compute_team_synergy(c_eff, player_mask, pairwise_delta_t)  # (B, 1)

        # 4. Online Anomaly Adjustment:
        # If F == online and question is factual: z_anomaly = delta_cheat
        if anomaly_delta is not None:
            z_anomaly = anomaly_delta
        else:
            is_online = (format_ids == 2).unsqueeze(-1).float()  # (B, 1)
            factual_factor = (
                is_factual.float() if is_factual is not None else torch.ones_like(is_online)
            )
            z_anomaly = is_online * factual_factor * self.online_anomaly_weight

        # 5. Final Solved Probability:
        # P_final = sigmoid(logit(P_gen) + S_team + z_anomaly) = sigmoid(z_know + S_team + z_anomaly)
        z_final = z_know + s_team + z_anomaly
        p_final = torch.sigmoid(z_final)

        return {
            "logits": z_final,
            "probs": p_final,
            "p_gen": p_gen,
            "z_know": z_know,
            "s_team": s_team,
            "z_anomaly": z_anomaly,
            "k_team": k_team,
        }

    def compute_l2_regularization(self) -> torch.Tensor:
        """Compute L2 weight decay penalty on embeddings and synergy matrix."""
        l2_loss = (
            torch.norm(self.player_k_emb.weight) ** 2
            + torch.norm(self.player_c_base.weight) ** 2
            + torch.norm(self.player_c_delta.weight) ** 2
            + torch.norm(self.question_q_emb.weight) ** 2
            + torch.norm(self.W_synergy) ** 2
        )
        return l2_loss
