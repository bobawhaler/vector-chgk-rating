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
        num_editors: int = 2000,
        num_questions: int = 1,
        dim_k: int = 16,
        dim_c: int = 8,
        num_formats: int = 3,
        padding_player_idx: int = 0,
        padding_editor_idx: int = 0,
        time_decay_config: Optional[TimeDecayConfig] = None,
    ) -> None:
        super().__init__()
        self.num_players = num_players
        self.num_editors = num_editors
        self.num_questions = num_questions
        self.dim_k = dim_k
        self.dim_c = dim_c
        self.num_formats = num_formats
        self.padding_player_idx = padding_player_idx
        self.padding_editor_idx = padding_editor_idx

        # Player Embeddings
        # Knowledge Latent Embedding: k_i in R^D ~ Normal(0, 0.1)
        self.player_k_emb = nn.Embedding(
            num_players, dim_k, padding_idx=padding_player_idx
        )
        nn.init.normal_(self.player_k_emb.weight, mean=0.0, std=0.1)

        # Online Knowledge Delta Embedding: Delta k_i^online in R^D ~ Zeros
        # Captures personal skill divergence in online format, with strong Bayesian shrinkage
        self.player_k_online = nn.Embedding(
            num_players, dim_k, padding_idx=padding_player_idx
        )
        nn.init.zeros_(self.player_k_online.weight)

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
            self.player_k_online.weight[padding_player_idx].zero_()
            self.player_c_base.weight[padding_player_idx].zero_()
            self.player_c_delta.weight[padding_player_idx].zero_()

        # Editor Embeddings (style / topic preference vector in R^D)
        # 0 is reserved for padding / unknown default editor
        self.editor_emb = nn.Embedding(
            num_editors, dim_k, padding_idx=padding_editor_idx
        )
        nn.init.normal_(self.editor_emb.weight, mean=0.0, std=0.1)

        # Editor baseline difficulty bias in R^1
        self.editor_b = nn.Embedding(
            num_editors, 1, padding_idx=padding_editor_idx
        )
        nn.init.constant_(self.editor_b.weight, 0.0)

        with torch.no_grad():
            self.editor_emb.weight[padding_editor_idx].zero_()
            self.editor_b.weight[padding_editor_idx].zero_()

        # Global default package requirement vector and baseline difficulty (for editors with 0 index)
        self.global_editor_vec = nn.Parameter(torch.zeros(dim_k))
        self.global_editor_b = nn.Parameter(torch.zeros(1))

        # Format Constraints (offline, sync, online)
        self.format_emb = nn.Embedding(num_formats, dim_c)
        nn.init.normal_(self.format_emb.weight, mean=0.0, std=0.1)

        # Global format baseline bias (e.g. format difficulty / environment baseline shift)
        self.format_bias = nn.Embedding(num_formats, 1)
        nn.init.zeros_(self.format_bias.weight)

        # Synergy Interaction Tensor: W_synergy in R^(M x M)
        self.W_synergy = nn.Parameter(torch.empty(dim_c, dim_c))
        nn.init.xavier_normal_(self.W_synergy, gain=0.1)

        # Dynamic Time Decay & Skill Tracking Module
        self.time_decay = TimeDecayModule(dim_k=dim_k, config=time_decay_config)

        # Online Anomaly Adjustment parameter/bias
        self.online_anomaly_weight = nn.Parameter(torch.zeros(1))

        # Experience scaling for synergy: S_im * (1.0 + beta_exp * log(1 + N_im))
        # Initialized to softplus^{-1}(0.2) ~ -1.5 so beta_exp starts positive and small
        self.raw_exp_weight = nn.Parameter(torch.tensor(-1.5))

        # Smooth-max temperature for knowledge pooling: tau * logsumexp(k / tau)
        # Initialized to softplus^{-1}(0.5) so tau starts around 0.5 (approaching hard max smoothly)
        self.raw_smoothmax_temp = nn.Parameter(torch.tensor(0.0))

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
        pairwise_exp: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute pairwise synergy matrix for team players: S_im = c_i^T * W * c_m * (1 + beta_exp * exp_im).

        Args:
            c_eff: Tensor of shape (B, K, M)
            player_mask: Optional boolean Tensor of shape (B, K) where True = valid player
            pairwise_delta_t: Optional Tensor of shape (B, K, K) containing days since joint game
            pairwise_exp: Optional Tensor of shape (B, K, K) containing log(1 + joint_games_count)

        Returns:
            Tensor of shape (B, K, K) containing pairwise synergy scores.
        """
        # C: (B, K, M), W: (M, M) -> C_W: (B, K, M)
        c_w = torch.matmul(c_eff, self.W_synergy)  # (B, K, M)
        # S_mat: (B, K, K) = C_W * C^T
        s_mat = torch.matmul(c_w, c_eff.transpose(-1, -2))  # (B, K, K)

        # Apply experience scaling from historical joint games: (1.0 + beta_exp * log(1 + N_games))
        if pairwise_exp is not None:
            beta_exp = F.softplus(self.raw_exp_weight)
            exp_scaling = 1.0 + beta_exp * pairwise_exp
            s_mat = s_mat * exp_scaling

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
        pairwise_exp: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute total team synergy: S_team = sum_{i < m} S_im.

        Args:
            c_eff: Tensor of shape (B, K, M)
            player_mask: Optional boolean Tensor of shape (B, K)
            pairwise_delta_t: Optional Tensor of shape (B, K, K)
            pairwise_exp: Optional Tensor of shape (B, K, K)

        Returns:
            Tensor of shape (B, 1)
        """
        s_mat = self.compute_team_synergy_matrix(
            c_eff, player_mask, pairwise_delta_t=pairwise_delta_t, pairwise_exp=pairwise_exp
        )  # (B, K, K)
        # Extract upper triangle strictly above diagonal (i < m)
        k = s_mat.size(-1)
        triu_indices = torch.triu_indices(k, k, offset=1, device=s_mat.device)
        s_team = s_mat[:, triu_indices[0], triu_indices[1]].sum(dim=-1, keepdim=True)
        return s_team  # (B, 1)

    def forward(
        self,
        player_ids: torch.Tensor,
        editor_ids: Optional[torch.Tensor] = None,
        format_ids: Optional[torch.Tensor] = None,
        question_ids: Optional[torch.Tensor] = None,
        is_factual: Optional[torch.Tensor] = None,
        anomaly_delta: Optional[torch.Tensor] = None,
        player_delta_t: Optional[torch.Tensor] = None,
        pairwise_delta_t: Optional[torch.Tensor] = None,
        pairwise_exp: Optional[torch.Tensor] = None,
        expected_difficulty: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for a batch of team observations conditioned on editors and format.

        Args:
            player_ids: LongTensor of shape (B, K) - roster player indices
            editor_ids: Optional LongTensor of shape (B, E) or (B,) - tournament editor indices
            format_ids: Optional LongTensor of shape (B,) - format indices (0=offline, 1=sync, 2=online)
            question_ids: Kept for legacy signature compatibility (unused)
            is_factual: Optional LongTensor/FloatTensor of shape (B, 1) - 1 if factual question
            anomaly_delta: Optional FloatTensor of shape (B, 1) - explicit cheat adjustment
            player_delta_t: Optional FloatTensor of shape (B, K) - elapsed days since last game per player
            pairwise_delta_t: Optional FloatTensor of shape (B, K, K) - elapsed days since last joint game per pair
            pairwise_exp: Optional FloatTensor of shape (B, K, K) - log(1 + joint games count) per pair
            expected_difficulty: Optional FloatTensor of shape (B, 1) or (B,) - optional difficulty override

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

        if format_ids is None:
            format_ids = torch.zeros(batch_size, dtype=torch.long, device=player_ids.device)

        # 1. Player Knowledge Embeddings & Temporal Decay
        k_player_embs = self.player_k_emb(player_ids)  # (B, K, D)

        # If format is online (format_id == 2), apply player online knowledge delta
        is_online_batch = (format_ids == 2).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        if hasattr(self, "player_k_online"):
            k_online_delta = self.player_k_online(player_ids)  # (B, K, D)
            k_player_embs = k_player_embs + is_online_batch * k_online_delta

        if player_delta_t is not None:
            # Apply asymmetric inactivity decay:
            # k_i(t) = k_i * exp(-lambda_decay * max(0, Delta_t - G_grace))
            k_player_embs = self.time_decay.apply_knowledge_decay(k_player_embs, player_delta_t)

            # In training mode, apply Brownian variance scaling:
            # eta_i(t) = sqrt(1.0 + alpha_gap * (Delta_t / 365.0))
            if self.training:
                gap_scale = self.time_decay.compute_training_gap_scale(player_delta_t).unsqueeze(-1)
                k_player_embs = k_player_embs * gap_scale

        # Team Knowledge Pooling: K_team(t) via Smooth-Max (LogSumExp with temperature tau)
        # tau = softplus(raw_smoothmax_temp) + 0.05
        # When tau -> 0, Smooth-Max becomes exact Hard Max
        tau = F.softplus(self.raw_smoothmax_temp) + 0.05
        has_players = player_mask.any(dim=-1, keepdim=True)  # (B, 1)

        # Mask padding players with -1e9
        masked_k = k_player_embs.masked_fill(~player_mask.unsqueeze(-1), -1e9)
        # Hard max component
        hard_max_k = torch.max(masked_k, dim=1).values  # (B, D)
        # Numerical stability shift: subtract hard_max_k
        scaled_diff = (masked_k - hard_max_k.unsqueeze(1)) / tau
        scaled_exp = torch.exp(scaled_diff) * player_mask.unsqueeze(-1).float()
        smooth_lse = tau * torch.log(torch.clamp(scaled_exp.sum(dim=1), min=1e-7))
        k_team = hard_max_k + smooth_lse  # (B, D)
        k_team = torch.where(has_players, k_team, torch.zeros_like(k_team))

        # 2. Package / Editor Conditioning:
        # If editors provided: pool their style embeddings and baseline difficulty
        if editor_ids is not None:
            if editor_ids.dim() == 1:
                editor_ids = editor_ids.unsqueeze(-1)  # (B, 1)
            ed_mask = (editor_ids != self.padding_editor_idx).unsqueeze(-1)  # (B, E, 1)
            ed_embs = self.editor_emb(editor_ids)  # (B, E, D)
            ed_bs = self.editor_b(editor_ids)      # (B, E, 1)

            # Average over valid editors
            valid_counts = ed_mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
            e_pkg = (ed_embs * ed_mask.float()).sum(dim=1) / valid_counts  # (B, D)
            b_pkg = (ed_bs * ed_mask.float()).sum(dim=1) / valid_counts    # (B, 1)

            # Where no editors specified (all 0 padding), add global default vector
            has_no_eds = (~ed_mask.any(dim=1))  # (B, 1)
            e_pkg = torch.where(has_no_eds, self.global_editor_vec.unsqueeze(0), e_pkg + self.global_editor_vec.unsqueeze(0))
            b_pkg = torch.where(has_no_eds, self.global_editor_b.unsqueeze(0), b_pkg + self.global_editor_b.unsqueeze(0))
        else:
            e_pkg = self.global_editor_vec.unsqueeze(0).expand(batch_size, -1)  # (B, D)
            b_pkg = self.global_editor_b.unsqueeze(0).expand(batch_size, -1)    # (B, 1)

        # Allow explicit difficulty override if supplied
        if expected_difficulty is not None:
            if expected_difficulty.dim() == 1:
                b_pkg = expected_difficulty.unsqueeze(-1)
            else:
                b_pkg = expected_difficulty

        # Solvability logit: dot(K_team, e_pkg) - b_pkg
        dot_ke = (k_team * e_pkg).sum(dim=-1, keepdim=True)  # (B, 1)
        z_know = dot_ke - b_pkg
        p_gen = torch.sigmoid(z_know)

        # 3. Team Synergy Matrix & S_team with Pairwise Joint Decay & Experience Scaling
        c_eff = self.get_effective_communication(player_ids, format_ids)  # (B, K, M)
        s_team = self.compute_team_synergy(
            c_eff, player_mask, pairwise_delta_t=pairwise_delta_t, pairwise_exp=pairwise_exp
        )  # (B, 1)

        # 4. Online Anomaly Adjustment:
        if anomaly_delta is not None:
            z_anomaly = anomaly_delta
        else:
            is_online = (format_ids == 2).unsqueeze(-1).float()  # (B, 1)
            factual_factor = (
                is_factual.float() if is_factual is not None else torch.ones_like(is_online)
            )
            z_anomaly = is_online * factual_factor * self.online_anomaly_weight

        # 5. Final Solved Probability:
        # P_final = sigmoid(logit(P_gen) + S_team + b_f + z_anomaly)
        b_f = self.format_bias(format_ids)  # (B, 1)
        z_final = z_know + s_team + b_f + z_anomaly
        p_final = torch.sigmoid(z_final)

        return {
            "logits": z_final,
            "probs": p_final,
            "p_gen": p_gen,
            "z_know": z_know,
            "s_team": s_team,
            "b_format": b_f,
            "z_anomaly": z_anomaly,
            "k_team": k_team,
        }

    def compute_l2_regularization(self) -> torch.Tensor:
        """Compute L2 weight decay penalty on embeddings and synergy matrix."""
        l2_loss = (
            torch.norm(self.player_k_emb.weight) ** 2
            + torch.norm(self.player_c_base.weight) ** 2
            + torch.norm(self.player_c_delta.weight) ** 2
            + torch.norm(self.editor_emb.weight) ** 2
            + torch.norm(self.editor_b.weight) ** 2
            + torch.norm(self.W_synergy) ** 2
        )
        if hasattr(self, "player_k_online"):
            # 10x stronger penalty on online knowledge deviation (Bayesian shrinkage towards offline base)
            l2_loss = l2_loss + 10.0 * torch.norm(self.player_k_online.weight) ** 2
        return l2_loss
