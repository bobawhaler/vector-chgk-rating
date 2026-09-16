"""Model inspection, synergy analysis, online anomaly detection, and embedding export."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from chgk_rating.config import (
    DEFAULT_META_PATH,
    DEFAULT_MODEL_PATH,
    FORMAT_OFFLINE,
    FORMAT_ONLINE,
    FORMAT_SYNC,
    FORMAT_TO_ID,
)
from chgk_rating.dataset import DatasetMetadata
from chgk_rating.model import DualEmbeddingChGKModel

logger = logging.getLogger(__name__)


def load_model_and_metadata(
    model_path: Path = DEFAULT_MODEL_PATH,
    meta_path: Path = DEFAULT_META_PATH,
    device: str = "cpu",
) -> Tuple[DualEmbeddingChGKModel, DatasetMetadata]:
    """Load trained DualEmbeddingChGKModel and DatasetMetadata from disk."""
    metadata = DatasetMetadata.load(meta_path)
    checkpoint = torch.load(model_path, map_location=device)
    cfg = checkpoint.get("config", {})

    model = DualEmbeddingChGKModel(
        num_players=cfg.get("num_players", len(metadata.player_indexer) + 1),
        num_questions=cfg.get("num_questions", len(metadata.question_indexer)),
        dim_k=cfg.get("dim_k", 16),
        dim_c=cfg.get("dim_c", 8),
        num_formats=cfg.get("num_formats", 3),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, metadata


def get_top_players(
    model: DualEmbeddingChGKModel,
    metadata: DatasetMetadata,
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    """Extract top-N players ranked by the L2 norm of their knowledge latent vector ||k_i||."""
    indexer = metadata.player_indexer
    with torch.no_grad():
        k_weights = model.player_k_emb.weight.cpu().numpy()  # (num_players, D)
        norms = np.linalg.norm(k_weights, axis=1)

    # Note: index 0 is padding, player indices are 1 .. len(indexer)
    player_ranks: List[Dict[str, Any]] = []
    for raw_id, idx in indexer.item_to_idx.items():
        if idx < len(norms):
            player_ranks.append({
                "player_id": raw_id,
                "player_idx": idx,
                "knowledge_norm": float(norms[idx]),
            })

    player_ranks.sort(key=lambda x: x["knowledge_norm"], reverse=True)
    for rank, p in enumerate(player_ranks[:top_n], start=1):
        p["rank"] = rank

    return player_ranks[:top_n]


def compute_team_synergy_details(
    model: DualEmbeddingChGKModel,
    metadata: DatasetMetadata,
    player_ids: List[Union[int, str]],
    format_name: str = FORMAT_OFFLINE,
) -> Dict[str, Any]:
    """Compute pairwise synergy matrix and total synergy for custom roster across formats."""
    fmt_id = metadata.format_map.get(format_name.lower(), 0)
    p_indexer = metadata.player_indexer

    mapped_indices: List[int] = []
    valid_player_ids: List[str] = []

    for p in player_ids:
        raw_str = str(p)
        idx = p_indexer.get_idx(raw_str, default=0)
        mapped_indices.append(idx)
        valid_player_ids.append(raw_str)

    # Pad or slice to 6 players
    padded_indices = mapped_indices[:6]
    while len(padded_indices) < 6:
        padded_indices.append(0)

    p_tensor = torch.tensor([padded_indices], dtype=torch.long)
    fmt_tensor = torch.tensor([fmt_id], dtype=torch.long)

    with torch.no_grad():
        c_eff = model.get_effective_communication(p_tensor, fmt_tensor)  # (1, 6, M)
        mask = p_tensor != 0
        s_mat = model.compute_team_synergy_matrix(c_eff, mask)[0].cpu().numpy()  # (6, 6)
        total_s = float(model.compute_team_synergy(c_eff, mask)[0, 0].item())

    # Build structured pairwise response for actual players provided
    n_players = len(mapped_indices[:6])
    matrix_subset = s_mat[:n_players, :n_players].tolist()

    pairwise_list: List[Dict[str, Any]] = []
    for i in range(n_players):
        for m in range(i + 1, n_players):
            pairwise_list.append({
                "player_1": valid_player_ids[i],
                "player_2": valid_player_ids[m],
                "synergy_score": float(s_mat[i, m]),
            })

    pairwise_list.sort(key=lambda x: x["synergy_score"], reverse=True)

    # Optimal seating arrangement heuristic: arrange top synergistic pairs adjacent
    return {
        "format": format_name,
        "format_id": fmt_id,
        "total_synergy": total_s,
        "players": valid_player_ids[:n_players],
        "synergy_matrix": matrix_subset,
        "pairwise_rankings": pairwise_list,
        "best_pair": pairwise_list[0] if pairwise_list else None,
    }


def detect_online_anomalies(
    df: pd.DataFrame,
    model: DualEmbeddingChGKModel,
    metadata: DatasetMetadata,
    min_games: int = 20,
) -> List[Dict[str, Any]]:
    """Detect online anomaly scores (delta_cheat) for teams showing unnatural online surges."""
    model.eval()
    p_indexer = metadata.player_indexer
    q_indexer = metadata.question_indexer

    team_stats: Dict[int, Dict[str, Any]] = {}

    with torch.no_grad():
        for _, row in df.iterrows():
            t_id = int(row["team_id"])
            fmt = str(row.get("tournament_type", FORMAT_OFFLINE)).lower()
            q_id = str(row["question_id"])
            if q_id not in q_indexer.item_to_idx:
                continue

            q_idx = q_indexer.get_idx(q_id)
            fmt_id = metadata.format_map.get(fmt, 0)
            solved = float(row["solved"])

            raw_players = row["player_ids"]
            if isinstance(raw_players, np.ndarray):
                raw_players = raw_players.tolist()
            elif not isinstance(raw_players, list):
                raw_players = [raw_players]

            p_indices = [p_indexer.get_idx(p, default=0) for p in raw_players[:6]]
            while len(p_indices) < 6:
                p_indices.append(0)

            p_ten = torch.tensor([p_indices], dtype=torch.long)
            q_ten = torch.tensor([q_idx], dtype=torch.long)
            f_ten = torch.tensor([fmt_id], dtype=torch.long)

            out = model(p_ten, q_ten, f_ten)
            pred_prob = float(out["probs"][0, 0].item())

            if t_id not in team_stats:
                team_stats[t_id] = {
                    "team_id": t_id,
                    "offline_actual": 0.0,
                    "offline_predicted": 0.0,
                    "offline_count": 0,
                    "online_actual": 0.0,
                    "online_predicted": 0.0,
                    "online_count": 0,
                }

            st = team_stats[t_id]
            if fmt == FORMAT_ONLINE:
                st["online_actual"] += solved
                st["online_predicted"] += pred_prob
                st["online_count"] += 1
            else:
                st["offline_actual"] += solved
                st["offline_predicted"] += pred_prob
                st["offline_count"] += 1

    anomalies: List[Dict[str, Any]] = []
    for t_id, st in team_stats.items():
        if st["offline_count"] >= min_games and st["online_count"] >= min_games:
            off_residual = (st["offline_actual"] - st["offline_predicted"]) / st["offline_count"]
            on_residual = (st["online_actual"] - st["online_predicted"]) / st["online_count"]
            delta_cheat = on_residual - off_residual

            anomalies.append({
                "team_id": t_id,
                "delta_cheat_score": float(delta_cheat),
                "online_residual": float(on_residual),
                "offline_residual": float(off_residual),
                "online_games": st["online_count"],
                "offline_games": st["offline_count"],
                "is_suspicious": bool(delta_cheat > 0.15),
            })

    anomalies.sort(key=lambda x: x["delta_cheat_score"], reverse=True)
    return anomalies


def export_embeddings_to_parquet(
    model: DualEmbeddingChGKModel,
    metadata: DatasetMetadata,
    output_dir: Path,
) -> Dict[str, Path]:
    """Export player and question embeddings to Parquet and CSV for visualization."""
    output_dir.mkdir(parents=True, exist_ok=True)
    p_indexer = metadata.player_indexer
    q_indexer = metadata.question_indexer

    with torch.no_grad():
        k_w = model.player_k_emb.weight.cpu().numpy()
        c_base_w = model.player_c_base.weight.cpu().numpy()
        c_delta_w = model.player_c_delta.weight.cpu().numpy()

        q_w = model.question_q_emb.weight.cpu().numpy()
        b_w = model.question_b.weight.cpu().numpy().squeeze(-1)
        a_w = model.question_a.cpu().numpy().squeeze(-1)

    # 1. Player embeddings table
    player_rows: List[Dict[str, Any]] = []
    for raw_p, idx in p_indexer.item_to_idx.items():
        if idx < len(k_w):
            k_vec = k_w[idx]
            c_b_vec = c_base_w[idx]
            c_d_vec = c_delta_w[idx]

            row: Dict[str, Any] = {
                "player_id": raw_p,
                "player_idx": idx,
                "knowledge_norm": float(np.linalg.norm(k_vec)),
                "comm_base_norm": float(np.linalg.norm(c_b_vec)),
                "format_sensitivity_norm": float(np.linalg.norm(c_d_vec)),
            }
            # Append latent components
            for d in range(len(k_vec)):
                row[f"k_{d}"] = float(k_vec[d])
            for m in range(len(c_b_vec)):
                row[f"c_base_{m}"] = float(c_b_vec[m])
                row[f"c_delta_{m}"] = float(c_d_vec[m])
            player_rows.append(row)

    p_df = pd.DataFrame(player_rows)
    p_parquet = output_dir / "player_embeddings.parquet"
    p_csv = output_dir / "player_embeddings.csv"
    p_df.to_parquet(p_parquet, index=False)
    p_df.to_csv(p_csv, index=False)

    # 2. Question embeddings table
    question_rows: List[Dict[str, Any]] = []
    for raw_q, idx in q_indexer.item_to_idx.items():
        if idx < len(q_w):
            q_vec = q_w[idx]
            row = {
                "question_id": raw_q,
                "question_idx": idx,
                "difficulty_bias": float(b_w[idx]),
                "discrimination_a": float(a_w[idx]),
                "requirement_norm": float(np.linalg.norm(q_vec)),
            }
            for d in range(len(q_vec)):
                row[f"q_{d}"] = float(q_vec[d])
            question_rows.append(row)

    q_df = pd.DataFrame(question_rows)
    q_parquet = output_dir / "question_embeddings.parquet"
    q_csv = output_dir / "question_embeddings.csv"
    q_df.to_parquet(q_parquet, index=False)
    q_df.to_csv(q_csv, index=False)

    logger.info("Exported embeddings to %s", output_dir)
    return {
        "player_parquet": p_parquet,
        "player_csv": p_csv,
        "question_parquet": q_parquet,
        "question_csv": q_csv,
    }
