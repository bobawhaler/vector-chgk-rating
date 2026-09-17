"""Dataset, chronological state tracking, and DataLoader collators for ChGK Rating Model."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from chgk_rating.config import DEFAULT_META_PATH, FORMAT_OFFLINE, FORMAT_TO_ID


def parse_date(d_str: Any) -> datetime:
    """Safely parse date string into datetime object."""
    s = str(d_str)[:10]
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        return datetime(2023, 1, 1)


class IDIndexer:
    """Bi-directional mapping between raw string/int IDs and contiguous tensor indices."""

    def __init__(self, start_idx: int = 0) -> None:
        self.start_idx = start_idx
        self.item_to_idx: Dict[str, int] = {}
        self.idx_to_item: Dict[int, str] = {}

    def fit(self, items: Sequence[Any]) -> None:
        for item in items:
            key = str(item)
            if key not in self.item_to_idx:
                idx = len(self.item_to_idx) + self.start_idx
                self.item_to_idx[key] = idx
                self.idx_to_item[idx] = key

    def get_idx(self, item: Any, default: Optional[int] = None) -> int:
        key = str(item)
        if default is not None:
            return self.item_to_idx.get(key, default)
        return self.item_to_idx[key]

    def get_item(self, idx: int) -> Optional[str]:
        return self.idx_to_item.get(idx)

    def __len__(self) -> int:
        return len(self.item_to_idx) + self.start_idx

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_idx": self.start_idx,
            "item_to_idx": self.item_to_idx,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> IDIndexer:
        indexer = cls(start_idx=data["start_idx"])
        indexer.item_to_idx = {str(k): int(v) for k, v in data["item_to_idx"].items()}
        indexer.idx_to_item = {int(v): str(k) for k, v in indexer.item_to_idx.items()}
        return indexer


class DatasetMetadata:
    """Container for serializing indexers, format vocabularies, and temporal player histories."""

    def __init__(
        self,
        player_indexer: IDIndexer,
        question_indexer: IDIndexer,
        format_map: Dict[str, int] = FORMAT_TO_ID,
        player_last_date: Optional[Dict[str, str]] = None,
        pair_last_date: Optional[Dict[str, str]] = None,
        player_activity_history: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> None:
        self.player_indexer = player_indexer
        self.question_indexer = question_indexer
        self.format_map = format_map
        self.player_last_date: Dict[str, str] = player_last_date or {}
        self.pair_last_date: Dict[str, str] = pair_last_date or {}
        self.player_activity_history: Dict[str, List[Dict[str, Any]]] = (
            player_activity_history or {}
        )

    def save(self, path: Path = DEFAULT_META_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "player_indexer": self.player_indexer.to_dict(),
                    "question_indexer": self.question_indexer.to_dict(),
                    "format_map": self.format_map,
                    "player_last_date": self.player_last_date,
                    "pair_last_date": self.pair_last_date,
                    "player_activity_history": self.player_activity_history,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: Path = DEFAULT_META_PATH) -> DatasetMetadata:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            player_indexer=IDIndexer.from_dict(data["player_indexer"]),
            question_indexer=IDIndexer.from_dict(data["question_indexer"]),
            format_map=data.get("format_map", FORMAT_TO_ID),
            player_last_date=data.get("player_last_date", {}),
            pair_last_date=data.get("pair_last_date", {}),
            player_activity_history=data.get("player_activity_history", {}),
        )


class ChGKDataset(Dataset):
    """PyTorch Dataset for atomic team-question observations with temporal tracking."""

    def __init__(
        self,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        max_players_per_team: int = 6,
        track_running_state: bool = True,
    ) -> None:
        self.max_players = max_players_per_team
        self.metadata = metadata

        p_idxer = metadata.player_indexer
        q_idxer = metadata.question_indexer
        fmt_map = metadata.format_map

        # Sort chronologically to compute realistic running gaps
        sorted_df = df.sort_values(by=["date", "tournament_id"]).reset_index(drop=True)

        rosters: List[List[int]] = []
        question_ids: List[int] = []
        format_ids: List[int] = []
        labels: List[float] = []
        player_delta_ts: List[List[float]] = []
        pairwise_delta_ts: List[List[List[float]]] = []

        # Running timestamps for computing Delta_t
        running_player_last: Dict[str, datetime] = {}
        running_pair_last: Dict[str, datetime] = {}

        q_ids = sorted_df["question_id"].astype(str).tolist()
        fmt_strs = (
            sorted_df["tournament_type"]
            .fillna(FORMAT_OFFLINE)
            .astype(str)
            .str.lower()
            .tolist()
            if "tournament_type" in sorted_df.columns
            else [FORMAT_OFFLINE] * len(sorted_df)
        )
        solved_vals = sorted_df["solved"].astype(float).tolist()
        date_strs = (
            sorted_df["date"].astype(str).tolist()
            if "date" in sorted_df.columns
            else ["2023-01-01"] * len(sorted_df)
        )
        player_id_lists = sorted_df["player_ids"].tolist()

        for q_id_raw, fmt_str, solved_val, curr_date_str, raw_players in zip(
            q_ids, fmt_strs, solved_vals, date_strs, player_id_lists
        ):
            if q_id_raw not in q_idxer.item_to_idx:
                continue

            q_idx = q_idxer.get_idx(q_id_raw)
            fmt_id = fmt_map.get(fmt_str, 0)
            curr_date = parse_date(curr_date_str)

            # Extract raw players
            if isinstance(raw_players, np.ndarray):
                raw_players = raw_players.tolist()
            elif not isinstance(raw_players, list):
                raw_players = [raw_players]

            mapped_p: List[int] = []
            valid_raw_p: List[str] = []
            p_deltas: List[float] = []

            for p in raw_players[:max_players_per_team]:
                p_str = str(p)
                p_mapped = p_idxer.get_idx(p_str, default=0)
                if p_mapped > 0:
                    mapped_p.append(p_mapped)
                    valid_raw_p.append(p_str)

                    # Compute Delta t_i = curr_date - T_last_i
                    if p_str in running_player_last:
                        delta_days = max(
                            0.0, (curr_date - running_player_last[p_str]).total_seconds() / 86400.0
                        )
                    else:
                        delta_days = 0.0
                    p_deltas.append(delta_days)

                    if track_running_state:
                        running_player_last[p_str] = curr_date

            # Compute pairwise joint gaps
            pair_mat = [[0.0] * max_players_per_team for _ in range(max_players_per_team)]
            n_val = len(valid_raw_p)
            for i in range(n_val):
                for m in range(n_val):
                    if i != m:
                        p1, p2 = sorted([valid_raw_p[i], valid_raw_p[m]])
                        pair_key = f"{p1}_{p2}"
                        if pair_key in running_pair_last:
                            p_days = max(
                                0.0,
                                (curr_date - running_pair_last[pair_key]).total_seconds() / 86400.0,
                            )
                        else:
                            p_days = 0.0
                        pair_mat[i][m] = p_days
                        if track_running_state and i < m:
                            running_pair_last[pair_key] = curr_date

            # Pad to max_players
            while len(mapped_p) < max_players_per_team:
                mapped_p.append(0)
                p_deltas.append(0.0)

            rosters.append(mapped_p)
            question_ids.append(q_idx)
            format_ids.append(fmt_id)
            labels.append(solved_val)
            player_delta_ts.append(p_deltas)
            pairwise_delta_ts.append(pair_mat)

        # Update metadata's global last seen timestamps if tracking
        if track_running_state:
            for p_str, dt_val in running_player_last.items():
                metadata.player_last_date[p_str] = dt_val.strftime("%Y-%m-%d")
            for pair_key, dt_val in running_pair_last.items():
                metadata.pair_last_date[pair_key] = dt_val.strftime("%Y-%m-%d")

        self.rosters = torch.tensor(rosters, dtype=torch.long)
        self.question_ids = torch.tensor(question_ids, dtype=torch.long)
        self.format_ids = torch.tensor(format_ids, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)
        self.player_delta_ts = torch.tensor(player_delta_ts, dtype=torch.float32)
        self.pairwise_delta_ts = torch.tensor(pairwise_delta_ts, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "player_ids": self.rosters[idx],
            "question_id": self.question_ids[idx],
            "format_id": self.format_ids[idx],
            "label": self.labels[idx],
            "player_delta_t": self.player_delta_ts[idx],
            "pairwise_delta_t": self.pairwise_delta_ts[idx],
        }
