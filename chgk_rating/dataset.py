"""Dataset and DataLoader collators for ChGK Rating Model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from chgk_rating.config import DEFAULT_META_PATH, FORMAT_OFFLINE, FORMAT_TO_ID


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
    """Container for serializing indexers and format vocabularies."""

    def __init__(
        self,
        player_indexer: IDIndexer,
        question_indexer: IDIndexer,
        format_map: Dict[str, int] = FORMAT_TO_ID,
    ) -> None:
        self.player_indexer = player_indexer
        self.question_indexer = question_indexer
        self.format_map = format_map

    def save(self, path: Path = DEFAULT_META_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "player_indexer": self.player_indexer.to_dict(),
                    "question_indexer": self.question_indexer.to_dict(),
                    "format_map": self.format_map,
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
        )


class ChGKDataset(Dataset):
    """PyTorch Dataset for atomic team-question observations."""

    def __init__(
        self,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        max_players_per_team: int = 6,
    ) -> None:
        self.max_players = max_players_per_team
        self.metadata = metadata

        # Pre-convert columns to efficient numpy arrays / tensors
        p_idxer = metadata.player_indexer
        q_idxer = metadata.question_indexer
        fmt_map = metadata.format_map

        # Filter out rows with unseen questions or empty players if needed
        rosters: List[List[int]] = []
        question_ids: List[int] = []
        format_ids: List[int] = []
        labels: List[float] = []

        for _, row in df.iterrows():
            q_id_raw = str(row["question_id"])
            if q_id_raw not in q_idxer.item_to_idx:
                continue

            q_idx = q_idxer.get_idx(q_id_raw)
            fmt_str = str(row.get("tournament_type", FORMAT_OFFLINE)).lower()
            fmt_id = fmt_map.get(fmt_str, 0)
            solved_val = float(row["solved"])

            # Map players, pad with 0 up to max_players
            raw_players = row["player_ids"]
            if isinstance(raw_players, np.ndarray):
                raw_players = raw_players.tolist()
            elif not isinstance(raw_players, list):
                raw_players = [raw_players]

            mapped_p: List[int] = []
            for p in raw_players[:max_players_per_team]:
                p_mapped = p_idxer.get_idx(p, default=0)
                if p_mapped > 0:
                    mapped_p.append(p_mapped)

            # Pad to max_players
            while len(mapped_p) < max_players_per_team:
                mapped_p.append(0)

            rosters.append(mapped_p)
            question_ids.append(q_idx)
            format_ids.append(fmt_id)
            labels.append(solved_val)

        self.rosters = torch.tensor(rosters, dtype=torch.long)
        self.question_ids = torch.tensor(question_ids, dtype=torch.long)
        self.format_ids = torch.tensor(format_ids, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "player_ids": self.rosters[idx],
            "question_id": self.question_ids[idx],
            "format_id": self.format_ids[idx],
            "label": self.labels[idx],
        }
