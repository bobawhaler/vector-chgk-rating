"""FastAPI application for ChGK Dual Latent Embedding Rating System."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from chgk_rating.analytics import compute_team_synergy_details, load_model_and_metadata
from chgk_rating.config import DEFAULT_META_PATH, DEFAULT_MODEL_PATH, FORMAT_OFFLINE
from chgk_rating.dataset import DatasetMetadata
from chgk_rating.model import DualEmbeddingChGKModel

logger = logging.getLogger(__name__)

# Global model and metadata state
state: Dict[str, Any] = {
    "model": None,
    "metadata": None,
}


def get_model_and_meta() -> tuple[DualEmbeddingChGKModel, DatasetMetadata]:
    """Retrieve or lazily load trained model and metadata."""
    if state["model"] is None or state["metadata"] is None:
        if not DEFAULT_MODEL_PATH.exists() or not DEFAULT_META_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Model or metadata not found. Please train a model first using CLI.",
            )
        model, metadata = load_model_and_metadata(DEFAULT_MODEL_PATH, DEFAULT_META_PATH)
        state["model"] = model
        state["metadata"] = metadata
    return state["model"], state["metadata"]


# Request / Response Schemas
class PredictMatchRequest(BaseModel):
    player_ids: List[Union[int, str]] = Field(
        ..., min_length=1, max_length=6, description="List of up to 6 player IDs"
    )
    question_ids: List[Union[int, str]] = Field(
        ..., min_length=1, description="List of question IDs to predict"
    )
    format: str = Field(
        default=FORMAT_OFFLINE, description="Tournament format: offline, sync, or online"
    )


class QuestionPrediction(BaseModel):
    question_id: str
    solved_probability: float
    raw_generation_prob: float
    knowledge_logit: float
    team_synergy: float


class PredictMatchResponse(BaseModel):
    players: List[str]
    format: str
    total_synergy: float
    predictions: List[QuestionPrediction]


class TeamSynergyRequest(BaseModel):
    player_ids: List[Union[int, str]] = Field(
        ..., min_length=2, max_length=6, description="List of 2 to 6 player IDs"
    )
    format: str = Field(
        default=FORMAT_OFFLINE, description="Tournament format: offline, sync, or online"
    )


class PairwiseSynergy(BaseModel):
    player_1: str
    player_2: str
    synergy_score: float


class TeamSynergyResponse(BaseModel):
    format: str
    total_synergy: float
    players: List[str]
    synergy_matrix: List[List[float]]
    pairwise_rankings: List[PairwiseSynergy]
    optimal_seating_order: List[str]


class SynergisticTeammate(BaseModel):
    teammate_id: str
    synergy_score: float


class PlayerProfileResponse(BaseModel):
    player_id: str
    knowledge_norm: float
    communication_base_norm: float
    format_sensitivity_norm: float
    top_synergistic_teammates: List[SynergisticTeammate]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Attempt to pre-load model if available
    if DEFAULT_MODEL_PATH.exists() and DEFAULT_META_PATH.exists():
        try:
            model, metadata = load_model_and_metadata(DEFAULT_MODEL_PATH, DEFAULT_META_PATH)
            state["model"] = model
            state["metadata"] = metadata
            logger.info("Preloaded trained model and metadata.")
        except Exception as e:
            logger.warning("Could not preload model: %s", e)
    yield


app = FastAPI(
    title="ChGK Dual Latent Embedding Rating System",
    description="Neural rating model for player skill, team synergy, and question difficulty in ChGK.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "ChGK Dual Latent Embedding Rating API is running."}


@app.get("/health")
def health() -> Dict[str, Any]:
    model_loaded = state["model"] is not None or DEFAULT_MODEL_PATH.exists()
    return {
        "status": "healthy",
        "model_available": model_loaded,
    }


@app.post("/predict_match", response_model=PredictMatchResponse)
def predict_match(req: PredictMatchRequest) -> PredictMatchResponse:
    model, metadata = get_model_and_meta()
    p_indexer = metadata.player_indexer
    q_indexer = metadata.question_indexer

    # Map players
    mapped_players: List[int] = []
    valid_p_ids: List[str] = []
    for p in req.player_ids:
        raw_p = str(p)
        idx = p_indexer.get_idx(raw_p, default=0)
        mapped_players.append(idx)
        valid_p_ids.append(raw_p)

    while len(mapped_players) < 6:
        mapped_players.append(0)

    fmt_id = metadata.format_map.get(req.format.lower(), 0)

    # Tensor representations
    p_tensor = torch.tensor([mapped_players], dtype=torch.long)  # (1, 6)
    f_tensor = torch.tensor([fmt_id], dtype=torch.long)          # (1,)

    predictions: List[QuestionPrediction] = []
    total_synergy_val = 0.0

    with torch.no_grad():
        for q in req.question_ids:
            q_raw = str(q)
            q_idx = q_indexer.get_idx(q_raw, default=None)
            if q_idx is None:
                # If question is unknown, use average question index 0
                q_idx = 0

            q_tensor = torch.tensor([q_idx], dtype=torch.long)
            out = model(p_tensor, q_tensor, f_tensor)

            p_final = float(out["probs"][0, 0].item())
            p_gen = float(out["p_gen"][0, 0].item())
            z_know = float(out["z_know"][0, 0].item())
            s_team = float(out["s_team"][0, 0].item())
            total_synergy_val = s_team

            predictions.append(
                QuestionPrediction(
                    question_id=q_raw,
                    solved_probability=p_final,
                    raw_generation_prob=p_gen,
                    knowledge_logit=z_know,
                    team_synergy=s_team,
                )
            )

    return PredictMatchResponse(
        players=valid_p_ids,
        format=req.format,
        total_synergy=total_synergy_val,
        predictions=predictions,
    )


@app.post("/team_synergy", response_model=TeamSynergyResponse)
def get_team_synergy(req: TeamSynergyRequest) -> TeamSynergyResponse:
    model, metadata = get_model_and_meta()
    details = compute_team_synergy_details(
        model, metadata, req.player_ids, format_name=req.format
    )

    # Greedy optimal circular seating order heuristic to maximize adjacent synergy
    players = details["players"]
    matrix = details["synergy_matrix"]
    n = len(players)

    seating_order: List[str] = []
    if n > 0:
        remaining = list(range(n))
        current = 0
        seating_order.append(players[current])
        remaining.remove(current)

        while remaining:
            # Find closest synergistic teammate among remaining
            best_next = max(remaining, key=lambda idx: matrix[current][idx])
            seating_order.append(players[best_next])
            remaining.remove(best_next)
            current = best_next

    return TeamSynergyResponse(
        format=details["format"],
        total_synergy=details["total_synergy"],
        players=players,
        synergy_matrix=matrix,
        pairwise_rankings=[PairwiseSynergy(**p) for p in details["pairwise_rankings"]],
        optimal_seating_order=seating_order,
    )


@app.get("/player/{player_id}/profile", response_model=PlayerProfileResponse)
def get_player_profile(player_id: str) -> PlayerProfileResponse:
    model, metadata = get_model_and_meta()
    p_indexer = metadata.player_indexer

    if str(player_id) not in p_indexer.item_to_idx:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found in model.")

    idx = p_indexer.get_idx(str(player_id))

    with torch.no_grad():
        k_vec = model.player_k_emb.weight[idx].cpu().numpy()
        c_base = model.player_c_base.weight[idx].cpu().numpy()
        c_delta = model.player_c_delta.weight[idx].cpu().numpy()

        k_norm = float(np.linalg.norm(k_vec))
        c_base_norm = float(np.linalg.norm(c_base))
        c_delta_norm = float(np.linalg.norm(c_delta))

        # Find top 5 synergistic partners across all players (offline format)
        all_c_base = model.player_c_base.weight.cpu().numpy()  # (N, M)
        w_syn = model.W_synergy.cpu().numpy()                  # (M, M)

        # synergy score: c_i * W * c_m^T
        target_cw = np.dot(c_base, w_syn)                     # (M,)
        scores = np.dot(all_c_base, target_cw)                 # (N,)

    # Rank partners excluding self and padding (idx 0)
    top_partner_indices = np.argsort(-scores)
    top_teammates: List[SynergisticTeammate] = []

    for p_idx in top_partner_indices:
        if p_idx == 0 or p_idx == idx:
            continue
        p_raw = p_indexer.get_item(int(p_idx))
        if p_raw:
            top_teammates.append(
                SynergisticTeammate(
                    teammate_id=p_raw,
                    synergy_score=float(scores[p_idx]),
                )
            )
        if len(top_teammates) >= 5:
            break

    return PlayerProfileResponse(
        player_id=str(player_id),
        knowledge_norm=k_norm,
        communication_base_norm=c_base_norm,
        format_sensitivity_norm=c_delta_norm,
        top_synergistic_teammates=top_teammates,
    )
