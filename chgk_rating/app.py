"""FastAPI application for ChGK Dual Latent Embedding Rating System with Dynamic Time Tracking."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from chgk_rating.analytics import (
    compute_team_synergy_details,
    get_player_historical_timeline,
    load_model_and_metadata,
)
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
    match_date: Optional[str] = Field(
        default=None, description="Match date in ISO format YYYY-MM-DD for dynamic inactivity decay"
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
    match_date: Optional[str]
    total_synergy: float
    predictions: List[QuestionPrediction]


class TeamSynergyRequest(BaseModel):
    player_ids: List[Union[int, str]] = Field(
        ..., min_length=2, max_length=6, description="List of 2 to 6 player IDs"
    )
    format: str = Field(
        default=FORMAT_OFFLINE, description="Tournament format: offline, sync, or online"
    )
    match_date: Optional[str] = Field(
        default=None, description="Match date for joint inactivity synergy decay"
    )


class PairwiseSynergy(BaseModel):
    player_1: str
    player_2: str
    synergy_score: float


class TeamSynergyResponse(BaseModel):
    format: str
    target_date: Optional[str]
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


class TimelinePoint(BaseModel):
    date: str
    days_inactive: float
    is_active: bool
    effective_knowledge_norm: float
    factual_norm: float
    logical_norm: float
    retention_ratio: float


class PlayerTimelineResponse(BaseModel):
    player_id: str
    timeline: List[TimelinePoint]


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
    description="Neural rating model with dynamic time decay for player skill, team synergy, and question difficulty in ChGK.",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "ChGK Dual Latent Embedding Rating API is running with Dynamic Time Tracking."}


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

    # Compute time gaps if match_date provided
    player_delta_t: Optional[torch.Tensor] = None
    pairwise_delta_t: Optional[torch.Tensor] = None

    if req.match_date:
        try:
            match_dt = datetime.strptime(str(req.match_date)[:10], "%Y-%m-%d")
        except ValueError:
            match_dt = datetime(2026, 9, 1)

        p_deltas = [0.0] * 6
        for i, p_str in enumerate(valid_p_ids[:6]):
            if p_str in metadata.player_last_date:
                last_dt = datetime.strptime(metadata.player_last_date[p_str][:10], "%Y-%m-%d")
                p_deltas[i] = max(0.0, (match_dt - last_dt).total_seconds() / 86400.0)
        player_delta_t = torch.tensor([p_deltas], dtype=torch.float32)

        # Pairwise gaps
        pair_mat = np.zeros((1, 6, 6), dtype=np.float32)
        n_p = len(valid_p_ids[:6])
        for i in range(n_p):
            for m in range(n_p):
                if i != m:
                    p1, p2 = sorted([valid_p_ids[i], valid_p_ids[m]])
                    key = f"{p1}_{p2}"
                    if key in metadata.pair_last_date:
                        last_p_dt = datetime.strptime(metadata.pair_last_date[key][:10], "%Y-%m-%d")
                        pair_mat[0, i, m] = max(0.0, (match_dt - last_p_dt).total_seconds() / 86400.0)
                    else:
                        pair_mat[0, i, m] = max(p_deltas[i], p_deltas[m])
        pairwise_delta_t = torch.tensor(pair_mat, dtype=torch.float32)

    # Tensor representations
    p_tensor = torch.tensor([mapped_players], dtype=torch.long)  # (1, 6)
    f_tensor = torch.tensor([fmt_id], dtype=torch.long)          # (1,)

    predictions: List[QuestionPrediction] = []
    total_synergy_val = 0.0

    with torch.no_grad():
        for q in req.question_ids:
            q_raw = str(q)
            if hasattr(model, "editor_emb"):
                # Current architecture: model is parameterized by editors, not question embeddings
                # We do forward pass with default editor (idx 0)
                ed_tensor = torch.zeros((1, 1), dtype=torch.long)
                out = model(
                    player_ids=p_tensor,
                    editor_ids=ed_tensor,
                    format_ids=f_tensor,
                    player_delta_t=player_delta_t,
                    pairwise_delta_t=pairwise_delta_t,
                )
            else:
                q_idx = q_indexer.get_idx(q_raw, default=0) if q_indexer else 0
                q_tensor = torch.tensor([q_idx], dtype=torch.long)
                out = model(
                    player_ids=p_tensor,
                    question_ids=q_tensor,
                    format_ids=f_tensor,
                    player_delta_t=player_delta_t,
                    pairwise_delta_t=pairwise_delta_t,
                )

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
        match_date=req.match_date,
        total_synergy=total_synergy_val,
        predictions=predictions,
    )


@app.post("/team_synergy", response_model=TeamSynergyResponse)
def get_team_synergy(req: TeamSynergyRequest) -> TeamSynergyResponse:
    model, metadata = get_model_and_meta()
    details = compute_team_synergy_details(
        model,
        metadata,
        req.player_ids,
        format_name=req.format,
        target_date=req.match_date,
    )

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
            best_next = max(remaining, key=lambda idx: matrix[current][idx])
            seating_order.append(players[best_next])
            remaining.remove(best_next)
            current = best_next

    return TeamSynergyResponse(
        format=details["format"],
        target_date=details.get("target_date"),
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

        all_c_base = model.player_c_base.weight.cpu().numpy()  # (N, M)
        w_syn = model.W_synergy.cpu().numpy()                  # (M, M)
        target_cw = np.dot(c_base, w_syn)                     # (M,)
        scores = np.dot(all_c_base, target_cw)                 # (N,)

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


@app.get("/player/{player_id}/historical_timeline", response_model=PlayerTimelineResponse)
def get_player_timeline(
    player_id: str,
    start_date: str = Query("2022-01-01", description="Timeline start date (YYYY-MM-DD)"),
    end_date: str = Query("2026-09-01", description="Timeline end date (YYYY-MM-DD)"),
    step_days: int = Query(30, ge=1, le=365, description="Step in days between timeline evaluations"),
) -> PlayerTimelineResponse:
    """Return historical skill trajectory showing growth vs inactivity decay curves."""
    model, metadata = get_model_and_meta()
    if str(player_id) not in metadata.player_indexer.item_to_idx:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found.")

    raw_timeline = get_player_historical_timeline(
        model, metadata, player_id, start_date=start_date, end_date=end_date, step_days=step_days
    )
    return PlayerTimelineResponse(
        player_id=str(player_id),
        timeline=[TimelinePoint(**pt) for pt in raw_timeline],
    )


# --- Editor Calibration & Tournament Simulation Endpoints ---

class EditorCalibrationResponse(BaseModel):
    editor_id: int
    name: str
    tournaments_count: int
    avg_forecast_dl: float
    avg_true_dl: float
    raw_bias: float
    calibrated_bias: float
    rmse: float
    mae: float
    accuracy_half_dl: float
    tendency: str


class SimulateTournamentRequest(BaseModel):
    teams: Dict[str, List[Union[int, str]]] = Field(
        ..., min_length=2, description="Dictionary mapping team names to player IDs"
    )
    format: str = Field(
        default=FORMAT_OFFLINE, description="Tournament format: offline, sync, or online"
    )
    target_dl: Optional[float] = Field(
        default=5.0, description="Target difficulty declared by tournament/editors"
    )
    editor_ids: Optional[List[int]] = Field(
        default=None, description="List of editor player IDs"
    )
    num_questions: int = Field(
        default=36, ge=1, le=120, description="Number of questions in tournament"
    )


class TeamStanding(BaseModel):
    team_name: str
    solve_prob: float
    expected_score: float
    mean_rank: float
    prob_1st_place: float
    prob_top_3: float


class SimulateTournamentResponse(BaseModel):
    calibrated_dl: float
    target_dl: Optional[float]
    editor_ids: List[int]
    num_questions: int
    standings: List[TeamStanding]


@app.get("/editor/{editor_id}/calibration", response_model=EditorCalibrationResponse)
def get_editor_calibration_route(editor_id: int) -> EditorCalibrationResponse:
    from chgk_rating.editor_analytics import get_editor_registry
    from dataclasses import asdict

    reg = get_editor_registry()
    prof = reg.get_editor_profile(editor_id)
    if prof is None:
        raise HTTPException(status_code=404, detail=f"Editor {editor_id} not found in calibration database.")

    return EditorCalibrationResponse(**asdict(prof))


@app.get("/editors/leaderboard", response_model=List[EditorCalibrationResponse])
def get_editors_leaderboard_route(
    min_tournaments: int = Query(5, ge=1, description="Minimum tournaments edited"),
    sort_by: str = Query("accuracy", description="Sort by: accuracy, rmse, count, or bias"),
    limit: int = Query(50, ge=1, le=200, description="Max results"),
) -> List[EditorCalibrationResponse]:
    from chgk_rating.editor_analytics import get_editor_registry
    from dataclasses import asdict

    reg = get_editor_registry()
    top = reg.get_top_editors(min_tournaments=min_tournaments, sort_by=sort_by, limit=limit)
    return [EditorCalibrationResponse(**asdict(p)) for p in top]


@app.post("/simulate_tournament", response_model=SimulateTournamentResponse)
def simulate_tournament_route(req: SimulateTournamentRequest) -> SimulateTournamentResponse:
    from chgk_rating.analytics import simulate_tournament

    model, metadata = get_model_and_meta()
    result = simulate_tournament(
        model=model,
        metadata=metadata,
        teams=req.teams,
        format_name=req.format,
        target_dl=req.target_dl,
        editor_ids=req.editor_ids,
        num_questions=req.num_questions,
    )
    return SimulateTournamentResponse(**result)
