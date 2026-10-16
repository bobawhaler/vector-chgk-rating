"""Tournament prediction engine for arbitrary ChGK tournament IDs.

Integrates Stage 1 Bayesian editor difficulty calibration with Stage 2
neural team embeddings, communication synergy, tactical tour optimization,
and Monte Carlo placement simulations.
"""

from __future__ import annotations

import itertools
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
import numpy as np
import pandas as pd
import torch

from chgk_rating.analytics import load_model_and_metadata
from chgk_rating.config import (
    DEFAULT_API_BASE_URL,
    DEFAULT_CACHE_DB_PATH,
    DEFAULT_META_PATH,
    DEFAULT_MODEL_PATH,
    FORMAT_OFFLINE,
    FORMAT_ONLINE,
    FORMAT_SYNC,
    FORMAT_TO_ID,
)
from chgk_rating.data_loader import normalize_tournament_format
from chgk_rating.dataset import DatasetMetadata
from chgk_rating.editor_analytics import EditorCalibrationRegistry, get_editor_registry
from chgk_rating.model import DualEmbeddingChGKModel

logger = logging.getLogger(__name__)


class TournamentPredictor:
    """End-to-end tournament outcome and standings predictor."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        meta_path: Path = DEFAULT_META_PATH,
        db_path: Path = DEFAULT_CACHE_DB_PATH,
        device: str = "cpu",
    ) -> None:
        self.model_path = Path(model_path)
        self.meta_path = Path(meta_path)
        self.db_path = Path(db_path)
        self.device = torch.device(device)

        self._model: Optional[DualEmbeddingChGKModel] = None
        self._metadata: Optional[DatasetMetadata] = None
        self._registry: Optional[EditorCalibrationRegistry] = None

    def _ensure_loaded(self) -> None:
        """Lazily load model weights, vocab indexers, and editor calibration registry."""
        if self._model is None or self._metadata is None:
            self._model, self._metadata = load_model_and_metadata(
                model_path=self.model_path,
                meta_path=self.meta_path,
                device=str(self.device),
            )
            self._model.eval()

        if self._registry is None:
            self._registry = get_editor_registry(self.db_path)
            self._registry.load_data()

    def fetch_tournament_data(
        self,
        tournament_id: int,
        force_refresh: bool = False,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Retrieve tournament metadata and team results/rosters from local cache or official API.

        Args:
            tournament_id: Official rating.chgk.info tournament ID.
            force_refresh: If True, bypass cache and fetch fresh data from API.

        Returns:
            Tuple of (tournament_metadata_dict, list_of_team_results_dicts).
        """
        t_meta: Optional[Dict[str, Any]] = None
        results_data: Optional[List[Dict[str, Any]]] = None

        meta_url = f"{DEFAULT_API_BASE_URL}/tournaments/{tournament_id}"
        results_url = f"{DEFAULT_API_BASE_URL}/tournaments/{tournament_id}/results?includeMasksAndControversials=1&includeTeamMembers=1"
        teams_url = f"{DEFAULT_API_BASE_URL}/tournaments/{tournament_id}/teams"
        requests_url = f"{DEFAULT_API_BASE_URL}/tournaments/{tournament_id}/requests"

        if not force_refresh and self.db_path.exists():
            con = sqlite3.connect(self.db_path)
            cur = con.cursor()

            # 1. Check ingested_tournaments
            cur.execute(
                "SELECT raw_data FROM ingested_tournaments WHERE tournament_id = ?",
                (tournament_id,),
            )
            row = cur.fetchone()
            if row:
                try:
                    t_meta = json.loads(row[0])
                except Exception:
                    pass

            # 2. Check api_cache for tournament info
            if t_meta is None:
                cur.execute("SELECT response_json FROM api_cache WHERE url = ?", (meta_url,))
                r = cur.fetchone()
                if r:
                    try:
                        t_meta = json.loads(r[0])
                    except Exception:
                        pass

            # 3. Check api_cache for results
            cur.execute("SELECT response_json FROM api_cache WHERE url = ?", (results_url,))
            r_res = cur.fetchone()
            if r_res:
                try:
                    results_data = json.loads(r_res[0])
                except Exception:
                    pass

            con.close()

        # If missing, fetch directly from official API
        if t_meta is None or results_data is None or force_refresh:
            try:
                with httpx.Client(timeout=15.0) as client:
                    if t_meta is None or force_refresh:
                        resp = client.get(meta_url)
                        if resp.status_code == 200:
                            t_meta = resp.json()
                            self._cache_api_response(meta_url, resp.text)
                        else:
                            raise ValueError(
                                f"Tournament {tournament_id} returned HTTP {resp.status_code}"
                            )

                    if results_data is None or force_refresh:
                        resp_res = client.get(results_url)
                        if resp_res.status_code == 200:
                            results_data = resp_res.json()
                            self._cache_api_response(results_url, resp_res.text)

                        # If results are empty (upcoming tournament), try /teams or /requests
                        if not results_data:
                            resp_teams = client.get(teams_url)
                            if resp_teams.status_code == 200:
                                results_data = resp_teams.json()
                                self._cache_api_response(teams_url, resp_teams.text)

                        if not results_data:
                            resp_req = client.get(requests_url)
                            if resp_req.status_code == 200:
                                results_data = resp_req.json()
                                self._cache_api_response(requests_url, resp_req.text)
            except Exception as exc:
                if t_meta is None:
                    raise RuntimeError(
                        f"Failed to fetch data for tournament {tournament_id}: {exc}"
                    ) from exc
                logger.warning(
                    "Network error fetching results for %d; proceeding with cached/empty results: %s",
                    tournament_id,
                    exc,
                )

        if t_meta is None:
            raise ValueError(f"Could not load metadata for tournament ID {tournament_id}")

        results_data = results_data or []
        return t_meta, results_data

    def _cache_api_response(self, url: str, response_text: str) -> None:
        """Persist API response into local SQLite cache."""
        try:
            if not self.db_path.exists():
                return
            con = sqlite3.connect(self.db_path)
            cur = con.cursor()
            columns = {col[1] for col in cur.execute("PRAGMA table_info(api_cache)").fetchall()}
            if "status_code" in columns:
                cur.execute(
                    "INSERT OR REPLACE INTO api_cache (url, status_code, response_json, created_at) VALUES (?, 200, ?, datetime('now'))",
                    (url, response_text),
                )
            elif "cached_at" in columns:
                cur.execute(
                    "INSERT OR REPLACE INTO api_cache (url, response_json, cached_at) VALUES (?, ?, datetime('now'))",
                    (url, response_text),
                )
            else:
                cur.execute(
                    "INSERT OR REPLACE INTO api_cache (url, response_json) VALUES (?, ?)",
                    (url, response_text),
                )
            con.commit()
            con.close()
        except Exception as exc:
            logger.debug("Failed to cache API response for %s: %s", url, exc)

    def predict_tournament(
        self,
        tournament_id: int,
        num_simulations: int = 10000,
        tactical: bool = True,
        force_refresh: bool = False,
        tournament_dl_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Perform end-to-end two-stage rating forecast for an arbitrary tournament.

        Args:
            tournament_id: Tournament ID on rating.chgk.info.
            num_simulations: Number of Monte Carlo simulation runs for rank probabilities.
            tactical: If True, evaluates per-tour editor style and optimal 6-person roster lineups.
            force_refresh: If True, refreshes tournament metadata from official API.
            tournament_dl_override: Optional difficulty override for sensitivity analysis.

        Returns:
            Structured dictionary with standings, predictions, and Monte Carlo probabilities.
        """
        self._ensure_loaded()
        assert self._model is not None and self._metadata is not None and self._registry is not None

        t_meta, results_data = self.fetch_tournament_data(
            tournament_id=tournament_id, force_refresh=force_refresh
        )

        t_name = str(t_meta.get("name") or f"Tournament {tournament_id}")
        t_date = str(t_meta.get("dateStart") or "2026-01-01")[:10]

        # Determine tournament format
        t_type = t_meta.get("type", {})
        t_type_name = t_type.get("name") if isinstance(t_type, dict) else str(t_type or "")
        t_format = normalize_tournament_format(t_type_name, t_name)
        fmt_id = self._metadata.format_map.get(t_format, 0)
        f_tensor = torch.tensor([fmt_id], dtype=torch.long, device=self.device)

        # Extract editors
        raw_editors = t_meta.get("editors") or []
        editor_ids = [
            int(e["id"]) for e in raw_editors if isinstance(e, dict) and "id" in e and e["id"]
        ]

        # Extract question count
        q_qty = t_meta.get("questionQty") or {}
        if isinstance(q_qty, dict) and q_qty:
            tour_questions = [int(v) for v in q_qty.values() if isinstance(v, (int, float))]
            total_questions = sum(tour_questions) if tour_questions else 36
        else:
            total_questions = 36
            tour_questions = [total_questions]

        # Determine Stage 1 Calibrated Difficulty
        forecast_dl = (
            float(t_meta["difficultyForecast"])
            if t_meta.get("difficultyForecast") is not None
            else None
        )
        actual_true_dl = (
            float(t_meta["trueDL"]) if t_meta.get("trueDL") is not None else None
        )

        if tournament_dl_override is not None:
            calibrated_dl = float(tournament_dl_override)
            std_dl = 0.65
        else:
            calibrated_dl, std_dl = self._registry.predict_tournament_dl_distribution(
                forecast_dl=forecast_dl,
                editor_ids=editor_ids,
                tournament_format=t_format,
            )

        # Editor tensor for general evaluation
        ed_indexer = self._metadata.editor_indexer
        mapped_eds = [ed_indexer.get_idx(str(e), default=0) for e in editor_ids]
        mapped_eds = [idx for idx in mapped_eds if idx > 0]
        while len(mapped_eds) < 6:
            mapped_eds.append(0)
        ed_tensor_pooled = torch.tensor([mapped_eds[:6]], dtype=torch.long, device=self.device)

        # Per-tour editor tensors. Credits name the editors of each question range;
        # without them, each listed tournament editor is treated as one tour.
        editor_tensors_by_tour = []
        credit = None
        if t_meta.get("credits"):
            from chgk_rating.tour_editors import credit_record

            credit = credit_record(t_meta)
        if tactical and credit is not None and tour_questions:
            from chgk_rating.tour_editors import editors_for_span

            cursor = 1
            for count in tour_questions:
                span_ids = editors_for_span(credit, cursor, cursor + int(count) - 1, editor_ids)
                mapped_tour = [ed_indexer.get_idx(str(item), default=0) for item in span_ids]
                mapped_tour = [idx for idx in mapped_tour if idx > 0][:6]
                while len(mapped_tour) < 6:
                    mapped_tour.append(0)
                editor_tensors_by_tour.append(
                    torch.tensor([mapped_tour], dtype=torch.long, device=self.device)
                )
                cursor += int(count)
        elif tactical and editor_ids:
            for ed_id in editor_ids:
                e_idx = ed_indexer.get_idx(str(ed_id), default=0)
                e_ten = torch.tensor([[e_idx, 0, 0, 0, 0, 0]], dtype=torch.long, device=self.device)
                editor_tensors_by_tour.append(e_ten)

        p_indexer = self._metadata.player_indexer
        teams_eval: List[Dict[str, Any]] = []
        skipped_teams: List[Tuple[Any, str]] = []

        # Parse team rosters and run neural inference
        for t in results_data:
            team_obj = t.get("team") if isinstance(t.get("team"), dict) else {}
            team_id = team_obj.get("id") or t.get("idTeam") or t.get("team_id")
            team_name = team_obj.get("name") or t.get("team_name") or "Unknown"
            town = (team_obj.get("town") or {}).get("name", "") if isinstance(team_obj.get("town"), dict) else ""

            members = t.get("teamMembers") or t.get("members") or []
            player_ids: List[Any] = []
            for m in members:
                if isinstance(m, dict):
                    p_obj = m.get("player") or m
                    if isinstance(p_obj, dict) and p_obj.get("id"):
                        player_ids.append(p_obj["id"])
                    elif "id" in m:
                        player_ids.append(m["id"])

            if not player_ids:
                skipped_teams.append((team_id, team_name))
                continue

            num_players = len(player_ids)
            valid_p_all = [str(p) for p in player_ids]

            # Roster evaluation
            is_tactical_applicable = (
                tactical
                and len(editor_tensors_by_tour) > 0
                and len(editor_tensors_by_tour) == len(tour_questions)
            )

            # Extract actual score if tournament was already completed
            act_score: Optional[int] = None
            mask = t.get("mask")
            current = t.get("current")
            if mask and isinstance(mask, str) and any(c in "01" for c in mask):
                act_score = sum(int(c) for c in mask if c in "01")
            elif isinstance(current, dict) and current.get("score") is not None:
                act_score = int(current["score"])

            with torch.no_grad():
                if is_tactical_applicable and num_players > 6:
                    # Tactical Lineup Optimization: choose subset of 6 players maximizing P(Solve) for each tour
                    combos = list(itertools.combinations(valid_p_all, 6))
                    tour_probs = []

                    for t_idx, e_ten in enumerate(editor_tensors_by_tour):
                        best_p = -1.0
                        best_synergy = 0.0
                        best_z_know = 0.0
                        best_k_norm = 0.0

                        for c_roster in combos:
                            p_indices = [p_indexer.get_idx(p, default=0) for p in c_roster]
                            exp_mat = np.zeros((1, 6, 6), dtype=np.float32)
                            for i in range(6):
                                for m in range(6):
                                    if i != m:
                                        p1, p2 = sorted([c_roster[i], c_roster[m]])
                                        exp_mat[0, i, m] = float(
                                            np.log1p(self._metadata.pair_games_count.get(f"{p1}_{p2}", 0))
                                        )

                            p_ten = torch.tensor([p_indices], dtype=torch.long, device=self.device)
                            exp_ten = torch.from_numpy(exp_mat).to(self.device)
                            t_dl_ten = torch.tensor([[calibrated_dl]], dtype=torch.float32, device=self.device)

                            out = self._model(
                                player_ids=p_ten,
                                editor_ids=e_ten,
                                format_ids=f_tensor,
                                pairwise_exp=exp_ten,
                                tournament_dl=t_dl_ten,
                            )
                            c_p = float(out["probs"].item())
                            if c_p > best_p:
                                best_p = c_p
                                best_synergy = float(out["s_team"].item())
                                best_z_know = float(out["z_know"].item())
                                best_k_norm = float(torch.norm(out["k_team"]).item())

                        tour_probs.append(best_p)

                    # Weighted mean solve probability across tours
                    expected_score = sum(
                        p_t * q_t for p_t, q_t in zip(tour_probs, tour_questions)
                    )
                    prob = expected_score / max(total_questions, 1)
                    synergy = best_synergy
                    z_know = best_z_know
                    k_norm = best_k_norm
                    w_syn = 0.21
                    tactics_desc = f"Оптимальные замены из {num_players} чел."

                    # TrueDL prediction
                    if actual_true_dl is not None:
                        t_true_dl_ten = torch.tensor([[actual_true_dl]], dtype=torch.float32, device=self.device)
                        expected_score_true = 0.0
                        for t_idx, e_ten in enumerate(editor_tensors_by_tour):
                            out_tr = self._model(
                                player_ids=p_ten,
                                editor_ids=e_ten,
                                format_ids=f_tensor,
                                pairwise_exp=exp_ten,
                                tournament_dl=t_true_dl_ten,
                            )
                            expected_score_true += float(out_tr["probs"].item()) * tour_questions[t_idx]
                    else:
                        expected_score_true = None

                else:
                    # Standard or <=6 players evaluation
                    valid_p = valid_p_all[:6]
                    p_indices = [p_indexer.get_idx(p, default=0) for p in valid_p]
                    while len(p_indices) < 6:
                        p_indices.append(0)

                    exp_mat = np.zeros((1, 6, 6), dtype=np.float32)
                    n_p = len(valid_p)
                    for i in range(n_p):
                        for m in range(n_p):
                            if i != m:
                                p1, p2 = sorted([valid_p[i], valid_p[m]])
                                exp_mat[0, i, m] = float(
                                    np.log1p(self._metadata.pair_games_count.get(f"{p1}_{p2}", 0))
                                )

                    p_ten = torch.tensor([p_indices], dtype=torch.long, device=self.device)
                    exp_ten = torch.from_numpy(exp_mat).to(self.device)
                    t_dl_ten = torch.tensor([[calibrated_dl]], dtype=torch.float32, device=self.device)

                    out = self._model(
                        player_ids=p_ten,
                        editor_ids=ed_tensor_pooled,
                        format_ids=f_tensor,
                        pairwise_exp=exp_ten,
                        tournament_dl=t_dl_ten,
                    )

                    prob = float(out["probs"].item())
                    expected_score = prob * total_questions
                    synergy = float(out["s_team"].item())
                    w_syn = float(out["w_syn"].item()) if "w_syn" in out else 0.21
                    z_know = float(out["z_know"].item())
                    k_norm = float(torch.norm(out["k_team"]).item())
                    tactics_desc = "Основной состав"

                    if actual_true_dl is not None:
                        t_true_dl_ten = torch.tensor([[actual_true_dl]], dtype=torch.float32, device=self.device)
                        out_true = self._model(
                            player_ids=p_ten,
                            editor_ids=ed_tensor_pooled,
                            format_ids=f_tensor,
                            pairwise_exp=exp_ten,
                            tournament_dl=t_true_dl_ten,
                        )
                        expected_score_true = float(out_true["probs"].item()) * total_questions
                    else:
                        expected_score_true = None

            teams_eval.append({
                "team_id": team_id,
                "team_name": team_name,
                "town": town,
                "players_count": num_players,
                "tactics": tactics_desc,
                "solve_prob": prob,
                "expected_score": expected_score,
                "expected_score_true_dl": expected_score_true,
                "actual_score": act_score,
                "diff_calibrated_dl": (expected_score - act_score) if act_score is not None else None,
                "diff_true_dl": (expected_score_true - act_score) if (expected_score_true is not None and act_score is not None) else None,
                "synergy": synergy,
                "w_syn": w_syn,
                "z_know": z_know,
                "knowledge_norm": k_norm,
            })

        if not teams_eval:
            raise ValueError(f"No valid team rosters found for tournament {tournament_id}")

        # Check if actual scores are meaningful (tournament has actually been played)
        has_real_scores = any(
            t["actual_score"] is not None and t["actual_score"] > 0 for t in teams_eval
        )
        if not has_real_scores:
            for t in teams_eval:
                t["actual_score"] = None
                t["diff_calibrated_dl"] = None
                t["diff_true_dl"] = None

        # Monte Carlo Placement Simulation
        np.random.seed(42)
        num_teams = len(teams_eval)
        sim_scores = np.zeros((num_simulations, num_teams), dtype=np.int32)

        for idx, t in enumerate(teams_eval):
            p = t["solve_prob"]
            sim_scores[:, idx] = np.random.binomial(total_questions, p, size=num_simulations)

        # Compute ranks per simulation
        # Higher score = lower rank index (rank 1 is best)
        ranks = np.zeros((num_simulations, num_teams), dtype=np.float32)
        for s in range(num_simulations):
            order = np.argsort(-sim_scores[s, :])
            ranks[s, order] = np.arange(1, num_teams + 1)

        for idx, t in enumerate(teams_eval):
            t["mean_rank"] = float(np.mean(ranks[:, idx]))
            t["prob_1st"] = float(np.mean(ranks[:, idx] == 1))
            t["prob_top3"] = float(np.mean(ranks[:, idx] <= 3))
            t["prob_top5"] = float(np.mean(ranks[:, idx] <= 5))
            t["p10"] = float(np.percentile(sim_scores[:, idx], 10))
            t["p90"] = float(np.percentile(sim_scores[:, idx], 90))

        # Sort standings by expected score
        df_standings = pd.DataFrame(teams_eval).sort_values("expected_score", ascending=False).reset_index(drop=True)
        standings_list = df_standings.to_dict(orient="records")

        return {
            "tournament_id": tournament_id,
            "tournament_name": t_name,
            "date": t_date,
            "format": t_format,
            "total_questions": total_questions,
            "forecast_dl": forecast_dl,
            "calibrated_dl": calibrated_dl,
            "std_dl": std_dl,
            "actual_true_dl": actual_true_dl,
            "editors_count": len(editor_ids),
            "skipped_teams": skipped_teams,
            "standings": standings_list,
        }

    def format_terminal_table(self, res: Dict[str, Any]) -> str:
        """Format prediction results into a readable text table."""
        has_actual = any(r.get("actual_score") is not None for r in res["standings"])
        has_truedl = res.get("actual_true_dl") is not None

        lines = []
        lines.append("=" * 105)
        lines.append(f"Турнир: {res['tournament_name']} (ID: {res['tournament_id']})")
        fc_str = f"{res['forecast_dl']:.2f}" if res.get("forecast_dl") is not None else "-"
        true_str = f"{res['actual_true_dl']:.2f}" if res.get("actual_true_dl") is not None else "-"
        lines.append(
            f"Формат: {res['format'].upper()} | Вопросов: {res['total_questions']} | Заявленная сл. (FC): {fc_str} | "
            f"Калибр. сл.: {res['calibrated_dl']:.2f} | Факт. сл. (TrueDL): {true_str}"
        )
        lines.append("-" * 105)

        if has_actual and has_truedl:
            header = f"{'#':<3} {'Команда':<28} | {'Город':<14} | {'Пред(Калибр)':<12} | {'Пред(TrueDL)':<12} | {'Факт':<5} | {'Дельта':<7} | {'P(1-е)':<6} | {'P(Топ-3)':<7} | {'Ср.место':<8}"
        elif has_actual:
            header = f"{'#':<3} {'Команда':<28} | {'Город':<14} | {'Пред(Калибр)':<12} | {'Факт':<5} | {'Дельта':<7} | {'P(1-е)':<6} | {'P(Топ-3)':<7} | {'Ср.место':<8}"
        else:
            header = f"{'#':<3} {'Команда':<28} | {'Город':<14} | {'Ожид.балл':<10} | {'90% интервал':<14} | {'Взятие %':<9} | {'P(1-е)':<6} | {'P(Топ-3)':<7} | {'Ср.место':<8}"

        lines.append(header)
        lines.append("-" * 105)

        for rank, r in enumerate(res["standings"], 1):
            t_name = r["team_name"][:28]
            t_town = (r.get("town") or "")[:14]
            exp_s = f"{r['expected_score']:.1f}"

            if has_actual and has_truedl:
                exp_tr = f"{r['expected_score_true_dl']:.1f}" if r.get("expected_score_true_dl") is not None else "-"
                act_s = f"{r['actual_score']}" if r.get("actual_score") is not None else "-"
                diff_s = f"{r['diff_calibrated_dl']:+.1f}" if r.get("diff_calibrated_dl") is not None else "-"
                lines.append(
                    f"{rank:2d}. {t_name:<28} | {t_town:<14} | {exp_s:<12} | {exp_tr:<12} | {act_s:<5} | {diff_s:<7} | {r['prob_1st']:5.1%} | {r['prob_top3']:6.1%} | {r['mean_rank']:5.1f}"
                )
            elif has_actual:
                act_s = f"{r['actual_score']}" if r.get("actual_score") is not None else "-"
                diff_s = f"{r['diff_calibrated_dl']:+.1f}" if r.get("diff_calibrated_dl") is not None else "-"
                lines.append(
                    f"{rank:2d}. {t_name:<28} | {t_town:<14} | {exp_s:<12} | {act_s:<5} | {diff_s:<7} | {r['prob_1st']:5.1%} | {r['prob_top3']:6.1%} | {r['mean_rank']:5.1f}"
                )
            else:
                interval = f"[{int(r['p10'])}-{int(r['p90'])}]"
                solve_p = f"{r['solve_prob']:.1%}"
                lines.append(
                    f"{rank:2d}. {t_name:<28} | {t_town:<14} | {exp_s:<10} | {interval:<14} | {solve_p:<9} | {r['prob_1st']:5.1%} | {r['prob_top3']:6.1%} | {r['mean_rank']:5.1f}"
                )

        lines.append("=" * 105)
        return "\n".join(lines)

    def export_json(self, res: Dict[str, Any], output_path: Union[str, Path]) -> Path:
        """Save prediction results to a JSON file."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        return out_p

    def export_html(self, res: Dict[str, Any], output_path: Union[str, Path]) -> Path:
        """Generate interactive, responsive HTML report for the tournament forecast."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        t_name = res.get("tournament_name", f"Tournament {res['tournament_id']}")
        t_id = res["tournament_id"]
        total_q = res.get("total_questions", 36)
        t_fmt = res.get("format", "offline").upper()
        calibrated_dl = res.get("calibrated_dl", 4.25)
        actual_true_dl = res.get("actual_true_dl")
        has_actual = any(r.get("actual_score") is not None for r in res["standings"])

        true_dl_str = f"{actual_true_dl:.2f}" if actual_true_dl is not None else "Не сыгран"

        rows_html = []
        for rank, r in enumerate(res["standings"], 1):
            rank_cls = f"rank-{rank}" if rank <= 3 else ""
            t_team = r["team_name"]
            t_town = r.get("town", "")
            exp_s = f"{r['expected_score']:.1f}"
            p1st = f"{r['prob_1st']:.1%}"
            ptop3 = f"{r['prob_top3']:.1%}"
            mean_r = f"{r['mean_rank']:.1f}"
            interval = f"{int(r['p10'])} &ndash; {int(r['p90'])}"

            if has_actual:
                act_s = f"{r['actual_score']}" if r.get("actual_score") is not None else "&mdash;"
                diff = r.get("diff_calibrated_dl")
                diff_str = f"{diff:+.1f}" if diff is not None else "&mdash;"
                diff_cls = "diff-pos" if diff is not None and diff > 0 else "diff-neg"
                extra_cells = f"""
                    <td class="num">{act_s}</td>
                    <td class="num {diff_cls}">{diff_str}</td>
                """
            else:
                extra_cells = f"""
                    <td class="num">{interval}</td>
                """

            rows_html.append(f"""
                <tr>
                    <td class="rank {rank_cls}">{rank}</td>
                    <td class="team-name">{t_team}</td>
                    <td class="town">{t_town}</td>
                    <td class="score">{exp_s}</td>
                    {extra_cells}
                    <td class="num">{p1st}</td>
                    <td class="num">{ptop3}</td>
                    <td class="num">{mean_r}</td>
                </tr>
            """)

        extra_header = """
            <th class="num">Факт</th>
            <th class="num">Дельта</th>
        """ if has_actual else """
            <th class="num">90% интервал</th>
        """

        html_code = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Прогноз ЧГК: {t_name}</title>
    <style>
        :root {{
            --primary: #2563eb;
            --primary-light: #eff6ff;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --border: #e2e8f0;
            --row-alt: #f1f5f9;
            --gold: #f59e0b;
            --silver: #94a3b8;
            --bronze: #d97706;
            --pos: #059669;
            --neg: #dc2626;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            margin: 0;
            padding: 30px 16px;
            display: flex;
            justify-content: center;
        }}
        .container {{
            max-width: 1040px;
            width: 100%;
            background: var(--card-bg);
            border-radius: 14px;
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.06);
            padding: 32px;
            border: 1px solid var(--border);
        }}
        header {{
            margin-bottom: 24px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 18px;
        }}
        h1 {{
            font-size: 24px;
            margin: 0 0 8px 0;
            color: var(--text-main);
        }}
        .badges {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 12px;
        }}
        .badge {{
            display: inline-block;
            background: var(--row-alt);
            color: var(--text-muted);
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 500;
        }}
        .badge-primary {{
            background: var(--primary-light);
            color: var(--primary);
            font-weight: 600;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 14px;
            margin-top: 16px;
        }}
        thead th {{
            background-color: var(--row-alt);
            padding: 10px 14px;
            font-weight: 600;
            color: var(--text-muted);
            border-bottom: 2px solid var(--border);
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.05em;
        }}
        tbody tr {{
            border-bottom: 1px solid var(--border);
            transition: background-color 0.15s ease;
        }}
        tbody tr:hover {{
            background-color: #f8fafc;
        }}
        tbody td {{
            padding: 12px 14px;
        }}
        .rank {{
            font-weight: bold;
            width: 32px;
            text-align: center;
        }}
        .rank-1 {{ color: var(--gold); }}
        .rank-2 {{ color: var(--silver); }}
        .rank-3 {{ color: var(--bronze); }}
        .team-name {{
            font-weight: 600;
            color: var(--text-main);
        }}
        .town {{
            color: var(--text-muted);
            font-size: 13px;
        }}
        .score {{
            font-weight: 700;
            font-variant-numeric: tabular-nums;
            color: var(--primary);
            text-align: right;
        }}
        .num {{
            font-variant-numeric: tabular-nums;
            text-align: right;
        }}
        .diff-pos {{ color: var(--pos); }}
        .diff-neg {{ color: var(--neg); }}
        footer {{
            margin-top: 24px;
            font-size: 12px;
            color: var(--text-muted);
            display: flex;
            justify-content: space-between;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{t_name}</h1>
            <div class="badges">
                <span class="badge badge-primary">ID: {t_id}</span>
                <span class="badge">Формат: {t_fmt}</span>
                <span class="badge">Дистанция: {total_q} вопросов</span>
                <span class="badge">Калибр. сложность: {calibrated_dl:.2f}</span>
                <span class="badge">Фактическая сложность: {true_dl_str}</span>
            </div>
        </header>
        <table>
            <thead>
                <tr>
                    <th class="rank">№</th>
                    <th>Команда</th>
                    <th>Город</th>
                    <th class="num">Прогноз</th>
                    {extra_header}
                    <th class="num">P(1-е)</th>
                    <th class="num">P(Топ-3)</th>
                    <th class="num">Ср. место</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows_html)}
            </tbody>
        </table>
        <footer>
            <span>Нейросетевая модель Dual Latent Smooth-Max & Stage 1 Calibration</span>
            <span>Монте-Карло симуляция (10,000 прогонов)</span>
        </footer>
    </div>
</body>
</html>
"""
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(html_code)
        return out_p
