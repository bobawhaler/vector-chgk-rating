"""Editor calibration and difficulty modeling for ChGK tournaments."""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DB = Path("data/chgk_cache.db")


@dataclass
class EditorProfile:
    """Statistical profile of a tournament editor's difficulty calibration."""

    editor_id: int
    name: str
    tournaments_count: int
    avg_forecast_dl: float
    avg_true_dl: float
    std_true_dl: float  # Historical standard deviation of package difficulty
    raw_bias: float  # avg(TrueDL - ForecastDL)
    calibrated_bias: float  # Bayesian smoothed bias beta_e
    rmse: float
    mae: float
    accuracy_half_dl: float  # Fraction within |Delta| <= 0.5
    tendency: str  # 'harder' (groby), 'easier' (svoyak/simple), or 'calibrated'


class EditorCalibrationRegistry:
    """Manages historical tournament editor metadata and calibrations."""

    def __init__(
        self,
        db_path: Path = DEFAULT_CACHE_DB,
        prior_n0: float = 3.0,
        prior_mu0: float = -0.02,
    ) -> None:
        self.db_path = Path(db_path)
        self.prior_n0 = prior_n0
        self.prior_mu0 = prior_mu0
        self._profiles: Dict[int, EditorProfile] = {}
        self._tournament_metadata: Dict[int, Dict[str, Any]] = {}
        self._is_loaded = False

    def load_data(self) -> None:
        """Load tournament metadata and calculate editor profiles."""
        if not self.db_path.exists():
            logger.warning("Cache DB %s not found. Editor profiles empty.", self.db_path)
            self._is_loaded = True
            return

        con = sqlite3.connect(self.db_path)
        cur = con.cursor()

        # Gather per-tournament records
        editor_deltas: Dict[int, List[float]] = {}
        editor_forecasts: Dict[int, List[float]] = {}
        editor_trues: Dict[int, List[float]] = {}
        editor_names: Dict[int, str] = {}

        for row in cur.execute("SELECT raw_data FROM ingested_tournaments"):
            try:
                t_data = json.loads(row[0])
            except Exception:
                continue

            t_id = t_data.get("id")
            fc_raw = t_data.get("difficultyForecast")
            true_raw = t_data.get("trueDL")
            editors = t_data.get("editors") or []

            fc = float(fc_raw) if fc_raw is not None and str(fc_raw).strip() != "" else None
            tdl = float(true_raw) if true_raw is not None and str(true_raw).strip() != "" else None

            self._tournament_metadata[int(t_id)] = {
                "id": t_id,
                "name": t_data.get("name", ""),
                "date_start": t_data.get("dateStart"),
                "forecast_dl": fc,
                "true_dl": tdl,
                "editor_ids": [int(e["id"]) for e in editors if isinstance(e, dict) and "id" in e],
            }

            if fc is not None and tdl is not None:
                delta = tdl - fc
                for ed in editors:
                    if not isinstance(ed, dict) or "id" not in ed:
                        continue
                    eid = int(ed["id"])
                    full_name = " ".join(
                        part for part in [ed.get("name"), ed.get("patronymic"), ed.get("surname")] if part
                    ).strip()
                    if full_name:
                        editor_names[eid] = full_name
                    elif eid not in editor_names:
                        editor_names[eid] = f"Editor {eid}"

                    editor_deltas.setdefault(eid, []).append(delta)
                    editor_forecasts.setdefault(eid, []).append(fc)
                    editor_trues.setdefault(eid, []).append(tdl)

        con.close()

        # Compute empirical bayes profiles
        profiles: Dict[int, EditorProfile] = {}
        for eid, deltas in editor_deltas.items():
            arr_deltas = np.array(deltas, dtype=np.float64)
            arr_fc = np.array(editor_forecasts[eid], dtype=np.float64)
            arr_tr = np.array(editor_trues[eid], dtype=np.float64)

            n = len(arr_deltas)
            raw_bias = float(np.mean(arr_deltas))
            # Bayesian shrinkage towards population prior_mu0
            calibrated_bias = float((np.sum(arr_deltas) + self.prior_n0 * self.prior_mu0) / (n + self.prior_n0))

            rmse = float(np.sqrt(np.mean(arr_deltas**2)))
            mae = float(np.mean(np.abs(arr_deltas)))
            acc_half = float(np.mean(np.abs(arr_deltas) <= 0.5))
            std_tr = float(np.std(arr_tr)) if len(arr_tr) > 1 else 0.65

            if calibrated_bias > 0.3:
                tendency = "harder"
            elif calibrated_bias < -0.3:
                tendency = "easier"
            else:
                tendency = "calibrated"

            profiles[eid] = EditorProfile(
                editor_id=eid,
                name=editor_names.get(eid, f"Editor {eid}"),
                tournaments_count=n,
                avg_forecast_dl=float(np.mean(arr_fc)),
                avg_true_dl=float(np.mean(arr_tr)),
                std_true_dl=std_tr,
                raw_bias=raw_bias,
                calibrated_bias=calibrated_bias,
                rmse=rmse,
                mae=mae,
                accuracy_half_dl=acc_half,
                tendency=tendency,
            )

        self._profiles = profiles
        self._is_loaded = True
        logger.info("Loaded calibrations for %d editors across %d tournaments.", len(profiles), len(self._tournament_metadata))

    def ensure_loaded(self) -> None:
        """Lazily load data if not yet loaded."""
        if not self._is_loaded:
            self.load_data()

    def get_editor_profile(self, editor_id: int) -> Optional[EditorProfile]:
        """Return calibration profile for a specific editor."""
        self.ensure_loaded()
        return self._profiles.get(editor_id)

    def get_tournament_metadata(self, tournament_id: int) -> Optional[Dict[str, Any]]:
        """Return tournament info from cache."""
        self.ensure_loaded()
        return self._tournament_metadata.get(tournament_id)

    def predict_tournament_truedl(
        self,
        forecast_dl: Optional[float],
        editor_ids: Optional[List[int]] = None,
        default_dl: float = 4.26,
    ) -> float:
        """Estimate expected TrueDL given target forecast_dl and editor set.

        If forecast_dl is None, estimates tournament difficulty directly from the
        historical TrueDL of the editors (Bayesian smoothed towards the global mean).
        """
        self.ensure_loaded()

        if forecast_dl is not None:
            # Editor calibration relative to declared target DL
            base_dl = float(forecast_dl)
            if not editor_ids:
                return float(base_dl + self.prior_mu0)

            biases = []
            for eid in editor_ids:
                prof = self._profiles.get(eid)
                if prof is not None:
                    biases.append(prof.calibrated_bias)
                else:
                    biases.append(self.prior_mu0)

            mean_bias = float(np.mean(biases)) if biases else self.prior_mu0
            return float(base_dl + mean_bias)
        else:
            # When editors DO NOT specify target difficulty:
            # Estimate directly from the editors' historical TrueDL distribution
            if not editor_ids:
                return default_dl

            ed_trues = []
            for eid in editor_ids:
                prof = self._profiles.get(eid)
                if prof is not None and prof.avg_true_dl > 0:
                    # Bayesian shrinkage towards global mean
                    shrunk_true = (prof.avg_true_dl * prof.tournaments_count + 3.0 * default_dl) / (prof.tournaments_count + 3.0)
                    ed_trues.append(shrunk_true)
                else:
                    ed_trues.append(default_dl)

            return float(np.mean(ed_trues)) if ed_trues else default_dl

    def predict_tournament_dl_distribution(
        self,
        forecast_dl: Optional[float],
        editor_ids: Optional[List[int]] = None,
        default_dl: float = 4.26,
        default_std: float = 0.65,
    ) -> Tuple[float, float]:
        """Estimate (mean, std) of TrueDL distribution for a tournament package.

        Returns:
            Tuple of (expected_mean_dl, expected_std_dl).
        """
        self.ensure_loaded()
        mean_dl = self.predict_tournament_truedl(forecast_dl, editor_ids, default_dl=default_dl)

        if not editor_ids:
            return mean_dl, default_std

        stds = []
        for eid in editor_ids:
            prof = self._profiles.get(eid)
            if prof is not None and prof.std_true_dl > 0:
                stds.append(prof.std_true_dl)
            else:
                stds.append(default_std)

        mean_std = float(np.mean(stds)) if stds else default_std
        return mean_dl, mean_std

    def get_top_editors(
        self,
        min_tournaments: int = 5,
        sort_by: str = "accuracy",
        ascending: bool = False,
        limit: int = 50,
    ) -> List[EditorProfile]:
        """Retrieve editor rankings filtered by tournament activity."""
        self.ensure_loaded()
        filtered = [p for p in self._profiles.values() if p.tournaments_count >= min_tournaments]

        if sort_by == "accuracy":
            filtered.sort(key=lambda p: p.accuracy_half_dl, reverse=not ascending)
        elif sort_by == "rmse":
            filtered.sort(key=lambda p: p.rmse, reverse=ascending)
        elif sort_by == "count":
            filtered.sort(key=lambda p: p.tournaments_count, reverse=not ascending)
        elif sort_by == "bias":
            filtered.sort(key=lambda p: p.calibrated_bias, reverse=not ascending)
        else:
            filtered.sort(key=lambda p: p.accuracy_half_dl, reverse=True)

        return filtered[:limit]


# Global singleton registry
_global_registry: Optional[EditorCalibrationRegistry] = None


def get_editor_registry(db_path: Path = DEFAULT_CACHE_DB) -> EditorCalibrationRegistry:
    """Access global singleton EditorCalibrationRegistry."""
    global _global_registry
    if _global_registry is None or _global_registry.db_path != db_path:
        _global_registry = EditorCalibrationRegistry(db_path=db_path)
    return _global_registry
