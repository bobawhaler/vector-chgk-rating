"""Re-export of the App Engine copy so training and prediction share one parser."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "gae" / "tour_editors.py"
_SPEC = importlib.util.spec_from_file_location("gae_tour_editors", _PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load {_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

credit_record = _MODULE.credit_record
editors_for_question = _MODULE.editors_for_question
editors_for_span = _MODULE.editors_for_span
player_ids = _MODULE.player_ids
