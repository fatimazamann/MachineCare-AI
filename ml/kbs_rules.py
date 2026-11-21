"""Bridge to the rule-based expert system (IEFD-KBS, extended with train air-compressor knowledge):
turn one minute of MetroPT-3 features into compressor measurements, run forward chaining and return
the certainty factor (CF) of an air leak, plus the full explanation for the dashboard."""

from __future__ import annotations

import os
import sys
from functools import lru_cache

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kbs import DiagnosisSystem  # noqa: E402
from kbs import certainty as cfm  # noqa: E402
from kbs.abstraction import abstract_measurements  # noqa: E402
from kbs.engine import InferenceEngine  # noqa: E402
from kbs.working_memory import Contribution, WorkingMemory  # noqa: E402

ASSET = "APU-01"
_SYS = None


def system():
    global _SYS
    if _SYS is None:
        _SYS = DiagnosisSystem()
    return _SYS


def measurements(row, include_lps: bool = False) -> dict:
    """Feature row -> the compressor parameters the knowledge base understands."""
    m = {
        "duty_pct_30": round(float(row["loaded_frac_30"]) * 100),
        "current_mean_a_30": round(float(row["current_mean_30"]), 1),
        "oil_temp_c_max_30": round(float(row["oil_temp_max_30"]), 1),
        "panel_pressure_min_bar_30": round(float(row["tp3_min_30"]), 2),
        "dryer_dv_pressure_bar_30": round(float(row["dv_pressure_mean_30"]), 2),
    }
    if include_lps and "LPS" in row:
        m["lps_active"] = "yes" if float(row["LPS"]) > 0 else "no"
    return m


@lru_cache(maxsize=200_000)
def _cfs(duty, cur, oil, tp3, dv) -> tuple:
    kb = system().kb
    frame = kb.asset_frame("GEN-APU")
    wm = WorkingMemory()
    rel = kb.policies["source_reliability"]["sensor"]
    meas = {"duty_pct_30": duty, "current_mean_a_30": cur, "oil_temp_c_max_30": oil,
            "panel_pressure_min_bar_30": tp3, "dryer_dv_pressure_bar_30": dv}
    for i, ab in enumerate(abstract_measurements(kb, frame, meas)):
        wm.put(ab.fact, Contribution(f"sensor:{ab.fact}:{i}", "sensor", cfm.clamp(ab.band_cf * rel), ab.detail))
    InferenceEngine(kb, wm, frame["class"]).forward_chain()
    return wm.cf("fault:air_leak"), wm.cf("fault:compressor_overheating")


def leak_cf(row) -> float:
    """CF of an air leak from forward chaining (LPS alarm is NOT used, same inputs as the ML)."""
    m = measurements(row)
    # round further for the cache: 2 % duty, 0.2 A, 0.5 °C, 0.05 bar steps do not change any rule
    return _cfs(round(m["duty_pct_30"] / 2) * 2, round(m["current_mean_a_30"] * 5) / 5, round(m["oil_temp_c_max_30"] * 2) / 2,
                round(m["panel_pressure_min_bar_30"] * 20) / 20, round(m["dryer_dv_pressure_bar_30"] * 20) / 20)[0]


def explain(meas: dict, policy: str = "balanced") -> dict:
    """Full expert-system result (faults, causes, HOW text, maintenance plan) for one set of readings."""
    return system().diagnose(ASSET, {}, meas, policy=policy)
