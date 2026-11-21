"""Loading, cleaning, labelling and feature engineering for the MetroPT-3 air-compressor data.

Raw data: 1.5 million sensor readings (one every ~10 s) from the Air Production Unit (APU) of a
Metro do Porto train, Feb-Sep 2020. The file has no failure column: labels are built here from the
company's failure reports (4 air leaks)."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV = os.path.join(ROOT, "data", "MetroPT3(AirCompressor).csv")

ANALOG = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]
DIGITAL = ["COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level", "Caudal_impulses"]

# Failure reports from the dataset documentation (Data Description_Metro.pdf)
# (start, end, type, repair time). The report column in the PDF is shifted by one row: "maintenance on
# 30 Apr" cannot belong to the 29 May failure, so it is read as the repair of failure 1.
FAILURES = [
    ("2020-04-18 00:00", "2020-04-18 23:59", "Air leak", "2020-04-30 12:00"),
    ("2020-05-29 23:30", "2020-05-30 06:00", "Air leak", None),
    ("2020-06-05 10:00", "2020-06-07 14:30", "Air leak", "2020-06-08 16:00"),
    ("2020-07-15 14:30", "2020-07-15 19:00", "Air leak", "2020-07-16 00:00"),
]
WARNING_H = 2          # minutes in the 2 h before a reported failure are labelled positive (García-Méndez et al., 2025)
WINDOW = "30min"       # look-back window for the rolling features
SHORT_GAP_MIN = 5      # gaps up to 5 min are interpolated, longer gaps are left empty

# Raw minute-level sensor columns used by the baseline (LPS is excluded everywhere: it is the train's own
# low-pressure alarm, so using it as an input would be cheating; we compare against it instead)
RAW = ["TP2", "TP3", "H1", "DV_pressure", "Oil_temperature", "Motor_current", "COMP", "DV_eletric",
       "Towers", "MPG", "Pressure_switch", "Oil_level", "Caudal_impulses"]
# Knowledge-guided feature set used by the main models. Domain rule: a leak ANYWHERE in the pneumatic
# system makes the compressor run loaded for longer, draw more current and heat up. Signals that depend on
# WHERE the leak is (dryer: DV_pressure, H1, Towers, Pressure_switch) are left out because they would only
# recognise leaks at the dryer. Oil_level is left out as unreliable (active 90 % of the time, although the
# documentation says it means "oil low").
KNOW = ["TP2", "TP3", "Oil_temperature", "Motor_current", "COMP", "DV_eletric", "MPG", "Caudal_impulses",
        "loaded_frac_30", "loaded_frac_5", "starts_30", "oil_temp_max_30", "oil_temp_rise_30",
        "current_mean_30", "current_std_30", "tp3_min_30", "tp3_mean_30", "tp3_std_30", "tp2_mean_30"]
ENG = RAW + [
    "loaded_frac_30", "loaded_frac_5", "starts_30", "oil_temp_max_30", "oil_temp_rise_30",
    "current_mean_30", "current_std_30", "tp3_min_30", "tp3_mean_30", "tp3_std_30", "tp2_mean_30",
    "h1_mean_30", "dv_pressure_mean_30", "towers_switches_30",
]


def load_raw(path: str = RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def quality_report(df: pd.DataFrame) -> dict:
    """What is wrong with the raw file (used in the report and the dashboard's Data tab)."""
    ts = df["timestamp"]
    step = ts.diff().dt.total_seconds()
    gaps = step[step > 60]
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "start": str(ts.min()), "end": str(ts.max()),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_timestamps": int(ts.duplicated().sum()),
        "median_step_s": float(step.median()),
        "step_range_s": [float(step.min()), float(step.max())],
        "irregular_steps": int((step != 10).sum() - 1),
        "gaps_over_1min": int(len(gaps)),
        "gaps_over_1h": int((step > 3600).sum()),
        "longest_gap_h": round(float(step.max()) / 3600, 1),
        "missing_hours": round(float(gaps.sum()) / 3600, 1),
        "negative_pressure_readings": int((df[["TP2", "TP3", "H1", "DV_pressure", "Reservoirs"]] < 0).sum().sum()),
        "tp3_reservoirs_corr": round(float(df["TP3"].corr(df["Reservoirs"])), 4),
        "oil_level_active_pct": round(float(df["Oil_level"].mean()) * 100, 1),
        "lps_active_rows": int(df["LPS"].sum()),
        "documented_rate": "1 Hz (15,169,480 points)",
        "actual_rate": f"about 1 reading per {step.median():.0f} s ({len(df):,} rows)",
    }


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, de-duplicate, clip sensor offsets and drop the redundant Reservoirs column."""
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp")
    for c in ("TP2", "TP3", "H1", "DV_pressure", "Reservoirs"):
        df[c] = df[c].clip(lower=0)             # small negative values are sensor offset (e.g. -0.012 bar)
    return df.drop(columns=["Reservoirs"])      # correlation with TP3 is ~1.0: same information twice


FROZEN_MIN = 30        # oil temperature AND panel pressure unchanged for 30+ min = logger stuck, not real data


def frozen_mask(m: pd.DataFrame) -> pd.Series:
    """True for minutes inside a run where both analogue signals are perfectly constant (stuck logger)."""
    same = (m["Oil_temperature"].diff() == 0) & (m["TP3"].diff() == 0)
    run_id = (~same).cumsum()
    run_len = same.groupby(run_id).transform("sum")
    return same & (run_len >= FROZEN_MIN)


def to_minutes(df: pd.DataFrame):
    """Irregular ~10 s readings -> regular 1-minute rows. Digital signals become 'fraction of the minute
    the signal was active'. Short gaps are interpolated; longer gaps and frozen-sensor periods are
    dropped, never invented. Returns (minutes, number of frozen minutes removed)."""
    m = df.set_index("timestamp").resample("1min").mean()
    m = m.interpolate(limit=SHORT_GAP_MIN, limit_area="inside")
    m = m.dropna(how="all", subset=[c for c in m.columns if c != "LPS"])
    fz = frozen_mask(m)
    return m[~fz], int(fz.sum())


def add_features(m: pd.DataFrame) -> pd.DataFrame:
    """Knowledge-guided rolling features over the previous 30 minutes (past data only, so the model can
    run live). Time-based windows never bridge a long gap because the gap rows do not exist."""
    m = m.copy()
    r30, r5 = m.rolling(WINDOW, min_periods=5), m.rolling("5min", min_periods=2)
    loaded = (m["DV_eletric"] > 0.5).astype(float)
    m["loaded_frac_30"] = r30["DV_eletric"].mean()                     # compressor duty cycle
    m["loaded_frac_5"] = r5["DV_eletric"].mean()
    m["starts_30"] = (loaded.diff() > 0).astype(float).rolling(WINDOW, min_periods=1).sum()
    m["oil_temp_max_30"] = r30["Oil_temperature"].max()
    m["oil_temp_rise_30"] = m["Oil_temperature"] - r30["Oil_temperature"].min()
    m["current_mean_30"] = r30["Motor_current"].mean()
    m["current_std_30"] = r30["Motor_current"].std()
    m["tp3_min_30"] = r30["TP3"].min()
    m["tp3_mean_30"] = r30["TP3"].mean()
    m["tp3_std_30"] = r30["TP3"].std()
    m["tp2_mean_30"] = r30["TP2"].mean()
    m["h1_mean_30"] = r30["H1"].mean()
    m["dv_pressure_mean_30"] = r30["DV_pressure"].mean()
    m["towers_switches_30"] = (m["Towers"].round().diff().abs() > 0).astype(float).rolling(WINDOW, min_periods=1).sum()
    return m.dropna(subset=ENG)


def add_labels(m: pd.DataFrame, warning_h: int = WARNING_H) -> pd.DataFrame:
    """failure = 1 from 2 h before each reported failure until its end; event = which failure (0 = none);
    unrepaired = 1 between the end of a failure and its repair (leak still there, label unknown)."""
    m = m.copy()
    m["failure"] = 0
    m["event"] = 0
    m["unrepaired"] = 0
    for i, (s, e, _, fix) in enumerate(FAILURES, 1):
        sel = (m.index >= pd.Timestamp(s) - pd.Timedelta(hours=warning_h)) & (m.index <= pd.Timestamp(e))
        m.loc[sel, "failure"] = 1
        m.loc[sel, "event"] = i
        if fix:
            m.loc[(m.index > pd.Timestamp(e)) & (m.index <= pd.Timestamp(fix)), "unrepaired"] = 1
    return m


def build_dataset(path: str = RAW_CSV, cache: bool = True):
    """Raw CSV -> cleaned, labelled 1-minute table with features. Returns (table, quality report)."""
    cache_path = os.path.join(ROOT, "data", "metro_minutes.pkl")
    if cache and os.path.exists(cache_path):
        return pd.read_pickle(cache_path)
    raw = load_raw(path)
    q = quality_report(raw)
    minutes, n_frozen = to_minutes(clean(raw))
    m = add_labels(add_features(minutes))
    q["frozen_minutes_removed"] = n_frozen
    q["minute_rows"] = int(len(m))
    q["positive_minutes"] = int(m["failure"].sum())
    if cache:
        pd.to_pickle((m, q), cache_path)
    return m, q


# ---- time-based split: train/tune on failures 1-3 (Feb-Jun), test on unseen July-August (failure 4) ----
TEST_START = pd.Timestamp("2020-07-01")
# leave-one-failure-out folds inside the training period: each block holds exactly one failure
CV_BLOCKS = [("2020-02-01", "2020-05-08"), ("2020-05-08", "2020-06-02"), ("2020-06-02", "2020-07-01")]


def time_folds(index: pd.DatetimeIndex):
    """(train_idx, val_idx) pairs: validate on one block, train on the other two."""
    folds = []
    for s, e in CV_BLOCKS:
        val = np.where((index >= pd.Timestamp(s)) & (index < pd.Timestamp(e)))[0]
        tr = np.where(~((index >= pd.Timestamp(s)) & (index < pd.Timestamp(e))))[0]
        folds.append((tr, val))
    return folds


def make_preprocessor(features, scale: bool = True):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return ColumnTransformer([("num", Pipeline(steps), list(features))])
