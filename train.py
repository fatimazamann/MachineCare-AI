"""MachineCare AI - train and evaluate every model on the MetroPT-3 train air-compressor data.

Run:  python train.py        (about 10-15 minutes on a laptop)
Writes results/results.json, results/summary.md, figures/*.png and models/leak_model.joblib."""

from __future__ import annotations

import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier, VotingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from ml import plots
from ml.kbs_rules import leak_cf
from ml.preprocess import ENG, FAILURES, KNOW, RAW, TEST_START, build_dataset, make_preprocessor, time_folds
from ml.portable import export_pipeline
from ml.smote import SMOTEClassifier

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
for d in ("results", "figures", "models"):
    os.makedirs(os.path.join(ROOT, d), exist_ok=True)
SEED = 42
SMOOTH = "15min"     # alarm filter: average of the last 15 minutes of probabilities
THRESHOLD = 0.5
# Leave-one-failure-out blocks over the whole period (each block holds exactly one reported failure)
LOFO_BLOCKS = [("2020-02-01", "2020-05-08"), ("2020-05-08", "2020-06-02"), ("2020-06-02", "2020-07-01"), ("2020-07-01", "2020-09-02")]


def pipe(model, feats, scale=True):
    return Pipeline([("prep", make_preprocessor(feats, scale)), ("model", model)])


def smooth(prob: pd.Series) -> pd.Series:
    return prob.rolling(SMOOTH, min_periods=1).mean()


def metrics(y, prob: pd.Series) -> dict:
    pred = (prob >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": round(accuracy_score(y, pred), 4), "balanced_accuracy": round(balanced_accuracy_score(y, pred), 4),
            "precision": round(precision_score(y, pred, zero_division=0), 4), "recall": round(recall_score(y, pred), 4),
            "f1": round(f1_score(y, pred), 4), "roc_auc": round(roc_auc_score(y, prob), 4),
            "pr_auc": round(average_precision_score(y, prob), 4), "confusion": [int(tn), int(fp), int(fn), int(tp)]}


def episodes(alarm: pd.Series, merge_min=60, min_len_min=15):
    """Turn minute alarms into alarm episodes (gaps < 60 min merged, episodes < 15 min ignored)."""
    t = alarm.index[alarm.values]
    if len(t) == 0:
        return []
    br = np.where(np.diff(t.values).astype("timedelta64[m]").astype(int) > merge_min)[0]
    starts = [t[0]] + [t[i + 1] for i in br]
    ends = [t[i] for i in br] + [t[-1]]
    return [(s, e) for s, e in zip(starts, ends) if e - s >= pd.Timedelta(minutes=min_len_min)]


def event_metrics(prob: pd.Series, unrepaired: pd.Series) -> dict:
    """Did an alarm cover each reported failure (from 24 h before its start to its end)? How early?
    How many alarm episodes happened outside any failure window (false alarms)?"""
    eps = episodes(prob >= THRESHOLD)
    out, windows = [], []
    for i, (s, e, _, _) in enumerate(FAILURES, 1):
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        if s < prob.index.min() or s > prob.index.max():
            continue
        ws = s - pd.Timedelta(hours=24)
        windows.append((ws, e))
        hit = [a for a, b in eps if a <= e and b >= ws]
        first = min(hit) if hit else None
        out.append({"failure": i, "start": str(s), "detected": bool(hit),
                    "lead_min": None if first is None else int((s - first).total_seconds() // 60)})
    fa = [(a, b) for a, b in eps if not any(a <= we and b >= ws for ws, we in windows) and not bool(unrepaired.get(a, 0))]
    precursors = sum(1 for a, _ in fa if any(pd.Timedelta(0) < pd.Timestamp(s) - a <= pd.Timedelta(days=42) for s, _, _, _ in FAILURES))
    days = (prob.index.max() - prob.index.min()).total_seconds() / 86400
    return {"events": out, "detected": sum(o["detected"] for o in out), "total": len(out),
            "false_alarm_episodes": len(fa), "false_alarms_per_week": round(len(fa) / days * 7, 2),
            "false_alarms_within_6_weeks_before_a_failure": precursors,
            "false_alarm_list": [[str(a), str(b)] for a, b in fa]}


def tune(name, est, grid, X, y, folds):
    t0 = time.perf_counter()
    gs = GridSearchCV(est, grid, scoring="f1", cv=folds, n_jobs=1, refit=True).fit(X, y)
    i = gs.best_index_
    print(f"  {name:<32} CV F1 {gs.best_score_:.3f}  {gs.best_params_}  ({time.perf_counter() - t0:.0f} s)", flush=True)
    clean = {k.replace("model__", "").replace("estimator__", ""): v for k, v in gs.best_params_.items()}
    return gs.best_estimator_, {"cv_score": round(float(gs.best_score_), 4), "cv_std": round(float(gs.cv_results_["std_test_score"][i]), 4),
                                "best_params": clean}


def main():
    t_start = time.perf_counter()
    print("1. Loading and cleaning MetroPT-3 ...", flush=True)
    m, quality = build_dataset()
    m["leak_cf"] = [leak_cf(r) for _, r in m.iterrows()]
    use = m["unrepaired"] == 0          # leak reported but not yet repaired -> true label unknown, left out
    tr = m[(m.index < TEST_START) & use]
    te = m[(m.index >= TEST_START) & use]
    ytr, yte = tr["failure"].values, te["failure"]
    folds = time_folds(tr.index)
    print(f"   {quality['rows']:,} raw rows -> {len(m):,} minute rows; train {len(tr):,} ({int(ytr.sum())} failure minutes), "
          f"test {len(te):,} ({int(yte.sum())} failure minutes)", flush=True)
    plots.eda(m, quality)

    print("2. Training and tuning (leave-one-failure-out CV inside Feb-Jun) ...", flush=True)
    models, info = {}, {}
    base = pipe(LogisticRegression(max_iter=3000), RAW).fit(tr[RAW], ytr)
    models["Baseline: Logistic Regression"] = (base, RAW)
    info["Baseline: Logistic Regression"] = {"cv_score": round(float(np.mean(cross_val_score(clone(base), tr[RAW], ytr, cv=folds, scoring="f1"))), 4),
                                             "best_params": {"features": "13 raw sensors", "class_weight": None}}
    specs = [
        ("Logistic Regression", pipe(LogisticRegression(max_iter=3000), KNOW),
         {"model__C": [0.01, 0.1, 1], "model__class_weight": [None, "balanced"]}),
        ("Decision Tree", pipe(DecisionTreeClassifier(random_state=SEED), KNOW, scale=False),
         {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [20, 200], "model__class_weight": [None, "balanced"]}),
        ("Random Forest", pipe(RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1), KNOW, scale=False),
         {"model__max_depth": [6, 12], "model__min_samples_leaf": [20, 100], "model__class_weight": ["balanced_subsample"]}),
        ("Random Forest + SMOTE", pipe(SMOTEClassifier(RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=20,
                                                                              random_state=SEED, n_jobs=-1)), KNOW),
         {"model__ratio": [0.2, 0.5]}),
        ("Gradient Boosting", pipe(HistGradientBoostingClassifier(random_state=SEED), KNOW, scale=False),
         {"model__learning_rate": [0.05, 0.1], "model__max_depth": [2, 3, None], "model__class_weight": [None, "balanced"]}),
    ]
    for name, est, grid in specs:
        mdl, i = tune(name, est, grid, tr[KNOW], ytr, folds)
        models[name], info[name] = (mdl, KNOW), i
    # soft-voting ensemble of the three strongest model families
    members = [(k.split()[0].lower(), clone(models[k][0])) for k in ("Logistic Regression", "Random Forest", "Gradient Boosting")]
    vote = VotingClassifier(members, voting="soft")
    cv = cross_val_score(vote, tr[KNOW], ytr, cv=folds, scoring="f1")
    models["Ensemble (soft voting)"] = (vote.fit(tr[KNOW], ytr), KNOW)
    info["Ensemble (soft voting)"] = {"cv_score": round(float(cv.mean()), 4), "cv_std": round(float(cv.std()), 4),
                                      "best_params": {"members": "Logistic Regression, Random Forest, Gradient Boosting"}}
    print(f"  {'Ensemble (soft voting)':<32} CV F1 {cv.mean():.3f}", flush=True)

    # unsupervised anomaly detector: learns normal behaviour only, needs no failure labels
    iso_prep = make_preprocessor(KNOW).fit(tr.loc[tr["failure"] == 0, KNOW])
    iso = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1).fit(iso_prep.transform(tr.loc[tr["failure"] == 0, KNOW]))
    tr_scores = -iso.score_samples(iso_prep.transform(tr[KNOW]))
    lo, hi = np.percentile(tr_scores[ytr == 0], [50, 99.5])     # 99.5th percentile of normal = alarm level

    def iso_prob(X):
        s = -iso.score_samples(iso_prep.transform(X))
        return np.clip(0.5 * (s - lo) / (hi - lo), 0, 1)        # maps the alarm level to 0.5

    print("3. Testing on unseen July-August data (failure 4) ...", flush=True)
    results, probs = {}, {}
    for name, (mdl, feats) in models.items():
        p = smooth(pd.Series(mdl.predict_proba(te[feats])[:, 1], index=te.index))
        probs[name] = p
        results[name] = dict(metrics(yte, p), **info[name], **{"event": event_metrics(p, m["unrepaired"])})
    p = smooth(pd.Series(iso_prob(te[KNOW]), index=te.index))
    probs["Isolation Forest (unsupervised)"] = p
    results["Isolation Forest (unsupervised)"] = dict(metrics(yte, p), cv_score=None, best_params={"trees": 200, "alarm": "99.5th pct of normal"},
                                                      event=event_metrics(p, m["unrepaired"]))
    best_name = max((n for n in models if not n.startswith("Baseline")), key=lambda n: info[n]["cv_score"])
    best, best_feats = models[best_name]
    rule_p = smooth(te["leak_cf"].clip(lower=0))
    probs["Expert system (rules)"] = rule_p
    results["Expert system (rules)"] = dict(metrics(yte, rule_p), cv_score=None, best_params={}, event=event_metrics(rule_p, m["unrepaired"]))
    hyb = (probs[best_name] + rule_p) / 2
    probs["Hybrid (ML + rules)"] = hyb
    results["Hybrid (ML + rules)"] = dict(metrics(yte, hyb), cv_score=None, best_params={"model": best_name, "fusion": "average of ML probability and rule CF"},
                                          event=event_metrics(hyb, m["unrepaired"]))
    print(f"\n  best model by CV F1: {best_name}")
    for n, r in results.items():
        print(f"  {n:<34} acc {r['accuracy']:.3f}  bal {r['balanced_accuracy']:.3f}  P {r['precision']:.3f}  R {r['recall']:.3f}  "
              f"F1 {r['f1']:.3f}  AUC {r['roc_auc']:.3f}  events {r['event']['detected']}/{r['event']['total']}  FA {r['event']['false_alarm_episodes']}", flush=True)

    print("4. Leave-one-failure-out over all four failures ...", flush=True)
    lofo = {"ml": [], "rules": [], "hybrid": []}
    lofo_prob = {k: pd.Series(np.nan, index=m.index) for k in lofo}
    allm = m[use]
    for s, e in LOFO_BLOCKS:
        sel = (allm.index >= s) & (allm.index < e)
        mdl = clone(best).fit(allm.loc[~sel, best_feats], allm.loc[~sel, "failure"])
        blk = allm[sel]
        pm = smooth(pd.Series(mdl.predict_proba(blk[best_feats])[:, 1], index=blk.index))
        pr = smooth(blk["leak_cf"].clip(lower=0))
        for k, p in (("ml", pm), ("rules", pr), ("hybrid", (pm + pr) / 2)):
            lofo_prob[k].loc[blk.index] = p
            lofo[k].append({"block": f"{s} to {e}", **metrics(blk["failure"], p)})
    lofo_summary = {}
    for k in lofo:
        p = lofo_prob[k].loc[allm.index]
        ev = event_metrics(p, m["unrepaired"])
        lofo_summary[k] = {"per_failure_f1": [f["f1"] for f in lofo[k]], "overall": metrics(allm["failure"], p), "event": ev}
        print(f"  {k:<7} F1 per failure {lofo_summary[k]['per_failure_f1']}  overall F1 {lofo_summary[k]['overall']['f1']:.3f}  "
              f"events {ev['detected']}/4  lead {[o['lead_min'] for o in ev['events']]}  false-alarm episodes {ev['false_alarm_episodes']} "
              f"({ev['false_alarms_within_6_weeks_before_a_failure']} within 6 weeks before a failure)", flush=True)

    print("5. Ablation, leakage check, robustness, importance ...", flush=True)
    ablation = {}
    for fname, feats in (("Raw sensors (13)", RAW), ("Knowledge-guided (19)", KNOW), ("All features (27)", ENG)):
        row = {}
        for mname, est in (("Logistic Regression", models["Logistic Regression"][0]), ("Gradient Boosting", models["Gradient Boosting"][0])):
            est = clone(est)
            est.set_params(prep=make_preprocessor(feats, scale=mname.startswith("Logistic")))
            f1s = []
            for s, e in LOFO_BLOCKS:
                sel = (allm.index >= s) & (allm.index < e)
                fit = est.fit(allm.loc[~sel, feats], allm.loc[~sel, "failure"])
                p = smooth(pd.Series(fit.predict_proba(allm.loc[sel, feats])[:, 1], index=allm.index[sel]))
                f1s.append(round(f1_score(allm.loc[sel, "failure"], (p >= THRESHOLD).astype(int)), 3))
            row[mname] = {"per_failure_f1": f1s, "mean_f1": round(float(np.mean(f1s)), 3)}
        ablation[fname] = row
        print(f"  {fname:<22} " + "  ".join(f"{k}: {v['per_failure_f1']} mean {v['mean_f1']}" for k, v in row.items()), flush=True)

    # what a random (non-time) split would wrongly report: same untuned Random Forest, two ways of splitting
    rf = lambda: RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    a_tr, a_te = train_test_split(allm, test_size=0.2, random_state=SEED, stratify=allm["failure"])
    leakage = {"random_split_f1": round(f1_score(a_te["failure"], rf().fit(a_tr[ENG], a_tr["failure"]).predict(a_te[ENG])), 4),
               "time_split_f1": round(f1_score(yte, rf().fit(tr[ENG], ytr).predict(te[ENG])), 4)}
    print(f"  untuned Random Forest, all features: random 80/20 split F1 {leakage['random_split_f1']} vs time split F1 {leakage['time_split_f1']}", flush=True)

    rng = np.random.default_rng(SEED)

    def corrupt(X, missing, noise):
        X = X.copy()
        if noise:
            X = X + rng.normal(0, 1, X.shape) * (noise * (X.max() - X.min()).values)
        if missing:
            X = X.mask(rng.random(X.shape) < missing)
        return X

    robust = {}
    for cond, miss, noise in (("Clean test data", 0, 0), ("5 % missing values", 0.05, 0), ("Sensor noise (2 % of range)", 0, 0.02),
                              ("10 % missing + 2 % noise", 0.10, 0.02)):
        r = {}
        for n, mdl, feats in (("Baseline", models["Baseline: Logistic Regression"][0], RAW), ("Best model", best, best_feats)):
            p = smooth(pd.Series(mdl.predict_proba(corrupt(te[feats], miss, noise))[:, 1], index=te.index))
            r[n] = round(f1_score(yte, (p >= THRESHOLD).astype(int)), 4)
        robust[cond] = r
        print(f"  {cond:<30} {r}", flush=True)

    imp = permutation_importance(best, te[best_feats], yte, scoring="average_precision", n_repeats=3, random_state=SEED, n_jobs=1)
    importance = sorted(({"feature": f, "mean": round(float(mu), 4), "std": round(float(sd), 4)}
                         for f, mu, sd in zip(best_feats, imp.importances_mean, imp.importances_std)), key=lambda d: -d["mean"])

    joblib.dump({"model": best, "features": best_feats, "name": best_name, "threshold": THRESHOLD, "smooth": SMOOTH},
                os.path.join(ROOT, "models", "leak_model.joblib"))
    # data for the dashboard: (1) a replay of the leave-one-failure-out warnings at 5-minute resolution,
    # (2) a sample of real minutes so a what-if input can be matched to the most similar real minute
    replay = allm[["loaded_frac_30", "Oil_temperature", "current_mean_30", "tp3_min_30", "failure"]].copy()
    for k in ("ml", "rules", "hybrid"):
        replay[k] = lofo_prob[k].loc[allm.index]
    replay.resample("5min").mean().dropna().round(3).to_csv(os.path.join(ROOT, "results", "replay.csv.gz"))
    keep = (m["failure"] == 1) | (m["loaded_frac_30"] >= 0.5) | (pd.Series(rng.random(len(m)), index=m.index) < 0.12)
    lookup = m.loc[keep, KNOW + ["dv_pressure_mean_30", "LPS"]].round(4)
    lookup.to_csv(os.path.join(ROOT, "models", "lookup.csv.gz"))            # plain CSV: loads with any pandas version
    export_pipeline(best, best_feats, best_name, os.path.join(ROOT, "models", "leak_model.json"), THRESHOLD, SMOOTH)

    out = {"data": quality, "split": {"train": f"2020-02-01 to {TEST_START.date()} (failures 1-3)", "test": f"{TEST_START.date()} to 2020-09-01 (failure 4)",
                                      "train_rows": int(len(tr)), "train_failure_minutes": int(ytr.sum()),
                                      "test_rows": int(len(te)), "test_failure_minutes": int(yte.sum()),
                                      "excluded_unrepaired_minutes": int((~use).sum())},
           "best_model": best_name, "results": results, "lofo": lofo_summary, "lofo_detail": lofo, "ablation": ablation,
           "leakage_check": leakage, "robustness": robust, "importance": importance,
           "runtime_s": round(time.perf_counter() - t_start)}
    with open(os.path.join(ROOT, "results", "results.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)

    # figures
    plots.model_comparison(results)
    plots.confusion(results[best_name]["confusion"], best_name)
    curves = {n: (roc_curve(yte, probs[n]), precision_recall_curve(yte, probs[n]), results[n]["roc_auc"], results[n]["pr_auc"])
              for n in ("Baseline: Logistic Regression", best_name, "Expert system (rules)", "Hybrid (ML + rules)")}
    plots.roc_pr(curves)
    plots.test_timeline(te, probs[best_name], rule_p, best_name)
    plots.lofo_timeline(allm, lofo_prob["hybrid"])
    plots.importance(importance)
    plots.robustness(robust)
    plots.ablation(ablation)
    plots.summary_md(out)
    print(f"\nDone in {out['runtime_s']} s. See results/summary.md and figures/.")


if __name__ == "__main__":
    main()
