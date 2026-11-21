"""Charts and the markdown summary for MachineCare AI (MetroPT-3)."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ml.preprocess import FAILURES, TEST_START  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, AQUA, GREY, RED = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2", "#c43c3c"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 10.5,
    "axes.titlecolor": INK, "legend.frameon": False,
})
SHORT = {"Baseline: Logistic Regression": "Baseline (LR, raw)", "Logistic Regression": "Log. Regression",
         "Random Forest + SMOTE": "RF + SMOTE", "Gradient Boosting": "Grad. Boosting", "Ensemble (soft voting)": "Ensemble",
         "Isolation Forest (unsupervised)": "Isolation Forest", "Expert system (rules)": "Expert system",
         "Hybrid (ML + rules)": "Hybrid ML+rules"}


def short(n):
    return SHORT.get(n, n)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=180, bbox_inches="tight")
    plt.close(fig)


def shade_failures(ax, alpha=0.18):
    for s, e, _, _ in FAILURES:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color=RED, alpha=alpha, lw=0)


def eda(m, q):
    # (1) six months at a glance: duty cycle and oil temperature, failures shaded
    h = m[["loaded_frac_30", "Oil_temperature"]].resample("1h").mean()
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 4.4), sharex=True)
    axes[0].plot(h.index, h["loaded_frac_30"] * 100, color=BLUE, lw=0.7)
    axes[0].set_ylabel("Loaded time (%)")
    axes[0].set_title("Compressor loaded time and oil temperature, Feb–Sep 2020 (red = reported air leaks)", loc="left")
    axes[1].plot(h.index, h["Oil_temperature"], color=ORANGE, lw=0.7)
    axes[1].set_ylabel("Oil temp (°C)")
    for ax in axes:
        shade_failures(ax, 0.35)
        ax.axvline(TEST_START, color=INK2, lw=1, ls="--")
    axes[0].text(TEST_START, 102, "  test period →", fontsize=8.5, color=INK2, va="bottom")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.tight_layout()
    save(fig, "fig_timeline.png")
    # (2) data problems + class balance
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.9))
    ax = axes[0]
    items = [("Missing hours (gaps)", q["missing_hours"]), ("Frozen-sensor hours", q["frozen_minutes_removed"] / 60),
             ("Gaps > 1 hour", q["gaps_over_1h"]), ("Irregular steps (k)", q["irregular_steps"] / 1000)]
    ax.barh([i[0] for i in items], [i[1] for i in items], color=BLUE, height=0.55)
    for i, (_, v) in enumerate(items):
        ax.text(v, i, f" {v:,.0f}", va="center", fontsize=8.5, color=INK)
    ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(i[1] for i in items) * 1.25)
    ax.set_title("(a) Problems in the raw data", loc="left")
    ax = axes[1]
    n, f = q["minute_rows"], q["positive_minutes"]
    ax.barh(["Normal", "Air leak (or 2 h before)"], [n - f, f], color=[GREY, ORANGE], height=0.55)
    for i, v in enumerate([n - f, f]):
        ax.text(v, i, f" {v:,} ({v / n:.1%})", va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, n * 1.35); ax.grid(axis="y", visible=False)
    ax.set_title("(b) 1-minute rows: heavy class imbalance", loc="left")
    fig.tight_layout()
    save(fig, "fig_data.png")


def model_comparison(res):
    names = list(res)
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    x = np.arange(len(names))
    w = 0.27
    for k, (metric, col, lab) in enumerate((("precision", BLUE, "Precision"), ("recall", ORANGE, "Recall"), ("f1", AQUA, "F1"))):
        ax.bar(x + (k - 1) * w, [res[n][metric] for n in names], width=w - 0.02, color=col, label=lab)
    for i, n in enumerate(names):
        ax.text(i + w, res[n]["f1"] + 0.015, f"{res[n]['f1']:.2f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([short(n) for n in names], rotation=25, ha="right")
    ax.set_ylim(0, 1.12)
    ax.grid(axis="x", visible=False)
    ax.set_title("Air-leak warning on the unseen July–August test period (minute level, after the 15-min alarm filter)", loc="left", fontsize=10)
    ax.legend(ncol=3, loc="upper left", fontsize=8.5)
    save(fig, "fig_model_comparison.png")


def confusion(cm, name):
    tn, fp, fn, tp = cm
    c = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(3.8, 3.1))
    ax.imshow(np.log1p(c), cmap="Blues")
    for (i, j), v in np.ndenumerate(c):
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=11, color="white" if np.log1p(v) > np.log1p(c.max()) * 0.6 else INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Leak"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Leak"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.grid(False)
    ax.set_title(f"{short(name)} – test minutes", loc="left", fontsize=9.5)
    save(fig, "fig_confusion.png")


def roc_pr(curves):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
    for (name, ((fpr, tpr, _), (p, r, _), auc, ap)), col in zip(curves.items(), (GREY, BLUE, ORANGE, AQUA)):
        axes[0].plot(fpr, tpr, color=col, lw=1.8, label=f"{short(name)} (AUC {auc:.3f})")
        axes[1].plot(r, p, color=col, lw=1.8, label=f"{short(name)} (AP {ap:.3f})")
    axes[0].plot([0, 1], [0, 1], color=GRID, lw=1, ls="--")
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate"); axes[0].set_title("(a) ROC curve", loc="left")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].set_title("(b) Precision–recall curve", loc="left")
    axes[0].legend(fontsize=7.2, loc="lower right"); axes[1].legend(fontsize=7.2, loc="upper right")
    fig.tight_layout()
    save(fig, "fig_roc_pr.png")


def test_timeline(te, p_ml, p_rule, name):
    s = pd.Timestamp(FAILURES[3][0])
    win = te.loc[s - pd.Timedelta(hours=10): s + pd.Timedelta(hours=10)].asfreq("1min")   # gaps stay empty
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 4.0), sharex=True)
    axes[0].plot(win.index, win["loaded_frac_30"] * 100, color=BLUE, lw=1.2, label="Loaded time, last 30 min (%)")
    axes[0].plot(win.index, win["Oil_temperature"], color=ORANGE, lw=1.2, label="Oil temperature (°C)")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].set_title("Failure 4 (15 July, never seen in training): sensors and warnings", loc="left")
    axes[1].plot(win.index, p_ml.reindex(win.index), color=BLUE, lw=1.4, label=f"{short(name)} probability")
    axes[1].plot(win.index, p_rule.reindex(win.index), color=AQUA, lw=1.4, label="Expert-system CF")
    axes[1].axhline(0.5, color=INK2, lw=0.8, ls="--")
    axes[1].set_ylim(-0.02, 1.05); axes[1].legend(fontsize=8, loc="upper left")
    for ax in axes:
        shade_failures(ax, 0.15)
        ax.axvspan(s - pd.Timedelta(hours=2), s, color=ORANGE, alpha=0.10, lw=0)
        ax.set_xlim(win.index.min(), win.index.max())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.tight_layout()
    save(fig, "fig_test_failure.png")


def lofo_timeline(allm, prob):
    d = prob.loc[allm.index].resample("1h").max()
    fig, ax = plt.subplots(figsize=(9.0, 2.6))
    ax.fill_between(d.index, 0, d.values, color=BLUE, alpha=0.6, lw=0, step="mid")
    ax.axhline(0.5, color=INK2, lw=0.8, ls="--")
    shade_failures(ax, 0.35)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Hybrid warning")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_title("Leave-one-failure-out: warnings over the whole period (red = reported leaks)", loc="left")
    save(fig, "fig_lofo_timeline.png")


def importance(imp):
    imp = imp[:12]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    labels = [d["feature"] for d in imp][::-1]
    vals = [d["mean"] for d in imp][::-1]
    eng = {l for l in labels if l.endswith(("_30", "_5"))}
    ax.barh(labels, vals, xerr=[d["std"] for d in imp][::-1], color=[ORANGE if l in eng else BLUE for l in labels], height=0.6,
            error_kw={"ecolor": MUTED, "lw": 1})
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Drop in PR-AUC when the feature is shuffled")
    ax.set_title("What the best model relies on (orange = engineered window features)", loc="left", fontsize=9.5)
    save(fig, "fig_importance.png")


def robustness(rb):
    conds = list(rb)
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    y = np.arange(len(conds))
    b = [rb[c]["Baseline"] for c in conds]
    m = [rb[c]["Best model"] for c in conds]
    ax.barh(y - 0.18, b, height=0.34, color=GREY, label="Baseline (LR, raw)")
    ax.barh(y + 0.18, m, height=0.34, color=BLUE, label="Best model")
    for i in range(len(conds)):
        ax.text(m[i] + 0.01, i + 0.18, f"{m[i]:.2f}", va="center", fontsize=8, color=INK)
        ax.text(b[i] + 0.01, i - 0.18, f"{b[i]:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(conds); ax.invert_yaxis()
    ax.set_xlim(0, 1.1); ax.grid(axis="y", visible=False)
    ax.set_xlabel("F1 on the test period")
    ax.set_title("Robustness to missing and noisy sensor readings", loc="left")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "fig_robustness.png")


def ablation(ab):
    sets = list(ab)
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    x = np.arange(4)
    w = 0.26
    cols = (GREY, BLUE, ORANGE)
    for k, (s, col) in enumerate(zip(sets, cols)):
        vals = ab[s]["Gradient Boosting"]["per_failure_f1"]
        ax.bar(x + (k - 1) * w, vals, width=w - 0.02, color=col, label=s)
    ax.set_xticks(x); ax.set_xticklabels([f"Failure {i}\n({pd.Timestamp(f[0]):%d %b})" for i, f in enumerate(FAILURES, 1)])
    ax.set_ylim(0, 1.05); ax.grid(axis="x", visible=False)
    ax.set_ylabel("F1 when held out")
    ax.set_title("Gradient Boosting, leave-one-failure-out: which feature set generalises?", loc="left", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    save(fig, "fig_ablation.png")


def summary_md(out):
    q, res, sp = out["data"], out["results"], out["split"]
    L = ["# MachineCare AI – results (MetroPT-3)", "",
         f"Raw data: {q['rows']:,} readings ({q['start']} to {q['end']}), {q['gaps_over_1min']} gaps "
         f"({q['missing_hours']} h missing, longest {q['longest_gap_h']} h), {q['frozen_minutes_removed']:,} frozen-sensor minutes removed. "
         f"After cleaning: {q['minute_rows']:,} one-minute rows, {q['positive_minutes']:,} labelled as air leak ({q['positive_minutes'] / q['minute_rows']:.1%}).", "",
         f"Split by time: train {sp['train']} ({sp['train_rows']:,} rows, {sp['train_failure_minutes']:,} leak minutes); "
         f"test {sp['test']} ({sp['test_rows']:,} rows, {sp['test_failure_minutes']} leak minutes). "
         f"{sp['excluded_unrepaired_minutes']:,} minutes between a failure and its repair are left out (label unknown).", "",
         "## Test results (unseen July–August, failure 4)", "",
         "| Model | CV F1 (leave-one-failure-out) | Accuracy | Balanced acc. | Precision | Recall | F1 | ROC-AUC | PR-AUC | Failure caught | Lead (min) | False-alarm episodes |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in res.items():
        cv = f"{r['cv_score']:.3f}" + (f" ± {r['cv_std']:.3f}" if r.get("cv_std") is not None else "") if r.get("cv_score") is not None else "–"
        ev = r["event"]
        lead = ev["events"][0]["lead_min"] if ev["events"] else None
        L.append(f"| {n} | {cv} | {r['accuracy']:.3f} | {r['balanced_accuracy']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
                 f"{r['roc_auc']:.3f} | {r['pr_auc']:.3f} | {'yes' if ev['detected'] else 'no'} | {lead if lead is not None else '–'} | {ev['false_alarm_episodes']} |")
    L += ["", f"Best model (chosen by cross-validated F1 on Feb–Jun, not by test score): **{out['best_model']}**", "",
          "Lead = minutes between the first warning and the reported start of the failure (negative = warning came after the reported start).", "",
          "## Leave-one-failure-out over all four failures", "", "| Approach | F1 per held-out failure | Overall F1 | Recall | Failures caught | Lead (min) | False-alarm episodes | …of which within 6 weeks before a failure |",
          "|---|---|---|---|---|---|---|---|"]
    for k, v in out["lofo"].items():
        ev = v["event"]
        L.append(f"| {k} | {v['per_failure_f1']} | {v['overall']['f1']:.3f} | {v['overall']['recall']:.3f} | {ev['detected']}/{ev['total']} | "
                 f"{[o['lead_min'] for o in ev['events']]} | {ev['false_alarm_episodes']} | {ev['false_alarms_within_6_weeks_before_a_failure']} |")
    L += ["", "## Feature-set ablation (leave-one-failure-out F1 per failure)", "", "| Feature set | Logistic Regression | Gradient Boosting |", "|---|---|---|"]
    for s, row in out["ablation"].items():
        L.append(f"| {s} | {row['Logistic Regression']['per_failure_f1']} (mean {row['Logistic Regression']['mean_f1']}) | "
                 f"{row['Gradient Boosting']['per_failure_f1']} (mean {row['Gradient Boosting']['mean_f1']}) |")
    lk = out["leakage_check"]
    L += ["", f"Leakage check (untuned Random Forest, all features): random 80/20 split F1 {lk['random_split_f1']:.3f} vs time-based split F1 "
          f"{lk['time_split_f1']:.3f}. A random split puts neighbouring minutes of the same failure in both train and test, so it looks perfect.", "",
          "## Robustness (test F1)", "", "| Condition | Baseline | Best model |", "|---|---|---|"]
    for c, r in out["robustness"].items():
        L.append(f"| {c} | {r['Baseline']:.3f} | {r['Best model']:.3f} |")
    L += ["", "## Most important features (permutation importance, best model, PR-AUC drop)", ""]
    L += [f"* {d['feature']}: {d['mean']:.3f} ± {d['std']:.3f}" for d in out["importance"][:10]]
    L += ["", f"Training time: {out['runtime_s']} s."]
    with open(os.path.join(ROOT, "results", "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
