# MachineCare AI: Predictive Maintenance System

CT-361 Artificial Intelligence & Expert Systems · Complex Computing Activity · NED University of Engineering & Technology

Group: Fatima Zaman (CT-24205), Eeshah Rashid (CT-24213), Javeria Soomro (CT-24216), Anoosha Hassan (CT-24108), Sana Nazeer (CT-24111)

A hybrid AI system that warns when the air compressor of a metro train is developing an **air leak**. It learns from
1.5 million real sensor readings (MetroPT-3), and a rule-based expert system explains each warning, names the likely
cause and recommends a safety-first maintenance action.

## How to run

Needs Python 3.9+ and four common packages:

```bash
python3 -m venv .venv && source .venv/bin/activate      # Windows: py -m venv .venv  then  .venv\Scripts\activate
pip install -r requirements.txt                           # scikit-learn, pandas, numpy, matplotlib
python train.py                                           # cleaning, labelling, training, tuning, evaluation, charts (~8 min)
python run_app.py                                         # dashboard at http://127.0.0.1:8050
```

The trained model and results are included, so the dashboard also runs without retraining. The first `train.py` run
caches the cleaned one-minute table in `data/metro_minutes.pkl` (delete it to rebuild from the raw CSV).

## 1. Problem

Metro trains use compressed air for brakes, doors and suspension. The Air Production Unit (APU) keeps the air at
8–10 bar. If air leaks out, the compressor has to run almost non-stop, overheats and wears out, and if the pressure falls
below 7 bar the train must be taken out of service. Leaks grow slowly, so an early warning lets the operator repair the
train at a planned stop instead of a breakdown in service.

**Task:** for every minute, predict whether the compressor is in an air-leak state or in the 2 hours before a reported
leak (binary classification, heavily imbalanced), then explain the warning and recommend an action.

## 2. Data

| Source | Type | Size | Use |
|---|---|---|---|
| MetroPT-3 Dataset (UCI, doi:10.24432/C5VW3R, CC BY 4.0) – real data from a Metro do Porto train, Feb–Sep 2020 | Multivariate sensor time series | 1,516,948 rows, 15 signals (7 analogue, 8 digital) | Training and evaluating the ML models |
| Company failure reports (in `data/Data Description_Metro.pdf`) | 4 air-leak failures with start/end times | 4 events | Building the labels |
| Expert knowledge base (`data/kb/`) | Rules with certainty factors | 93 rules (11 for the compressor) | Feature choice, rule-based warnings, explanations, root cause, maintenance plan |

Signals: TP2 (compressor pressure), TP3 (pneumatic panel pressure), H1, DV_pressure (dryer discharge), Reservoirs,
Oil_temperature, Motor_current, and the digital signals COMP, DV_eletric (compressor loaded), Towers, MPG, LPS
(low-pressure alarm), Pressure_switch, Oil_level, Caudal_impulses (air flow).

### Why this data is not clean (what we found and fixed)

| Problem in the raw file | What we did |
|---|---|
| **No failure column at all** – the data is unlabelled | Built labels from the 4 failure reports: leak minutes plus the 2 hours before each failure (following García-Méndez et al., 2025) |
| Documentation says 1 Hz / 15.2 M points; the file has one reading every ~10 s (1.5 M rows) | Used the real timestamps |
| Irregular time steps (8–22 s, 179,426 irregular steps) | Resampled to regular 1-minute rows |
| 331 gaps, 909 hours missing in total, longest 48 h | Gaps ≤ 5 min interpolated; longer gaps left empty, never invented |
| 10,177 minutes of frozen sensors (oil temperature and pressure flat for hours, e.g. 22–25 June) | Detected (both signals constant ≥ 30 min) and removed |
| Negative pressure readings (sensor offset, e.g. −0.012 bar) | Clipped to 0 |
| TP3 and Reservoirs are the same signal (correlation 1.0) | Dropped Reservoirs |
| Oil_level is "active" 90 % of the time although the documentation says active = oil low | Not used (unreliable) |
| Failure report table has duplicated numbering and a repair date ("30 Apr") that cannot belong to the 29 May failure | Read as shifted by one row; minutes between a failure and its repair (11,506) left out because their true label is unknown |
| Heavy class imbalance: 2.2 % leak minutes | Class weights, SMOTE, F1 / balanced accuracy instead of accuracy |

After cleaning: **243,177 one-minute rows**, 5,309 labelled as leak.

## 3. Method

1. **Features** (`ml/preprocess.py`): rolling 30-minute and 5-minute windows using past data only, so the system can run
   live: compressor loaded time (duty cycle), compressor starts, motor-current mean/std, oil-temperature max and rise,
   panel-pressure min/mean/std, compressor-pressure mean.
2. **Knowledge-guided feature set (19 features).** Rule from the knowledge base: *a leak anywhere makes the compressor run
   longer, draw more current and heat up*. Signals that only show leaks **at the dryer** (DV_pressure, H1, Towers,
   Pressure_switch) are left out of the ML model and used only by the expert system to find the cause.
3. **Time-based split.** Train Feb–Jun (failures 1–3), test Jul–Aug (failure 4, never seen). A random split is wrong for
   time series (see the leakage check).
4. **Tuning:** GridSearchCV with **leave-one-failure-out cross-validation** (3 time blocks, each holding one failure), scored by F1.
5. **Models:** baseline Logistic Regression (raw sensors, no imbalance handling), tuned Logistic Regression, Decision Tree,
   Random Forest, Random Forest + SMOTE, Gradient Boosting (HistGradientBoosting), soft-voting ensemble, and an
   unsupervised Isolation Forest trained on normal data only.
6. **Alarm filter:** probabilities are averaged over the last 15 minutes, then compared with 0.5.
7. **Expert system** (`kbs/`, `data/kb/`): forward chaining with MYCIN certainty factors over compressor facts
   (continuous loading, high motor load, hot oil, panel pressure below 8.2 bar, dryer purge high, LPS alarm),
   root-cause rules (dryer valve vs downstream pipe leak), a safety rule (pressure < 7 bar → take the train out of
   service) and a weighted maintenance recommender.
8. **Hybrid:** average of the ML probability and the expert-system CF; the expert system adds the explanation, cause and plan.

## 4. Results

### Test period: July–August (72,574 minutes, 391 leak minutes, failure 4 never seen in training)

| Model | CV F1 | Accuracy | Balanced acc. | Precision | Recall | F1 | ROC-AUC | Failure 4 caught? | False-alarm episodes |
|---|---|---|---|---|---|---|---|---|---|
| Baseline: Logistic Regression (raw) | 0.455 | 0.995 | 0.500 | 0.000 | 0.000 | 0.000 | 0.911 | no | 0 |
| **Logistic Regression (best by CV)** | **0.557** | **0.996** | **0.859** | **0.599** | **0.721** | **0.654** | 0.901 | **yes, 11 min early** | **3** |
| Decision Tree | 0.432 | 0.988 | 0.497 | 0.000 | 0.000 | 0.000 | 0.451 | no | 8 |
| Random Forest | 0.328 | 0.993 | 0.598 | 0.257 | 0.200 | 0.225 | 0.830 | late | 2 |
| Random Forest + SMOTE | 0.168 | 0.994 | 0.499 | 0.000 | 0.000 | 0.000 | 0.796 | no | 2 |
| Gradient Boosting | 0.405 | 0.995 | 0.714 | 0.514 | 0.430 | 0.468 | 0.531 | late | 2 |
| Ensemble (soft voting) | 0.415 | 0.995 | 0.724 | 0.560 | 0.450 | 0.499 | 0.935 | late | 2 |
| Isolation Forest (unsupervised) | – | 0.987 | 0.646 | 0.154 | 0.302 | 0.204 | 0.990 | late | 21 |
| Expert system (rules only) | – | 0.989 | 0.958 | 0.317 | 0.926 | 0.472 | 0.994 | yes, 91 min early | 24 |
| Hybrid (ML + rules) | – | 0.995 | 0.875 | 0.494 | 0.754 | 0.597 | 0.996 | yes, 24 min early | 5 |

Accuracy is misleading: always saying "normal" already scores 0.995. F1, balanced accuracy, recall and "caught / false
alarms" are the real measures. Baseline → best: F1 0.00 → 0.65, balanced accuracy 0.50 → 0.86.

### All four failures (leave-one-failure-out: each failure held out, model trained on the other three)

| Approach | F1 per failure (1, 2, 3, 4) | Overall F1 | Recall | Failures caught | False-alarm episodes (of which ≤ 6 weeks before a failure) |
|---|---|---|---|---|---|
| Logistic Regression | 0.47, 0.26, 0.96, 0.65 | 0.622 | 0.925 | 4/4 | 20 (18) |
| Expert system | 0.44, 0.24, 0.94, 0.47 | 0.581 | 0.945 | 4/4 | 65 (48) |
| Hybrid | 0.47, 0.25, 0.96, 0.60 | 0.614 | 0.932 | 4/4 | 24 (21) |

18 of the 20 "false alarms" happen in the six weeks before a reported failure and look exactly like a leak (compressor
loaded 100 %, dryer cycling, oil hot). They are very likely the same leak growing before anyone reported it, so the
real precision is higher than the strict numbers above.

### Critical findings

* **Simple beats complex here.** With only four failures, Logistic Regression generalised best; deep trees and boosting
  memorised failure-specific patterns.
* **Knowledge-guided features are what made the July failure detectable.** Leave-one-failure-out F1 for failure 4:
  raw sensors 0.00, all 27 features 0.00, knowledge-guided 19 features 0.65 (Logistic Regression). Failures 1–3 were
  dryer leaks (high DV_pressure); failure 4 was a downstream leak (DV_pressure 0). Models that used DV_pressure learned
  "leak = high DV_pressure" and missed failure 4 completely.
* **Leakage check.** The same untuned Random Forest scores F1 0.984 with a random 80/20 split but 0.000 with a time
  split: neighbouring minutes of the same failure end up in train and test. Random splits must not be used for time series.
* **Robustness.** Best model F1 on the test period: 0.654 clean, 0.653 with 5 % missing values, 0.651 with sensor noise,
  0.672 with 10 % missing + noise (the 30-minute window features average out noise).
* **Most important features** (permutation importance, PR-AUC drop): oil_temp_max_30 0.255, tp2_mean_30 0.212,
  Oil_temperature 0.206, tp3_std_30 0.164, loaded_frac_30 0.036.
* **SMOTE did not help**: synthetic leak minutes made the forest over-confident on training-specific patterns.

All numbers: `results/summary.md` and `results/results.json`. Charts: `figures/`.

## 5. Dashboard (`python run_app.py`)

* **Check readings:** enter five compressor values (or pick an example). Shows the ML probability (using the most similar
  real minute in the data), the expert-system CF, the hybrid score, the rules that fired, the likely cause, any safety
  constraint and the maintenance plan.
* **Replay:** real data around each failure, an unreported episode and a normal day, with ML and rule warnings.
* **Model results** and **Data** tabs: tables and charts from training.

## 6. Ethics and limitations

* No personal data. A missed leak can lead to a train losing brake/door air, so the system only **warns**; a person
  makes the decision, and the pressure < 7 bar safety rule always comes first.
* Only four labelled failures, all air leaks, from one train; results need confirming on more trains and failure types.
* Labels are incomplete (unreported leak episodes) and the feature-set choice was informed by looking at all four
  failures, so the July test is not fully blind for that one decision.
* The 2-hour warning label comes from the literature; real lead times were short (11 min early for failure 4 with ML,
  91 min with the rules).

## 7. Project structure

```
train.py              full pipeline: clean → label → features → tune → test → LOFO → ablation → charts
run_app.py            starts the dashboard
ml/preprocess.py      loading, quality report, cleaning, resampling, frozen-sensor removal, features, labels, splits
ml/smote.py           SMOTE that runs only inside training folds
ml/kbs_rules.py       bridge from a minute of data to the expert system
ml/plots.py           charts and results/summary.md
kbs/                  expert-system engine (frames, rules, certainty factors, forward/backward chaining, recommender)
data/kb/              knowledge base (ontology, facts, faults, causes, actions, rules.kb, thresholds, sources)
data/                 MetroPT3(AirCompressor).csv and its documentation
app/                  dashboard (Python standard-library web server + one HTML page)
models/ results/ figures/ docs/   trained model, results, charts, dashboard screenshots
```

## References

* N. Davari, B. Veloso, R. P. Ribeiro, P. M. Pereira, J. Gama, "Predictive maintenance based on anomaly detection using deep learning for air production unit in the railway industry," IEEE DSAA 2021, doi:10.1109/DSAA53316.2021.9564181.
* B. Veloso, R. P. Ribeiro, P. M. Pereira, J. Gama, "The MetroPT dataset for predictive maintenance," Scientific Data 9, 764 (2022), doi:10.1038/s41597-022-01877-3.
* **S. García-Méndez, F. de Arriba-Pérez, F. Leal, B. Veloso, B. Malheiro, J. C. Burguillo-Rial, "An explainable machine learning framework for railway predictive maintenance using data streams from the metro operator of Portugal," Scientific Reports (2025), doi:10.1038/s41598-025-08084-1.** (main paper we follow: labels from failure reports with a 2-hour pre-failure window, window statistics as features, tree/ensemble models, explanations)
* **M. Jakobs, B. Veloso, J. Gama, "Interpretable Rules for Online Failure Prediction: A Case Study on the Metro do Porto dataset," arXiv:2502.07394 (2025).** (rule-based explanations of failure alarms)
* A. C. Akshitha Mary et al., "Predictive Modeling for Metro Performance Using MetroPT3 Dataset," IJACSA 16(8), 2025.
* N. V. Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique," JAIR 16, 321–357, 2002.
* F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," JMLR 12, 2825–2830, 2011.
