# MachineCare AI – results (MetroPT-3)

Raw data: 1,516,948 readings (2020-02-01 00:00:00 to 2020-09-01 03:59:50), 331 gaps (909.5 h missing, longest 48.0 h), 10,177 frozen-sensor minutes removed. After cleaning: 243,177 one-minute rows, 5,309 labelled as air leak (2.2%).

Split by time: train 2020-02-01 to 2020-07-01 (failures 1-3) (159,097 rows, 4,918 leak minutes); test 2020-07-01 to 2020-09-01 (failure 4) (72,574 rows, 391 leak minutes). 11,506 minutes between a failure and its repair are left out (label unknown).

## Test results (unseen July–August, failure 4)

| Model | CV F1 (leave-one-failure-out) | Accuracy | Balanced acc. | Precision | Recall | F1 | ROC-AUC | PR-AUC | Failure caught | Lead (min) | False-alarm episodes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline: Logistic Regression | 0.455 | 0.995 | 0.500 | 0.000 | 0.000 | 0.000 | 0.911 | 0.500 | no | – | 0 |
| Logistic Regression | 0.557 ± 0.285 | 0.996 | 0.859 | 0.599 | 0.721 | 0.654 | 0.901 | 0.675 | yes | 11 | 3 |
| Decision Tree | 0.432 ± 0.114 | 0.988 | 0.497 | 0.000 | 0.000 | 0.000 | 0.451 | 0.005 | no | – | 8 |
| Random Forest | 0.328 ± 0.047 | 0.993 | 0.598 | 0.257 | 0.200 | 0.225 | 0.830 | 0.113 | yes | -172 | 2 |
| Random Forest + SMOTE | 0.168 ± 0.116 | 0.994 | 0.499 | 0.000 | 0.000 | 0.000 | 0.796 | 0.185 | no | – | 2 |
| Gradient Boosting | 0.405 ± 0.130 | 0.995 | 0.714 | 0.514 | 0.430 | 0.468 | 0.531 | 0.249 | yes | -98 | 2 |
| Ensemble (soft voting) | 0.415 ± 0.152 | 0.995 | 0.724 | 0.560 | 0.450 | 0.499 | 0.935 | 0.371 | yes | -95 | 2 |
| Isolation Forest (unsupervised) | – | 0.987 | 0.646 | 0.154 | 0.302 | 0.204 | 0.990 | 0.184 | yes | -26 | 21 |
| Expert system (rules) | – | 0.989 | 0.958 | 0.317 | 0.926 | 0.472 | 0.994 | 0.429 | yes | 91 | 24 |
| Hybrid (ML + rules) | – | 0.995 | 0.875 | 0.494 | 0.754 | 0.597 | 0.996 | 0.640 | yes | 24 | 5 |

Best model (chosen by cross-validated F1 on Feb–Jun, not by test score): **Logistic Regression**

Lead = minutes between the first warning and the reported start of the failure (negative = warning came after the reported start).

## Leave-one-failure-out over all four failures

| Approach | F1 per held-out failure | Overall F1 | Recall | Failures caught | Lead (min) | False-alarm episodes | …of which within 6 weeks before a failure |
|---|---|---|---|---|---|---|---|
| ml | [0.4709, 0.2581, 0.9565, 0.6543] | 0.622 | 0.925 | 4/4 | [-23, -3, -2, 11] | 20 | 18 |
| rules | [0.4411, 0.2358, 0.9412, 0.472] | 0.581 | 0.945 | 4/4 | [-23, -10, -14, 91] | 65 | 48 |
| hybrid | [0.4651, 0.2495, 0.9562, 0.5972] | 0.614 | 0.932 | 4/4 | [-23, -7, -8, 24] | 24 | 21 |

## Feature-set ablation (leave-one-failure-out F1 per failure)

| Feature set | Logistic Regression | Gradient Boosting |
|---|---|---|
| Raw sensors (13) | [0.485, 0.261, 0.969, 0.0] (mean 0.429) | [0.524, 0.277, 0.734, 0.0] (mean 0.384) |
| Knowledge-guided (19) | [0.471, 0.258, 0.957, 0.654] (mean 0.585) | [0.364, 0.258, 0.963, 0.468] (mean 0.513) |
| All features (27) | [0.463, 0.265, 0.962, 0.0] (mean 0.422) | [0.342, 0.268, 0.541, 0.0] (mean 0.288) |

Leakage check (untuned Random Forest, all features): random 80/20 split F1 0.984 vs time-based split F1 0.000. A random split puts neighbouring minutes of the same failure in both train and test, so it looks perfect.

## Robustness (test F1)

| Condition | Baseline | Best model |
|---|---|---|
| Clean test data | 0.000 | 0.654 |
| 5 % missing values | 0.000 | 0.653 |
| Sensor noise (2 % of range) | 0.000 | 0.651 |
| 10 % missing + 2 % noise | 0.000 | 0.672 |

## Most important features (permutation importance, best model, PR-AUC drop)

* oil_temp_max_30: 0.255 ± 0.002
* tp2_mean_30: 0.212 ± 0.004
* Oil_temperature: 0.206 ± 0.004
* tp3_std_30: 0.164 ± 0.004
* loaded_frac_30: 0.036 ± 0.005
* TP2: 0.012 ± 0.002
* COMP: 0.004 ± 0.001
* MPG: 0.004 ± 0.001
* DV_eletric: 0.004 ± 0.001
* oil_temp_rise_30: 0.000 ± 0.000

Training time: 453 s.
