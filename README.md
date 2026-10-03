# Football match models: modeling and experimentation environment

> **Scope note.** This repository is **only the experimentation and model-training environment**.
> It is not a production service. There is no API, no scheduler, no live inference, and no
> deployment code here. Everything in this folder exists to _explore data, select features,
> train candidate models, compare them honestly, and persist the winners_ so that a separate
> production system can consume them later.

---

## The video

This is the machine learning project behind the Sleeping Dev video
**SQL Full Course With a Real ML Project (Football Data)**.

The course teaches SQL on the database this project reads from, and ends by building the kind of table
these models are trained on: one row per match, with features computed only from earlier matches. The
models here start where that table ends.

Three numbers from the course put the results of this repository in context. They come from different
sets of matches, so read them as a scale and not as a ranking:

| Approach | Matches called right |
| --- | --- |
| Always pick the home team | 43.5% |
| One feature written in SQL: points in the previous five matches | 46.7% |
| The models in this repository, about a thousand features, on their test set | about 54% |

The course also shows what happens when that one SQL feature includes the match it is trying to
predict: the score jumps to 65.6%, above the real models. That is the leak the audit in
[Feature selection](#feature-selection--the-core-idea) looks for.

---

## What this environment does

We model football (soccer) matches in two fundamentally different ways, and the environment is
built around that split:

| Problem type       | Question it answers      | Target                               | Output |
| ------------------ | ------------------------ | ------------------------------------ | ------ |
| **Classification** | 1X2 (Home / Draw / Away) | A probability vector over 3 outcomes |

KPI: Robust Probability Score (RPS)
| **Distribution prediction** | _How many?_ | Goals, corners, cards, shots | A full probability distribution over counts (0, 1, 2, …), from which Over/Under markets are derived. KPIS (Robust Probability Score and NLL)

The second family of models, we do not predict
a single number like "2.7 corners". We predict the **parameters of a count distribution**
(Poisson `λ` (mean), or Negative Binomial `μ, α` (mean & variation)), which gives us `P(Y = k)` for every `k`. Home and away
components are modeled separately and then combined by **convolution** to obtain the distribution
of the match total — which is what an Over/Under line actually needs. (the convolution process is not found on the repository).

---

## Directory map

```
.
├── configs/       Per-target YAML configuration (the control panel)
├── data/          Input feature datasets (parquet). Not included in this repository
├── notebooks/     The actual experiments — one notebook per target
├── outputs/       Everything the experiments produce, from json file with model results, to pickle files
├── src/           Reusable library code imported by the notebooks
└── requirements.txt
```

### `configs/` — the control panel

One folder per target (`1x2`, `goals`, `corners`, `cards`). Nothing that matters is hardcoded in
Python; it lives here as YAML.

- **`<target>_config.yaml`** — the master file for a target. It declares which columns form the
  target, the Over/Under thresholds, which distribution families to compare, cross-validation
  settings, base hyperparameters for every model, numeric guards, and the entire feature-selection
  configuration.
- **`optuna_search_spaces.yaml`** — hyperparameter search ranges for tuning. Widening a search
  range never requires touching code.
- **`best_params_*.yaml`** — the winning hyperparameters found by tuning, checkpointed so an
  experiment can be resumed without re-running a multi-hour search.

The practical consequence: **adapting the whole pipeline to a new count target is a config change,
not a code change.** Copy `goals_config.yaml`, point `components[*].column` at the new columns,
adjust the thresholds, and the same pipeline runs.

### `data/` — inputs

**The data is not included in this repository**, and neither is the feature-engineering pipeline that
builds it. The notebooks expect pre-built feature datasets in parquet form under `data/`, split by scope:
`clubs/` for club football, `countries/` for national teams. These are wide tables, on the order of a
thousand-plus engineered features per match, which is why feature selection is the centerpiece of this
environment. The notebooks are kept with their outputs so the results can be read without the data.

### `notebooks/` — the experiments

Organized as `notebooks/<scope>/<target>/`. Each notebook is the narrative of one modeling
problem, run end to end and left with its outputs visible so results are auditable without
re-execution. They all follow the same arc:

> **Baseline → Feature Selection → Model Selection → Hyperparameter Tuning → Error Analysis → Test Evaluation → Save Winner**

The notebooks are deliberately thin: they orchestrate, plot, and narrate. The statistical
machinery they call lives in `src/`.

### `outputs/` — everything produced

- **`feature_selection_subset/`** — the surviving feature sets per target, persisted as parquet so
  downstream steps never repeat the (expensive) selection. Not included here: only the result files are.
- **`models/`** — serialized model bundles (`.pkl`, not included here). A bundle is self-contained: the fitted model,
  the imputer fitted on train, the ordered feature list, and metadata (hyperparameters, metrics,
  date) so any saved model can be traced back to the run that produced it.
- **`model_metrics/`** — JSON metric records for every stage (baseline, validation, tuning, test)
  plus diagnostic plots such as calibration curves.

### `src/` — the library

| Module               | Responsibility                                                                                                           |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `config/`            | Loads and validates the YAML config; resolves output paths from the target name                                          |
| `preprocessing/`     | Target reconstruction and cleaning (e.g. rebuilding 90-minute counts from halves, so extra time never leaks in)          |
| `feature_selection/` | The two selection pipelines described below, plus their shared helpers                                                   |
| `training/`          | Temporal cross-validation, leakage-free target encoding, and a unified interface over LightGBMLSS / XGBoostLSS / NGBoost |
| `metrics/`           | RPS, NLL, calibration, and count-distribution metrics (Poisson & Negative Binomial)                                      |
| `hypertunning/`      | Optuna-driven search, with search spaces read from YAML                                                                  |
| `evaluation/`        | Evaluation over already-computed predictions, plus model and result persistence                                          |
| `error_analysis/`    | Post-hoc diagnosis: calibration, segmentation by league, overdispersion, temporal drift, bias                            |

---

## Feature selection — the core idea

We start with roughly a **thousand-plus candidate features and only a few thousand rows**. In that
regime, any feature-ranking method will happily hand you a list of "important" features that are
pure noise. So the selection process is designed around one question:

> **Would this feature still look important if the target were meaningless?**

Both pipelines answer it the same way — by building an explicit **null distribution** and forcing
every feature to beat it. Two pipelines exist because the two problem types need different
machinery, but they share the same skeleton and the same shared helpers
(`src/feature_selection/common.py`).

### Shared skeleton

| Step                                                              | What it does                                                                                                                                                | Why                                                                                                                                                                                                                                                                                                                                                                                         |
| ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **0. Sparsity filter**                                            | Drops near-constant and almost-entirely-missing columns                                                                                                     | No information to extract. Evaluated on **train only** — val and test never participate, so nothing leaks                                                                                                                                                                                                                                                                                   |
| **0.5 Leakage audit** _(count targets)_                           | Trains a **single-feature** model on each of the most correlated features and measures how much it improves prediction                                      | The failure that ruins a football model isn't statistical — it's a rolling average that accidentally includes the current match, or closing odds mixed in with opening odds. A permutation test cannot catch this, because such a feature genuinely _is_ predictive. This step flags implausibly strong single features and reports them; **it deletes nothing**, the call is the analyst's |
| **1. Mutual information pre-filter**                              | Cheap screen: drops features with MI ≈ 0 against the target                                                                                                 | MI near zero means statistical independence. Removes the obvious dead weight before the expensive steps                                                                                                                                                                                                                                                                                     |
| **2. Temporal correlation filter**                                | Groups near-duplicate features via hierarchical clustering and keeps one representative per group                                                           | Correlation is measured across **disjoint time blocks** and the median is taken, so two features are only called redundant if they're redundant _in most regimes_, not just one. Clustering (rather than a greedy pass) makes the result independent of column order and handles transitivity: if A≈B and B≈C, all three land in one group                                                  |
| **3. Block-permutation importance + multiple-testing correction** | The real test. See below                                                                                                                                    | This is where noise is actually removed                                                                                                                                                                                                                                                                                                                                                     |
| **4. Stability filter**                                           | Splits training history into consecutive windows, re-measures importance in each, and keeps features that are **both important and consistently important** | Score is `mean(percentile) − k · std(percentile)`. Using the mean alone would _actively reward_ regime-dependent features: a feature ranking `[0.95, 0.10, 0.95, 0.10]` averages 0.61 and survives, while a steady 0.35 gets cut. Penalizing the spread inverts that                                                                                                                        |
| **5. Validation**                                                 | Retrains on validation data and compares candidate feature sets                                                                                             | See "the honesty check" below                                                                                                                                                                                                                                                                                                                                                               |

### Step 3 in plain terms — the null-model test

For each feature we want to know whether its measured importance is larger than what pure chance
would produce. So:

1. Train the model on the real data and record each feature's importance (mean absolute SHAP
   contribution).
2. **Shuffle the target hundreds of times** and retrain each time, recording importance again.
   These runs are the "what does noise look like" reference.
3. A feature is kept only if its real importance is extreme relative to its own null distribution.

Two details matter:

- **The shuffle preserves time structure.** We rotate _contiguous blocks_ of the target rather than
  shuffling row by row. An i.i.d. shuffle destroys the target's autocorrelation and produces an
  artificially _weak_ null — any feature with a slow trend (seasonal drift, rule changes, league
  mix) beats it without having real predictive power. Block rotation raises the bar to an honest
  level. Block length is estimated automatically from the target: from label run-length for
  classification, from autocorrelation for counts. When the target has no temporal persistence, the
  block collapses to 1 and the test correctly degenerates to an i.i.d. shuffle.

- **Testing a thousand features at once inflates false positives.** With 1,000 features at a 5%
  threshold you expect ~50 spurious "discoveries". We apply a **Benjamini–Hochberg** correction to
  control the false discovery rate. There's also a **preflight check** that runs _before_ the
  compute is spent: it verifies the configuration is mathematically capable of rejecting anything
  at all. (With too few permutations, the smallest achievable p-value is larger than what the
  correction demands — the test then cannot select a single feature, by construction, no matter how
  strong the signal.)

### What differs between the two pipelines

|                         | **Classification** (`feature_selection.py`)                                 | **Distributions / counts** (`poisson_feature_selection.py`)                                                                    |
| ----------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Used for                | 1X2                                                                         | Goals, corners, cards, shots                                                                                                   |
| Model objective         | Multiclass                                                                  | Poisson (importance measured in log-rate space)                                                                                |
| Relevance screen        | `mutual_info_classif`                                                       | `mutual_info_regression`, averaged over several seeds (MI is stochastic)                                                       |
| Null shuffle constraint | Must preserve all classes — LightGBM requires every label present           | Must preserve variance — a constant target collapses the model to an intercept and would make _every_ feature look significant |
| Cluster representative  | Chosen by MI (the target is nominal, so correlation with it is meaningless) | Chosen by Spearman with the target (a count is ordinal, so correlation is meaningful)                                          |
| Validation metric       | Log loss, accuracy, macro-F1                                                | RPS and Poisson deviance                                                                                                       |
| Leakage audit           | —                                                                           | Yes (step 0.5)                                                                                                                 |

### The honesty check (step 5)

The selection isn't trusted until it proves itself against **two** references:

1. **`baseline_all`** — the same model trained on _all_ features. If the selection can't beat this,
   it's pruning away signal.
2. **`random_k`** — the same model trained on a _randomly chosen_ set of the same size, averaged
   over several draws. This one is the important control: without it you cannot distinguish
   _"selection found real signal"_ from _"having fewer features helps regardless of which ones"_.

Only a selection that beats **both** justifies the pipeline. The notebooks print this comparison
explicitly, including the cases where the answer is negative.

---

## Guiding principles

These recur throughout the code and explain most of its design decisions:

- **Time is never violated.** Splits are chronological, cross-validation is `TimeSeriesSplit`,
  correlations and stability are measured across time blocks, and every filter is fitted on train
  only.
- **Silent failures are made loud.** Categorical schemas are explicitly aligned across
  train/val/test and asserted, because a mismatch surfaces as an unrelated error much later.
  Unseen category levels — which silently become NaN and still yield a plausible-looking
  prediction — are counted and reported.
- **Reproducibility is not optional.** Seeds are fixed and threaded through; permuted targets are
  generated in the main process so results are identical whether one worker runs or eight.
- **Every step is auditable.** Each pipeline returns a report object carrying the feature count
  after every stage, the intermediate tables, and the raw null matrix — so significance criteria
  can be re-examined later without paying the compute cost again.

---

## Running an experiment

```bash
pip install -r requirements.txt
```

Then open the notebook for the target you're working on
(`notebooks/clubs/<target>/<Target>ModelCreation.ipynb`) and run it top to bottom. It reads its
parameters from `configs/<target>/`, pulls data from `data/`, imports its machinery from `src/`,
and writes everything it produces to `outputs/`.

# Considerations

1. The national-team models for each market have not been refactored yet (they use a different
   feature-engineering pipeline because of the gaps between matches).
2. The focus is on European club competitions (Spain, England, Germany, Italy and France) and the main
   national-team tournaments.
3. Running a notebook end to end needs the datasets, which are not published. Without them, read the
   notebooks and the JSON files in `outputs/` for the results.
