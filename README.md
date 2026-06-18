# Adversarial Attacks on Class-Imbalanced Credit Default Models

**Authors:** Tim Angevare, Rick Hezeman, Ruben Stevenaar, Roy Mulder
**Course:** Applications of AI for Business (AAIB) — University of Twente

---

## Research Overview

Credit default prediction models are trained on highly class-imbalanced datasets: borrowers who default are far rarer than those who repay. Practitioners apply resampling techniques (SMOTE, random under-sampling, etc.) to improve minority-class recall, but this raises a question that this project investigates:

**Do resampling methods harden or weaken models against adversarial attacks?**
   A high-risk borrower (predicted to default) has an incentive to modify the features they control like loan amount or income in order to flip the model's prediction and obtain credit or a lower interest rate.


The study uses the [LendingClub dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club) and evaluates four classifiers (XGBoost, Logistic Regression, Random Forrist, and Neural Network) across five resampling strategies (Baseline, Class Weight, Undersampling, SMOTEENN, KMEANS SMOTE) and five levels of synthetic class imbalance (1%, 2%, 5%, 10%, 20%).

---

## Methodology

### Dataset

The LendingClub accepted-loans dataset is filtered to loans with status `Fully Paid` or `Charged Off`. The original minority fraction (defaults) is approximately 20%. To simulate realistic financial imbalance, the dataset is undersampled to five target minority fractions: **1%, 2%, 5%, 10%, 20%**.

### Pipeline (as implemented in `Advesarial_attacks.ipynb`)

```
Raw data
  └─ Feature selection & engineering
       ├─ Ordinal encoding  (emp_length, sub_grade, term)
       ├─ One-hot encoding  (purpose, home_ownership, addr_state, …)
       ├─ Log transform     (annual_inc, revol_bal)
       └─ Feature engineering (fico_mid, credit_history_yrs)
            └─ Per imbalance ratio [0.01, 0.02, 0.05, 0.10, 0.20]
                 ├─ Undersample majority / minority to target ratio
                 ├─ Kolmogorov–Smirnov bias check vs. original distribution
                 └─ 80 / 20 stratified train–test split
                      └─ GridSearchCV on 10% imbalance (class-weight, PR-AUC)
                           └─ Per model × per resampling method
                                ├─ DIR on resampled training labels (pre-CV)
                                ├─ 5-fold stratified CV  → metrics
                                ├─ SHAP feature importances (last fold)
                                └─ Genetic-algorithm (GA) perturbation adversarial attack
```

### Models

| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear; interpretable via SHAP LinearExplainer |
| Random Forest | Ensemble; SHAP TreeExplainer |
| XGBoost | Gradient-boosted trees; SHAP TreeExplainer |
| Neural Network | 2-layer MLP (skorch); SHAP KernelExplainer |

### Resampling methods

| Method | Type |
|---|---|
| Baseline | None — raw imbalanced data |
| RandomUnderSampler | Under-sample the majority class |
| SMOTE-ENN | Combined over/under-sampling |
| KMeans-SMOTE | Cluster-aware oversampling |
| ClassWeight | Cost-sensitive learning (no resampling) |

### Adversarial attack

A **genetic-algorithm (GA) perturbation attack** (`GAdvExample` in `src/ga_attack.py`). For each correctly-predicted default the GA evolves a population of perturbed copies of the instance: only the **mutable** features (`loan_amnt`, `annual_inc`, `dti`, `revol_bal`, `revol_util`, `open_acc`, `total_acc`) may change — every other column is black-listed — and new values are sampled from the training distribution. Crossover and mutation explore feature combinations, while the fitness function rewards flipping the prediction to non-default (0) and penalises the number and magnitude of changes (the proximity penalty is gradually relaxed via `GA_BETA` until a flip is found). The attack succeeds when the predicted class flips from default (1) to non-default (0); among the runs that flip, the most proximal counterfactual (fewest changed features) is reported. The primary metric is **attack success rate** — the fraction of true-default instances that can be flipped.

### Fairness metric (DIR)

The Disparate Impact Ratio is computed on the **resampled training data** — after the sampler runs but before any model is trained. It measures whether resampling changes the favourable-label rate differently across groups:

```
DIR = favourable_label_rate(unprivileged) / favourable_label_rate(privileged)
```

Groups evaluated:
- `emp_length`: short (< 5 years) vs. long employment (unprivileged / privileged)
- `annual_inc`: below-median vs. above-median income

DIR < 0.8 flags potential disparate impact introduced by the resampler.

---

## Repository Structure

```
.
├── Advesarial_attacks.ipynb   # Original exploratory notebook
├── src/
│   ├── config.py              # All constants: imbalance fracs, model names, paths
│   ├── preprocessing.py       # Data loading, feature engineering, bias check
│   ├── samplers_def.py        # Sampler objects (with n_jobs)
│   ├── models_def.py          # Model constructors + apply_class_weight
│   └── pipeline_utils.py      # CV, SHAP, DIR, adversarial attack logic
├── setup.py                   # Phase A: preprocess data + run gridsearch
├── main.py                    # Phase B: single contained experiment (HPC entry point)
├── aggregate.py               # Phase C: collect results → CSVs + plots
├── hq_task.sh                 # HyperQueue task wrapper
├── hyperq_job_hpc.sh          # SLURM job that starts HQ workers
├── setup_job.sh               # SLURM job for Phase A (optional)
├── requirements_hpc.txt       # Python dependencies for the cluster
├── Data/
│   └── accepted.csv           # Raw LendingClub data (not committed)
├── data/processed/            # Generated by setup.py
│   ├── X_full_0.01.parquet
│   ├── y_full_0.01.parquet
│   ├── bias_0.01.json
│   ├── …
│   └── feature_names.json
├── config/
│   └── best_params.json       # Generated by setup.py (gridsearch results)
├── results/
│   └── exp_XXXX/              # One directory per experiment (generated by main.py)
│       ├── metrics.csv
│       ├── shap.csv
│       ├── dir.csv
│       ├── attacks.csv
│       └── meta.json
└── outputs/                   # Generated by aggregate.py
    ├── model_results.csv
    ├── shap_importances.csv
    ├── dir_fairness_results.csv
    ├── adversarial_attack_results.csv
    └── plots/
```

---

## Running Locally (notebook)

```bash
pip install -r requirements.txt
jupyter notebook Advesarial_attacks.ipynb
```

Place `accepted.csv` in `Data/` before running. The notebook runs the full pipeline end-to-end in a single process and produces `model_results.csv` and `adversarial_attack_results.csv` in the project root.

---

## Running on the HPC Cluster SURF (SLURM + HyperQueue)

The notebook pipeline is refactored into three sequential phases. Phases A and C run once; Phase B runs as 100 parallel SLURM tasks (one per experiment combination).

### Experiment grid

```
5 imbalance fractions  ×  4 models  ×  5 resampling methods  =  100 experiments
```

Each experiment is fully self-contained: it loads preprocessed data, runs 5-fold CV, computes SHAP values, evaluates DIR, and runs the adversarial attack — writing all results to its own `results/exp_XXXX/` directory.

### Phase A — Setup

Preprocesses the raw data and runs GridSearchCV to find optimal hyperparameters for XGBoost, Logistic Regression, and Random Forest on the 10% imbalance dataset.

```
setup.py
  ├─ load_and_clean()  +  engineer_features()
  ├─ create_imbalanced_dataset() × 5 fractions  → parquet files
  ├─ bias check (KS test) per fraction          → bias_*.json
  └─ GridSearchCV (n_jobs=-1) × 3 models        → best_params.json
```

Outputs saved to `data/processed/` and `config/best_params.json`. Only needs to run once before any Phase B jobs.

### Phase B — Experiments (100 parallel HQ tasks)

```
main.py --experiment_id=N --n_cores=8
  ├─ Load X_full_{frac}.parquet + best_params.json
  ├─ 80/20 train–test split (fixed seed)
  ├─ build_pipeline(model, sampler)
  ├─ DIR on resampled training labels  → dir.csv
  ├─ 5-fold stratified CV              → metrics.csv
  ├─ SHAP (last CV fold)               → shap.csv
  ├─ Refit on full train set
  └─ Adversarial attack                → attacks.csv
```

Each task is mapped from `experiment_id` → `(frac, model, sampler)` via a fixed flat index. HyperQueue pins each task to 8 cores (`--cpus=8`), so 192/8 = 24 tasks run concurrently on a single genoa node.

### Phase C — Aggregation

```
aggregate.py
  ├─ Collect results/exp_*/metrics.csv   → outputs/model_results.csv
  ├─ Collect results/exp_*/shap.csv      → outputs/shap_importances.csv
  ├─ Collect results/exp_*/dir.csv       → outputs/dir_fairness_results.csv
  ├─ Collect results/exp_*/attacks.csv   → outputs/adversarial_attack_results.csv
  ├─ Report failed experiments
  └─ Generate plots (PR-AUC, attack success rate, global SHAP)
```

### Dispatch commands

```bash
# --- On the login node ---

# 1. Start the HyperQueue server (keep terminal open or use nohup)
module load 2023 && module load HyperQueue/0.19.0
nohup hq server start &

# 2. (Optional) Run Phase A as a SLURM job
sbatch setup_job.sh
# Or interactively: python setup.py

# 3. Submit the 100-task HQ array
hq submit \
    --array 0-99 \
    --stdout=logs/%{TASK_ID}.out \
    --stderr=logs/%{TASK_ID}.err \
    --pin taskset --cpus=8 \
    --time-limit=1300min \
    hq_task.sh

# 4. Submit the SLURM worker job
sbatch hyperq_job_hpc.sh

# 5. Monitor
squeue -u $USER
hq job progress 1

# 6. After all tasks finish, collect results
python aggregate.py
```

### Resource calculation

| Parameter | Value |
|---|---|
| Cores per node | 192 |
| Cores per task (`--cpus`) | 8 |
| Concurrent tasks | 192 ÷ 8 = **24** |
| Total tasks | 5 × 4 × 5 = **100** |
| Parallel rounds | ⌈100 ÷ 24⌉ = **5** |
| Worst-case time/task | ~90 min (SMOTE-ENN + Random Forest) |
| Estimated wall time | ~7.5 h |
| Requested (`#SBATCH --time`) | 20:00:00 |

To change cores-per-task, edit `N_CORES` in `hq_task.sh` **and** `--cpus` in the `hq submit` command — both must match.

---

## Configuration

All tuneable constants live in `src/config.py`:

| Constant | Default | Description |
|---|---|---|
| `MINORITY_FRACS` | `[0.01, 0.02, 0.05, 0.1, 0.2]` | Target minority fractions |
| `N_FOLDS` | `5` | CV folds |
| `N_ATTACK_CASES` | `50` | Instances attacked per experiment |
| `GA_SOL_PER_POP` | `30` | GA population size per generation |
| `GA_NUM_PARENTS_MATING` | `10` | Parents selected for crossover each generation |
| `GA_NUM_GENERATIONS` | `100` | Max generations per run (early-stops once flipped) |
| `GA_N_RUNS` | `5` | Independent GA restarts per attacked instance |
| `GA_BETA` | `0.95` | Proximity-constraint relaxation factor (0 < β < 1) |
| `N_SHAP_MAX` | `200` | Test samples for tree/linear SHAP |
| `N_SHAP_NN` | `20` | Test samples for NN KernelExplainer |
| `MUTABLE_FEATURES` | see config | Features the attacker can modify |

---

## Dependencies

Install with:

```bash
# Local
pip install -r requirements.txt

# HPC cluster
pip install -r requirements_hpc.txt  # adds pyarrow for parquet support
```

Key packages: `scikit-learn 1.9`, `imbalanced-learn 0.14`, `xgboost 3.2`, `shap 0.52`, `skorch 1.4`, `torch 2.12`.
