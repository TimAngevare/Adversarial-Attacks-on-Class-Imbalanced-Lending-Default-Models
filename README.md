# Adversarial Attacks on Class-Imbalanced Credit Default Models

**Authors:** Tim Angevare, Rick Hezeman, Ruben Stevenaar, Roy Mulder
**Course:** Applications of AI for Business (AAIB) — University of Twente

---

## Research Overview

Credit default prediction models are trained on highly class-imbalanced datasets: borrowers who default are far rarer than those who repay. Practitioners apply resampling techniques (SMOTE, random under-sampling, etc.) to improve minority-class recall, but this raises a question that this project investigates:

**Do resampling methods harden or weaken models against adversarial attacks?**
   A high-risk borrower (predicted to default) has an incentive to modify the features they control like loan amount or income in order to flip the model's prediction and obtain credit or a lower interest rate.


The study uses the [LendingClub dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club) and evaluates four classifiers (XGBoost, Logistic Regression, Random Forrist, and Neural Network) across five resampling strategies (Baseline, Class Weight, Undersampling, SMOTEENN, KMEANS SMOTE) and five levels of synthetic class imbalance (1%, 2%, 5%, 10%, 20%).

The Jupyter notebook, 'Notebook_Final.ipynb' Implements the full pipeline. For improved resources on the HPC cluster we rewrote the notebook into a multiple steps with various paython files.

---

## Methodology

### Dataset

The LendingClub accepted-loans dataset is filtered to loans with status `Fully Paid` or `Charged Off`. The original minority fraction (defaults) is approximately 20%. To simulate realistic financial imbalance, the dataset is undersampled to five target minority fractions: **1%, 2%, 5%, 10%, 20%**.

### Pipeline (as implemented in `Notebook_Final.ipynb`)

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
                                └─ Adversarial attacks: GA search + ZOO (ART), compared
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

### Adversarial attacks

Two attacks are run and compared under the **same threat model**: only three **actionable** features may change — `loan_amnt` (decrease ≤ 25%), `annual_inc` (increase ≤ 10%) and `revol_bal` (decrease ≤ 25%), as set by `ACTION_SPECS` — and at most `MAX_CHANGED_ACTIONS` (2) of them at once. Dependent columns are **repaired** to stay self-consistent: `installment` tracks the loan amount, `dti` tracks income, and `revol_util` tracks the balance. Both attacks search the same normalised `[0, 1]` action space and are evaluated on the **same sample** of eligible defaults, so their metrics are directly comparable. Targets are chosen with an **F1-optimal decision threshold** calibrated on a held-out validation split; an attack succeeds when it drives the predicted default probability below that threshold. The primary metric is **attack success rate**.

**1. PermuteAttack-inspired GA** (`permute_attack` in `src/pipeline_utils.py`), after Hashemi & Fathi (2020) [[1]](#references). A genetic search evolves a population of action vectors with elitism, uniform crossover and Gaussian mutation over several restarts; the objective rewards a low default probability and penalises perturbation size (`PROXIMITY_PENALTY`). The most proximal flipping candidate is reported. *(Referred to as "GA" in the result columns and plots.)*

**2. ZOO-inspired finite-difference attack** (`zoo_attack` in `src/pipeline_utils.py`), after Chen et al. (2017) [[2]](#references). A black-box attack that estimates the objective's gradient by central finite differences over `predict_proba` queries and steps the action vector with an Adam update, re-projected into the constraint set each iteration, across several restarts.

Both attacks are pure-NumPy and query the model only through `predict_proba`, so they apply uniformly to every model type. They are re-implementations inspired by the cited works rather than the original code. Per-experiment results carry method-suffixed columns, e.g. `Attack success rate (GA)` and `Attack success rate (ZOO)`; the comparison plot overlays both (colour = model, solid = GA, dashed = ZOO).

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
├── Notebook_Final.ipynb       # Original exploratory notebook
├── HPC/
│   ├── src/
│   │   ├── config.py              # All constants: imbalance fracs, model names, paths
│   │   ├── preprocessing.py       # Data loading, feature engineering, bias check
│   │   ├── samplers_def.py        # Sampler objects (with n_jobs)
│   │   ├── models_def.py          # Model constructors + apply_class_weight
│   │   └── pipeline_utils.py      # CV, SHAP, DIR, adversarial attack logic
│   ├── setup.py                   # Phase A: preprocess data + run gridsearch
│   ├── main.py                    # Phase B: single contained experiment (HPC entry point)
│   ├── aggregate.py               # Phase C: collect results → CSVs + plots
│   ├── hq_task.sh                 # HyperQueue task wrapper
│   ├── hyperq_job_hpc.sh          # SLURM job that starts HQ workers
│   └── setup_job.sh               # SLURM job for Phase A (optional)
├── requirements.txt           # Python dependencies (local + HPC, incl. pyarrow)
├── Data/
│   └── accepted.csv           # Raw LendingClub data
├── data/processed/            # Generated by setup.py, run from the repo root
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
├── logs/                      # HQ / SLURM task stdout & stderr
├── pictures/                  # Static figures referenced by the notebook
├── milestones/                # Course deliverables (proposal + milestone PDFs)
└── outputs/                   # Generated by aggregate.py, run from the repo root
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
jupyter notebook Notebook_Final.ipynb
```

`accepted.csv` is already committed under `Data/`. The notebook runs the full pipeline end-to-end in a single process and produces `model_results.csv` and `adversarial_attack_results.csv` in the project root.

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
HPC/setup.py
  ├─ load_and_clean()  +  engineer_features()
  ├─ create_imbalanced_dataset() × 5 fractions  → parquet files
  ├─ bias check (KS test) per fraction          → bias_*.json
  └─ GridSearchCV (n_jobs=-1) × 3 models        → best_params.json
```

Outputs saved to `data/processed/` and `config/best_params.json`. Only needs to run once before any Phase B jobs.

### Phase B — Experiments (100 parallel HQ tasks)

```
HPC/main.py --experiment_id=N --n_cores=16
  ├─ Load X_full_{frac}.parquet + best_params.json
  ├─ 80/20 train–test split (fixed seed)
  ├─ build_pipeline(model, sampler)
  ├─ DIR on resampled training labels  → dir.csv
  ├─ 5-fold stratified CV              → metrics.csv
  ├─ SHAP (last CV fold)               → shap.csv
  ├─ Refit on full train set
  └─ Adversarial attack                → attacks.csv
```

Each task is mapped from `experiment_id` → `(frac, model, sampler)` via a fixed flat index. HyperQueue pins each task to 16 cores (`--cpus=16`, set via `N_CORES` in `hq_task.sh`), so 192/16 = 12 tasks run concurrently on a single genoa node.

### Phase C — Aggregation

```
HPC/aggregate.py
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

# 2. (Optional) Run Phase A as a SLURM job (all paths below are relative to the repo root)
sbatch HPC/setup_job.sh
# Or interactively: python HPC/setup.py

# 3. Submit the 100-task HQ array
hq submit \
    --array 0-99 \
    --stdout=logs/%{TASK_ID}.out \
    --stderr=logs/%{TASK_ID}.err \
    --pin taskset --cpus=16 \
    --time-limit=1300min \
    HPC/hq_task.sh

# 4. Submit the SLURM worker job
sbatch HPC/hyperq_job_hpc.sh

# 5. Monitor
squeue -u $USER
hq job progress 1

# 6. After all tasks finish, collect results
python HPC/aggregate.py
```

### Resource calculation

| Parameter | Value |
|---|---|
| Cores per node | 192 |
| Cores per task (`--cpus`) | 16 |
| Concurrent tasks | 192 ÷ 16 = **12** |
| Total tasks | 5 × 4 × 5 = **100** |
| Parallel rounds | ⌈100 ÷ 12⌉ = **9** |
| Worst-case time/task | ~90 min (SMOTE-ENN + Random Forest) |
| Estimated wall time | ~13.5 h |
| Requested (`#SBATCH --time`) | 20:00:00 |

To change cores-per-task, edit `N_CORES` in `hq_task.sh` **and** `--cpus` in the `hq submit` command — both must match.

---

## Configuration

All tuneable constants live in `src/config.py`:

| Constant | Default | Description |
|---|---|---|
| `MINORITY_FRACS` | `[0.01, 0.02, 0.05, 0.1, 0.2]` | Target minority fractions |
| `N_FOLDS` | `5` | CV folds |
| `N_ATTACK_CASES` | `100` | Instances attacked per experiment |
| `ACTION_SPECS` | see config | Mutable features + allowed direction & max relative change |
| `MAX_CHANGED_ACTIONS` | `2` | Max features changed at once (sparsity) |
| `PROXIMITY_PENALTY` | `0.02` | Weight on perturbation size in the attack objective |
| `GA_POP_SIZE` | `50` | GA population size per generation |
| `GA_GENERATIONS` | `100` | Max generations per restart (early-stops once flipped) |
| `GA_RESTARTS` | `5` | Independent GA restarts per attacked instance |
| `GA_MUTATION_PROB` / `GA_MUTATION_SCALE` | `0.20` / `0.15` | GA mutation rate and Gaussian scale |
| `GA_ELITE_FRAC` | `0.20` | Fraction of population carried over as elites |
| `ZOO_MAX_ITER` | `200` | ZOO finite-difference iterations per restart |
| `ZOO_RESTARTS` | `3` | Independent ZOO restarts per attacked instance |
| `ZOO_LEARNING_RATE` | `0.03` | ZOO Adam step size |
| `ZOO_FD_STEP` | `0.05` | ZOO central finite-difference step |
| `N_SHAP_MAX` | `200` | Test samples for tree/linear SHAP |
| `N_SHAP_NN` | `20` | Test samples for NN KernelExplainer |
| `MUTABLE_FEATURES` | `list(ACTION_SPECS)` | Features the attacker can modify (highlights SHAP plots) |

---

## Dependencies

Install with:

```bash
pip install -r requirements.txt
```

A single `requirements.txt` covers both local and HPC use (it already includes `pyarrow` for the parquet data files).

Key packages: `scikit-learn 1.9`, `imbalanced-learn 0.14`, `xgboost 3.2`, `shap 0.52`, `skorch 1.4`, `torch 2.12`. The adversarial attacks are pure-NumPy, so `adversarial-robustness-toolbox` / `pygad` are not required.

---

## References

The two adversarial attacks implemented here are based on:

1. **PermuteAttack (GA counterfactual search).** M. Hashemi and A. Fathi, *"PermuteAttack: Counterfactual Explanation of Machine Learning Credit Scorecards,"* arXiv:2008.10138, 2020. Paper: <https://arxiv.org/abs/2008.10138> · Code: <https://github.com/masoudhashemi/PermuteAttack/>

2. **ZOO (Zeroth-Order Optimization black-box attack).** P.-Y. Chen, H. Zhang, Y. Sharma, J. Yi, and C.-J. Hsieh, *"ZOO: Zeroth Order Optimization Based Black-box Attacks to Deep Neural Networks without Training Substitute Models,"* in *Proc. 10th ACM Workshop on Artificial Intelligence and Security (AISec '17)*, 2017, pp. 15–26. <https://doi.org/10.1145/3128572.3140448>

<details>
<summary>BibTeX</summary>

```bibtex
@misc{hashemi2020permuteattackcounterfactualexplanationmachine,
      title={PermuteAttack: Counterfactual Explanation of Machine Learning Credit Scorecards},
      author={Masoud Hashemi and Ali Fathi},
      year={2020},
      eprint={2008.10138},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2008.10138},
}

@inproceedings{10.1145/3128572.3140448,
      author = {Chen, Pin-Yu and Zhang, Huan and Sharma, Yash and Yi, Jinfeng and Hsieh, Cho-Jui},
      title = {ZOO: Zeroth Order Optimization Based Black-box Attacks to Deep Neural Networks without Training Substitute Models},
      year = {2017},
      isbn = {9781450352024},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      url = {https://doi.org/10.1145/3128572.3140448},
      doi = {10.1145/3128572.3140448},
      booktitle = {Proceedings of the 10th ACM Workshop on Artificial Intelligence and Security},
      pages = {15--26},
      numpages = {12},
      keywords = {substitute model, neural network, deep learning, black-box attack, adversarial learning},
      location = {Dallas, Texas, USA},
      series = {AISec '17}
}
```

</details>
