import os

RANDOM_STATE = 42

MINORITY_FRACS = [0.01, 0.02, 0.05, 0.1, 0.2]

# Imputation: median SimpleImputer used in every preprocessing pipeline.
# (Replaced the KNNImputer, which was too slow on the full dataset.)

# KMeansSMOTE clustering: number of clusters fixed at 30 via the elbow method
# (distortion/inertia, see Advesarial_attacks.ipynb), so we hardcode it here
# instead of re-searching on the cluster every run.
KMEANS_SMOTE_N_CLUSTERS = 30

SAMPLER_NAMES = [
    "Baseline",
    "RandomUnderSampler",
    "SMOTEENN",
    "KMeansSMOTE",
    "ClassWeight",
]

MODEL_NAMES = [
    "XGBClassifier",
    "LogisticRegression",
    "RandomForestClassifier",
    "NeuralNetClassifier",
]

N_FOLDS = 5

# SHAP settings
N_SHAP_MAX = 200
N_SHAP_NN = 20
N_SHAP_NN_BG = 50

# ---------------------------------------------------------------------------
# Adversarial attack settings
# ---------------------------------------------------------------------------
# Both attacks search for a realistic, *actionable* counterfactual: a minimal,
# direction-constrained change to a few mutable loan features that flips the
# model's decision from "default" to "repaid".
DEFAULT_CLASS = 1            # unfavourable label the attacker tries to escape
N_ATTACK_CASES = 100         # attacked instances per experiment
MAX_CHANGED_ACTIONS = 2      # sparsity: at most this many features may change

# Per-feature action constraints: allowed direction and max relative change.
ACTION_SPECS = {
    "loan_amnt":  {"direction": "decrease", "max_change": 0.25},
    "annual_inc": {"direction": "increase", "max_change": 0.10},
    "revol_bal":  {"direction": "decrease", "max_change": 0.25},
}

# Features stored on a log1p scale in model space (decoded when computing bounds).
LOG_FEATURES = {"annual_inc", "revol_bal"}

# A row must have these columns non-null to be an eligible attack target.
ATTACK_REQUIRED_COLUMNS = [
    "loan_amnt", "annual_inc", "revol_bal", "installment", "dti", "revol_util",
]

# Features the attacker can directly modify (also used to highlight SHAP plots).
MUTABLE_FEATURES = list(ACTION_SPECS)

LOWER_Q = 0.01               # training quantiles that clamp the action bounds
UPPER_Q = 0.99
PROXIMITY_PENALTY = 0.02     # weight on perturbation size in the attack objective

# PermuteAttack-inspired genetic algorithm.
GA_POP_SIZE = 50
GA_GENERATIONS = 100
GA_RESTARTS = 5
GA_MUTATION_PROB = 0.20
GA_MUTATION_SCALE = 0.15
GA_ELITE_FRAC = 0.20

# ZOO-inspired finite-difference attack (Adam-optimised gradient estimate).
ZOO_MAX_ITER = 200
ZOO_RESTARTS = 3
ZOO_LEARNING_RATE = 0.03
ZOO_FD_STEP = 0.05

# Models for which the ZOO attack is skipped. ZOO issues thousands of single-row
# predict_proba queries per instance (~4200 with the settings above); on
# RandomForest each query fans out over joblib, so the full ZOO budget blows
# through the job wall-time. GA still runs for these models; their ZOO columns
# are left empty in the results. Empty this set to re-enable ZOO everywhere.
ZOO_SKIP_MODELS = {"RandomForestClassifier"}

FAVOURABLE_LABEL = 0

DIR_GROUP_SPECS = {
    "emp_length": {
        "method": "threshold",
        "threshold": 5,
        "unprivileged_if": "lower",
        "unprivileged_name": "emp_length < 5 years",
        "privileged_name": "emp_length >= 5 years",
    },
    "annual_inc": {
        "method": "quantile",
        "quantile": 0.50,
        "unprivileged_if": "lower",
        "unprivileged_name": "lower annual income",
        "privileged_name": "higher annual income",
    },
}

# Paths
DATA_RAW = "Data/accepted.csv"

# Processed data and results are saved to LOCAL_DATA_DIR if set, otherwise to the default paths below.
DATA_PROCESSED_DIR = os.environ.get("LOCAL_DATA_DIR", "data/processed")

RESULTS_DIR = "results"
OUTPUTS_DIR = "outputs"
PARAMS_FILE = "config/best_params.json"
