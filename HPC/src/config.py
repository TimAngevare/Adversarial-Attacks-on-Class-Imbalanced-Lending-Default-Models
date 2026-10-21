# Define the HPC experiment settings and paths.
import os

RANDOM_STATE = 42

MINORITY_FRACS = [0.01, 0.02, 0.05, 0.1, 0.2]

# Reserve the test pool before constructing prevalence datasets
TEST_SIZE = 0.20
# Hold out part of the training pool for final-model threshold selection
VALIDATION_SIZE = 0.20
# Tune hyperparameters at this default prevalence
TUNING_FRACTION = 0.1

# Set the number of KMeans-SMOTE clusters
KMEANS_SMOTE_N_CLUSTERS = 30
# Use a fixed density exponent to avoid unstable cluster weights
KMEANS_SMOTE_DENSITY_EXPONENT = 2
# Keep SMOTE replacement disabled unless explicitly enabled
KMEANS_SMOTE_ALLOW_FALLBACK = False
# Label outputs separately when the optional replacement is used
KMEANS_SMOTE_FALLBACK_LABEL = "KMeansSMOTE (SMOTE fallback)"

# Keep sampler order fixed because experiment IDs determine attack seeds
SAMPLER_NAMES = [
    "Baseline",
    "ClassWeight",
    "RandomUnderSampler",
    "SMOTEENN",
    "KMeansSMOTE",
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

# Data preparation (notebook Chapter 2)
MISSING_THRESHOLD = 0.10
NEAR_CONSTANT_THRESHOLD = 0.90
HIGH_CARDINALITY_COUNT = 10_000

CORR_DROP_COLS = [
    "out_prncp_inv", "total_pymnt_inv", "funded_amnt", "funded_amnt_inv", "num_sats",
    "num_rev_tl_bal_gt_0", "tot_hi_cred_lim", "total_il_high_credit_limit", "bc_util",
    "percent_bc_gt_75", "bc_open_to_buy", "total_bc_limit", "num_op_rev_tl", "avg_cur_bal",
    "last_fico_range_high", "last_fico_range_low", "num_bc_sats", "num_actv_bc_tl",
    "total_rev_hi_lim", "num_bc_tl", "num_tl_30dpd", "collection_recovery_fee",
]
DROP_FOR_LEAKAGE = [
    "total_pymnt", "total_rec_prncp", "total_rec_int", "last_pymnt_d",
    "last_pymnt_amnt", "last_credit_pull_d", "issue_d", "recoveries",
]
# Exclude LendingClub pricing from the main predictor set
DROP_FOR_PROXY_RISK = ["int_rate", "grade", "sub_grade", "installment"]
PRICING_COLUMNS = ["int_rate", "installment"]
DROP_MANUAL = [
    "zip_code", "tot_coll_amt", "tot_cur_bal", "total_bal_ex_mort",
    "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op",
    "mo_sin_rcnt_tl", "mths_since_recent_bc", "mths_since_recent_inq",
    "num_accts_ever_120_pd", "num_actv_rev_tl", "num_bc_tl", "num_il_tl",
    "num_rev_accts", "num_tl_30dpd", "num_tl_op_past_12m", "num_tl_90g_dpd_24m",
    "pct_tl_nvr_dlq",
]
EMP_LENGTH_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9, "10+ years": 10,
}

# Fairness (notebook Section 6.3)
FAVOURABLE_LABEL = 0
DEFAULT_CLASS = 1            # unfavourable label the attacker tries to escape
N_BOOTSTRAP = 1000
FAIRNESS_CI = 0.95

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

# Adversarial attacks (notebook Chapter 7)
# Sample eligible defaults for GA and ZOO; references use all eligible defaults
N_ATTACK_CASES = 100         # None = GA and ZOO on every eligible default
MAX_CHANGED_ACTIONS = 2

ACTION_SPECS = {
    "loan_amnt":  {"direction": "decrease", "max_change": 0.25},
    "annual_inc": {"direction": "increase", "max_change": 0.10},
    "revol_bal":  {"direction": "decrease", "max_change": 0.25},
}
LOG_FEATURES = {"annual_inc", "revol_bal"}
ATTACK_REQUIRED_COLUMNS = list(ACTION_SPECS)
MUTABLE_FEATURES = list(ACTION_SPECS)

# Update dependent features only when their parent changes
DEPENDENT_FEATURES = {
    "annual_inc": [("dti", "inverse")],
    "revol_bal": [("revol_util", "proportional")],
}
CHANGE_TOL = 1e-9

LOWER_Q = 0.01
UPPER_Q = 0.99
PROXIMITY_PENALTY = 0.02
ATTACK_OBJECTIVE = "logit"

GA_POP_SIZE = 50
GA_GENERATIONS = 100
GA_RESTARTS = 5
GA_MUTATION_PROB = 0.20
GA_MUTATION_SCALE = 0.15
GA_ELITE_FRAC = 0.20

ZOO_MAX_ITER = 200
ZOO_RESTARTS = 3
ZOO_LEARNING_RATE = 0.03
ZOO_FD_STEP = 0.05

GRID_LEVELS = 21
GRID_MAX_ROWS_PER_CALL = 200_000

# Set the log-odds tolerance for reference verification and threshold ties
SEPARABILITY_TOL = 1e-6

# List any models to exclude from ZOO
ZOO_SKIP_MODELS = set()

# Use fresh workers for XGBoost, LR and NN; fork RF workers to share model memory
ATTACK_START_METHOD = {
    "XGBClassifier": "spawn",
    "LogisticRegression": "spawn",
    "RandomForestClassifier": "fork",
    "NeuralNetClassifier": "spawn",
}

# Pricing sensitivity (notebook Section 6.1.6): separate HQ array
PRICING_VARIANTS = {
    "Pre-pricing (main)": [],
    "With installment (previous feature set)": ["installment"],
    "Post-pricing (int_rate included)": ["int_rate"],
}
PRICING_SENSITIVITY_MODELS = ["LogisticRegression"]
PRICING_SENSITIVITY_SAMPLERS = ["Baseline", "ClassWeight"]

# Paths
DATA_RAW = "Data/accepted.csv"

# Read node-local processed data when LOCAL_DATA_DIR is set
DATA_PROCESSED_DIR = os.environ.get("LOCAL_DATA_DIR", "data/processed")

# Store task outputs separately from processed data and collected results
RESULTS_DIR = "results_v2"
OUTPUTS_DIR = "outputs"
PARAMS_FILE = "config/best_params.json"
