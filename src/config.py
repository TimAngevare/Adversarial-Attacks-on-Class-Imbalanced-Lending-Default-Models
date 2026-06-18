RANDOM_STATE = 42

MINORITY_FRACS = [0.01, 0.02, 0.05, 0.1, 0.2]

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

# Models to optimise
GRIDSEARCH_MODELS = ["XGBClassifier", "LogisticRegression", "RandomForestClassifier"]

N_FOLDS = 5

# SHAP settings
N_SHAP_MAX = 200
N_SHAP_NN = 20
N_SHAP_NN_BG = 50

# Adversarial attack settings
N_ATTACK_CASES = 50

# Genetic-algorithm (GA) perturbation attack settings.
# The attack searches for a minimal-change counterfactual that flips the
# prediction by evolving a population of perturbed copies of the input.
GA_SOL_PER_POP = 30          # population size per generation
GA_NUM_PARENTS_MATING = 10   # parents selected for crossover each generation
GA_NUM_GENERATIONS = 100     # max generations per run (early-stops once flipped)
GA_N_RUNS = 5                # independent GA restarts per attacked instance
GA_BETA = 0.95               # proximity-constraint relaxation factor (0<beta<1)

# ZOO (Zeroth-Order Optimization) attack settings — Adversarial Robustness Toolbox.
# A black-box, gradient-estimation attack: it numerically estimates gradients of
# the model's output through repeated predict_proba queries and perturbs the input
# along them. Like the GA attack it is constrained to the mutable features only.
ZOO_MAX_ITER = 40            # optimisation iterations per instance
ZOO_LEARNING_RATE = 1e-1     # step size for the estimated-gradient update
ZOO_BINARY_SEARCH_STEPS = 1  # search steps over the proximity trade-off constant
ZOO_INITIAL_CONST = 1e-3     # initial proximity/attack trade-off constant
ZOO_NB_PARALLEL = None       # coords perturbed in parallel (None -> n mutable feats)

MUTABLE_FEATURES = [
    "loan_amnt", "annual_inc", "dti", "revol_bal",
    "revol_util", "open_acc", "total_acc",
]

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
DATA_PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results"
OUTPUTS_DIR = "outputs"
PARAMS_FILE = "config/best_params.json"
