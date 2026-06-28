from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import KMeansSMOTE
from sklearn.cluster import MiniBatchKMeans

from src.config import RANDOM_STATE, KMEANS_SMOTE_N_CLUSTERS


def get_samplers(n_jobs: int = 1) -> dict:
    return {
        "Baseline": None,
        "RandomUnderSampler": RandomUnderSampler(
            sampling_strategy="majority",
            random_state=RANDOM_STATE,
        ),
        "SMOTEENN": SMOTEENN(
            sampling_strategy="minority",
            random_state=RANDOM_STATE,
            n_jobs=n_jobs,
        ),
        "KMeansSMOTE": KMeansSMOTE(
            sampling_strategy="minority",
            random_state=RANDOM_STATE,
            cluster_balance_threshold=0.01,
            kmeans_estimator=MiniBatchKMeans(
                n_clusters=KMEANS_SMOTE_N_CLUSTERS,
                random_state=RANDOM_STATE,
                n_init=3,
            ),
            n_jobs=n_jobs,
        ),
        "ClassWeight": None,
    }
