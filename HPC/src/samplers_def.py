import warnings

from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import KMeansSMOTE, SMOTE
from sklearn.cluster import MiniBatchKMeans

from src.config import (
    RANDOM_STATE, KMEANS_SMOTE_N_CLUSTERS, KMEANS_SMOTE_DENSITY_EXPONENT,
    KMEANS_SMOTE_ALLOW_FALLBACK,
)

# Record reasons for any enabled replacement of KMeans-SMOTE by SMOTE
FALLBACK_EVENTS = []


# Record the optional SMOTE replacement when KMeans-SMOTE fails
class GuardedKMeansSMOTE(KMeansSMOTE):

    # Resample with KMeans-SMOTE and handle the optional replacement
    def fit_resample(self, X, y, **params):
        try:
            return super().fit_resample(X, y, **params)
        except (ValueError, RuntimeError) as exc:
            if not KMEANS_SMOTE_ALLOW_FALLBACK:
                raise
            FALLBACK_EVENTS.append(str(exc))
            warnings.warn(f"KMeansSMOTE failed ({exc}); replaced by SMOTE.", RuntimeWarning)
            fallback = SMOTE(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                k_neighbors=self.k_neighbors,
            )
            return fallback.fit_resample(X, y, **params)


# Define the five imbalance-handling strategies
def get_samplers(n_jobs: int = 1) -> dict:
    return {
        "Baseline": None,
        "ClassWeight": None,
        "RandomUnderSampler": RandomUnderSampler(
            sampling_strategy="majority",
            random_state=RANDOM_STATE,
        ),
        "SMOTEENN": SMOTEENN(
            sampling_strategy="minority",
            random_state=RANDOM_STATE,
            n_jobs=n_jobs,
        ),
        "KMeansSMOTE": GuardedKMeansSMOTE(
            sampling_strategy="minority",
            random_state=RANDOM_STATE,
            cluster_balance_threshold=0.01,
            density_exponent=KMEANS_SMOTE_DENSITY_EXPONENT,
            kmeans_estimator=MiniBatchKMeans(
                n_clusters=KMEANS_SMOTE_N_CLUSTERS,
                random_state=RANDOM_STATE,
                n_init=3,
            ),
            n_jobs=n_jobs,
        ),
    }
