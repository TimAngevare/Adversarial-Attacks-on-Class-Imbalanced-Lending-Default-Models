import warnings

from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import KMeansSMOTE, SMOTE
from sklearn.cluster import MiniBatchKMeans

from src.config import RANDOM_STATE, KMEANS_SMOTE_N_CLUSTERS


class SafeKMeansSMOTE(KMeansSMOTE):

    def fit_resample(self, X, y, **params):
        try:
            return super().fit_resample(X, y, **params)
        except (ValueError, RuntimeError) as exc:
            warnings.warn(
                f"KMeansSMOTE failed ({exc}); falling back to SMOTE.",
                RuntimeWarning,
            )
            fallback = SMOTE(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
            return fallback.fit_resample(X, y, **params)


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
        "KMeansSMOTE": SafeKMeansSMOTE(
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
