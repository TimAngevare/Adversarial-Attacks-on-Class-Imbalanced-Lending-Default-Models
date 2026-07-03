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
            # NB: SMOTE dropped the `n_jobs` parameter in imbalanced-learn >=0.12
            # (KMeansSMOTE/SMOTEENN still accept it), so forwarding it here makes
            # the fallback itself raise TypeError and the fit fails. Use
            # k_neighbors from the KMeansSMOTE config (default 2) so the fallback
            # stays valid on small minority clusters.
            fallback = SMOTE(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                k_neighbors=self.k_neighbors,
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
            # Pin the density exponent to a small constant. The default ("auto")
            # scales the exponent with the data; on our high-dimensional,
            # standardised features that makes `mean_distance ** exponent` overflow
            # to inf, so every cluster sparsity collapses to 0 and the normalisation
            # 0/0 -> NaN sample count. That NaN is exactly what triggered the SMOTE
            # fallback at the higher imbalance fractions. A fixed small exponent
            # keeps the sparser-cluster weighting while staying numerically stable.
            density_exponent=2,
            kmeans_estimator=MiniBatchKMeans(
                n_clusters=KMEANS_SMOTE_N_CLUSTERS,
                random_state=RANDOM_STATE,
                n_init=3,
            ),
            n_jobs=n_jobs,
        ),
        "ClassWeight": None,
    }
