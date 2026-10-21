import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, LRScheduler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import RANDOM_STATE


# Define two 64-unit ReLU layers with dropout
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 64),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, X):
        return self.net(X)


# Build the four classifiers using the selected parameters and core count
def get_models(n_features: int, n_cores: int = 1, best_params: dict = None) -> dict:

    params = best_params or {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    xgb_kw = dict(params.get("XGBClassifier", {}))
    xgb_kw.setdefault("random_state", RANDOM_STATE)
    xgb_kw.setdefault("eval_metric", "logloss")
    xgb_kw.pop("nthread", None)
    xgb_kw["n_jobs"] = n_cores

    lr_kw = dict(params.get("LogisticRegression", {}))
    lr_kw.setdefault("random_state", RANDOM_STATE)

    rf_kw = dict(params.get("RandomForestClassifier", {}))
    rf_kw.setdefault("random_state", RANDOM_STATE)
    rf_kw["n_jobs"] = n_cores

    net = NeuralNetClassifier(
        NeuralNetwork,
        module__input_dim=n_features,
        criterion=nn.CrossEntropyLoss,
        max_epochs=100,
        lr=0.001,
        device=device,
        optimizer=Adam,
        batch_size=512,
        iterator_train__shuffle=True,
        iterator_train__num_workers=0,
        callbacks=[
            EarlyStopping(monitor="valid_loss", patience=10),
            LRScheduler(policy=ReduceLROnPlateau, monitor="valid_loss"),
        ],
        verbose=0,
    )

    return {
        "XGBClassifier": XGBClassifier(**xgb_kw),
        "LogisticRegression": LogisticRegression(**lr_kw),
        "RandomForestClassifier": RandomForestClassifier(**rf_kw),
        "NeuralNetClassifier": net,
    }


# Set or clear class weights for the selected model
def apply_class_weight(estimator, sampler_name: str, y) -> object:
    y_arr = np.asarray(y)
    name = estimator.__class__.__name__
    if sampler_name == "ClassWeight":
        if name in ("LogisticRegression", "RandomForestClassifier"):
            estimator.set_params(class_weight="balanced")
        elif name == "XGBClassifier":
            ratio = float((y_arr == 0).sum()) / float((y_arr == 1).sum())
            estimator.set_params(scale_pos_weight=ratio)
        elif name == "NeuralNetClassifier":
            _, counts = np.unique(y_arr, return_counts=True)
            w = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32)
            estimator.set_params(criterion__weight=w)
    else:
        if name in ("LogisticRegression", "RandomForestClassifier"):
            estimator.set_params(class_weight=None)
        elif name == "XGBClassifier":
            estimator.set_params(scale_pos_weight=1.0)
        elif name == "NeuralNetClassifier":
            estimator.set_params(criterion__weight=None)
    return estimator


# Set prediction thread counts for the fitted model
def set_model_threads(estimator, n_threads: int) -> None:
    name = estimator.__class__.__name__
    if name == "XGBClassifier":
        estimator.set_params(n_jobs=n_threads)
        try:
            estimator.get_booster().set_param({"nthread": n_threads})
        except Exception:
            pass
    elif name == "RandomForestClassifier":
        estimator.set_params(n_jobs=n_threads)
    elif name == "NeuralNetClassifier":
        torch.set_num_threads(n_threads)
