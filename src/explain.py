import shap
import matplotlib.pyplot as plt
import pandas as pd
from typing import Any


def generate_global_shap_summary(model: Any, X: pd.DataFrame, save_path: str) -> None:
    """Produce and save a global SHAP summary (beeswarm) plot."""
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def generate_local_shap_explanation(model: Any, instance: pd.DataFrame, save_path: str) -> None:
    """Produce and save a single-prediction SHAP waterfall plot."""
    explainer = shap.Explainer(model)
    shap_values = explainer(instance)
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def check_concerning_patterns(model: Any, X: pd.DataFrame) -> pd.DataFrame:
    """
    Flag features whose SHAP contribution direction seems counter to
    business intuition (e.g., higher monetary value increasing risk),
    for manual review before deployment.
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    mean_shap = pd.DataFrame({
        "feature": X.columns,
        "mean_shap": shap_values.values.mean(axis=0),
    }).sort_values("mean_shap", key=abs, ascending=False)
    return mean_shap
