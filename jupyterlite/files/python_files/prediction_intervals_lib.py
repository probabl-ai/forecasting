from feature_engineering_lib import feature_engineering_outputs, time_range
from multiple_horizons_prediction_lib import neg_mape
from single_horizon_prediction_lib import TimeSeriesSplitter, get_cv_results, get_hgb_params
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import d2_pinball_score, mean_absolute_percentage_error
from tutorial_helpers import (
    binned_coverage,
    collect_cv_predictions,
    plot_lorenz_curve,
    plot_predictions,
    plot_reliability_diagram,
    plot_residuals_vs_predicted,
)
from tutorial_helpers import binned_coverage, coverage
import altair
import functools
import importlib
import numpy as np
import polars as pl
import skrub
import tutorial_helpers
import warnings


def split_by_quantile(pred):
    quantile_cols = {}
    for c in pred.columns:
        quantile_cols.setdefault(c.split("__")[1], []).append(c)
    return {
        q: pred.select(cols).rename(lambda c: c.split("__")[0]) for q, cols in quantile_cols.items()
    }


def neg_mape_median(y_true, y_pred):
    "return the negative MAPE for the prediction for the quantile 0.5"
    return neg_mape(y_true, split_by_quantile(y_pred)["q_0.5"])


def neg_mape_median_scorer(estimator, X, y):
    return neg_mape_median(y, estimator.predict(X))


def pinball(y_true, y_pred):
    quantile_predictions = split_by_quantile(y_pred)
    scores = {}
    for q, q_pred in quantile_predictions.items():
        scores[f"d2_pinball_score__average__{q}"] = d2_pinball_score(
            y_true, q_pred, alpha=float(q.removeprefix("q_"))
        )
        detail = d2_pinball_score(y_true, q_pred, multioutput="raw_values")
        scores.update(
            {f"d2_pinball_score__{c}__{q}": float(s) for c, s in zip(y_true.columns, detail)}
        )
    return scores


def pinball_scorer(estimator, X, y):
    return pinball(y, estimator.predict(X))


class HGBQuantileRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, quantiles=(0.05, 0.5, 0.95), hgb_params=None):
        self.quantiles = quantiles
        self.hgb_params = hgb_params

    def fit(self, X, y):
        self.quantiles_ = sorted(self.quantiles)
        params = (self.hgb_params or {}) | {"loss": "quantile"}
        self.estimators_ = {
            q: HistGradientBoostingRegressor(quantile=q, **params).fit(X, y)
            for q in self.quantiles_
        }
        return self

    def predict(self, X):
        result = np.asarray([e.predict(X) for e in self.estimators_.values()])
        result.sort(axis=0)
        return pl.DataFrame(result, schema=[f"q_{q}" for q in self.quantiles_])


def concat_horizons(all_pred, mode=skrub.eval_mode()):
    """
    Consolidate predictions of models for different horizons in one dataframe.
    """
    if mode == "fit":
        return all_pred
    return pl.concat(
        [v.rename(f"{h}h__{{}}".format) for h, v in all_pred.items()], how="horizontal"
    )


def make_multi_horizon_pred(features, y, regressor):
    """
    Create a full DataOp for predicting the specified horizons and quantiles.
    """
    predictions = {
        h: feat.skb.drop(["prediction_time", "target_time"])
        .skb.apply(regressor, y=y[f"{h}h"])
        .skb.set_name(f"pred_{h}h")
        for h, feat in features.items()
    }
    return skrub.deferred(concat_horizons)(predictions)
