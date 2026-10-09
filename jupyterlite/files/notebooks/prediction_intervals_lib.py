from datetime import UTC, timedelta
from datetime import datetime
from feature_engineering_lib import feature_engineering_outputs, time_range
from single_horizon_prediction_lib import TimeSeriesSplitter
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import d2_pinball_score, mean_absolute_percentage_error
from tutorial_helpers import (
    binned_coverage,
    collect_cv_predictions,
    plot_lorenz_curve,
    plot_reliability_diagram,
    plot_residuals_vs_predicted,
)
from tutorial_helpers import binned_coverage, coverage
import altair
import functools
import importlib
import numpy as np
import plotly.graph_objects as go
import polars as pl
import re
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


def neg_mape(y_true, y_pred, quantile_regression=False):
    if quantile_regression:
        quantile_predictions = split_by_quantile(y_pred)
        scores = {}
        for q, q_pred in quantile_predictions.items():
            q_neg_mape = neg_mape(y_true, q_pred, quantile_regression=False)
            scores.update({f"{k}__{q}": v for k, v in q_neg_mape.items()})
            if q == "q_0.5":
                # Pick the median if available for comparison with non-quantile
                # models
                scores.update(q_neg_mape)
        return scores
    average = mean_absolute_percentage_error(y_true, y_pred)
    detail = mean_absolute_percentage_error(y_true, y_pred, multioutput="raw_values")
    return {"neg_mape__average": -average} | {
        f"neg_mape__{c}": -float(s) for c, s in zip(y_true.columns, detail)
    }


def neg_mape_scorer(estimator, X, y, quantile_regression=False):
    return neg_mape(y, estimator.predict(X), quantile_regression=quantile_regression)


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
    Create a full DataOp for predicting the specified horizons.
    """
    predictions = {
        h: feat.skb.drop(["prediction_time", "target_time"])
        .skb.apply(regressor, y=y[f"{h}h"])
        .skb.set_name(f"pred_{h}h")
        for h, feat in features.items()
    }
    return skrub.deferred(concat_horizons)(predictions)


def concat_X_y_predictions(X_test, y_test, prediction):
    return pl.concat(
        [
            X_test,
            y_test,
            prediction.rename("pred_{}".format),
        ],
        how="horizontal",
    )


def cross_val_predict(data_op, environment=None):
    """
    Get cross-validated predictions for different horizons.
    """
    all_predictions, all_scores = [], []
    for i, split in enumerate(data_op.skb.iter_cv_splits(environment=environment)):
        learner = data_op.skb.make_learner().fit(split["train"])
        score, predictions = learner.score(split["test"], return_predictions=True)
        all_predictions.append(
            concat_X_y_predictions(
                split["X_test"], split["y_test"], predictions["predict"]
            ).with_columns(split=pl.lit(i)),
        )
        print(split["X_test"]["prediction_time"].min().isoformat())
        print(score)
        all_scores.append(score | {"split": i})
    all_predictions = pl.concat(all_predictions, how="vertical")
    all_scores = pl.DataFrame(all_scores)
    return all_predictions, all_scores


def plot_predictions(results, horizons=None, start="2025-03-01"):
    if start is not None:
        results = results.filter(
            pl.col("prediction_time") > datetime.fromisoformat(start).replace(tzinfo=UTC)
        )
    if horizons is None:
        horizons = sorted(
            {
                int(m.group(1))
                for c in results.columns
                if (m := re.match(r"^pred_(\d+)h.*$", c)) is not None
            }
        )
    fig = go.Figure()
    for i, h in enumerate(horizons):
        target_time = (results["prediction_time"] + timedelta(hours=h)).to_list()
        if not i:
            fig.add_trace(
                go.Scatter(
                    x=target_time,
                    y=results[f"{h}h"].to_list(),
                    mode="lines+markers",
                    line={"dash": "dash"},
                    name="true_load_mw",
                    hovertemplate="%{x|%Y-%m-%d} (%{x|%A}): %{y}<extra></extra>",
                )
            )
        for col in filter(lambda c: f"pred_{h}h" in c, results.columns):
            fig.add_trace(
                go.Scatter(
                    x=target_time,
                    y=results[col].to_list(),
                    mode="lines+markers",
                    name=col,
                    hovertemplate="%{x|%Y-%m-%d} (%{x|%A}): %{y}<extra></extra>",
                )
            )
    fig.update_layout(height=600, title="CV predicted load mw")
    return fig
