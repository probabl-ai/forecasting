# %% [markdown]
#
# # Computing prediction intervals using quantile regression
#
# ## Environment setup
#
# We need to install some extra dependencies for this notebook if needed (when
# running jupyterlite).

# %%
# %pip install -q https://pypi.anaconda.org/ogrisel/simple/polars/1.24.0/polars-1.24.0-cp39-abi3-emscripten_3_1_58_wasm32.whl
# %pip install -q skrub altair holidays plotly nbformat

# %%
import functools
import importlib
import warnings

import altair
import numpy as np
import polars as pl
import skrub

import tutorial_helpers

importlib.reload(tutorial_helpers)

from feature_engineering_lib import feature_engineering_outputs, time_range
from multiple_horizons_prediction_lib import neg_mape
from single_horizon_prediction_lib import TimeSeriesSplitter, get_cv_results, get_hgb_params
from tutorial_helpers import (
    binned_coverage,
    collect_cv_predictions,
    plot_lorenz_curve,
    plot_predictions,
    plot_reliability_diagram,
    plot_residuals_vs_predicted,
)

# Ignore warnings from pkg_resources triggered by Python 3.13's multiprocessing.
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")


# %% [markdown]
# ### Define the quantile regressors
#
# In this section, we show how one can use a gradient boosting but modify the loss
# function to predict different quantiles and thus obtain an uncertainty quantification
# of the predictions.
#
# In terms of evaluation, we reuse the MAPE score. However, they it is not helpful
# to assess the reliability of quantile models. For this purpose, we use a derivate of
# the metric minimized by the quantile regressors: the pinball loss. We use the D2 score that is
# easier to interpret since the best possible score is bounded by 1 and a score of 0
# corresponds to constant predictions at the target quantile.

# %%
from sklearn.metrics import d2_pinball_score, mean_absolute_percentage_error


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


# %%
TIME_HORIZONS = (1, 12, 24)
features, y = feature_engineering_outputs(TIME_HORIZONS, TimeSeriesSplitter())

# %% [markdown]
#
# We follow a multiple regressor approach and define separate
# models per quantile as follows:
#
# - a model predicting the 5th percentile of the load
# - a model predicting the median of the load
# - a model predicting the 95th percentile of the load
#
# We evaluate the performance of the quantile regressors via cross-validation.

# %%
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor


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


quantiles = (0.05, 0.5, 0.95)

hgb_params = get_hgb_params()

hgb_q_regressor = HGBQuantileRegressor(quantiles=quantiles, hgb_params=hgb_params)


# %%
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


pred = (
    make_multi_horizon_pred(features, y, regressor=hgb_q_regressor)
    .skb.with_scoring(neg_mape_median_scorer)
    .skb.with_scoring(pinball_scorer)
)

pred

# %% [markdown]
#
# Let's first collect all the cross-validated predictions to make further inspection.


# %%
cv_predictions_hgbr = get_cv_results(pred)

# %% [markdown]
# Now, we can inspect the cross-validated predictions and plot them for the different quantiles.

# %%
plot_predictions(cv_predictions_hgbr[0], horizon=12, start="2023-01-01").show()

# %% [markdown]
# Now, let's collect the cross-validated predictions and plot the residual vs predicted
# values for the different models into a report.

# %%
cv_predictions_hgbr[0].head(5)


# %% [markdown]
#
# Focusing on the different D2 scores, we observe that each model minimize the D2 score
# associated to the target quantile that we set. For instance, the model predicting the
# 5th percentile obtained the highest D2 pinball score with `alpha=0.05`. It is expected
# but a confirmation of what loss each model minimizes.
#
# Now, let's collect the cross-validated predictions and plot the residual vs predicted
# values for the different models.

# %%
plot_residuals_vs_predicted(cv_predictions_hgbr[0], 1, quantile=0.05).interactive().properties(
    title=("Residuals vs Predicted Values from cross-validation predictions" " for quantile 0.05")
)

# %%
plot_residuals_vs_predicted(cv_predictions_hgbr[0], 1, quantile=0.5).interactive().properties(
    title=("Residuals vs Predicted Values from cross-validation predictions for median")
)

# %%
plot_residuals_vs_predicted(cv_predictions_hgbr[0], 1, quantile=0.95).interactive().properties(
    title=("Residuals vs Predicted Values from cross-validation predictions" " for quantile 0.95")
)

# %% [markdown]
#
# We observe an expected behaviour: the residuals are centered and symmetric around 0
# for the median model while not centered and biased for the 5th and 95th percentiles
# models.
#
# %% [markdown]
# Now, we assess if the actual coverage of the models is close to the target coverage of
# 90%. In addition, we compute the average width of the bands.


# %%
import altair

from tutorial_helpers import binned_coverage, coverage, mean_width

preds = cv_predictions_hgbr[0]
horizon = 1

# --- Overall coverage per fold ---
for (split_idx,), fold_df in preds.group_by("split", maintain_order=True):
    cov = coverage(
        fold_df[f"{horizon}h"].to_numpy(),
        fold_df[f"pred_{horizon}h__q_0.05"].to_numpy(),
        fold_df[f"pred_{horizon}h__q_0.95"].to_numpy(),
    )
    print(f"Split {split_idx}: {cov:.1%} coverage (90% interval)")

# --- Binned coverage plot ---
folds = [fold_df for (_,), fold_df in preds.group_by("split", maintain_order=True)]
binned = binned_coverage(
    y_true_folds=[f[f"{horizon}h"].to_numpy() for f in folds],
    y_quantile_low=[f[f"pred_{horizon}h__q_0.05"].to_numpy() for f in folds],
    y_quantile_high=[f[f"pred_{horizon}h__q_0.95"].to_numpy() for f in folds],
)

altair.Chart(binned).mark_line(point=True).encode(
    x=altair.X("bin_center:Q", title="True load (MW)"),
    y=altair.Y("coverage:Q", title="Coverage", scale=altair.Scale(domain=[0, 1])),
    color=altair.Color("fold_idx:N"),
).properties(title=f"Binned coverage — {horizon}h horizon, 90% interval")
# %% [markdown]
#
# We observe that the lower and higher bins, so low and high load, have the worse
# coverage with a high variability.
#
# ### Reliability diagrams and Lorenz curves for quantile regression

# %%
plot_reliability_diagram(
    cv_predictions_hgbr[0], 1, forecast_quantile=0.50
).interactive().properties(
    title="Reliability diagram for quantile 0.50 from cross-validation predictions"
)

# %%
plot_reliability_diagram(
    cv_predictions_hgbr[0], 1, forecast_quantile=0.05
).interactive().properties(
    title="Reliability diagram for quantile 0.05 from cross-validation predictions"
)

# %%
plot_reliability_diagram(
    cv_predictions_hgbr[0], 1, forecast_quantile=0.95
).interactive().properties(
    title="Reliability diagram for quantile 0.95 from cross-validation predictions"
)

# %%
plot_lorenz_curve(cv_predictions_hgbr[0], 1, quantile=0.50).interactive().properties(
    title="Lorenz curve for quantile 0.50 from cross-validation predictions"
)

# %%
plot_lorenz_curve(cv_predictions_hgbr[0], 1, quantile=0.05).interactive().properties(
    title="Lorenz curve for quantile 0.05 from cross-validation predictions"
)

# %%
plot_lorenz_curve(cv_predictions_hgbr[0], 1, quantile=0.95).interactive().properties(
    title="Lorenz curve for quantile 0.95 from cross-validation predictions"
)

# %% [markdown]
# ## Post-hoc interval recalibration
#
# The reliability diagrams and the binned coverage show that the raw 90% interval is
# not equally reliable everywhere: coverage degrades for low and high loads. We
# compare two complementary corrections of the intervals:
#
# - A **reliability-based recalibrator**. For each quantile level tau, we bin the
#   out-of-fold predictions and measure, in each bin, the actual tau-quantile of the
#   residuals. This is the shift in MW that would make the bin calibrated. A smooth
#   spline model maps the predicted value to this shift, so the correction depends
#   on the load level. Adapted from
#   [`calibration.py`](https://github.com/probabl-ai/skore-example-electricity-load/blob/main/src/electricity_load_forecasting/calibration.py)
#   of the skore electricity load example.
# - **Conformalized quantile regression (CQR)**. A single symmetric margin is added to
#   both ends of the interval so that a chosen target coverage holds on a calibration
#   set. It can target any coverage, which lets us trace a coverage versus width
#   trade-off.
#
# The recalibrator is a regular scikit-learn estimator wrapping the quantile
# regressor, so it is fitted inside each training fold of the outer cross-validation:
# its out-of-fold predictions only use past data.

# %%
import re
from datetime import timedelta

from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

# Hourly rows: 7 days between train and test, as in TimeSeriesSplitter.
_DEFAULT_GAP_ROWS = 24 * 7


def _quantile_levels(pred):
    levels = {}
    for column in pred.columns:
        match = re.fullmatch(r"q_(.+)", column)
        if match is None:
            raise ValueError(f"Expected quantile columns named 'q_<level>', got {column!r}")
        levels[column] = float(match.group(1))
    return levels


class Recalibrator(RegressorMixin, BaseEstimator):
    """Post-hoc correction of a quantile estimator based on its reliability.

    Wraps a single-target quantile regressor whose ``predict`` returns a polars
    DataFrame with columns ``q_<level>``, such as ``HGBQuantileRegressor``.

    1. Collect out-of-fold predictions of ``estimator`` with ``cv``. Test folds
       need not partition the data: rows that are never tested are not used.
    2. For each quantile level tau, bin the out-of-fold predictions ``p`` into
       ``n_bins`` bins with as many points each. In each bin, measure the actual
       tau-quantile of the residuals ``y - p`` (in MW): the shift that would
       make the bin calibrated.
    3. Fit a smooth model (spline + RidgeCV, weighted by bin size) mapping the
       predicted value to the shift.
    4. Refit ``estimator`` on all the data.

    At prediction time the learned shift is added to the predictions, then the
    quantiles are sorted so that they do not cross.

    Parameters
    ----------
    estimator : quantile regressor
    cv : temporal cross-validation splitter, default=None
        Default: ``TimeSeriesSplit(n_splits=5, gap=24 * 7)``.
    n_bins : int, default=10
        Keep it small: a tail quantile needs several hundred points per bin.
        At least 5 (the number of spline basis functions).
    n_last_folds : int or None, default=4
        Only calibrate on the last folds of ``cv``, so that the correction
        follows recent behaviour, and to save time.
    """

    def __init__(self, estimator, cv=None, n_bins=10, n_last_folds=4):
        self.estimator = estimator
        self.cv = cv
        self.n_bins = n_bins
        self.n_last_folds = n_last_folds

    def fit(self, X, y):
        if self.n_bins < 5:
            raise ValueError(f"n_bins must be at least 5, got {self.n_bins}")
        if self.n_last_folds is not None and self.n_last_folds < 1:
            raise ValueError(f"n_last_folds must be at least 1, got {self.n_last_folds}")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        cv = self.cv if self.cv is not None else TimeSeriesSplit(5, gap=_DEFAULT_GAP_ROWS)

        folds = list(cv.split(X, y))
        n_skipped = 0
        if self.n_last_folds is not None:
            n_skipped = max(len(folds) - self.n_last_folds, 0)
            folds = folds[n_skipped:]
        self.n_folds_used_ = len(folds)

        y_oof, pred_oof, row_oof, fold_oof = [], [], [], []
        for fold, (train, test) in enumerate(folds, start=n_skipped):
            fold_estimator = clone(self.estimator).fit(X[train], y[train])
            pred_oof.append(fold_estimator.predict(X[test]))
            y_oof.append(y[test])
            row_oof.append(np.asarray(test))
            fold_oof.append(np.full(len(test), fold))
        if not y_oof:
            raise ValueError("cv produced no splits")
        pred_oof = pl.concat(pred_oof, how="vertical")
        y_oof = np.concatenate(y_oof)
        # keep the out-of-fold data for inspection
        self.oof_ = pred_oof.with_columns(
            row=pl.Series(np.concatenate(row_oof)),
            inner_fold=pl.Series(np.concatenate(fold_oof)),
            y=pl.Series(y_oof),
        )

        self.quantile_levels_ = _quantile_levels(pred_oof)
        self.shift_models_ = {}
        self.bin_summary_ = {}
        for column, tau in self.quantile_levels_.items():
            p = pred_oof[column].to_numpy()
            model, summary = self._fit_shift_model(p, y_oof - p, tau)
            self.shift_models_[column] = model
            self.bin_summary_[column] = summary

        self.estimator_ = clone(self.estimator).fit(X, y)
        return self

    def _fit_shift_model(self, p, residuals, tau):
        # bins with an equal number of points; np.unique in case of ties
        edges = np.unique(np.quantile(p, np.linspace(0, 1, self.n_bins + 1)))
        bin_ids = np.searchsorted(edges[1:-1], p, side="right")
        rows = []
        for b in np.unique(bin_ids):
            in_bin = bin_ids == b
            rows.append(
                {
                    "mean_prediction": float(p[in_bin].mean()),
                    "count": int(in_bin.sum()),
                    "shift": float(np.quantile(residuals[in_bin], tau)),
                    "empirical_coverage": float((residuals[in_bin] <= 0).mean()),
                }
            )
        summary = pl.DataFrame(rows)
        if len(summary) < 5:
            # ties can leave too few distinct bins for the spline
            warnings.warn(
                f"Only {len(summary)} distinct bins for quantile {tau}: using a constant shift.",
                stacklevel=2,
            )
            model = DummyRegressor(strategy="constant", constant=float(np.quantile(residuals, tau)))
            model.fit(summary[["mean_prediction"]].to_numpy(), summary["shift"].to_numpy())
            return model, summary

        # 5 basis functions: n_knots + degree - 1 = 3 + 3 - 1
        model = make_pipeline(
            SplineTransformer(n_knots=3, degree=3, extrapolation="constant"),
            RidgeCV(alphas=np.logspace(-3, 3, 13)),
        )
        # Weights normalized to a mean of 1: the ridge penalty is not scaled by
        # the weights, so raw counts would make the alpha grid ineffective.
        weights = summary["count"].to_numpy() / summary["count"].mean()
        model.fit(
            summary["mean_prediction"].to_numpy()[:, None],
            summary["shift"].to_numpy(),
            ridgecv__sample_weight=weights,
        )
        return model, summary

    def predict(self, X):
        pred = self.estimator_.predict(np.asarray(X, dtype=float))
        corrected = pred.to_numpy().astype(float, copy=True)
        for i, column in enumerate(pred.columns):
            corrected[:, i] += self.shift_models_[column].predict(corrected[:, [i]])
        # shifts are learned independently per quantile: avoid crossing
        order = np.argsort([self.quantile_levels_[c] for c in pred.columns])
        corrected[:, order] = np.sort(corrected[:, order], axis=1)
        return pl.DataFrame({c: corrected[:, i] for i, c in enumerate(pred.columns)})


recalibrated_pred = make_multi_horizon_pred(
    features, y, regressor=Recalibrator(hgb_q_regressor)
).skb.with_scoring(pinball_scorer)

# %% [markdown]
# We collect the cross-validated predictions of the recalibrated pipeline. They have
# the same layout as the raw ones, so both can be compared fold by fold.

# %%
cv_predictions_recalibrated = get_cv_results(recalibrated_pred)

# %% [markdown]
# ### Coverage versus interval width
#
# We now compare four intervals on the same assessment rows: the raw 90% interval, the
# recalibrated one, and each of them widened or narrowed with CQR to several target
# coverages. The earliest half of the chronological folds is used to estimate the CQR
# margin, and the later folds are held out for assessment, after an embargo equal to
# the longest forecast horizon. This is a time-series experiment, so temporal drift
# can limit the guarantees of exchangeable conformal prediction.
#
# For each horizon, the Pareto frontier keeps the intervals that no other interval
# dominates, that is, with at least as much coverage and no greater width.

# %%
LOWER, UPPER = "q_0.05", "q_0.95"


def conformal_margin(calibration, horizon, alpha):
    """Margin (MW) such that the widened interval covers 1 - alpha of calibration."""
    y_true = calibration[f"{horizon}h"].to_numpy()
    lower = calibration[f"pred_{horizon}h__{LOWER}"].to_numpy()
    upper = calibration[f"pred_{horizon}h__{UPPER}"].to_numpy()
    scores = np.maximum(lower - y_true, y_true - upper)
    rank = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return float(np.partition(scores, rank - 1)[rank - 1]) if rank <= len(scores) else np.inf


def calibration_evaluation_split(preds, calibration_fold_ids, evaluation_fold_ids, embargo):
    calibration = preds.filter(pl.col("split").is_in(calibration_fold_ids))
    cutoff = calibration["prediction_time"].max() + embargo
    evaluation = preds.filter(
        pl.col("split").is_in(evaluation_fold_ids) & (pl.col("prediction_time") > cutoff)
    )
    return calibration, evaluation


split_ids = sorted(cv_predictions_hgbr[0]["split"].unique().to_list())
n_calibration_folds = len(split_ids) // 2
if n_calibration_folds == 0:
    raise ValueError("Post-hoc recalibration requires at least two CV folds")
embargo = timedelta(hours=max(TIME_HORIZONS))
interval_predictions = {}
for name, cv_predictions in (
    ("raw", cv_predictions_hgbr[0]),
    ("recalibrated", cv_predictions_recalibrated[0]),
):
    interval_predictions[name] = calibration_evaluation_split(
        cv_predictions, split_ids[:n_calibration_folds], split_ids[n_calibration_folds:], embargo
    )
    if interval_predictions[name][1].is_empty():
        raise ValueError("No evaluation predictions remain after the forecast-horizon embargo")


def interval_point(horizon, label, nominal, evaluation, margin=0.0):
    y_true = evaluation[f"{horizon}h"].to_numpy()
    lower = evaluation[f"pred_{horizon}h__{LOWER}"].to_numpy() - margin
    upper = evaluation[f"pred_{horizon}h__{UPPER}"].to_numpy() + margin
    return {
        "horizon": f"{horizon}h",
        "interval": label,
        "nominal_coverage": nominal,
        "coverage": coverage(y_true, lower, upper),
        "mean_width_mw": mean_width(y_true, lower, upper),
    }


interval_points = []
cqr_miscoverage_levels = (0.5, 0.3, 0.2, 0.1, 0.05, 0.02)
for horizon in TIME_HORIZONS:
    for name, (calibration, evaluation) in interval_predictions.items():
        interval_points.append(interval_point(horizon, name, 0.9, evaluation))
        for alpha in cqr_miscoverage_levels:
            margin = conformal_margin(calibration, horizon, alpha)
            if np.isfinite(margin):
                interval_points.append(
                    interval_point(horizon, f"{name} + CQR", 1 - alpha, evaluation, margin)
                )

pl.DataFrame(interval_points).filter(pl.col("nominal_coverage") == 0.9)

# %% [markdown]
# The table above compares the four 90% intervals. The chart sweeps the target
# coverage and highlights the Pareto frontier of each horizon.

# %%
pareto_frontier = []
for horizon in (f"{h}h" for h in TIME_HORIZONS):
    horizon_points = [p for p in interval_points if p["horizon"] == horizon]
    for point in horizon_points:
        point["on_pareto_front"] = not any(
            other["coverage"] >= point["coverage"]
            and other["mean_width_mw"] <= point["mean_width_mw"]
            and (
                other["coverage"] > point["coverage"]
                or other["mean_width_mw"] < point["mean_width_mw"]
            )
            for other in horizon_points
        )
        duplicate = any(
            other["horizon"] == horizon
            and other["coverage"] == point["coverage"]
            and other["mean_width_mw"] == point["mean_width_mw"]
            for other in pareto_frontier
        )
        if point["on_pareto_front"] and not duplicate:
            pareto_frontier.append(point)

points_chart = (
    altair.Chart(altair.Data(values=interval_points))
    .mark_point(filled=True, size=90)
    .encode(
        x=altair.X("mean_width_mw:Q", title="Mean interval width (MW)"),
        y=altair.Y("coverage:Q", title="Empirical coverage", scale=altair.Scale(domain=[0, 1])),
        color=altair.Color("horizon:N", title="Forecast horizon"),
        shape=altair.Shape("interval:N", title="Interval"),
        tooltip=["horizon:N", "interval:N", "nominal_coverage:Q", "coverage:Q", "mean_width_mw:Q"],
    )
)
frontier_chart = (
    altair.Chart(altair.Data(values=pareto_frontier))
    .mark_line(point=True, strokeWidth=2)
    .encode(
        x="mean_width_mw:Q",
        y="coverage:Q",
        color=altair.Color("horizon:N", title="Forecast horizon"),
        detail="horizon:N",
        order=altair.Order("mean_width_mw:Q"),
    )
)
(points_chart + frontier_chart).properties(title="Coverage versus interval width by horizon")
