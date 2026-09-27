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
from datetime import datetime
import functools
import re
import warnings
from pathlib import Path

import altair
import skrub
import numpy as np
import polars as pl

import tutorial_helpers
import importlib
importlib.reload(tutorial_helpers)

from tutorial_helpers import (
    binned_coverage,
    plot_lorenz_curve,
    plot_reliability_diagram,
    plot_residuals_vs_predicted,
    collect_cv_predictions,
)


from feature_engineering_lib import feature_engineering_outputs, time_range

from next_horizon_prediction_lib import TimeSeriesSplitter


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
from sklearn.metrics import mean_absolute_percentage_error, d2_pinball_score

def split_by_quantile(pred):
    quantile_cols = {}
    for c in pred.columns:
        quantile_cols.setdefault(c.split("__")[1], []).append(c)
    return {
        q: pred.select(cols).rename(lambda c: c.split("__")[0])
        for q, cols in quantile_cols.items()
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
            {
                f"d2_pinball_score__{c}__{q}": float(s)
                for c, s in zip(y_true.columns, detail)
            }
        )
    return scores


def pinball_scorer(estimator, X, y):
    return pinball(y, estimator.predict(X))

# %%
TIME_HORIZONS = (1,12,24)
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.base import BaseEstimator, RegressorMixin

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
        return pl.DataFrame(
            {
                f"q_{quantile}": result[index]
                for index, quantile in enumerate(self.quantiles_)
            }
        )


quantiles=(0.05, 0.5, 0.95)

learning_rate = skrub.choose_float(
    0.01, 0.7, default=0.1, log=True, name="learning_rate"
)
max_leaf_nodes = skrub.choose_int(3, 300, default=30, log=True, name="max_leaf_nodes")
hgb_params = dict(
    random_state=0,
    max_iter=300,
    learning_rate=learning_rate,
    max_leaf_nodes=max_leaf_nodes,
)

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
    Create a full DataOp for predicting the specified horizons.
    """
    predictions = {
        h: feat.skb.drop(["prediction_time", "target_time"])
        .skb.apply(regressor, y=y[f"{h}h"])
        .skb.set_name(f"pred_{h}h")
        for h, feat in features.items()
    }
    return skrub.deferred(concat_horizons)(predictions)




pred = make_multi_horizon_pred(features, y, regressor=hgb_q_regressor).skb.with_scoring(
            functools.partial(neg_mape_scorer, quantile_regression=True)
        ).skb.with_scoring(pinball_scorer)

# %% [markdown]
# ### Independent randomized search per quantile
#
# The baseline above shares hyperparameters across quantiles. Here, we search each
# quantile independently using its own D² pinball score, then combine the selected
# models' predictions on the same held-out folds. Each search also chooses among
# temperature, temperature plus wind speed, and all weather features.

# %%
search_environment = {"start": "2023-01-01", "end": "2025-05-31"}
outer_split = next(pred.skb.iter_cv_splits(environment=search_environment))
quantile_searches = {}
quantile_test_predictions = {}

for quantile in quantiles:
    quantile_name = f"q{quantile:g}".replace(".", "_")
    quantile_regressor = HGBQuantileRegressor(
        quantiles=(quantile,),
        hgb_params={
            "max_iter": skrub.choose_int(
                100, 500, default=300, log=True,
                name=f"max_iter_{quantile_name}",
            ),
            "learning_rate": skrub.choose_float(
                0.01, 0.7, default=0.1, log=True,
                name=f"learning_rate_{quantile_name}",
            ),
            "max_leaf_nodes": skrub.choose_int(
                3, 300, default=30, log=True,
                name=f"max_leaf_nodes_{quantile_name}",
            ),
        },
    )
    quantile_pred = make_multi_horizon_pred(
        features, y, regressor=quantile_regressor
    ).skb.with_scoring(pinball_scorer)
    quantile_search = quantile_pred.skb.make_randomized_search(
        backend="optuna",
        n_iter=10,
        n_jobs=1,
        refit=f"d2_pinball_score__average__q_{quantile}",
        study_name=f"quantile_{quantile_name}",
    )
    quantile_search.fit(outer_split["train"])
    quantile_searches[quantile] = quantile_search
    quantile_test_predictions[quantile] = quantile_search.predict(
        outer_split["test"]
    )

def combine_quantile_predictions(predictions):
    combined = pl.concat(predictions, how="horizontal")
    for horizon in TIME_HORIZONS:
        columns = [f"{horizon}h__q_{quantile}" for quantile in quantiles]
        sorted_predictions = np.sort(combined.select(columns).to_numpy(), axis=1)
        combined = combined.with_columns(
            [
                pl.Series(name=column, values=sorted_predictions[:, index])
                for index, column in enumerate(columns)
            ]
        )
    return combined


combined_quantile_test_predictions = combine_quantile_predictions(
    [quantile_test_predictions[q] for q in quantiles]
)
combined_quantile_test_predictions

# %% [markdown]
# ### Quantile prediction with a RandomForest
#
# [scikit-learn PR #32903](https://github.com/scikit-learn/scikit-learn/pull/32903)
# proposes native pinball-loss splitting for `DecisionTreeRegressor` (and therefore
# `RandomForestRegressor`), the same quantile loss `HistGradientBoostingRegressor`
# already supports. As of writing, the PR is still an open, unmerged draft, so the
# installed scikit-learn's `RandomForestRegressor` has no `quantile` parameter yet.
#
# As a working alternative, we implement a quantile regression forest
# (Meinshausen, 2006): fit one standard `RandomForestRegressor`, then at predict
# time pool the training targets that land in the same leaves as each test point
# across all trees, and read off empirical quantiles from that weighted
# distribution. A PR review noted that leaves smaller than `1 / min(alpha, 1 -
# alpha)` samples bias the quantile estimate, so we size `min_samples_leaf`
# accordingly for our most extreme quantile (5%).
#
# For computational reasons we evaluate this experiment on the same held-out
# split used for the per-quantile randomized search above, rather than the full
# walk-forward cross-validation.

# %%
from sklearn.ensemble import RandomForestRegressor


class RandomForestQuantileRegressor(RegressorMixin, BaseEstimator):
    """Quantile regression forest (Meinshausen, 2006).

    Fits a single `RandomForestRegressor`; at predict time, empirical quantiles
    are read off the pooled training targets found in the leaves reached by
    each test sample across all trees.
    """

    def __init__(self, quantiles=(0.05, 0.5, 0.95), rf_params=None):
        self.quantiles = quantiles
        self.rf_params = rf_params

    def fit(self, X, y):
        self.quantiles_ = sorted(self.quantiles)
        self.forest_ = RandomForestRegressor(**(self.rf_params or {})).fit(X, y)
        y = np.asarray(y)
        train_leaves = self.forest_.apply(X)
        order = np.argsort(y)
        self.sorted_y_ = y[order]
        rank_of = np.empty(len(y), dtype=np.int64)
        rank_of[order] = np.arange(len(y))
        # per tree: leaf id -> ranks (in sorted_y_) of training samples in that leaf
        self.leaf_ranks_ = [
            {
                leaf: rank_of[np.flatnonzero(tree_leaves == leaf)]
                for leaf in np.unique(tree_leaves)
            }
            for tree_leaves in train_leaves.T
        ]
        return self

    def predict(self, X):
        test_leaves = self.forest_.apply(X)
        n_test, n_estimators = test_leaves.shape
        n_train = len(self.sorted_y_)
        quantile_targets = np.asarray(self.quantiles_)
        result = np.empty((n_test, len(self.quantiles_)))
        for i in range(n_test):
            weights = np.zeros(n_train)
            for tree_idx, leaf in enumerate(test_leaves[i]):
                ranks = self.leaf_ranks_[tree_idx].get(leaf)
                if ranks is not None and len(ranks):
                    weights[ranks] += 1.0 / len(ranks)
            cumulative = np.cumsum(weights)
            cumulative /= cumulative[-1]
            positions = np.minimum(
                np.searchsorted(cumulative, quantile_targets), n_train - 1
            )
            result[i] = self.sorted_y_[positions]
        return pl.DataFrame(
            {
                f"q_{quantile}": result[:, index]
                for index, quantile in enumerate(self.quantiles_)
            }
        )


# Guideline from the PR review: leaves smaller than 1 / min(alpha, 1 - alpha)
# bias the leaf-based quantile estimate.
rf_min_samples_leaf = max(20, int(np.ceil(1 / min(min(quantiles), 1 - max(quantiles)))))
rf_q_regressor = RandomForestQuantileRegressor(
    quantiles=quantiles,
    rf_params=dict(
        n_estimators=200,
        min_samples_leaf=rf_min_samples_leaf,
        random_state=0,
        n_jobs=-1,
    ),
)

rf_pred = make_multi_horizon_pred(features, y, regressor=rf_q_regressor).skb.with_scoring(
    pinball_scorer
)
rf_learner = rf_pred.skb.make_learner().fit(outer_split["train"])
rf_test_predictions = rf_learner.predict(outer_split["test"])

hgb_learner = pred.skb.make_learner().fit(outer_split["train"])
hgb_test_predictions = hgb_learner.predict(outer_split["test"])

from tutorial_helpers import coverage, mean_width

rf_vs_hgb_results = []
for horizon in TIME_HORIZONS:
    y_test_horizon = outer_split["y_test"][f"{horizon}h"].to_numpy()
    for model_name, predictions in (
        ("RandomForestQuantileRegressor", rf_test_predictions),
        ("HGBQuantileRegressor", hgb_test_predictions),
    ):
        lower = predictions[f"{horizon}h__q_0.05"].to_numpy()
        upper = predictions[f"{horizon}h__q_0.95"].to_numpy()
        rf_vs_hgb_results.append(
            {
                "horizon": f"{horizon}h",
                "model": model_name,
                "coverage": coverage(y_test_horizon, lower, upper),
                "mean_width_mw": mean_width(y_test_horizon, lower, upper),
            }
        )

pl.DataFrame(rf_vs_hgb_results)

# %% [markdown]
#
# Let's first collect all the cross-validated predictions to make further inspection.

# %%
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

cv_predictions_hgbr = cross_val_predict(pred,
                                        environment={"start": "2023-01-01", "end": "2025-05-31"})

# %% [markdown]
# Now, we can inspect the cross-validated predictions and plot them for the different quantiles.

# %%
import plotly.graph_objects as go
from datetime import UTC, timedelta
def plot_predictions(results, horizons=None, start="2025-03-01"):
    if start is not None:
        results = results.filter(
            pl.col("prediction_time")
            > datetime.fromisoformat(start).replace(tzinfo=UTC)
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
    fig.update_layout(height=600, title=f"CV predicted load mw")
    return fig

plot_predictions(cv_predictions_hgbr[0], horizons=(12,), start="2023-01-01").show()

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
plot_residuals_vs_predicted(cv_predictions_hgbr[0],1,quantile=0.05).interactive().properties(
    title=(
        "Residuals vs Predicted Values from cross-validation predictions"
        " for quantile 0.05"
    )
)

# %%
plot_residuals_vs_predicted(cv_predictions_hgbr[0],1,quantile=0.5).interactive().properties(
    title=("Residuals vs Predicted Values from cross-validation predictions for median")
)

# %%
plot_residuals_vs_predicted(cv_predictions_hgbr[0],1,quantile=0.95).interactive().properties(
    title=(
        "Residuals vs Predicted Values from cross-validation predictions"
        " for quantile 0.95"
    )
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
from tutorial_helpers import coverage, mean_width, binned_coverage
import altair

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

# %% [markdown]
# ### Post-hoc interval recalibration
#
# We use conformalized quantile regression (CQR) to adjust the 90% interval after
# fitting. The earliest half of the chronological cross-validation folds estimates
# the correction; the later folds are held out to compare raw and calibrated coverage
# and width, after an embargo equal to the maximum forecast horizon. This is a
# time-series experiment, so temporal drift can limit the calibration guarantees of
# exchangeable conformal prediction.

# %%
def conformalize_interval(calibration, evaluation, horizon, alpha=0.1):
    target_col = f"{horizon}h"
    lower_col = f"pred_{horizon}h__q_0.05"
    upper_col = f"pred_{horizon}h__q_0.95"

    y_calibration = calibration[target_col].to_numpy()
    lower_calibration = calibration[lower_col].to_numpy()
    upper_calibration = calibration[upper_col].to_numpy()
    scores = np.maximum(
        lower_calibration - y_calibration,
        y_calibration - upper_calibration,
    )
    rank = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    correction = (
        float(np.partition(scores, rank - 1)[rank - 1])
        if rank <= len(scores)
        else np.inf
    )

    calibrated = evaluation.with_columns(
        (pl.col(lower_col) - correction).alias(f"calibrated_{lower_col}"),
        (pl.col(upper_col) + correction).alias(f"calibrated_{upper_col}"),
    )
    return calibrated, correction


split_ids = sorted(preds["split"].unique().to_list())
calibration_fold_count = len(split_ids) // 2
if calibration_fold_count == 0:
    raise ValueError("Post-hoc recalibration requires at least two CV folds")

calibration_fold_ids = split_ids[:calibration_fold_count]
evaluation_fold_ids = split_ids[calibration_fold_count:]
calibration_predictions = preds.filter(pl.col("split").is_in(calibration_fold_ids))
calibration_target_end = calibration_predictions["prediction_time"].max() + timedelta(
    hours=max(TIME_HORIZONS)
)
evaluation_predictions = preds.filter(
    pl.col("split").is_in(evaluation_fold_ids)
    & (pl.col("prediction_time") > calibration_target_end)
)
if evaluation_predictions.is_empty():
    raise ValueError("No evaluation predictions remain after the forecast-horizon embargo")
calibrated_evaluation = evaluation_predictions
calibration_results = []

for horizon in TIME_HORIZONS:
    calibrated_evaluation, correction = conformalize_interval(
        calibration_predictions, calibrated_evaluation, horizon
    )
    y_evaluation = calibrated_evaluation[f"{horizon}h"].to_numpy()
    raw_lower = calibrated_evaluation[f"pred_{horizon}h__q_0.05"].to_numpy()
    raw_upper = calibrated_evaluation[f"pred_{horizon}h__q_0.95"].to_numpy()
    calibrated_lower = calibrated_evaluation[
        f"calibrated_pred_{horizon}h__q_0.05"
    ].to_numpy()
    calibrated_upper = calibrated_evaluation[
        f"calibrated_pred_{horizon}h__q_0.95"
    ].to_numpy()
    calibration_results.append(
        {
            "horizon": horizon,
            "correction_mw": correction,
            "raw_coverage": coverage(y_evaluation, raw_lower, raw_upper),
            "calibrated_coverage": coverage(
                y_evaluation, calibrated_lower, calibrated_upper
            ),
            "raw_mean_width_mw": mean_width(y_evaluation, raw_lower, raw_upper),
            "calibrated_mean_width_mw": mean_width(
                y_evaluation, calibrated_lower, calibrated_upper
            ),
        }
    )

pl.DataFrame(calibration_results)

# %% [markdown]
# ### Coverage versus interval width
#
# We sweep several CQR target coverages and compare them with the raw 90% interval
# on the same assessment rows. For each horizon, the Pareto frontier keeps intervals
# that are not dominated by another interval with at least as much coverage and no
# greater width.

# %%
pareto_points = []
target_miscoverage_levels = (0.5, 0.3, 0.2, 0.1, 0.05, 0.02)

for horizon in TIME_HORIZONS:
    y_evaluation = evaluation_predictions[f"{horizon}h"].to_numpy()
    raw_lower = evaluation_predictions[f"pred_{horizon}h__q_0.05"].to_numpy()
    raw_upper = evaluation_predictions[f"pred_{horizon}h__q_0.95"].to_numpy()
    pareto_points.append(
        {
            "horizon": f"{horizon}h",
            "interval": "raw 90%",
            "nominal_coverage": 0.9,
            "coverage": coverage(y_evaluation, raw_lower, raw_upper),
            "mean_width_mw": float(np.abs(raw_upper - raw_lower).mean()),
        }
    )

    for alpha in target_miscoverage_levels:
        calibrated, _ = conformalize_interval(
            calibration_predictions, evaluation_predictions, horizon, alpha=alpha
        )
        calibrated_lower = calibrated[
            f"calibrated_pred_{horizon}h__q_0.05"
        ].to_numpy()
        calibrated_upper = calibrated[
            f"calibrated_pred_{horizon}h__q_0.95"
        ].to_numpy()
        pareto_points.append(
            {
                "horizon": f"{horizon}h",
                "interval": f"CQR {1 - alpha:.0%}",
                "nominal_coverage": 1 - alpha,
                "coverage": coverage(
                    y_evaluation, calibrated_lower, calibrated_upper
                ),
                "mean_width_mw": float(
                    np.abs(calibrated_upper - calibrated_lower).mean()
                ),
            }
        )

pareto_frontier = []
for horizon in (f"{h}h" for h in TIME_HORIZONS):
    horizon_points = [
        point for point in pareto_points if point["horizon"] == horizon
    ]
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
        if point["on_pareto_front"]:
            same_frontier_point = any(
                other["coverage"] == point["coverage"]
                and other["mean_width_mw"] == point["mean_width_mw"]
                for other in pareto_frontier
                if other["horizon"] == horizon
            )
            if not same_frontier_point:
                pareto_frontier.append(point)

points_chart = altair.Chart(altair.Data(values=pareto_points)).mark_point(
    filled=True, size=90
).encode(
    x=altair.X("mean_width_mw:Q", title="Mean interval width (MW)"),
    y=altair.Y(
        "coverage:Q",
        title="Empirical coverage",
        scale=altair.Scale(domain=[0, 1]),
    ),
    color=altair.Color("horizon:N", title="Forecast horizon"),
    shape=altair.Shape("interval:N", title="Interval setting"),
    tooltip=["horizon:N", "interval:N", "nominal_coverage:Q", "coverage:Q", "mean_width_mw:Q"],
)
frontier_chart = altair.Chart(altair.Data(values=pareto_frontier)).mark_line(
    point=True, strokeWidth=2
).encode(
    x="mean_width_mw:Q",
    y="coverage:Q",
    color=altair.Color("horizon:N", title="Forecast horizon"),
    detail="horizon:N",
    order=altair.Order("mean_width_mw:Q"),
)
(points_chart + frontier_chart).properties(
    title="Coverage versus interval width by horizon"
)

# --- Binned coverage plot ---
folds = [
    fold_df
    for (_, ), fold_df in preds.group_by("split", maintain_order=True)
]
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

# %%

# %% [markdown]
# ## Skrub report
#
# Precomputed reports: [Jupyter Book](../../_static/reports/prediction_intervals/index.html)
# and [JupyterLite](../reports/prediction_intervals/index.html). On a local Python
# installation, run the next cell to regenerate this report from the current pipeline.

# %%
import sys
from pathlib import Path
from IPython.display import FileLink, display

if sys.platform == "emscripten":
    print("Use the precomputed report link above in JupyterLite.")
else:
    repository_root = next(
        parent for parent in (Path.cwd(), *Path.cwd().parents)
        if (parent / "book").is_dir()
    )
    report = pred.skb.full_report(
        open=False,
        output_dir=repository_root / "book" / "_static" / "reports" / "prediction_intervals",
        overwrite=True,
        title="Quantile prediction intervals pipeline",
    )
    display(FileLink(str(report["report_path"])))
