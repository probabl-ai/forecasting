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

from tutorial_helpers import binned_coverage, coverage

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
