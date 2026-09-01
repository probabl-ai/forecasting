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
        return pl.DataFrame(result, schema=[f"q_{q}" for q in self.quantiles_])


quantiles=(0.05, 0.5, 0.95)

learning_rate = skrub.choose_float(
    0.01, 0.7, default=0.1, log=True, name="learning_rate"
)
max_leaf_nodes = skrub.choose_int(3, 300, default=30, log=True, name="max_leaf_nodes")
hgb_params = dict(
    random_state=0,
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
from tutorial_helpers import coverage, binned_coverage
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


# %% [markdown]
#
# ## Quantile regression as classification
#
# In the following, we turn a quantile regression problem for all possible
# quantile levels into a multiclass classification problem by discretizing the
# target variable into bins and interpolating the cumulative sum of the bin
# membership probability to estimate the CDF of the distribution of the
# continuous target variable conditioned on the features.
#
# Ideally, the classifier should be efficient when trained on a large number of
# classes (induced by the number of bins). Therefore we use a Random Forest
# classifier as the default base estimator.
#
# There are several advantages to this approach:
# - a single model is trained and can jointly estimate quantiles for all
#   quantile levels (assuming a well tuned number of bins);
# - the quantile levels can be chosen at prediction time, which allows for a
#   flexible quantile regression model;
# - in practice, the resulting predictions are often reasonably well calibrated
#   as we will see in the reliability diagrams below.
#
# One possible drawback is that current implementations of gradient boosting
# models tend to be very slow to train with a large number of classes. Random
# Forests are much more efficient in this case, but they do not always provide
# the best predictive performance. It could be the case that combining this
# approach with tabular neural networks can lead to competitive results.
#
# However, the current scikit-learn API is not expressive enough to to handle
# the output shape of the quantile prediction function. We therefore cannot
# make it fit into a skrub pipeline.

# %%
from scipy.interpolate import interp1d
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils.validation import check_consistent_length
from sklearn.utils import check_random_state
import numpy as np


class BinnedQuantileRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        estimator=None,
        n_bins=100,
        default_quantiles=(0.05, 0.5, 0.95),
        random_state=None,
    ):
        self.n_bins = n_bins
        self.estimator = estimator
        self.default_quantiles = default_quantiles
        self.random_state = random_state

    def fit(self, X, y):
        # Lightweight input validation: most of the input validation will be
        # handled by the sub estimators.
        random_state = check_random_state(self.random_state)
        check_consistent_length(X, y)
        self.target_binner_ = KBinsDiscretizer(
            n_bins=self.n_bins,
            strategy="quantile",
            subsample=200_000,
            encode="ordinal",
            quantile_method="averaged_inverted_cdf",
            random_state=random_state,
        )

        y_binned = (
            self.target_binner_.fit_transform(np.asarray(y).reshape(-1, 1))
            .ravel()
            .astype(np.int32)
        )

        # Fit the multiclass classifier to predict the binned targets from the
        # training set.
        if self.estimator is None:
            estimator = RandomForestClassifier(random_state=random_state)
        else:
            estimator = clone(self.estimator)
        self.estimator_ = estimator.fit(X, y_binned)
        return self

    def predict(self, X, quantiles=None):
        if quantiles is None:
            quantiles = self.default_quantiles
        check_is_fitted(self, "estimator_")
        edges = self.target_binner_.bin_edges_[0]
        n_bins = edges.shape[0] - 1
        expected_shape = (X.shape[0], n_bins)
        y_proba_raw = self.estimator_.predict_proba(X)

        # Some might stay empty on the training set. Typically, classifiers do
        # not learn to predict an explicit 0 probability for unobserved classes
        # so we have to post process their output:
        if y_proba_raw.shape != expected_shape:
            y_proba = np.zeros(shape=expected_shape)
            y_proba[:, self.estimator_.classes_] = y_proba_raw
        else:
            y_proba = y_proba_raw

        # Build the mapper for inverse CDF mapping, from cumulated
        # probabilities to continuous prediction.
        y_cdf = np.zeros(shape=(X.shape[0], edges.shape[0]))
        y_cdf[:, 1:] = np.cumsum(y_proba, axis=1)
        result = np.asarray([interp1d(y_cdf_i, edges)(quantiles) for y_cdf_i in y_cdf])
        return pl.DataFrame(result, schema=[f"q_{q}" for q in quantiles])

# %%
quantiles = (0.05, 0.5, 0.95)
bqr = BinnedQuantileRegressor(
    RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        max_features=0.2,
        n_jobs=-1,
        random_state=0,
    ),
    n_bins=30,
)
bqr

# %%
from sklearn.model_selection import cross_validate
def limit_train_size(df, size=9000, mode=skrub.eval_mode()):
    if mode in ("fit", "fit_transform", "preview"):
        return df.tail(size)
    else:
        return df

# We limit the training size and predict all three horizons for the BQR model.
features = {h: features[h].skb.apply_func(limit_train_size) for h in TIME_HORIZONS}
y = y.skb.apply_func(limit_train_size)

pred = make_multi_horizon_pred(features, y, regressor=bqr).skb.with_scoring(
            functools.partial(neg_mape_scorer, quantile_regression=True)
        ).skb.with_scoring(pinball_scorer)
cv_predictions_bqr = cross_val_predict(pred,
                                        environment={"start": "2023-01-01", "end": "2025-05-31"})

# %%
preds = cv_predictions_bqr[0]
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
# Let's assess the calibration of the quantile regression model:

# %%
plot_reliability_diagram(
    cv_predictions_bqr[0], 1, forecast_quantile=0.50
).interactive().properties(
    title="Reliability diagram for quantile 0.50 from cross-validation predictions"
)

# %%
plot_reliability_diagram(
    cv_predictions_bqr[0], 1, forecast_quantile=0.05
).interactive().properties(
    title="Reliability diagram for quantile 0.05 from cross-validation predictions"
)

# %%
plot_reliability_diagram(
    cv_predictions_bqr[0], 1, forecast_quantile=0.95
).interactive().properties(
    title="Reliability diagram for quantile 0.95 from cross-validation predictions"
)

# %% [markdown]
#
# We can complement this assessment with the Lorenz curves, which only assess
# the ranking power of the predictions, irrespective of their absolute values.

# %%
plot_lorenz_curve(cv_predictions_bqr[0], 1, quantile=0.50).interactive().properties(
    title="Lorenz curve for quantile 0.50 from cross-validation predictions"
)

# %%
plot_lorenz_curve(cv_predictions_bqr[0], 1, quantile=0.05).interactive().properties(
    title="Lorenz curve for quantile 0.05 from cross-validation predictions"
)

# %%
plot_lorenz_curve(cv_predictions_bqr[0], 1, quantile=0.95).interactive().properties(
    title="Lorenz curve for quantile 0.95 from cross-validation predictions"
)

# %%
