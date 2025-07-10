# %% [markdown]
#
# # Computing prediction intervals using quantile regression
#
# ## Environment setup
#
# We need to install some extra dependencies for this notebook if needed (when
# running jupyterlite). We need the development version of skrub to be able to
# use the skrub expressions.

# %%
# %pip install -q https://pypi.anaconda.org/ogrisel/simple/polars/1.24.0/polars-1.24.0-cp39-abi3-emscripten_3_1_58_wasm32.whl
# %pip install -q https://pypi.anaconda.org/ogrisel/simple/skrub/0.6.dev0/skrub-0.6.dev0-py3-none-any.whl
# %pip install -q altair holidays plotly nbformat

# %%
import warnings

import polars as pl
import altair
import cloudpickle
import numpy as np

from tutorial_helpers import (
    binned_coverage,
    plot_lorenz_curve,
    plot_reliability_diagram,
    plot_residuals_vs_predicted,
    collect_cv_predictions,
)

# Ignore warnings from pkg_resources triggered by Python 3.13's multiprocessing.
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")


# %%
with open("feature_engineering_pipeline.pkl", "rb") as f:
    feature_engineering_pipeline = cloudpickle.load(f)


features = feature_engineering_pipeline["features"]
targets = feature_engineering_pipeline["targets"]
prediction_time = feature_engineering_pipeline["prediction_time"]
horizons = feature_engineering_pipeline["horizons"]
target_column_name_pattern = feature_engineering_pipeline["target_column_name_pattern"]


# %% [markdown]
# ### Define the quantile regressors
#
# In this section, we show how one can use a gradient boosting but modify the loss
# function to predict different quantiles and thus obtain an uncertainty quantification
# of the predictions.
#
# In terms of evaluation, we reuse the R2 and MAPE scores. However, they are not helpful
# to assess the reliability of quantile models. For this purpose, we use a derivate of
# the metric minimize by those models: the pinball loss. We use the D2 score that is
# easier to interpret since the best possible score is bounded by 1 and a score of 0
# corresponds to constant predictions at the target quantile.

# %%
horizon_of_interest = horizons[-1]  # Focus on the 24-hour horizon
target_column_name = target_column_name_pattern.format(horizon=horizon_of_interest)
predicted_target_column_name = "predicted_" + target_column_name
target = targets[target_column_name].skb.mark_as_y()
target

# %%
from sklearn.metrics import get_scorer, make_scorer
from sklearn.metrics import mean_absolute_percentage_error, d2_pinball_score

scoring = {
    "r2": get_scorer("r2"),
    "mape": make_scorer(mean_absolute_percentage_error),
    "d2_pinball_05": make_scorer(d2_pinball_score, alpha=0.05),
    "d2_pinball_50": make_scorer(d2_pinball_score, alpha=0.50),
    "d2_pinball_95": make_scorer(d2_pinball_score, alpha=0.95),
}

# %% [markdown]
#
# We know define three different models:
#
# - a model predicting the 5th percentile of the load
# - a model predicting the median of the load
# - a model predicting the 95th percentile of the load

# %%
from sklearn.ensemble import HistGradientBoostingRegressor


common_params = dict(
    loss="quantile", learning_rate=0.1, max_leaf_nodes=100, random_state=0
)
predictions_hgbr_05 = features.skb.apply(
    HistGradientBoostingRegressor(**common_params, quantile=0.05),
    y=target,
)
predictions_hgbr_50 = features.skb.apply(
    HistGradientBoostingRegressor(**common_params, quantile=0.5),
    y=target,
)
predictions_hgbr_95 = features.skb.apply(
    HistGradientBoostingRegressor(**common_params, quantile=0.95),
    y=target,
)

# %% [markdown]
#
# Now, let's make a plot of the predictions for each model and thus we need to gather
# all the predictions in a single dataframe.

# %%
results = pl.concat(
    [
        targets.skb.select(cols=["prediction_time", target_column_name]).skb.preview(),
        predictions_hgbr_05.rename({target_column_name: "quantile_05"}).skb.preview(),
        predictions_hgbr_50.rename({target_column_name: "median"}).skb.preview(),
        predictions_hgbr_95.rename({target_column_name: "quantile_95"}).skb.preview(),
    ],
    how="horizontal",
).tail(24 * 7)

# %% [markdown]
#
# Now, we plot the observed values and the predicted median with a line. In addition,
# we plot the 5th and 95th percentiles as a shaded area. It means that between those
# two bounds, we expect to find 90% of the observed values.
#
# We plot this information on a portion of the training data to observe the uncertainty
# quantification of the predictions.

# %%
median_chart = (
    altair.Chart(results)
    .transform_fold([target_column_name, "median"])
    .mark_line(tooltip=True)
    .encode(x="prediction_time:T", y="value:Q", color="key:N")
)

# Add a column for the band legend
results_with_band = results.with_columns(pl.lit("90% interval").alias("band_type"))

quantile_band_chart = (
    altair.Chart(results_with_band)
    .mark_area(opacity=0.4, tooltip=True)
    .encode(
        x="prediction_time:T",
        y="quantile_05:Q",
        y2="quantile_95:Q",
        color=altair.Color("band_type:N", scale=altair.Scale(range=["lightgreen"])),
    )
)

combined_chart = quantile_band_chart + median_chart
combined_chart.resolve_scale(color="independent").interactive()

# %% [markdown]
#
# While we should expend this plot on the test data and on several portions of the
# dataset, we observe a potential interesting pattern: during the week days, the
# 5th percentile is further from the median than during the weekend. However, for the
# 95th percentile, the opposite is observed.
#
# ### Evaluation via cross-validation
#
# We evaluate the performance of the quantile regressors via cross-validation.

# %%
# %%
from sklearn.model_selection import TimeSeriesSplit


max_train_size = 2 * 52 * 24 * 7  # max ~2 years of training data
test_size = 24 * 7 * 24  # 24 weeks of test data
gap = 7 * 24  # 1 week gap between train and test sets
ts_cv_5 = TimeSeriesSplit(
    n_splits=5, max_train_size=max_train_size, test_size=test_size, gap=gap
)

# %%
cv_results_hgbr_05 = predictions_hgbr_05.skb.cross_validate(
    cv=ts_cv_5,
    scoring=scoring,
    return_pipeline=True,
    verbose=1,
    n_jobs=-1,
)
cv_results_hgbr_50 = predictions_hgbr_50.skb.cross_validate(
    cv=ts_cv_5,
    scoring=scoring,
    return_pipeline=True,
    verbose=1,
    n_jobs=-1,
)
cv_results_hgbr_95 = predictions_hgbr_95.skb.cross_validate(
    cv=ts_cv_5,
    scoring=scoring,
    return_pipeline=True,
    verbose=1,
    n_jobs=-1,
)

# %%
cv_results_hgbr_05[
    [col for col in cv_results_hgbr_05.columns if col.startswith("test_")]
].mean(axis=0).round(3)

# %%
cv_results_hgbr_50[
    [col for col in cv_results_hgbr_50.columns if col.startswith("test_")]
].mean(axis=0).round(3)

# %%
cv_results_hgbr_95[
    [col for col in cv_results_hgbr_95.columns if col.startswith("test_")]
].mean(axis=0).round(3)

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
cv_predictions_hgbr_05 = collect_cv_predictions(
    cv_results_hgbr_05["pipeline"], ts_cv_5, predictions_hgbr_05, prediction_time
)
cv_predictions_hgbr_50 = collect_cv_predictions(
    cv_results_hgbr_50["pipeline"], ts_cv_5, predictions_hgbr_50, prediction_time
)
cv_predictions_hgbr_95 = collect_cv_predictions(
    cv_results_hgbr_95["pipeline"], ts_cv_5, predictions_hgbr_95, prediction_time
)

# %%
plot_residuals_vs_predicted(cv_predictions_hgbr_05).interactive().properties(
    title=(
        "Residuals vs Predicted Values from cross-validation predictions"
        " for quantile 0.05"
    )
)

# %%
plot_residuals_vs_predicted(cv_predictions_hgbr_50).interactive().properties(
    title=("Residuals vs Predicted Values from cross-validation predictions for median")
)

# %%
plot_residuals_vs_predicted(cv_predictions_hgbr_95).interactive().properties(
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
# Note that we could obtain similar plots using scikit-learn's `PredictionErrorDisplay`.
# This display allows to also plot the observed values vs predicted values as well.

# %%
cv_predictions_hgbr_05_concat = pl.concat(cv_predictions_hgbr_05, how="vertical")
cv_predictions_hgbr_50_concat = pl.concat(cv_predictions_hgbr_50, how="vertical")
cv_predictions_hgbr_95_concat = pl.concat(cv_predictions_hgbr_95, how="vertical")

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import PredictionErrorDisplay


for kind in ["actual_vs_predicted", "residual_vs_predicted"]:
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    PredictionErrorDisplay.from_predictions(
        y_true=cv_predictions_hgbr_05_concat["load_mw"].to_numpy(),
        y_pred=cv_predictions_hgbr_05_concat["predicted_load_mw"].to_numpy(),
        kind=kind,
        ax=axs[0],
    )
    axs[0].set_title("0.05 quantile regression")

    PredictionErrorDisplay.from_predictions(
        y_true=cv_predictions_hgbr_50_concat["load_mw"].to_numpy(),
        y_pred=cv_predictions_hgbr_50_concat["predicted_load_mw"].to_numpy(),
        kind=kind,
        ax=axs[1],
    )
    axs[1].set_title("Median regression")

    PredictionErrorDisplay.from_predictions(
        y_true=cv_predictions_hgbr_95_concat["load_mw"].to_numpy(),
        y_pred=cv_predictions_hgbr_95_concat["predicted_load_mw"].to_numpy(),
        kind=kind,
        ax=axs[2],
    )
    axs[2].set_title("0.95 quantile regression")

    fig.suptitle(f"{kind} for GBRT minimzing different quantile losses")

# %% [markdown]
#
# Those plots carry the same information than the previous ones.
#
# Now, we assess if the actual coverage of the models is close to the target coverage of
# 90%. In addition, we compute the average width of the bands.


# %%
def coverage(y_true, y_quantile_low, y_quantile_high):
    y_true = np.asarray(y_true)
    y_quantile_low = np.asarray(y_quantile_low)
    y_quantile_high = np.asarray(y_quantile_high)
    return float(
        np.logical_and(y_true >= y_quantile_low, y_true <= y_quantile_high)
        .mean()
        .round(4)
    )


def mean_width(y_true, y_quantile_low, y_quantile_high):
    y_true = np.asarray(y_true)
    y_quantile_low = np.asarray(y_quantile_low)
    y_quantile_high = np.asarray(y_quantile_high)
    return float(np.abs(y_quantile_high - y_quantile_low).mean().round(1))


# %%
coverage(
    cv_predictions_hgbr_50_concat["load_mw"].to_numpy(),
    cv_predictions_hgbr_05_concat["predicted_load_mw"].to_numpy(),
    cv_predictions_hgbr_95_concat["predicted_load_mw"].to_numpy(),
)

# %% [markdown]
#
# We see that the obtained coverage (~77%) on the cross-validated predictions is much
# lower than the target coverage of 90%. It means that the pair of regressors is not
# jointly calibrated to estimate the 90% interval.

# %%
mean_width(
    cv_predictions_hgbr_50_concat["load_mw"].to_numpy(),
    cv_predictions_hgbr_05_concat["predicted_load_mw"].to_numpy(),
    cv_predictions_hgbr_95_concat["predicted_load_mw"].to_numpy(),
)

# %% [markdown]
#
# In terms of interpretable measure, the mean width provides a measure in the original
# unit of the target variable in MW that is ~5,100 MW.
#
# We can go a bit further and bin the cross-validated predictions and check if some
# specific bins show a better or worse coverage.

# %%
binned_coverage_results = binned_coverage(
    [df["load_mw"].to_numpy() for df in cv_predictions_hgbr_50],
    [df["predicted_load_mw"].to_numpy() for df in cv_predictions_hgbr_05],
    [df["predicted_load_mw"].to_numpy() for df in cv_predictions_hgbr_95],
    n_bins=10,
)
binned_coverage_results

# %% [markdown]
#
# Let's make a plot to check those data visually.

# %%
coverage_by_bin = binned_coverage_results.copy()
coverage_by_bin["bin_label"] = coverage_by_bin.apply(
    lambda row: f"[{row.bin_left:.0f}, {row.bin_right:.0f}]", axis=1
)

# %%
ax = coverage_by_bin.boxplot(column="coverage", by="bin_label", whis=1000)
ax.axhline(y=0.9, color="red", linestyle="--", label="Target coverage (0.9)")
ax.set(
    xlabel="Load bins (MW)",
    ylabel="Coverage",
    title="Coverage Distribution by Load Bins",
)
ax.set_title("Coverage Distribution by Load Bins")
ax.legend()
plt.suptitle("")  # Remove automatic suptitle from boxplot
_ = plt.xticks(rotation=45)

# %% [markdown]
#
# We observe that the lower and higher bins, so low and high load, have the worse
# coverage with a high variability.
#
# ### Reliability diagrams and Lorenz curves for quantile regression

# %%
plot_reliability_diagram(
    cv_predictions_hgbr_50, kind="quantile", quantile_level=0.50
).interactive().properties(
    title="Reliability diagram for quantile 0.50 from cross-validation predictions"
)

# %%
plot_reliability_diagram(
    cv_predictions_hgbr_05, kind="quantile", quantile_level=0.05
).interactive().properties(
    title="Reliability diagram for quantile 0.05 from cross-validation predictions"
)

# %%
plot_reliability_diagram(
    cv_predictions_hgbr_95, kind="quantile", quantile_level=0.95
).interactive().properties(
    title="Reliability diagram for quantile 0.95 from cross-validation predictions"
)

# %%
plot_lorenz_curve(cv_predictions_hgbr_50).interactive().properties(
    title="Lorenz curve for quantile 0.50 from cross-validation predictions"
)

# %%
plot_lorenz_curve(cv_predictions_hgbr_05).interactive().properties(
    title="Lorenz curve for quantile 0.05 from cross-validation predictions"
)

# %%
plot_lorenz_curve(cv_predictions_hgbr_95).interactive().properties(
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
from sklearn.base import BaseEstimator, RegressorMixin, clone
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
        quantile=0.5,
        random_state=None,
    ):
        self.n_bins = n_bins
        self.estimator = estimator
        self.quantile = quantile
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

    def predict_quantiles(self, X, quantiles=(0.05, 0.5, 0.95)):
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
        return np.asarray([interp1d(y_cdf_i, edges)(quantiles) for y_cdf_i in y_cdf])

    def predict(self, X):
        return self.predict_quantiles(X, quantiles=(self.quantile,)).ravel()


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

X, y = features.skb.eval(), target.skb.eval()

cv_results_bqr = cross_validate(
    bqr,
    X,
    y,
    cv=ts_cv_5,
    scoring={
        "d2_pinball_50": make_scorer(d2_pinball_score, alpha=0.5),
    },
    return_estimator=True,
    return_indices=True,
    verbose=1,
    n_jobs=-1,
)

# %%
cv_predictions_bqr_all = [
    cv_predictions_bqr_05 := [],
    cv_predictions_bqr_50 := [],
    cv_predictions_bqr_95 := [],
]
for fold_ix, (qreg, test_idx) in enumerate(
    zip(cv_results_bqr["estimator"], cv_results_bqr["indices"]["test"])
):
    print(f"CV iteration #{fold_ix}")
    print(f"Test set size: {test_idx.shape[0]} rows")
    print(
        f"Test time range: {prediction_time.skb.eval()[test_idx][0, 0]} to "
        f"{prediction_time.skb.eval()[test_idx][-1, 0]} "
    )
    y_pred_all_quantiles = qreg.predict_quantiles(X[test_idx], quantiles=quantiles)

    coverage_score = coverage(
        y[test_idx],
        y_pred_all_quantiles[:, 0],
        y_pred_all_quantiles[:, 2],
    )
    print(f"Coverage: {coverage_score:.3f}")

    mean_width_score = mean_width(
        y[test_idx],
        y_pred_all_quantiles[:, 0],
        y_pred_all_quantiles[:, 2],
    )
    print(f"Mean prediction interval width: " f"{mean_width_score:.1f} MW")

    for q_idx, (quantile, predictions) in enumerate(
        zip(quantiles, cv_predictions_bqr_all)
    ):
        observed = y[test_idx]
        predicted = y_pred_all_quantiles[:, q_idx]
        predictions.append(
            pl.DataFrame(
                {
                    "prediction_time": prediction_time.skb.eval()[test_idx],
                    "load_mw": observed,
                    "predicted_load_mw": predicted,
                }
            )
        )
        print(f"d2_pinball score: {d2_pinball_score(observed, predicted):.3f}")
    print()

# %% [markdown
# Let's assess the calibration of the quantile regression model:

# %%
plot_reliability_diagram(
    cv_predictions_bqr_50, kind="quantile", quantile_level=0.50
).interactive().properties(
    title="Reliability diagram for quantile 0.50 from cross-validation predictions"
)

# %%
plot_reliability_diagram(
    cv_predictions_bqr_05, kind="quantile", quantile_level=0.05
).interactive().properties(
    title="Reliability diagram for quantile 0.05 from cross-validation predictions"
)

# %%
plot_reliability_diagram(
    cv_predictions_bqr_95, kind="quantile", quantile_level=0.95
).interactive().properties(
    title="Reliability diagram for quantile 0.95 from cross-validation predictions"
)

# %% [markdown]
#
# We can complement this assessment with the Lorenz curves, which only assess
# the ranking power of the predictions, irrespective of their absolute values.

# %%
plot_lorenz_curve(cv_predictions_bqr_50).interactive().properties(
    title="Lorenz curve for quantile 0.50 from cross-validation predictions"
)

# %%
plot_lorenz_curve(cv_predictions_bqr_05).interactive().properties(
    title="Lorenz curve for quantile 0.05 from cross-validation predictions"
)

# %%
plot_lorenz_curve(cv_predictions_bqr_95).interactive().properties(
    title="Lorenz curve for quantile 0.95 from cross-validation predictions"
)
