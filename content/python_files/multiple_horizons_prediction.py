# %% [markdown]
#
# # Multiple horizons predictive modeling
#
# ## Environment setup
#
# We need to install some extra dependencies for this notebook if needed (when
# running jupyterlite).

# %%
# %pip install -q https://pypi.anaconda.org/ogrisel/simple/polars/1.24.0/polars-1.24.0-cp39-abi3-emscripten_3_1_58_wasm32.whl
# %pip install -q skrub altair holidays plotly nbformat

# %%
import datetime
import warnings

import altair
import cloudpickle
import numpy as np
import pandas as pd
import polars as pl
import pyarrow  # noqa: F401
import skrub
import tzdata  # noqa: F401

from tutorial_helpers import plot_horizon_forecast

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
#
# ## Predicting multiple horizons with direct forecasting
#
# Instead of fitting a single multi-output estimator, we will fit one model
# for each horizon.
#
# We start by defining a couple of small helpers to:
#
# - build a direct forecasting data op for a given estimator and horizon;
# - evaluate all horizons with the same cross-validation procedure;
# - rebuild a wide prediction table so that we can still visualize the full
#   forecast curve at a given timestamp. (is there a better way to do this
#   perhaps, currently it is slow)

# %%
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import get_scorer, make_scorer, mean_absolute_percentage_error
from sklearn.model_selection import BaseCrossValidator


def to_numpy_1d(values):
    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]
    elif isinstance(values, pl.DataFrame):
        values = values.to_series()

    if isinstance(values, pd.Series):
        return values.to_numpy()
    if isinstance(values, pl.Series):
        return values.to_numpy()

    array = np.asarray(values)
    if array.ndim == 2 and array.shape[1] == 1:
        return array[:, 0]
    return array


prediction_timestamps = pd.to_datetime(to_numpy_1d(prediction_time.skb.eval()), utc=True)


class DateBasedSplitter(BaseCrossValidator):
    def __init__(self, prediction_timestamps, min_train_days=365 * 2, test_length_hours=24 * 7, gap_days=7):
        self.prediction_timestamps = pd.Series(prediction_timestamps)
        self.min_train_days = min_train_days
        self.test_length_hours = test_length_hours
        self.gap_days = gap_days

    def split(self, X=None, y=None, groups=None):
        first_test_start = self.prediction_timestamps.min() + pd.Timedelta(
            days=self.min_train_days + self.gap_days
        )
        test_starts = pd.date_range(
            start=first_test_start,
            end=self.prediction_timestamps.max(),
            freq=pd.Timedelta(hours=self.test_length_hours),
            inclusive="left",
        )

        for test_start in test_starts:
            train_end = test_start - pd.Timedelta(days=self.gap_days)
            test_end = test_start + pd.Timedelta(hours=self.test_length_hours)

            train_idx = np.flatnonzero(self.prediction_timestamps < train_end)
            test_idx = np.flatnonzero(
                (self.prediction_timestamps >= test_start)
                & (self.prediction_timestamps < test_end)
            )

            if len(train_idx) and len(test_idx):
                yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return sum(1 for _ in self.split(X=X, y=y, groups=groups))


def make_direct_predictions(estimator_factory):
    predictions_by_horizon = {}
    for horizon in horizons:
        target_column_name = target_column_name_pattern.format(horizon=horizon)
        predictions_by_horizon[horizon] = features.skb.apply(
            estimator_factory(),
            y=targets[target_column_name].skb.mark_as_y(),
        )
    return predictions_by_horizon


def cross_validate_direct_predictions(predictions_by_horizon, cv):
    cv_results_by_horizon = {}
    long_results = []

    for horizon, prediction in predictions_by_horizon.items():
        cv_results = prediction.skb.cross_validate(
            cv=cv,
            scoring={
                "mape": make_scorer(mean_absolute_percentage_error),
                "r2": get_scorer("r2"),
            },
            return_train_score=True,
            return_learner=True,
            verbose=1,
            n_jobs=-1,
        )
        cv_results_by_horizon[horizon] = cv_results
        scores = cv_results.drop(columns=["learner"]).copy()
        scores["horizon"] = horizon
        scores["fold"] = np.arange(len(scores))
        long_results.append(
            scores.melt(
                id_vars=["horizon", "fold"],
                var_name="score_name",
                value_name="score",
            )
        )

    return cv_results_by_horizon, pd.concat(long_results, ignore_index=True)


def make_named_predictions(predictions_by_horizon):
    data = {}
    for horizon, prediction in predictions_by_horizon.items():
        target_column_name = target_column_name_pattern.format(horizon=horizon)
        data[f"predicted_{target_column_name}"] = to_numpy_1d(prediction.skb.eval())
    return skrub.as_data_op(pl.DataFrame(data))


def plot_scores_by_horizon(cv_scores, title_template):
    plotted_scores = cv_scores.copy()
    plotted_scores[["dataset", "metric"]] = plotted_scores["score_name"].str.split(
        "_", n=1, expand=True
    )
    plotted_scores["horizon_label"] = plotted_scores["horizon"].map(lambda h: f"{h}h")

    for metric_name, dataset_type in [("mape", "test"), ("r2", "test"), ("mape", "train"), ("r2", "train")]:
        data_to_plot = plotted_scores[
            (plotted_scores["dataset"] == dataset_type)
            & (plotted_scores["metric"] == metric_name)
        ]
        chart = (
            altair.Chart(
                data_to_plot,
                title=title_template.format(
                    dataset=dataset_type.title(), metric=metric_name.upper()
                ),
            )
            .mark_boxplot(extent="min-max")
            .encode(
                x=altair.X(
                    "horizon_label:N",
                    title="Horizon",
                    sort=[f"{h}h" for h in horizons],
                ),
                y=altair.Y("score:Q", title=f"{metric_name.upper()} score"),
                color=altair.Color("horizon_label:N", legend=None),
            )
        )
        display(chart)

hgbr_predictions_by_horizon = make_direct_predictions(
    lambda: HistGradientBoostingRegressor(
        random_state=0,
        loss=skrub.choose_from(["squared_error", "poisson", "gamma"], name="loss"),
        learning_rate=skrub.choose_float(
            0.01, 0.7, default=0.1, log=True, name="learning_rate"
        ),
        max_leaf_nodes=skrub.choose_int(
            3, 300, default=30, log=True, name="max_leaf_nodes"
        ),
    )
)

# %% [markdown]
#
# Since each horizon now has its own fitted graph, we rebuild a wide prediction
# table with one predicted column per horizon. This is only for visualization.

# %%
target_column_names = [target_column_name_pattern.format(horizon=h) for h in horizons]
named_predictions_hgbr = make_named_predictions(hgbr_predictions_by_horizon)

# %% [markdown]
#
# Let's visualize the direct forecast curve at a given timestamp.

# %%
plot_at_time = datetime.datetime(2021, 4, 19, 0, 0, tzinfo=datetime.timezone.utc)
plot_horizon_forecast(
    targets,
    named_predictions_hgbr,
    plot_at_time,
    target_column_name_pattern,
).skb.preview()

# %% [markdown]
#
# On this curve, the red line corresponds to the observed values past to the the date
# for which we would like to forecast. The orange line corresponds to the observed
# values for the next 24 hours and the blue line corresponds to the predicted values
# reconstructed from the 24 independent direct models.
#
# Since we are visualizing in-sample predictions, the forecast is expected to look
# optimistic. The real question is how each horizon performs under time-based
# cross-validation.
#
# Instead of splitting by row counts, we use a splitter based on the actual
# timestamps.

# %%
ts_cv_5 = DateBasedSplitter(prediction_timestamps)

# %%
hgbr_cv_results_by_horizon, hgbr_cv_scores = cross_validate_direct_predictions(
    hgbr_predictions_by_horizon,
    ts_cv_5,
)

# %% [markdown]
#
# This direct strategy is also expensive, and quite long, 
# perhaps there is a way to make it more parallelised?

# %%
hgbr_cv_scores.round(3)

# %% [markdown]
#
# We can now plot the distribution of the cross-validated scores for each
# horizon.

# %%
from IPython.display import display

plot_scores_by_horizon(
    hgbr_cv_scores,
    title_template="{dataset} {metric} scores by horizon (direct HistGradientBoostingRegressor)",
)

# %% [markdown]
#
# ## Direct forecasting with `RandomForestRegressor`
#
# We can compare the gradient boosting baseline against another direct
# forecasting family. We keep the same framing: one random forest per horizon.
#
# Repeat the previous analysis using a `RandomForestRegressor`. Fix the parameter
# `min_samples_leaf` to 30 to limit the depth.
#
# Once you created the model, plot the horizon forecast for a given date and time.
# In addition, compute the cross-validated predictions and plot the R2 and MAPE
# scores for each horizon.
#
# Does this model perform better or worse than the previous model?

# %%
from sklearn.ensemble import RandomForestRegressor

# %%
rf_predictions_by_horizon = make_direct_predictions(
    lambda: RandomForestRegressor(min_samples_leaf=30, random_state=0, n_jobs=-1)
)

# %%
named_predictions_rf = make_named_predictions(rf_predictions_by_horizon)

# %%
plot_at_time = datetime.datetime(2021, 4, 24, 0, 0, tzinfo=datetime.timezone.utc)
plot_horizon_forecast(
    targets,
    named_predictions_rf,
    plot_at_time,
    target_column_name_pattern,
).skb.preview()

# %%
rf_cv_results_by_horizon, rf_cv_scores = cross_validate_direct_predictions(
    rf_predictions_by_horizon,
    ts_cv_5,
)

# %%
rf_cv_scores.round(3)

# %%
plot_scores_by_horizon(
    rf_cv_scores,
    title_template="{dataset} {metric} scores by horizon (direct RandomForestRegressor)",
)

# %% [markdown]
#
