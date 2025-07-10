# %% [markdown]
#
# # Multiple horizons predictive modeling
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
import datetime
import warnings

import altair
import cloudpickle
import pyarrow  # noqa: F401
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
# ## Predicting multiple horizons with a grid of single output models
#
# Usually, it is really common to predict values for multiple horizons at once. The most
# naive approach is to train as many models as there are horizons. To achieve this,
# scikit-learn provides a meta-estimator called `MultiOutputRegressor` that can be used
# to train a single model that predicts multiple horizons at once.
#
# In short, we only need to provide multiple targets where each column corresponds to
# an horizon and this meta-estimator will train an independent model for each column.
# However, we could expect that the quality of the forecast might degrade as the horizon
# increases.
#
# Let's train a gradient boosting regressor for each horizon.

# %%
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

multioutput_predictions = features.skb.apply(
    MultiOutputRegressor(
        estimator=HistGradientBoostingRegressor(random_state=0), n_jobs=-1
    ),
    y=targets.skb.drop(cols=["prediction_time", "load_mw"]).skb.mark_as_y(),
)

# %% [markdown]
#
# Now, let's just rename the columns for the predictions to make it easier to plot
# the horizon forecast.

# %%
target_column_names = [target_column_name_pattern.format(horizon=h) for h in horizons]
predicted_target_column_names = [
    f"predicted_{target_column_name}" for target_column_name in target_column_names
]
named_predictions = multioutput_predictions.rename(
    {k: v for k, v in zip(target_column_names, predicted_target_column_names)}
)

# %% [markdown]
#
# Let's plot the horizon forecast on a training data to check the validity of the
# output.

# %%
plot_at_time = datetime.datetime(2021, 4, 19, 0, 0, tzinfo=datetime.timezone.utc)
plot_horizon_forecast(
    targets,
    named_predictions,
    plot_at_time,
    target_column_name_pattern,
).skb.preview()

# %% [markdown]
#
# On this curve, the red line corresponds to the observed values past to the the date
# for which we would like to forecast. The orange line corresponds to the observed
# values for the next 24 hours and the blue line corresponds to the predicted values
# for the next 24 hours.
#
# Since we are using a strong model and very few training data to check the validity
# we observe that our model perfectly fits the training data.
#
# So, we are now ready to assess the performance of this multi-output model and we need
# to cross-validate it. Since we do not want to aggregate the metrics for the different
# horizons, we need to create a scikit-learn scorer in which we set
# `multioutput="raw_values"` to get the scores for each horizon.
#
# Passing this scorer to the `cross_validate` function returns all horizons scores.

# %%
from sklearn.model_selection import TimeSeriesSplit


max_train_size = 2 * 52 * 24 * 7  # max ~2 years of training data
test_size = 24 * 7 * 24  # 24 weeks of test data
gap = 7 * 24  # 1 week gap between train and test sets
ts_cv_5 = TimeSeriesSplit(
    n_splits=5, max_train_size=max_train_size, test_size=test_size, gap=gap
)

# %%
from sklearn.metrics import r2_score, mean_absolute_percentage_error


def multioutput_scorer(regressor, X, y, score_func, score_name):
    y_pred = regressor.predict(X)
    return {
        f"{score_name}_horizon_{h}h": score
        for h, score in enumerate(
            score_func(y, y_pred, multioutput="raw_values"), start=1
        )
    }


def scoring(regressor, X, y):
    return {
        **multioutput_scorer(regressor, X, y, mean_absolute_percentage_error, "mape"),
        **multioutput_scorer(regressor, X, y, r2_score, "r2"),
    }


multioutput_cv_results = multioutput_predictions.skb.cross_validate(
    cv=ts_cv_5,
    scoring=scoring,
    return_train_score=True,
    verbose=1,
    n_jobs=-1,
)

# %% [markdown]
#
# One thing that we observe is that training such multi-output model is expensive. It is
# expected since each horizon involves a different model and thus a training.

# %%
multioutput_cv_results.round(3)

# %% [markdown]
#
# Instead of reading the results in the table, we can plot the scores depending on the
# type of data and the metric.

# %%
import itertools
from IPython.display import display

for metric_name, dataset_type in itertools.product(["mape", "r2"], ["train", "test"]):
    columns = multioutput_cv_results.columns[
        multioutput_cv_results.columns.str.startswith(f"{dataset_type}_{metric_name}")
    ]
    data_to_plot = multioutput_cv_results[columns]
    data_to_plot.columns = [
        col.replace(f"{dataset_type}_", "")
        .replace(f"{metric_name}_", "")
        .replace("_", " ")
        for col in columns
    ]

    data_long = data_to_plot.melt(var_name="horizon", value_name="score")
    chart = (
        altair.Chart(
            data_long,
            title=f"{dataset_type.title()} {metric_name.upper()} scores by horizon",
        )
        .mark_boxplot(extent="min-max")
        .encode(
            x=altair.X(
                "horizon:N",
                title="Horizon",
                sort=altair.Sort(
                    [f"horizon {h}h" for h in range(1, data_to_plot.shape[1])]
                ),
            ),
            y=altair.Y("score:Q", title=f"{metric_name.upper()} Score"),
            color=altair.Color("horizon:N", legend=None),
        )
    )

    display(chart)

# %% [markdown]
#
# An interesting and unexpected observation is that the MAPE error on the test
# data is first increases and then decreases once past the horizon 18h. We
# would not necessarily expect this behaviour.
#
# ## Native multi-output handling using `RandomForestRegressor`
#
# In the previous section, we showed how to wrap a `HistGradientBoostingRegressor`
# in a `MultiOutputRegressor` to predict multiple horizons. With such a strategy, it
# means that we trained independent `HistGradientBoostingRegressor`, one for each
# horizon.
#
# `RandomForestRegressor` natively supports multi-output regression: instead of
# independently training a model per horizon, it will train a joint model that
# predicts all horizons at once.
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
# Write your code here.
#
#
#
#
#
#
#
#
#
#
#

# %%
multioutput_predictions_rf = features.skb.apply(
    RandomForestRegressor(min_samples_leaf=30, random_state=0, n_jobs=-1),
    y=targets.skb.drop(cols=["prediction_time", "load_mw"]).skb.mark_as_y(),
)

# %%
named_predictions_rf = multioutput_predictions_rf.rename(
    {k: v for k, v in zip(target_column_names, predicted_target_column_names)}
)

# %%
plot_at_time = datetime.datetime(2021, 4, 24, 0, 0, tzinfo=datetime.timezone.utc)
plot_horizon_forecast(
    targets,
    named_predictions_rf,
    plot_at_time,
    target_column_name_pattern,
).skb.preview()

# %%
multioutput_cv_results_rf = multioutput_predictions_rf.skb.cross_validate(
    cv=ts_cv_5,
    scoring=scoring,
    return_train_score=True,
    verbose=1,
    n_jobs=-1,
)

# %%
multioutput_cv_results_rf.round(3)

# %%
import itertools
from IPython.display import display

for metric_name, dataset_type in itertools.product(["mape", "r2"], ["train", "test"]):
    columns = multioutput_cv_results_rf.columns[
        multioutput_cv_results_rf.columns.str.startswith(
            f"{dataset_type}_{metric_name}"
        )
    ]
    data_to_plot = multioutput_cv_results_rf[columns]
    data_to_plot.columns = [
        col.replace(f"{dataset_type}_", "")
        .replace(f"{metric_name}_", "")
        .replace("_", " ")
        for col in columns
    ]

    data_long = data_to_plot.melt(var_name="horizon", value_name="score")
    chart = (
        altair.Chart(
            data_long,
            title=f"{dataset_type.title()} {metric_name.upper()} Scores by Horizon",
        )
        .mark_boxplot(extent="min-max")
        .encode(
            x=altair.X(
                "horizon:N",
                title="Horizon",
                sort=altair.Sort(
                    [f"horizon {h}h" for h in range(1, data_to_plot.shape[1])]
                ),
            ),
            y=altair.Y("score:Q", title=f"{metric_name.upper()} Score"),
            color=altair.Color("horizon:N", legend=None),
        )
    )

    display(chart)

# %% [markdown]
#
# We observe that the performance of the `RandomForestRegressor` is not better in terms
# of scores or computational cost. The trend of the scores along the horizon is also
# different from the `HistGradientBoostingRegressor`: the scores worsen as the horizon
# increases.
