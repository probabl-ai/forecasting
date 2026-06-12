# %% [markdown]
#
# # Multiple horizons predictive modeling
#
# ## Environment setup
#
mean_absolute_percentage_error(
# %%
)


# %%
import re
import datetime
import warnings

import altair
import cloudpickle
import pyarrow  # noqa: F401
import tzdata  # noqa: F401
import skrub
import numpy as np
import polars as pl

from sklearn.ensemble import HistGradientBoostingRegressor

from time_range import time_range
from load_electricity_and_resample import load_electricity_load_data, resample
from make_X_y import get_X_y
from add_features import add_features, fetch_city_weather
from train_test_split import TimeSeriesSplitter

from tutorial_helpers import plot_horizon_forecast


# Ignore warnings from pkg_resources triggered by Python 3.13's multiprocessing.
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

# %% [markdown]
#
# ## Predicting multiple horizons with a grid of single output models

# We now have a pipeline that makes predictions for 1 horizon. To predict
# multiple horizons, we just need to make one prediction for each horizon and
# group them in a single dataframe.


# %%
def apply_predictor(X, y, horizon):
    return (
        X.skb.apply_func(
            add_features,
            horizon=horizon,
            load_electricity_load_history=load_electricity_load_history,
            cities=cities,
            temperature_only=temperature_only,
            city_weather_fetcher=city_weather_fetcher
        )
        .skb.set_name(f"feat_{horizon}h")
        .skb.drop(["prediction_time", "target_time"])
        .skb.apply(regressor, y=y.skb.apply_func(log_transform_maybe, use_log_transform))
        .skb.apply_func(exp_transform_maybe, use_log_transform)
        .skb.set_name(f"pred_{horizon}h")
    )

def concat_horizons(predictions):
    """
    Consolidate predictions of models for different horizons in one dataframe.
    """
    return pl.DataFrame({f"{h}h": v for h, v in predictions.items()})


def make_multi_horizon_pred(horizons):
    """
    Create a full DataOp for predicting the specified horizons.
    """
    X_y = prediction_time.skb.apply_func(get_X_y, load_electricity_load_history, horizons)
    X = X_y["X"].skb.mark_as_X(cv=TimeSeriesSplitter())
    y = X_y["y"].skb.mark_as_y()
    predictions = {h: apply_predictor(X, y[f"{h}h"], h) for h in horizons}
    return skrub.deferred(concat_horizons)(predictions).skb.set_name("pred_multi_horizon")

# %% [markdown]
#
# We fetch the data and test the data pipeline for three example time horizons 1,12,24

# %%
TIME_HORIZONS = (1,12,24)
load_electricity_load_history = skrub.as_data_op(load_electricity_load_data).skb.set_name(
    "load_electricity_load_data"
)().skb.apply_func(resample)

range_start = skrub.var("start", "2021-03-23")
range_end = skrub.var("end", "2025-05-31")

prediction_time = skrub.deferred(time_range)(range_start, range_end)
X_y = prediction_time.skb.apply_func(get_X_y, load_electricity_load_history, TIME_HORIZONS)

temperature_only = skrub.choose_bool(name="temperature_only", default=True)
cities = skrub.choose_from(["all", ["paris", "lyon", "marseille"]], name="cities")
city_weather_fetcher = skrub.as_data_op(fetch_city_weather).skb.set_name(
    "city_weather_fetcher"
    )

# %% [markdown]
#
# We inspect the pipeline on an example with only 3 horizons so that it is fast
# and reasonably easy to visualize. Later we will cross-validate a pipeline for
# all horizons between 1 and 25 hours.

# %%
loss = skrub.choose_from(["squared_error", "poisson", "gamma"], name="loss")

regressor = HistGradientBoostingRegressor(
    random_state=0,
    loss=loss,
    learning_rate=skrub.choose_float(
        0.01, 0.7, default=0.1, log=True, name="learning_rate"
    ),
    max_leaf_nodes=skrub.choose_int(3, 300, default=30, log=True, name="max_leaf_nodes"),
)

# If the log is squared_error, we want to try with and without log-transforming the targets.
# Otherwise no log-transform.

use_log_transform = loss.match(
    {"squared_error": skrub.choose_bool(name="use_log_transform", default=True)},
    default=False,
)

pred = make_multi_horizon_pred(TIME_HORIZONS)
pred

# %% [markdown]
#
# We want to define a scorer that will produce the Mean Absolute Percentage
# Error (MAPE) for each of the horizons, and also averaged across horizons. To
# easily try our metric function on actual values and debug it, we collect
# ground truth and predictions on an example train/test split.

# %%
split = pred.skb.train_test_split()
learner = pred.skb.make_learner().fit(split["train"])
predicted_y_test = learner.predict(split["test"])
predicted_y_test

# %% [markdown]
#
# For multioutput regression, we can `mean_absolute_percentage_error` to return
# the error for each target, without averaging:

# %%
from sklearn.metrics import mean_absolute_percentage_error

mean_absolute_percentage_error(
    split["y_test"].drop("allow_object"), predicted_y_test, multioutput="raw_values"
)



# %% [markdown]
# We will therefore use `multioutput='raw_values'` and return all the errors in
# a dictionary, after adding the averaged error.
#
# Once we have defined this function of true and predicted electricity loads,
# (what scikit-learn calls a 'metric'), we wrap it in a 'scorer', a function
# that takes an estimator, X and y. Scorers can return a single score, or a
# dictionary mapping metric names (in our case 'neg_mape_1h', 'neg_mape_2h', ...) to
# scores.

# %%
def neg_mape(y_true, y_pred):
    average = mean_absolute_percentage_error(y_true, y_pred)
    detail = mean_absolute_percentage_error(y_true, y_pred, multioutput="raw_values")
    return {"neg_mape_average": -average} | {
        f"neg_mape_{c}": -float(s) for c, s in zip(y_true.columns, detail)
    }


def neg_mape_scorer(estimator, X, y):
    y = y.drop("allow_object")
    return neg_mape(y, estimator.predict(X))


# We set this as the default scorer on our pipeline.
pred = make_multi_horizon_pred(TIME_HORIZONS).skb.with_scoring(neg_mape_scorer)
pred


print(pred.skb.describe_param_grid())

# %% [markdown]
# Now that we have configured the scorer, we can check the score of our
# pipeline on the example split:

# %%
split["test"]['_skrub_y'] = split["test"]['_skrub_y'].drop("allow_object")
pred.skb.make_learner().fit(split["train"]).score(split["test"])

# %% [markdown]
## **Out-of-sample check:**
# it is always good that our pipeline can make a prediction on some truly
# left-out data, as a sanity check which could find bugs in the way we set it
# up or did the cross-validation.

# %%
history_dates = skrub.as_data_op(load_electricity_load_data).skb.set_name(
    "load_electricity_load_data")()["time"].skb.preview()
history_dates.max()

# %%
new_date = (
    (history_dates - datetime.timedelta(seconds=1)).dt.truncate("1h")
    + datetime.timedelta(hours=1)
).max()
new_date

# %%
# fit on all available data
learner = pred.skb.make_learner(fitted=True)
future_pred = learner.predict({"start": new_date, "end": None})
future_pred

# %%
import plotly.graph_objects as go


def plot_line(x, y):
    return go.Scatter(
        x=x,
        y=y,
        mode="lines+markers",
        name=y.name,
        hovertemplate="%{x|%Y-%m-%dT%H} (%{x|%A}): %{y}<extra></extra>",
    )

def transpose_pred(prediction_date, prediction):
    date = [
        prediction_date + datetime.timedelta(hours=int(c.removesuffix("h")))
        for c in prediction.columns
    ]
    load = prediction.row()
    return pl.DataFrame({"time": date, "load_mw": load})


future_pred_tall = transpose_pred(new_date, future_pred)

# %%
history_tail = load_electricity_load_history.skb.preview().filter(
    pl.col("time") > new_date - datetime.timedelta(days=8)
)

fig = go.Figure()
fig.add_trace(plot_line(history_tail["time"], history_tail["load_mw"]))
fig.add_trace(plot_line(future_pred_tall["time"], future_pred_tall["load_mw"]))
fig.update_layout(height=700)

# %% [markdown]
# We want to get the predictions in long rather than wide format indexed by
# date for a single prediction date might be a frequent need. We can add a
# little post-processor to the pipeline to optionally do that.

# %%
def post_process(pred):
    if range_end is not None:
        return pred
    return transpose_pred(prediction_time["time"].to_list()[0], pred)


pred = (
    make_multi_horizon_pred(TIME_HORIZONS)
    .skb.apply_func(post_process)
    .skb.with_scoring(neg_mape_scorer)
)
pred.skb.cross_validate()

# %% [markdown]
# Now we make the predictions and plot them. We notice that the 1h horizon
# qualitatively seems to stick better to the ground truth, which is expected
# and also corresponds to what we see in the MAPE.

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
        split["test"]['_skrub_y'] = split["test"]['_skrub_y'].drop("allow_object")
        prediction = data_op.skb.make_learner().fit(split["train"]).predict(split["test"])
        all_predictions.append(
            concat_X_y_predictions(
                split["X_test"], split["y_test"].drop("allow_object"), prediction
            ).with_columns(split=pl.lit(i)),
        )
        split_neg_mape = neg_mape(split["y_test"].drop("allow_object"), prediction)
        split_start = split["X_test"]["prediction_time"].min()
        fmt_mape = " ".join(
            f"{k.removeprefix('neg_mape_')}: {-v:.1%}" for k, v in split_neg_mape.items()
        )
        print(f"{split_start:%Y-%m-%d}: {fmt_mape}")
        all_scores.append(split_neg_mape | {"split": i})
    all_predictions = pl.concat(all_predictions, how="vertical")
    all_scores = pl.DataFrame(all_scores)
    return all_predictions, all_scores


cv_predictions, cv_scores = cross_val_predict(pred)

def plot_predictions(cv_predictions, horizons=None):
    if horizons is None:
        horizons = [
            int(m.group(1))
            for c in cv_predictions.columns
            if (m := re.match(r"^pred_(\d+)h$", c)) is not None
        ]
    fig = go.Figure()
    for i, h in enumerate(horizons):
        target_time = cv_predictions["prediction_time"] + datetime.timedelta(hours=h)
        if i == 0:
            fig.add_trace(
                plot_line(target_time, cv_predictions[f"{h}h"].rename("true_load"))
            )
        fig.add_trace(plot_line(target_time, cv_predictions[f"pred_{h}h"]))
    fig.update_layout(height=700)
    return fig


plot_predictions(cv_predictions)

# %% [markdown]
#
# We can save this pipeline for future reuse:

# %%
import pickle
from data_paths import results_dir

with open(results_dir() / "learner_3_horizons.pickle", "wb") as f:
    pickle.dump(pred.skb.make_learner(), f)

# %% [markdown]
# Finally, we run the cross-validation for all 24 horizons

# %%
HORIZONS = tuple(range(1, 25))

pred_24_horizons = (
    make_multi_horizon_pred(HORIZONS)
    .skb.apply_func(post_process)
    .skb.with_scoring(neg_mape_scorer)
)
cv_scores = pred_24_horizons.skb.cross_validate(verbose=2)
cv_scores

# %%
learner = pred_24_horizons.skb.make_learner(fitted=True)
future_pred = learner.predict({"start": new_date, "end": None})
fig = go.Figure()
fig.add_trace(plot_line(history_tail["time"], history_tail["load_mw"]))
fig.add_trace(plot_line(future_pred["time"], future_pred["load_mw"]))
fig.update_layout(height=700)


# %% [markdown]
# We can plot horizon vs MAPE to see if shorter horizons are easier to predict:

# %%
from matplotlib import pyplot as plt

(cv_scores.filter(regex="test_neg_mape_.*h") * -1).rename(
    columns=lambda c: c.removeprefix("test_neg_mape_")
).boxplot()
plt.xticks(rotation=45)
plt.xlabel("Horizon")
plt.ylabel("MAPE")

# %%
with open(results_dir() / "learner_24_horizons.pickle", "wb") as f:
    pickle.dump(pred_24_horizons.skb.make_learner(), f)

# %% [markdown]
## Hyperparameter Tuning
#
# We load the dataop we dumped in the previous notebook and search for the best
# hyperparameters with optuna.

# %%
import pickle
from data_paths import results_dir

with open(results_dir() / "learner_3_horizons.pickle", "rb") as f:
    pred = pickle.load(f).data_op


env = {"start": "2021-03-23", "end": "2025-05-31"}

# %% [markdown]
# We keep the last split of the default splitter as a held-out test set on
# which to validate the selected pipeline.

# %%
for outer_split in pred.skb.iter_cv_splits(env):
    pass

outer_split["X_test"]

# %% [markdown]
# We use persistent storage for our optuna database so we can resume or inspect
# it after the current process exits.

# %%
#storage = f"sqlite:///{results_dir / 'optuna.sqlite'}"
#print(f"Check search progress with:\noptuna-dashboard {storage}")
study_name = f"randomized_search"

search = pred.skb.make_randomized_search(
    backend="optuna",
    n_iter=64,
    n_jobs=1,
    refit="neg_mape_average",
    storage=None,#storage,
    study_name=study_name,
)

search.fit(outer_split["train"])
with open(results_dir / "randomized_search.pickle", "wb") as f:
    pickle.dump(search, f)

with open(results_dir / "best_learner.pickle", "wb") as f:
    pickle.dump(search.best_learner_, f)

search.results_.to_csv("search_results.csv", index=False)
search.score(outer_split["test"])

# %%
search.plot_results()

# %% [markdown]
# 
# Make an example prediction

# %%
search.predict({"start": "2025-06-27T15:00:00", "end": None})

    
