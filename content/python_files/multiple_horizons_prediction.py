# %% [markdown]
#
# # Multiple horizons predictive modeling
#
# ## Environment setup
#
# We need to install some extra dependencies for this notebook if needed (when
# running jupyterlite).

# %%
# %pip install -q skrub altair holidays plotly nbformat polars

# %%
import re
import datetime
import warnings
from pathlib import Path

import altair
import cloudpickle
import pyarrow  # noqa: F401
import tzdata  # noqa: F401
import skrub
import numpy as np
import polars as pl

from sklearn.ensemble import HistGradientBoostingRegressor


from feature_engineering_lib import feature_engineering_outputs, load_electricity_history_data

from next_horizon_prediction_lib import TimeSeriesSplitter, get_regressor, get_cv_results

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

def concat_horizons(predictions):
    """
    Consolidate predictions of models for different horizons in one dataframe.
    """
    return pl.DataFrame({f"{h}h": v for h, v in predictions.items()})
def make_multi_horizon_pred(features, y):
    """
    Create a full DataOp for predicting the specified horizons.
    """
    regressor = get_regressor()
    predictions = {
        h: feat.skb.drop(["prediction_time", "target_time"])
        .skb.apply(regressor, y=y[f"{h}h"])
        .skb.set_name(f"pred_{h}h")
        for h, feat in features.items()
    }
    return skrub.deferred(concat_horizons)(predictions)

# %% [markdown]
#
# We inspect the pipeline on an example with only 3 horizons so that it is fast
# and reasonably easy to visualize. Later we will cross-validate a pipeline for
# all horizons between 1 and 25 hours.

# %%
TIME_HORIZONS = (1,12,24)
features, y = feature_engineering_outputs(TIME_HORIZONS, cv_splitter=TimeSeriesSplitter())

pred = make_multi_horizon_pred(features, y)
pred.skb.preview()

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

mean_absolute_percentage_error(split["y_test"], predicted_y_test, multioutput="raw_values")



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
    return neg_mape(y, estimator.predict(X))


# We set this as the default scorer on our pipeline.
pred = make_multi_horizon_pred(features, y).skb.with_scoring(neg_mape_scorer)


# %% [markdown]
# Now that we have configured the scorer, we can check the score of our
# pipeline on the example split:

# %%
pred.skb.make_learner().fit(split["train"]).score(split["test"])

# %% [markdown]
## **Out-of-sample check:**
# it is always good that our pipeline can make a prediction on some truly
# left-out data, as a sanity check which could find bugs in the way we set it
# up or did the cross-validation.

# %%
electricity_load_history = load_electricity_history_data()
history_dates = electricity_load_history["time"]
history_dates.max()

# %%
new_date = (
    (history_dates - datetime.timedelta(seconds=1)).dt.truncate("1h")
).max()
new_date

# %%
# fit a model for 24 horizons on all available data
features_24_horizons, y_24_horizons = feature_engineering_outputs(range(1, 25), TimeSeriesSplitter())
pred_24_horizons = make_multi_horizon_pred(features_24_horizons, y_24_horizons).skb.with_scoring(neg_mape_scorer)
learner = pred_24_horizons.skb.make_learner(fitted=True)
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
    load = prediction.row(0)
    return pl.DataFrame({"time": date, "load_mw": load})


future_pred_tall = transpose_pred(new_date, future_pred)

# %%
history_tail = electricity_load_history.filter(
    pl.col("time") > new_date - datetime.timedelta(days=8)
)

fig = go.Figure()
fig.add_trace(plot_line(history_tail["time"], history_tail["load_mw"]))
fig.add_trace(plot_line(future_pred_tall["time"], future_pred_tall["load_mw"]))
fig.update_layout(height=700)

# %% [markdown]
# Now we make the predictions and plot them. We notice that the 1h horizon
# qualitatively seems to stick better to the ground truth, which is expected
# and also corresponds to what we see in the MAPE.

# %%
cv_predictions, cv_scores = get_cv_results(pred)

# %%

def plot_predictions(cv_predictions, horizons=None, start="2025-03-01"):
    if start is not None:
        cv_predictions = cv_predictions.filter(
            pl.col("prediction_time")
            > datetime.datetime.fromisoformat(start).astimezone(datetime.UTC)
        )

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
# Finally, we run the cross-validation for all 24 horizons

# %%
cv_scores = pred_24_horizons.skb.cross_validate(verbose=2)
cv_scores

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

# %% [markdown]
## Hyperparameter Tuning
#
# We load the dataop we dumped in the previous notebook and search for the best
# hyperparameters with optuna.

# %%
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
# %pip install -q optuna
# import optuna
# print(f"optuna version: {optuna.__version__}")

# %%
#storage = f"sqlite:///{results_dir / 'optuna.sqlite'}"
#print(f"Check search progress with:\noptuna-dashboard {storage}")
study_name = f"randomized_search"

search = pred.skb.make_randomized_search(
    backend="optuna",
    n_iter=10,
    n_jobs=1,
    refit="neg_mape_average",
    storage=None,#storage,
    study_name=study_name,
)

search.fit(outer_split["train"])
search.score(outer_split["test"])

# %%
search.plot_results()

# %% [markdown]
# 
# Make an example prediction

# %%
search.predict({"start": "2025-06-27T15:00:00", "end": None})

    
