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
import datetime
import re
import warnings
from pathlib import Path

import altair
import cloudpickle
import numpy as np
import polars as pl
import pyarrow  # noqa: F401
import skrub
import tzdata  # noqa: F401
from sklearn.ensemble import HistGradientBoostingRegressor

from feature_engineering_lib import (
    feature_engineering_outputs,
    fetch_demand_history,
)
from single_horizon_prediction_lib import (
    TimeSeriesSplitter,
    get_cv_results,
    get_regressor,
)
from tutorial_helpers import plot_multi_horizon_prediction, plot_predictions

# Ignore warnings from pkg_resources triggered by Python 3.13's multiprocessing.
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

# %% [markdown]
#
# ## Predicting multiple horizons with a grid of single output models

# We now have a pipeline that makes predictions for 1 horizon. To predict
# multiple horizons, we just need to make one prediction for each horizon and
# group them in a single dataframe. This follows the strategy of
# [Direct multi-step forecasting](https://skforecast.org/latest/introduction-forecasting/introduction-forecasting#direct-multi-step-forecasting)
# where an independent model is trained for each horizon.
# ![Direct multi-step forecasting](https://skforecast.org/latest/img/diagram-direct-multi-step-forecasting.png)


# %%


def concat_horizons(predictions):
    """
    Consolidate predictions of models for different horizons in one dataframe.
    """
    return pl.DataFrame({f"{h}h": v for h, v in predictions.items()})


def make_multi_horizon_pred(features, y, max_iter=100):
    """
    Create a full DataOp for predicting the specified horizons.

    increase max_iter for better predictions
    """
    regressor = get_regressor(max_iter=max_iter)
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
TIME_HORIZONS = (1, 12, 24)
features, y = feature_engineering_outputs(TIME_HORIZONS, cv_splitter=TimeSeriesSplitter())

pred = make_multi_horizon_pred(features, y)
pred

# %% [markdown]
#
# We want to define a scorer that will produce the Mean Absolute Percentage
# Error (MAPE) for each of the horizons, and also averaged across horizons.
#

# %%
from sklearn.metrics import mean_absolute_percentage_error


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
# Now we make the predictions and plot them. We notice that the 1h horizon
# qualitatively seems to stick better to the ground truth, which is expected
# and also corresponds to what we see in the MAPE.

# %%
cv_predictions, cv_scores = get_cv_results(pred)

# %%
plot_predictions(cv_predictions, horizon=12)

# %% [markdown]
#
# ## Evaluating and plotting the 24-horizon pipeline

# %%
features_24_horizons, y_24_horizons = feature_engineering_outputs(
    range(1, 25), TimeSeriesSplitter()
)
pred_24_horizons = make_multi_horizon_pred(features_24_horizons, y_24_horizons).skb.with_scoring(
    neg_mape_scorer
)

# %% [markdown]
#
# We can plot the prediction for the first 24 hours past the end of our data
#
# It is always good that our pipeline can make a prediction on some truly
# left-out data, as a sanity check which could find bugs in the way we set it
# up or did the cross-validation.

# %%
history = fetch_demand_history()
history_dates = history["time"]
new_date = ((history_dates - datetime.timedelta(seconds=1)).dt.truncate("1h")).max()
new_date

# %%
learner = pred_24_horizons.skb.make_learner(fitted=True)
prediction = learner.predict({"start": new_date, "end": None})
prediction

# %%
plot_multi_horizon_prediction(history, new_date, prediction)

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
outer_split = pred.skb.train_test_split(env)
outer_split["X_test"]

# %% [markdown]
# We use persistent storage for our optuna database so we can resume or inspect
# it after the current process exits.

# %%
# %pip install -q optuna
# import optuna
# print(f"optuna version: {optuna.__version__}")

# %% [markdown]
# To keep the notebook fast we reload an existing search that has 100
# iterations (specified by storage and study_name), and resume it to add 2 more
# iterations.

# %%
search = pred.skb.make_randomized_search(
    backend="optuna",
    storage="sqlite:///multiple_horizons_prediction.optuna",
    study_name="randomized_search",
    n_iter=2,
    n_jobs=1,
    refit="neg_mape_average",
)

search.fit(outer_split["train"])
search.score(outer_split["test"])

# %%
search.plot_results()
