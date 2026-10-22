# %% [markdown]

# # Time series forecasting with TabICL on electricity load

# ## Environment setup

# %%
# %pip install -q https://pypi.anaconda.org/ogrisel/simple/polars/1.24.0/polars-1.24.0-cp39-abi3-emscripten_3_1_58_wasm32.whl
# %pip install -q skrub altair holidays plotly nbformat
# %pip install "tabicl[forecast]"

# %%
import functools
import importlib
import re
import warnings
from datetime import UTC, datetime, timedelta

import numpy as np
import plotly.graph_objects as go
import polars as pl
import skrub
from tabicl import TabICLRegressor

import tutorial_helpers

importlib.reload(tutorial_helpers)
from feature_engineering_lib import feature_engineering_outputs
from prediction_intervals_lib import (
    concat_horizons,
    neg_mape_median_scorer,
    pinball_scorer,
)
from single_horizon_prediction_lib import TimeSeriesSplitter, get_cv_results
from tutorial_helpers import plot_lorenz_curve, plot_predictions, plot_reliability_diagram

warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")


# %%
def tabicl_quantiles_to_df(prediction, quantiles, mode=skrub.eval_mode()):
    if mode == "fit":
        return prediction
    return pl.DataFrame(prediction, schema=[f"q_{q}" for q in quantiles])


def make_multi_horizon_pred_tabicl(features, y, quantiles):
    quantiles_to_predict = skrub.as_data_op(quantiles).skb.set_name("quantiles")
    predictor = TabICLRegressor(n_estimators=1)
    predict_kwargs = {"output_type": "quantiles", "alphas": quantiles_to_predict}
    predictions = {
        h: feat.skb.drop(["prediction_time", "target_time"])
        .skb.apply(skrub.ToFloat())
        .to_numpy()
        .skb.apply(predictor, y=y[f"{h}h"], predict_kwargs=predict_kwargs)
        .skb.apply_func(tabicl_quantiles_to_df, quantiles_to_predict)
        .skb.set_name(f"pred_{h}h")
        for h, feat in features.items()
    }
    return skrub.deferred(concat_horizons)(predictions)


# %%
TIME_HORIZONS = (1, 12, 24)
QUANTILES = (0.05, 0.5, 0.95)

features, y = feature_engineering_outputs(
    TIME_HORIZONS,
    TimeSeriesSplitter(),
    max_train_size=tutorial_helpers.TABICL_TRAIN_SIZE,  # 9000 by default, 1500 in CI
)

pred = (
    make_multi_horizon_pred_tabicl(
        features,
        y,
        quantiles=QUANTILES,
    )
    .skb.with_scoring(neg_mape_median_scorer)
    .skb.with_scoring(pinball_scorer)
)

cv_predictions_tabicl = get_cv_results(pred)

# %%
plot_predictions(cv_predictions_tabicl[0], horizon=12, start="2023-01-01").show()

# %%
plot_reliability_diagram(
    cv_predictions_tabicl[0],
    1,
    forecast_quantile=0.50,
).interactive().properties(title="TabICL reliability diagram for quantile 0.50")

# %%
plot_reliability_diagram(
    cv_predictions_tabicl[0],
    1,
    forecast_quantile=0.05,
).interactive().properties(title="TabICL reliability diagram for quantile 0.05")

# %%
plot_reliability_diagram(
    cv_predictions_tabicl[0],
    1,
    forecast_quantile=0.95,
).interactive().properties(title="TabICL reliability diagram for quantile 0.95")

# %%
plot_lorenz_curve(
    cv_predictions_tabicl[0],
    1,
    quantile=0.50,
).interactive().properties(title="TabICL Lorenz curve for quantile 0.50")

# %%
plot_lorenz_curve(
    cv_predictions_tabicl[0],
    1,
    quantile=0.05,
).interactive().properties(title="TabICL Lorenz curve for quantile 0.05")

# %%
plot_lorenz_curve(
    cv_predictions_tabicl[0],
    1,
    quantile=0.95,
).interactive().properties(title="TabICL Lorenz curve for quantile 0.95")
