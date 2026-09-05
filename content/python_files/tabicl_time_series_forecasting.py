"""
Time series forecasting with TabICL on electricity load
=======================================================
"""

# %%
# %pip install -q https://pypi.anaconda.org/ogrisel/simple/polars/1.24.0/polars-1.24.0-cp39-abi3-emscripten_3_1_58_wasm32.whl
# %pip install -q skrub altair holidays plotly nbformat
# %pip install "tabicl[forecast]"

# %%
from datetime import UTC, datetime, timedelta
import functools
import re
import warnings

import numpy as np
import plotly.graph_objects as go
import skrub
import polars as pl

from tabicl import TabICLRegressor

from tutorial_helpers import plot_lorenz_curve, plot_reliability_diagram
from feature_engineering_lib import feature_engineering_outputs
from next_horizon_prediction_lib import TimeSeriesSplitter
from prediction_intervals_lib import (
    concat_horizons,
    cross_val_predict,
    neg_mape,
    neg_mape_scorer,
    pinball,
    pinball_scorer,
    plot_predictions,
)

warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")


def tabicl_quantiles_to_df(prediction, quantiles, mode=skrub.eval_mode()):
    if mode == "fit":
        return prediction
    return pl.DataFrame(prediction, schema=[f"q_{q}" for q in quantiles])


def limit_train_size(df, size=9000, mode=skrub.eval_mode()):
    if mode in ("fit", "fit_transform", "preview"):
        return df.tail(size)
    return df


def sanitize_tabicl_matrix(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(X).all():
        X = np.where(np.isfinite(X), X, 0.0)
    return X


def make_multi_horizon_pred_tabicl(features, y, quantiles):
    quantiles_to_predict = skrub.as_data_op(quantiles).skb.set_name("quantiles")
    predictor = TabICLRegressor(n_estimators=1)
    predict_kwargs = {"output_type": "quantiles", "alphas": quantiles_to_predict}
    predictions = {
        h: feat.skb.drop(["prediction_time", "target_time"])
        .skb.apply(skrub.ToFloat())
        .to_numpy()
        .skb.apply_func(sanitize_tabicl_matrix)
        .skb.apply(predictor, y=y[f"{h}h"], predict_kwargs=predict_kwargs)
        .skb.apply_func(tabicl_quantiles_to_df, quantiles_to_predict)
        .skb.set_name(f"pred_{h}h")
        for h, feat in features.items()
    }
    return skrub.deferred(concat_horizons)(predictions)


def concat_X_y_predictions(X_test, y_test, prediction):
    return pl.concat(
        [
            X_test,
            y_test,
            prediction.rename("pred_{}".format),
        ],
        how="horizontal",
    )


# %%
TIME_HORIZONS = (1, 12, 24)
QUANTILES = (0.05, 0.5, 0.95)

features, y = feature_engineering_outputs(TIME_HORIZONS, TimeSeriesSplitter())
features = {h: features[h].skb.apply_func(limit_train_size) for h in TIME_HORIZONS}
y = y.skb.apply_func(limit_train_size)

pred = (
    make_multi_horizon_pred_tabicl(
        features,
        y,
        quantiles=QUANTILES,
    )
    .skb.with_scoring(functools.partial(neg_mape_scorer, quantile_regression=True))
    .skb.with_scoring(pinball_scorer)
)

cv_predictions_tabicl = cross_val_predict(
    pred,
    environment={"start": "2023-01-01", "end": "2025-05-31"},
)

# %%
plot_predictions(
    cv_predictions_tabicl[0], horizons=TIME_HORIZONS, start="2023-01-01"
).show()

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
