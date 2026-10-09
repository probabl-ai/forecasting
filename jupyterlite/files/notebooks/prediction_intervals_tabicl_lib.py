from datetime import UTC, datetime, timedelta
from feature_engineering_lib import feature_engineering_outputs
from prediction_intervals_lib import (
    concat_horizons,
    cross_val_predict,
    neg_mape,
    neg_mape_scorer,
    pinball,
    pinball_scorer,
    plot_predictions,
)
from single_horizon_prediction_lib import TimeSeriesSplitter
from tabicl import TabICLRegressor
from tutorial_helpers import plot_lorenz_curve, plot_reliability_diagram
import functools
import importlib
import numpy as np
import plotly.graph_objects as go
import polars as pl
import re
import skrub
import tutorial_helpers
import warnings


def tabicl_quantiles_to_df(prediction, quantiles, mode=skrub.eval_mode()):
    if mode == "fit":
        return prediction
    return pl.DataFrame(prediction, schema=[f"q_{q}" for q in quantiles])


def limit_train_size(
    df,
    size=tutorial_helpers.TABICL_TRAIN_SIZE,  # 9000 by default, 1500 in CI
    mode=skrub.eval_mode(),
):
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
