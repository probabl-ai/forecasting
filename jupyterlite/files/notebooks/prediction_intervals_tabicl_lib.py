from datetime import UTC, datetime, timedelta
from feature_engineering_lib import feature_engineering_outputs
from prediction_intervals_lib import (
    concat_horizons,
    neg_mape_median_scorer,
    pinball_scorer,
)
from single_horizon_prediction_lib import TimeSeriesSplitter, get_cv_results
from tabicl import TabICLRegressor
from tutorial_helpers import plot_lorenz_curve, plot_predictions, plot_reliability_diagram
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
