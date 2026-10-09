from feature_engineering_lib import (
    feature_engineering_outputs,
    load_electricity_history_data,
)
from matplotlib import pyplot as plt
from pathlib import Path
from single_horizon_prediction_lib import (
    TimeSeriesSplitter,
    get_cv_results,
    get_regressor,
)
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error
from tutorial_helpers import plot_horizon_forecast
import altair
import cloudpickle
import datetime
import numpy as np
import plotly.graph_objects as go
import polars as pl
import pyarrow  # noqa: F401
import re
import skrub
import tzdata  # noqa: F401
import warnings


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


def neg_mape(y_true, y_pred):
    average = mean_absolute_percentage_error(y_true, y_pred)
    detail = mean_absolute_percentage_error(y_true, y_pred, multioutput="raw_values")
    return {"neg_mape_average": -average} | {
        f"neg_mape_{c}": -float(s) for c, s in zip(y_true.columns, detail)
    }


def neg_mape_scorer(estimator, X, y):
    return neg_mape(y, estimator.predict(X))


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
            fig.add_trace(plot_line(target_time, cv_predictions[f"{h}h"].rename("true_load")))
        fig.add_trace(plot_line(target_time, cv_predictions[f"pred_{h}h"]))
    fig.update_layout(height=700)
    return fig
