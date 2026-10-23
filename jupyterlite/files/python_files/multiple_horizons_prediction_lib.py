from feature_engineering_lib import (
    feature_engineering_outputs,
    fetch_demand_history,
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
from tutorial_helpers import plot_multi_horizon_prediction, plot_predictions
import altair
import cloudpickle
import datetime
import numpy as np
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


def neg_mape(y_true, y_pred):
    average = mean_absolute_percentage_error(y_true, y_pred)
    detail = mean_absolute_percentage_error(y_true, y_pred, multioutput="raw_values")
    return {"neg_mape_average": -average} | {
        f"neg_mape_{c}": -float(s) for c, s in zip(y_true.columns, detail)
    }


def neg_mape_scorer(estimator, X, y):
    return neg_mape(y, estimator.predict(X))
