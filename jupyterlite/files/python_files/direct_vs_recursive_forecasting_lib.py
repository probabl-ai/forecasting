from mlforecast import MLForecast
from mlforecast.lag_transforms import (
    RollingMax,
    RollingMean,
    RollingMin,
    RollingStd,
)
from mlforecast.target_transforms import Differences
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.tree import DecisionTreeRegressor
from time import perf_counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import threadpoolctl
import tzdata  # noqa: F401
import warnings


def generate_synthetic_1(
    segment_length=SEGMENT_LENGTH,
    n_segments=100,
    low_noise_level=0.01,
    high_noise_level=0.1,
    seed=None,
):
    """Generate synthetic time series data with two types of segments

    - segment type "a" has a prefix centered around 0 and a suffix centered
      around 1.
    - segment type "b" has a prefix centered around 0 with high variance and a
      suffix centered around -1.

    The variance of the prefix is therefore predictive of the suffix.

    The suffix values predictive of the next segment prefix's mean (always 0).
    """
    rng = np.random.default_rng(seed)
    total_length = segment_length * n_segments
    segment_types = rng.choice(["a", "b"], n_segments)
    prefix_length = segment_length // 2
    suffix_length = segment_length - prefix_length

    segments = []
    for segment_type in segment_types:
        if segment_type == "a":
            # Prefix is centered around 0 with low variance
            segments.append(rng.normal(loc=0, scale=low_noise_level, size=prefix_length))
            # Suffix is centered around 1 with low variance
            segments.append(rng.normal(loc=1, scale=low_noise_level, size=suffix_length))
        elif segment_type == "b":
            # Prefix is also centered around 0 but with high variance
            segments.append(rng.normal(loc=0, scale=high_noise_level, size=prefix_length))
            # Suffix is centered around -1 with low variance
            segments.append(rng.normal(loc=-1, scale=low_noise_level, size=suffix_length))
    return pd.DataFrame(
        {
            "time": np.arange(total_length),
            "y": np.concatenate(segments),
            "series_id": np.zeros(total_length, dtype=np.int32),
        }
    )


def collect_predictions(mlf, data_test, test_offset=0):
    """Collect predictions from the MLForecast object."""
    all_predictions = []
    UPDATE_CHUNK_SIZE = 5
    while test_offset < len(data_test):

        new_predictions = mlf.predict(PREDICTION_HORIZON)
        new_predictions["horizon"] = np.arange(new_predictions.shape[0]) + 1
        new_predictions = new_predictions.merge(data_test, on=["time", "series_id"], how="left")
        all_predictions.append(new_predictions)

        # Update the forecaster with the new observations
        mlf.update(data_test.iloc[test_offset : test_offset + UPDATE_CHUNK_SIZE])
        test_offset += UPDATE_CHUNK_SIZE

    return all_predictions


def score_predictions(all_predictions, model_name):
    """Compute the mean absolute error of the predictions."""
    all_predictions = pd.concat(all_predictions)
    all_predictions["absolute_error"] = np.abs(all_predictions["y"] - all_predictions[model_name])
    return all_predictions.dropna().groupby("horizon")


def plot_some_predictions(all_predictions, data_test, model_name, nrows=12, title=None):

    fig, axes = plt.subplots(nrows=nrows, figsize=(15, 5 * nrows))
    for row_idx, predictions in enumerate(all_predictions):
        predictions = predictions.drop("y", axis=1)
        merged_data = data_test.copy()
        merged_data = merged_data.merge(predictions, on=["time", "series_id"], how="left")
        merged_data.drop(["series_id"], axis=1).iloc[: SEGMENT_LENGTH * 3].plot(
            x="time", y=["y", model_name], ax=axes[row_idx]
        )
        axes[row_idx].set_title(title)
        axes[row_idx].set_ylim(-1.2, 1.2)

        if row_idx >= nrows - 1:
            break
