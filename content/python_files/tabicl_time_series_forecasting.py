"""
Time series forecasting
=======================
"""

# %% [markdown]
#
# This tutorial shows how to do zero-shot univariate forecasting with
# :class:`tabicl.TabICLForecaster`.
#
# Zero-shot forecasting is useful when you want a fast baseline without
# training a dedicated model for each series. It is a good starting point for
# exploration, but it should not replace repeated backtesting or model
# comparison on a real forecasting benchmark.
#
# Make sure the forecast dependencies are installed:
#
# .. code-block:: bash
#
#    pip install "tabicl[forecast]"

# %% [markdown]
# Time series forecasting as tabular regression
# ---------------------------------------------
#
# :class:`tabicl.TabICLForecaster` is inspired by
# `TabPFN-TS <https://arxiv.org/abs/2501.02945v3>`__.
#
# The forecaster wraps :class:`tabicl.TabICLRegressor` and turns forecasting
# into a tabular regression problem:
#
# - each row is one timestamp,
# - the target column is the series value,
# - time-aware features are added automatically,
# - the model directly outputs forecasts for future timestamps.
#
# In this tutorial, we use `skrub` to make the synthetic-data recipe explicit
# and inspect the resulting table. We then materialize the result as a pandas
# DataFrame before calling :meth:`tabicl.TabICLForecaster.predict_df`, whose
# forecasting interface is built around in-memory tabular data.
#
# .. note::
#
#    Compared with :class:`tabicl.TabICLClassifier` and
#    :class:`tabicl.TabICLRegressor`, the forecasting interface is newer and has
#    not yet been evaluated on a large public benchmark. We may later provide
#    evaluations and enhancements for time series forecasting.

# %%

import numpy as np
import pandas as pd
import skrub
from skrub import TableReport
from tabicl import TabICLForecaster
from tabicl.forecast import plot_forecast

# %% [markdown]
# Build a synthetic univariate series
# -----------------------------------
#
# We create a daily series that mixes a linear trend, a weekly seasonality,
# an annual seasonality, and Gaussian noise.
#
# This defined the modelling task: the model should extrapolate trend and recurring
# patterns into the future.
#
# The parameters that define the synthetic scenario are declared with
# :func:`skrub.var` so that the recipe easy to modify.
# This is a small but useful DataOps pattern: important assumptions can
# be explicitly set instead of being hidden inside the data-generation code.
# %%
random_seed = skrub.var("random_seed", 0)
n_timesteps = skrub.var("n_timesteps", 365 * 2)
trend_slope = skrub.var("trend_slope", 0.05)
weekly_amplitude = skrub.var("weekly_amplitude", 5.0)
annual_amplitude = skrub.var("annual_amplitude", 10.0)
noise_scale = skrub.var("noise_scale", 1.5)
prediction_length = skrub.var("prediction_length", 30)


@skrub.deferred
def make_synthetic_series(
	random_seed,
	n_timesteps,
	trend_slope,
	weekly_amplitude,
	annual_amplitude,
	noise_scale,
):
	rng = np.random.default_rng(random_seed)
	t = np.arange(n_timesteps)
	dates = pd.date_range(start="2022-01-01", periods=n_timesteps, freq="D")

	target = (
		trend_slope * t
		+ weekly_amplitude * np.sin(2 * np.pi * t / 7)
		+ annual_amplitude * np.sin(2 * np.pi * t / 365)
		+ rng.normal(scale=noise_scale, size=n_timesteps)
	)

	return pd.DataFrame({"timestamp": dates, "target": target})

synthetic_series = make_synthetic_series(
	random_seed,
	n_timesteps,
	trend_slope,
	weekly_amplitude,
	annual_amplitude,
	noise_scale,
)
synthetic_series

# %% [markdown]
# TabICL input format
# -------------------
#
# This synthetic table follows the input format required by
# :meth:`tabicl.TabICLForecaster.predict_df`:
#
# - required columns: ``timestamp``, ``target``
# - optional column: ``item_id`` for multiple series
# - use ``prediction_length`` for a regular future horizon, or ``future_df``
#   when future timestamps or known covariates are already available
#
# We also assume a regular daily frequency. In practice, missing timestamps or
# irregular event times should usually be addressed before applying a forecasting
# model that expects an evenly spaced history.
#
# At this point, the object is still a `skrub` expression, so we can inspect the
# planned computation before evaluating it.

# %% [markdown]
# Inspect the synthetic dataset
# -----------------------------
#
# `skrub.TableReport` gives a quick overview of the timestamp dtype, target
# distribution, and any missing values before we call the forecasting model.
#
# Even for synthetic data, this is a useful habit because it confirms that the
# generated table matches the assumptions of the downstream forecasting API.

# %%
TableReport(synthetic_series.skb.eval())

# %% [markdown]
# Materialize the input table for TabICL
# --------------------------------------
#
# The forecast API consumes an in-memory pandas DataFrame, so we evaluate the
# `skrub` expression graph at this boundary.
#
# This allows us to set the main uses as:
#
# - `skrub` helps declare and inspect the tabular data workflow,
# - TabICL handles the forecasting model itself.
# %%
df = synthetic_series.skb.eval()
prediction_length_value = prediction_length.skb.eval()
df.head()

# %% [markdown]
# Define the forecast horizon and hold-out period
# -----------------------------------------------
#
# We keep the last ``prediction_length`` points as a pseudo-future set.
#
# - ``context_df`` is the observed history provided to the forecaster
# - ``test_df`` is only used for visual comparison
#
# This is a simple single-window backtest. It is enough to illustrate the API,
# but real evaluation should use repeated rolling-origin splits so that forecast
# quality is not judged from a single final window.
#
# This split is also intentionally chronological: future rows are never allowed
# to leak into the context used by the model.

# %%
context_df = df.iloc[:-prediction_length_value]
test_df = df.iloc[-prediction_length_value:]

# %% [markdown]
# Forecast with TabICLForecaster
# ------------------------------
#
# :meth:`tabicl.TabICLForecaster.predict_df` returns one row per future
# timestamp, with a point forecast and quantile columns for uncertainty.
#
# Inspecting the returned table is useful to understand which summary columns are
# available before plotting or computing forecast metrics. In a real project,
# this is also the point where you would verify how prediction intervals are
# named and whether they are well calibrated on repeated backtests.

# %%
forecaster = TabICLForecaster()
pred_df = forecaster.predict_df(
	context_df,
	prediction_length=prediction_length_value,
)
pred_df.head()


# %% [markdown]
# Plot context, forecast, and held-out truth
# ------------------------------------------
#
# A visual comparison is often the fastest way to detect problems that a single
# aggregate metric would hide, such as phase shifts, trend drift, or overly wide
# uncertainty bands.
# %%
fig, axes = plot_forecast(context_df=context_df, pred_df=pred_df, test_df=test_df)

# %% [markdown]
# Interpreting the result
# -----------------------
#
# A good forecast should continue the upward trend and recover the recurring
# seasonal pattern visible in the held-out future values. When reading the
# figure, look for four things:
#
# - whether the forecast continues the global trend,
# - whether the weekly pattern remains aligned with the held-out truth,
# - whether the uncertainty bands widen as the horizon grows,
# - whether there is a systematic offset or lag in the forecast trajectory.
#
# In this example, ``TabICLForecaster`` tracks the synthetic trend and seasonality
# well enough to serve as a useful zero-shot baseline.
#
# On real data, the next step would usually be to repeat the same experiment on
# several backtest windows and compare the result to a simpler baseline such as
# a seasonal naive forecast.