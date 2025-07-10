# %% [markdown]
# # Feature engineering for electricity load forecasting
#
# The purpose of this notebook is to demonstrate how to use `skrub` and
# `polars` to perform feature engineering for electricity load forecasting.
#
# We will build a set of features (and targets) from different data sources:
#
# - Historical weather data for 10 medium to large urban areas in France;
# - Holidays and standard calendar features for France;
# - Historical electricity load data for the whole of France.
#
# All these data sources cover a time range from March 23, 2021 to May 31,
# 2025.
#
# Since our maximum forecasting horizon is 24 hours, we consider that the
# future weather data is known at a chosen prediction time. Similarly, the
# holidays and calendar features are known at prediction time for any point in
# the future.
#
# Therefore, exogenous features derived from the weather and calendar data can
# be used to engineer "future covariates". Since the load data is our
# prediction target, we will can also use it to engineer "past covariates" such
# as lagged features and rolling aggregations. The future values of the load
# data (with respect to the prediction time) are used as targets for the
# forecasting model.
#
# ## Environment setup
#
# We need to install some extra dependencies for this notebook if needed (when
# running jupyterlite). We need the development version of skrub to be able to
# use the skrub expressions.

# %%
# %pip install -q https://pypi.anaconda.org/ogrisel/simple/polars/1.24.0/polars-1.24.0-cp39-abi3-emscripten_3_1_58_wasm32.whl
# %pip install -q https://pypi.anaconda.org/ogrisel/simple/skrub/0.6.dev0/skrub-0.6.dev0-py3-none-any.whl
# %pip install -q altair holidays plotly nbformat

# %% [markdown]
#
# The following 3 imports are only needed to workaround some limitations when
# using polars in a pyodide/jupyterlite notebook.
#
# TODO: remove those workarounds once pyodide 0.28 is released with support for
# the latest polars version.

# %%
import datetime
import tzdata  # noqa: F401
import pandas as pd
from pyarrow.parquet import read_table

import altair
import numpy as np
import polars as pl
import skrub
from pathlib import Path
import holidays
import warnings

from plotly.io import write_json, read_json  # noqa: F401

from tutorial_helpers import (
    binned_coverage,
    plot_lorenz_curve,
    plot_reliability_diagram,
    plot_residuals_vs_predicted,
    plot_binned_residuals,
    plot_horizon_forecast,
    collect_cv_predictions,
)

# Ignore warnings from pkg_resources triggered by Python 3.13's multiprocessing.
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")


# %% [markdown]
# ## Shared time range for all historical data sources
#
# Let's define a hourly time range from March 23, 2021 to May 31, 2025 that
# will be used to join the electricity load data and the weather data. The time
# range is in UTC timezone to avoid any ambiguity when joining with the weather
# data that is also in UTC.
#
# We wrap the resulting polars dataframe in a `skrub` expression to benefit
# from the built-in `skrub.TableReport` display in the notebook. Using the
# `skrub` expression system will also be useful for other reasons: all
# operations in this notebook chain operations chained together in a directed
# acyclic graph that is automatically tracked by `skrub`. This allows us to
# extract the resulting pipeline and apply it to new data later on, exactly
# like a trained scikit-learn pipeline. The main difference is that we do so
# incrementally and while eagerly executing and inspecting the results of each
# step as is customary when working with dataframe libraries such as polars and
# pandas in Jupyter notebooks.

# %%
historical_data_start_time = skrub.var(
    "historical_data_start_time", pl.datetime(2021, 3, 23, hour=0, time_zone="UTC")
)
historical_data_end_time = skrub.var(
    "historical_data_end_time", pl.datetime(2025, 5, 31, hour=23, time_zone="UTC")
)


# %%
@skrub.deferred
def build_historical_time_range(
    historical_data_start_time,
    historical_data_end_time,
    time_interval="1h",
    time_zone="UTC",
):
    """Define an historical time range shared by all data sources."""
    return pl.DataFrame().with_columns(
        pl.datetime_range(
            start=historical_data_start_time,
            end=historical_data_end_time,
            time_zone=time_zone,
            interval=time_interval,
        ).alias("time"),
    )


time = build_historical_time_range(historical_data_start_time, historical_data_end_time)
time

# %% [markdown]
#
# If you run the above locally with pydot and graphviz installed, you can
# visualize the expression graph of the `time` variable by expanding the "Show
# graph" button.
#
# Let's now load the data records for the time range defined above.
#
# To avoid network issues when running this notebook, the necessary data files
# have already been downloaded and saved in the `datasets` folder. See the
# README.md file for instructions to download the data manually if you want to
# re-run this notebook with more recent data.

# %%
data_source_folder = skrub.var("data_source_folder", Path("../datasets"))

for data_file in sorted(data_source_folder.skb.eval().iterdir()):
    print(data_file)

# %% [markdown]
#
# We define a list of 10 medium to large urban areas to approximately cover
# most regions in France with a slight focus on most populated regions that are
# likely to drive electricity demand.

# %%
city_names = skrub.var(
    "city_names",
    [
        "paris",
        "lyon",
        "marseille",
        "toulouse",
        "lille",
        "limoges",
        "nantes",
        "strasbourg",
        "brest",
        "bayonne",
    ],
)


@skrub.deferred
def load_weather_data(time, city_names, data_source_folder):
    """Load and horizontal stack historical weather forecast data for each city."""
    all_city_weather = time
    for city_name in city_names:
        all_city_weather = all_city_weather.join(
            pl.from_arrow(
                read_table(f"{data_source_folder}/weather_{city_name}.parquet")
            )
            .with_columns([pl.col("time").dt.cast_time_unit("us")])
            .rename(lambda x: x if x == "time" else "weather_" + x + "_" + city_name),
            on="time",
        )
    return all_city_weather


all_city_weather = load_weather_data(time, city_names, data_source_folder)
all_city_weather


# %% [markdown]
# ## Calendar and holidays features
#
# We leverage the `holidays` package to enrich the time range with some
# calendar features such as public holidays in France. We also add some
# features that are useful for time series forecasting such as the day of the
# week, the day of the year, and the hour of the day.
#
# Note that the `holidays` package requires us to extract the date for the
# French timezone.
#
# Similarly for the calendar features: all the time features are extracted from
# the time in the French timezone, since it is likely that electricity usage
# patterns are influenced by inhabitants' daily routines aligned with the local
# timezone.


# %%
@skrub.deferred
def prepare_french_calendar_data(time):
    fr_time = pl.col("time").dt.convert_time_zone("Europe/Paris")
    fr_year_min = time.select(fr_time.dt.year().min()).item()
    fr_year_max = time.select(fr_time.dt.year().max()).item()
    holidays_fr = holidays.country_holidays(
        "FR", years=range(fr_year_min, fr_year_max + 1)
    )
    return time.with_columns(
        [
            fr_time.dt.hour().alias("cal_hour_of_day"),
            fr_time.dt.weekday().alias("cal_day_of_week"),
            fr_time.dt.ordinal_day().alias("cal_day_of_year"),
            fr_time.dt.year().alias("cal_year"),
            fr_time.dt.date().is_in(holidays_fr.keys()).alias("cal_is_holiday"),
        ],
    )


calendar = prepare_french_calendar_data(time)
calendar


# %% [markdown]
#
# ## Electricity load data
#
# Finally we load the electricity load data. This data will both be used as a
# target variable but also to craft some lagged and window-aggregated features.
# %%
@skrub.deferred
def load_electricity_load_data(time, data_source_folder):
    """Load and aggregate historical load data from the raw CSV files."""
    load_data_files = [
        data_file
        for data_file in sorted(data_source_folder.iterdir())
        if data_file.name.startswith("Total Load - Day Ahead")
        and data_file.name.endswith(".csv")
    ]
    return time.join(
        (
            pl.concat(
                [
                    pl.from_pandas(pd.read_csv(data_file, na_values=["N/A", "-"])).drop(
                        ["Day-ahead Total Load Forecast [MW] - BZN|FR"]
                    )
                    for data_file in load_data_files
                ]
            ).select(
                [
                    pl.col("Time (UTC)")
                    .str.split(by=" - ")
                    .list.first()
                    .str.to_datetime("%d.%m.%Y %H:%M", time_zone="UTC")
                    .alias("time"),
                    pl.col("Actual Total Load [MW] - BZN|FR").alias("load_mw"),
                ]
            )
        ),
        on="time",
    )

# %% [markdown]
#
# Let's load the data and check if there are missing values since we will use
# this data as the target variable for our forecasting model.

# %%
electricity_raw = load_electricity_load_data(time, data_source_folder)
electricity_raw.filter(pl.col("load_mw").is_null())

# %% [markdown]
#
# So apparently there a few missing measurements. Let's use linear
# interpolation to fill those missing values.

# %%
electricity_raw.filter(
    (pl.col("time") > pl.datetime(2021, 10, 30, hour=10, time_zone="UTC"))
    & (pl.col("time") < pl.datetime(2021, 10, 31, hour=10, time_zone="UTC"))
).skb.eval().plot.line(x="time:T", y="load_mw:Q")

# %%
electricity = electricity_raw.with_columns([pl.col("load_mw").interpolate()])
electricity.filter(
    (pl.col("time") > pl.datetime(2021, 10, 30, hour=10, time_zone="UTC"))
    & (pl.col("time") < pl.datetime(2021, 10, 31, hour=10, time_zone="UTC"))
).skb.eval().plot.line(x="time:T", y="load_mw:Q")

# %% [markdown]
#
# **Remark**: interpolating missing values in the target column that we will
# use to train and evaluate our models can bias the learning problem and make
# our cross-validation metrics misrepresent the performance of the deployed
# predictive system.
#
# A potentially better approach would be to keep the missing values in the
# dataset and use a sample_weight mask to keep a contiguous dataset while
# ignoring the time periods with missing values when training or evaluating the
# model.

# %% [markdown]
#
# ## Lagged features
#
# We can now create some lagged features from the electricity load data.
#
# We will create 3 hourly lagged features, 1 daily lagged feature, and 1 weekly
# lagged feature. We will also create a rolling median and inter-quartile
# feature over the last 24 hours and over the last 7 days.


# %%
def iqr(col, *, window_size: int):
    """Inter-quartile range (IQR) of a column."""
    return col.rolling_quantile(0.75, window_size=window_size) - col.rolling_quantile(
        0.25, window_size=window_size
    )


electricity_lagged = electricity.with_columns(
    [pl.col("load_mw").shift(i).alias(f"load_mw_lag_{i}h") for i in range(1, 4)]
    + [
        pl.col("load_mw").shift(24).alias("load_mw_lag_1d"),
        pl.col("load_mw").shift(24 * 7).alias("load_mw_lag_1w"),
        pl.col("load_mw")
        .rolling_median(window_size=24)
        .alias("load_mw_rolling_median_24h"),
        pl.col("load_mw")
        .rolling_median(window_size=24 * 7)
        .alias("load_mw_rolling_median_7d"),
        iqr(pl.col("load_mw"), window_size=24).alias("load_mw_iqr_24h"),
        iqr(pl.col("load_mw"), window_size=24 * 7).alias("load_mw_iqr_7d"),
    ],
)
electricity_lagged

# %%
altair.Chart(electricity_lagged.tail(100).skb.preview()).transform_fold(
    [
        "load_mw",
        "load_mw_lag_1h",
        "load_mw_lag_2h",
        "load_mw_lag_3h",
        "load_mw_lag_1d",
        "load_mw_lag_1w",
        "load_mw_rolling_median_24h",
        "load_mw_rolling_median_7d",
        "load_mw_rolling_iqr_24h",
        "load_mw_rolling_iqr_7d",
    ],
    as_=["key", "load_mw"],
).mark_line(tooltip=True).encode(x="time:T", y="load_mw:Q", color="key:N").interactive()

# %% [markdown]
#
# ## Important remark about lagged features engineering and system lag
#
# When working with historical data, we often have access to all the past
# measurements in the dataset. However, when we want to use the lagged features
# in a forecasting model, we need to be careful about the length of the
# **system lag**: the time between a timestamped measurement is made in the
# real world and the time the record is made available to the downstream
# application (in our case, a deployed predictive pipeline).
#
# System lag is rarely explicitly represented in the data sources even if such
# delay can be as large as several hours or even days and can sometimes be
# irregular. For instance, if there is a human intervention in the data
# recording process, holidays and weekends can punctually add significant
# delay.
#
# If the system lag is larger than the maximum feature engineering lag, the
# resulting features be filled with missing values once deployed. More
# importantly, if the system lag is not handled explicitly, those resulting
# missing values will only be present in the features computed for the
# deployed system but not present in the features computed to train and
# backtest the system before deployment.
#
# This structural discrepancy can severely degrade the performance of the
# deployed model compared to the performance estimated from backtesting on the
# historical data.
#
# We will set this problem aside for now but discuss it again in a later
# section of this tutorial.

# %% [markdown]
# ## Investigating outliers in the lagged features
#
# Let's use the `skrub.TableReport` tool to look at the plots of the marginal
# distribution of the lagged features.

# %%
from skrub import TableReport

TableReport(electricity_lagged.skb.eval())

# %% [markdown]
#
# Let's extract the dates where the inter-quartile range of the load is
# greater than 15,000 MW.

# %%
electricity_lagged.filter(pl.col("load_mw_iqr_7d") > 15_000)[
    "time"
].dt.date().unique().sort().to_list().skb.eval()

# %% [markdown]
#
# We observe 3 date ranges with high inter-quartile range. Let's plot the
# electricity load and the lagged features for the first data range along with
# the weather data for Paris.

# %%
altair.Chart(
    electricity_lagged.filter(
        (pl.col("time") > pl.datetime(2021, 12, 1, time_zone="UTC"))
        & (pl.col("time") < pl.datetime(2021, 12, 31, time_zone="UTC"))
    ).skb.eval()
).transform_fold(
    [
        "load_mw",
        "load_mw_iqr_7d",
    ],
).mark_line(
    tooltip=True
).encode(
    x="time:T", y="value:Q", color="key:N"
).interactive()

# %%
altair.Chart(
    all_city_weather.filter(
        (pl.col("time") > pl.datetime(2021, 12, 1, time_zone="UTC"))
        & (pl.col("time") < pl.datetime(2021, 12, 31, time_zone="UTC"))
    ).skb.eval()
).transform_fold(
    [f"weather_temperature_2m_{city_name}" for city_name in city_names.skb.eval()],
).mark_line(
    tooltip=True
).encode(
    x="time:T", y="value:Q", color="key:N"
).interactive()

# %% [markdown]
#
# Based on the plots above, we can see that the electricity load was high just
# before the Christmas holidays due to low temperatures. Then the load suddenly
# dropped because temperatures went higher right at the start of the
# end-of-year holidays.
#
# So those outliers do not seem to be caused to a data quality issue but rather
# due to a real change in the electricity load demand. We could conduct similar
# analysis for the other date ranges with high inter-quartile range but we will
# skip that for now.
#
# If we had observed significant data quality issues over extended periods of
# time could have been addressed by removing the corresponding rows from the
# dataset. However, this would make the lagged and windowing feature
# engineering challenging to reimplement correctly. A better approach would be
# to keep a contiguous dataset assign 0 weights to the affected rows when
# fitting or evaluating the trained models via the use of the `sample_weight`
# parameter.

# %% [markdown]
# ## Final dataset
#
# We now assemble the dataset that will be used to train and evaluate the forecasting
# models via backtesting.

# %%
prediction_start_time = skrub.var(
    "prediction_start_time", historical_data_start_time.skb.eval() + pl.duration(days=7)
)
prediction_end_time = skrub.var(
    "prediction_end_time", historical_data_end_time.skb.eval() - pl.duration(hours=24)
)


@skrub.deferred
def define_prediction_time_range(prediction_start_time, prediction_end_time):
    return pl.DataFrame().with_columns(
        pl.datetime_range(
            start=prediction_start_time,
            end=prediction_end_time,
            time_zone="UTC",
            interval="1h",
        ).alias("prediction_time"),
    )


prediction_time = define_prediction_time_range(
    prediction_start_time, prediction_end_time
).skb.subsample(n=1000, how="head")
prediction_time


# %%
@skrub.deferred
def build_features(
    prediction_time,
    electricity_lagged,
    all_city_weather,
    calendar,
    future_feature_horizons=[1, 24],
):

    return (
        prediction_time.join(
            electricity_lagged, left_on="prediction_time", right_on="time"
        )
        .join(
            all_city_weather.select(
                [pl.col("time")]
                + [
                    pl.col(c).shift(-h).alias(c + f"_future_{h}h")
                    for c in all_city_weather.columns
                    if c != "time"
                    for h in future_feature_horizons
                ]
            ),
            left_on="prediction_time",
            right_on="time",
        )
        .join(
            calendar.select(
                [pl.col("time")]
                + [
                    pl.col(c).shift(-h).alias(c + f"_future_{h}h")
                    for c in calendar.columns
                    if c != "time"
                    for h in future_feature_horizons
                ]
            ),
            left_on="prediction_time",
            right_on="time",
        )
    ).drop("prediction_time")


features = build_features(
    prediction_time=prediction_time,
    electricity_lagged=electricity_lagged,
    all_city_weather=all_city_weather,
    calendar=calendar,
).skb.mark_as_X()

features

# %% [markdown]
#
# Let's build training and evaluation targets for all possible horizons from 1
# to 24 hours.

# %%
horizons = range(1, 25)
target_column_name_pattern = "load_mw_horizon_{horizon}h"


@skrub.deferred
def build_targets(prediction_time, electricity, horizons):
    return prediction_time.join(
        electricity.with_columns(
            [
                pl.col("load_mw")
                .shift(-h)
                .alias(target_column_name_pattern.format(horizon=h))
                for h in horizons
            ]
        ),
        left_on="prediction_time",
        right_on="time",
    )


targets = build_targets(prediction_time, electricity, horizons)
targets


# %% [markdown]
# For now, let's focus on the last horizon (24 hours) to train a model
# predicting the electricity load at the next 24 hours.
# %%
horizon_of_interest = horizons[-1]  # Focus on the 24-hour horizon
target_column_name = target_column_name_pattern.format(horizon=horizon_of_interest)
predicted_target_column_name = "predicted_" + target_column_name
target = targets[target_column_name].skb.mark_as_y()
target

# %% [markdown]
#
# Let's define our first single output prediction pipeline. This pipeline
# chains our previous feature engineering steps with a `skrub.DropCols` step to
# drop some columns that we do not want to use as features, and a
# `HistGradientBoostingRegressor` model from scikit-learn.
#
# The `skrub.choose_from`, `skrub.choose_float`, and `skrub.choose_int`
# functions are used to define hyperparameters that can be tuned via
# cross-validated randomized search.

# %%
from sklearn.ensemble import HistGradientBoostingRegressor
import skrub.selectors as s


features_with_dropped_cols = features.skb.apply(
    skrub.DropCols(
        cols=skrub.choose_from(
            {
                "none": s.glob(""),  # No column has an empty name.
                "load": s.glob("load_*"),
                "rolling_load": s.glob("load_mw_rolling_*"),
                "weather": s.glob("weather_*"),
                "temperature": s.glob("weather_temperature_*"),
                "moisture": s.glob("weather_moisture_*"),
                "cloud_cover": s.glob("weather_cloud_cover_*"),
                "calendar": s.glob("cal_*"),
                "holiday": s.glob("cal_is_holiday*"),
                "future_1h": s.glob("*_future_1h"),
                "future_24h": s.glob("*_future_24h"),
                "non_paris_weather": s.glob("weather_*") & ~s.glob("weather_*_paris_*"),
            },
            name="dropped_cols",
        )
    )
)

hgbr_predictions = features_with_dropped_cols.skb.apply(
    HistGradientBoostingRegressor(
        random_state=0,
        loss=skrub.choose_from(["squared_error", "poisson", "gamma"], name="loss"),
        learning_rate=skrub.choose_float(
            0.01, 1, default=0.1, log=True, name="learning_rate"
        ),
        max_leaf_nodes=skrub.choose_int(
            3, 300, default=30, log=True, name="max_leaf_nodes"
        ),
    ),
    y=target,
)
hgbr_predictions

# %% [markdown]
#
# The `predictions` expression captures the whole expression graph that
# includes the feature engineering steps, the target variable, and the model
# training step.
#
# In particular, the input data keys for the full pipeline can be
# inspected as follows:

# %%
hgbr_predictions.skb.get_data().keys()

# %% [markdown]
#
# Furthermore, the hyper-parameters of the full pipeline can be retrieved as
# follows:

# %%
hgbr_pipeline = hgbr_predictions.skb.get_pipeline()
hgbr_pipeline.describe_params()

# %% [markdown]
#
# When running this notebook locally, you can also interactively inspect all
# the steps of the DAG using the following (once uncommented):

# %%
# predictions.skb.full_report()

# %% [markdown]
#
# Since we passed input values to all the upstream `skrub` variables, `skrub`
# automatically evaluates the whole expression graph graph (train and predict
# on the same data) so that we can interactively check that everything will
# work as expected.
#
# Let's use altair to visualize the predictions against the target values for
# the last 24 hours of the prediction time range used to train the model. This
# allows us can (over)fit the data with the features at hand.

# %%
altair.Chart(
    pl.concat(
        [
            targets.skb.preview(),
            hgbr_predictions.rename(
                {target_column_name: predicted_target_column_name}
            ).skb.preview(),
        ],
        how="horizontal",
    ).tail(24 * 7)
).transform_fold(
    [target_column_name, predicted_target_column_name],
).mark_line(
    tooltip=True
).encode(
    x="prediction_time:T", y="value:Q", color="key:N"
).interactive()

# %% [markdown]
#
# ## Assessing the model performance via cross-validation
#
# Being able to fit the training data is not enough. We need to assess the
# ability of the training pipeline to learn a predictive model that can
# generalize to unseen data.
#
# Furthermore, we want to assess the uncertainty of this estimate of the
# generalization performance via time-based cross-validation, also known as
# backtesting.
#
# scikit-learn provides a `TimeSeriesSplit` splitter providing a convenient way to
# split temporal data: in the different folds, the training data always precedes the
# test data. It implies that the size of the training data is getting larger as the
# fold index increases. The scikit-learn utility allows to define a couple of
# parameters to control the size of the training and test data and as well as a gap
# between the training and test data to potentially avoid leakage if our model relies
# on lagged features.
#
# In the example below, we define that the training data should be at most 2 years
# worth of data and the test data should be 24 weeks long. We also define a gap of
# 1 week between the training.
#
# Let's check those statistics by iterating over the different folds provided by the
# splitter.

# %%
from sklearn.model_selection import TimeSeriesSplit


max_train_size = 2 * 52 * 24 * 7  # max ~2 years of training data
test_size = 24 * 7 * 24  # 24 weeks of test data
gap = 7 * 24  # 1 week gap between train and test sets
ts_cv_5 = TimeSeriesSplit(
    n_splits=5, max_train_size=max_train_size, test_size=test_size, gap=gap
)

for fold_idx, (train_idx, test_idx) in enumerate(
    ts_cv_5.split(prediction_time.skb.eval())
):
    print(f"CV iteration #{fold_idx}")
    train_datetimes = prediction_time.skb.eval()[train_idx]
    test_datetimes = prediction_time.skb.eval()[test_idx]
    print(
        f"Train: {train_datetimes.shape[0]} rows, "
        f"Test: {test_datetimes.shape[0]} rows"
    )
    print(f"Train time range: {train_datetimes[0, 0]} to " f"{train_datetimes[-1, 0]} ")
    print(f"Test time range: {test_datetimes[0, 0]} to " f"{test_datetimes[-1, 0]} ")
    print()

# %% [markdown]
#
# Once the cross-validation strategy is defined, we pass it to the `cross_validate`
# function provided by `skrub` to compute the cross-validated scores. Here, we define
# the mean absolute percentage error that is interpretable. However, this metric is
# not a proper scoring rule. We therefore look at the R2 score and the Tweedie deviance
# score.

# %%
from sklearn.metrics import make_scorer, mean_absolute_percentage_error, get_scorer
from sklearn.metrics import d2_tweedie_score


hgbr_cv_results = hgbr_predictions.skb.cross_validate(
    cv=ts_cv_5,
    scoring={
        "mape": make_scorer(mean_absolute_percentage_error),
        "r2": get_scorer("r2"),
        "d2_poisson": make_scorer(d2_tweedie_score, power=1.0),
        "d2_gamma": make_scorer(d2_tweedie_score, power=2.0),
    },
    return_train_score=True,
    return_pipeline=True,
    verbose=1,
    n_jobs=-1,
)
hgbr_cv_results.round(3)

# %% [markdown]
#
# TODO: comment the results obtained via cross-validation.
#
# We further analyze our cross-validated model by collecting the predictions on each
# split.


# %%
hgbr_cv_predictions = collect_cv_predictions(
    hgbr_cv_results["pipeline"], ts_cv_5, hgbr_predictions, prediction_time
)
hgbr_cv_predictions[0]

# %% [markdown]
#
# The first curve is called the Lorenz curve. It shows on the x-axis the fraction of
# observations sorted by predicted values and on the y-axis the cumulative observed
# load proportion.

# %%
plot_lorenz_curve(hgbr_cv_predictions).interactive()

# %% [markdown]
#
# The diagonal on the plot corresponds to a model predicting a constant value that is
# therefore not an informative model. The oracle model corresponds to the "perfect"
# model that would provide the an output identical to the observed values. Thus, the
# ranking of such hypothetical model is the best possible ranking. However, you should
# note that the oracle model is not the line passing through the right-hand corner of
# the plot. Instead, this curvature is defined by the distribution of the observations.
# Indeed, more the observations are composed of small values and a couple of large
# values, the more the oracle model is closer to the right-hand corner of the plot.
#
# A true model is navigating between the diagonal and the oracle model. The area between
# the diagonal and the Lorenz curve of a model is called the Gini index.
#
# For our model, we observe that each oracle model is not far from the diagonal. It
# means that the observed values do not contain a couple of large values with high
# variability. Therefore, it informs us that the complexity of our problem at hand is
# not too high. Looking at the Lorenz curve of each model, we observe that it is quite
# close to the oracle model. Therefore, the gradient boosting regressor is
# discriminative for our task.
#
# Then, we have a look at the reliability diagram. This diagram shows on the x-axis the
# mean predicted load and on the y-axis the mean observed load.

# %%
plot_reliability_diagram(hgbr_cv_predictions).interactive().properties(
    title="Reliability diagram from cross-validation predictions"
)

# %% [markdown]
#
# The diagonal on the reliability diagram corresponds to the best possible model: for
# a level of predicted load that fall into a bin, then the mean observed load is also
# in the same bin. If the line is above the diagonal, it means that our model is
# predicted a value too low in comparison to the observed values. If the line is below
# the diagonal, it means that our model is predicted a value too high in comparison to
# the observed values.
#
# For our cross-validated model, we observe that each reliability curve is close to the
# diagonal. We only observe a mis-calibration for the extremum values.

# %%
plot_residuals_vs_predicted(hgbr_cv_predictions).interactive().properties(
    title="Residuals vs Predicted Values from cross-validation predictions"
)

# %%
plot_binned_residuals(hgbr_cv_predictions, by="hour").interactive().properties(
    title="Residuals by hour of the day from cross-validation predictions"
)

# %%
plot_binned_residuals(hgbr_cv_predictions, by="month").interactive().properties(
    title="Residuals by hour of the day from cross-validation predictions"
)

# %%
ts_cv_2 = TimeSeriesSplit(
    n_splits=2, test_size=test_size, max_train_size=max_train_size, gap=24
)
# randomized_search_hgbr = hgbr_predictions.skb.get_randomized_search(
#     cv=ts_cv_2,
#     scoring="r2",
#     n_iter=100,
#     fitted=True,
#     verbose=1,
#     n_jobs=-1,
# )
# # %%
# randomized_search_hgbr.results_.round(3)

# %%
# fig = randomized_search_hgbr.plot_results().update_layout(margin=dict(l=200))
# write_json(fig, "parallel_coordinates_hgbr.json")

# %%
fig = read_json("parallel_coordinates_hgbr.json")
fig.update_layout(margin=dict(l=200))

# %%
# nested_cv_results = skrub.cross_validate(
#     environment=predictions.skb.get_data(),
#     pipeline=randomized_search,
#     cv=ts_cv_5,
#     scoring={
#         "r2": get_scorer("r2"),
#         "mape": make_scorer(mean_absolute_percentage_error),
#     },
#     n_jobs=-1,
#     return_pipeline=True,
# ).round(3)
# nested_cv_results

# %%
# for outer_fold_idx in range(len(nested_cv_results)):
#     print(
#         nested_cv_results.loc[outer_fold_idx, "pipeline"]
#         .results_.loc[0]
#         .round(3)
#         .to_dict()
#     )

# %% [markdown]
#
# ### Exercise: non-linear feature engineering coupled with linear predictive model
#
# Now, it is your turn to make a predictive model. Towards this end, we request you
# to preprocess the input features with non-linear feature engineering:
#
# - the first step is to impute the missing values using a `SimpleImputer`. Make sure
#   to include the indicator of missing values in the feature set (i.e. look at the
#   `add_indicator` parameter);
# - use a `SplineTransformer` to create non-linear features. Use the default parameters
#   but make sure to set `sparse_output=True` since it subsequent processing will be
#   faster and more memory efficient with such data structure;
# - use a `VarianceThreshold` to remove features with potential constant features;
# - use a `SelectKBest` to select the most informative features. Set `k` to be chosen
#   from a log-uniform distribution between 100 and 1,000 (i.e. use `skrub.choose_int`);
# - use a `Nystroem` to approximate an RBF kernel. Set `n_components` to be chosen
#   from a log-uniform distribution between 10 and 200 (i.e. use `skrub.choose_int`).
# - finally, use a `Ridge` as the final predictive model. Set `alpha` to be
#   chosen from a log-uniform distribution between 1e-6 and 1e3 (i.e. use
#   `skrub.choose_float`).
#
# Use a scikit-learn `Pipeline` using `make_pipeline` to chain the steps together.
#
# Once the predictive model is defined, apply it on the `feature_with_dropped_cols`
# expression. Do not forget to define that `target` is the `y` variable.


# %%
# Here we provide all the imports for creating the predictive model.
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

# %%
# Write your code here.
#
#
#
#
#
#
#
#
#
#
#

# %%
predictions_ridge = features_with_dropped_cols.skb.apply(
    make_pipeline(
        SimpleImputer(add_indicator=True),
        SplineTransformer(sparse_output=True),
        VarianceThreshold(threshold=1e-6),
        SelectKBest(
            k=skrub.choose_int(100, 1_000, log=True, name="n_selected_splines")
        ),
        Nystroem(
            n_components=skrub.choose_int(
                10, 200, log=True, name="n_components", default=150
            )
        ),
        Ridge(
            alpha=skrub.choose_float(1e-6, 1e3, log=True, name="alpha", default=1e-2)
        ),
    ),
    y=target,
)
predictions_ridge

# %% [markdown]
#
# Now that you defined the predictive model, let's make a similar analysis than earlier.
# First, let's make a sanity check that plot forecast of our model on a subset of the
# training data to make a sanity check.

# %%
# Write your code here.
#
#
#
#
#
#
#
#
#
#
#

# %%
altair.Chart(
    pl.concat(
        [
            targets.skb.preview(),
            predictions_ridge.rename(
                {target_column_name: predicted_target_column_name}
            ).skb.preview(),
        ],
        how="horizontal",
    ).tail(24 * 7)
).transform_fold(
    [target_column_name, predicted_target_column_name],
).mark_line(
    tooltip=True
).encode(
    x="prediction_time:T", y="value:Q", color="key:N"
).interactive()


# %% [markdown]
#
# Now, let's evaluate the performance of the model using cross-validation. Use the
# time-based cross-validation splitter `ts_cv_5` defined earlier. Make sure to compute
# the R2 score and the mean absolute percentage error. Return the training scores as
# well as the fitted pipeline such that we can make additional analysis.
#
# Does this model perform better or worse than the previous model?
# Is it underfitting or overfitting?

# %%
# Write your code here.
#
#
#
#
#
#
#
#
#
#
#

# %%
cv_results_ridge = predictions_ridge.skb.cross_validate(
    cv=ts_cv_5,
    scoring={
        "r2": get_scorer("r2"),
        "mape": make_scorer(mean_absolute_percentage_error),
    },
    return_train_score=True,
    return_pipeline=True,
    verbose=1,
    n_jobs=-1,
)

# %% [markdown]
#
# Compute all cross-validated predictions to plot the Lorenz curve and the
# reliability diagram for this pipeline.
#
# To do so, you can use the function `collect_cv_predictions` to collect the
# predictions and then call the `plot_lorenz_curve` and
# `plot_reliability_diagram` functions to plot the results.

# %%
# Write your code here.
#
#
#
#
#
#
#
#
#
#
#

# %%
cv_predictions_ridge = collect_cv_predictions(
    cv_results_ridge["pipeline"], ts_cv_5, predictions_ridge, prediction_time
)

# %%
plot_lorenz_curve(cv_predictions_ridge).interactive()

# %%
plot_reliability_diagram(cv_predictions_ridge).interactive().properties(
    title="Reliability diagram from cross-validation predictions"
)

# %% [markdown]
#
# Now, let's perform a randomized search on the hyper-parameters of the model. The code
# to perform the search is shown below. Since it will be pretty computationally
# expensive, we are reloading the results of the parallel coordinates plot.

# %%
# randomized_search_ridge = predictions_ridge.skb.get_randomized_search(
#     cv=ts_cv_2,
#     scoring="r2",
#     n_iter=100,
#     fitted=True,
#     verbose=1,
#     n_jobs=-1,
# )

# %%
# fig = randomized_search_ridge.plot_results().update_layout(margin=dict(l=200))
# write_json(fig, "parallel_coordinates_ridge.json")

# %%
fig = read_json("parallel_coordinates_ridge.json")
fig.update_layout(margin=dict(l=200))

# %% [markdown]
#
# We observe that the default values of the hyper-parameters are in the optimal
# region explored by the randomized search. This is a good sign that the model
# is well-specified and that the hyper-parameters are not too sensitive to
# small changes of those values.
#
# We could further assess the stability of those optimal hyper-parameters by
# running a nested cross-validation, where we would perform a randomized search
# for each fold of the outer cross-validation loop as below but this is
# computationally expensive.

# %%
# nested_cv_results_ridge = skrub.cross_validate(
#     environment=predictions_ridge.skb.get_data(),
#     pipeline=randomized_search_ridge,
#     cv=ts_cv_5,
#     scoring={
#         "r2": get_scorer("r2"),
#         "mape": make_scorer(mean_absolute_percentage_error),
#     },
#     n_jobs=-1,
#     return_pipeline=True,
# ).round(3)

# %%
# nested_cv_results_ridge.round(3)

# %% [markdown]
#
# ## Predicting multiple horizons with a multi-output model
#
# Usually, it is really common to predict values for multiple horizons at once. The most
# naive approach is to train as many models as there are horizons. To achieve this,
# scikit-learn provides a meta-estimator called `MultiOutputRegressor` that can be used
# to train a single model that predicts multiple horizons at once.
#
# In short, we only need to provide multiple targets where each column corresponds to
# an horizon and this meta-estimator will train an independent model for each column.
# However, we could expect that the quality of the forecast might degrade as the horizon
# increases.
#
# Let's train a gradient boosting regressor for each horizon.

# %%
from sklearn.multioutput import MultiOutputRegressor

multioutput_predictions = features_with_dropped_cols.skb.apply(
    MultiOutputRegressor(
        estimator=HistGradientBoostingRegressor(random_state=0), n_jobs=-1
    ),
    y=targets.skb.drop(cols=["prediction_time", "load_mw"]).skb.mark_as_y(),
).skb.set_name("multioutput_hgbr")

# %% [markdown]
#
# Now, let's just rename the columns for the predictions to make it easier to plot
# the horizon forecast.

# %%
target_column_names = [target_column_name_pattern.format(horizon=h) for h in horizons]
predicted_target_column_names = [
    f"predicted_{target_column_name}" for target_column_name in target_column_names
]
named_predictions = multioutput_predictions.rename(
    {k: v for k, v in zip(target_column_names, predicted_target_column_names)}
)

# %% [markdown]
#
# Let's plot the horizon forecast on a training data to check the validity of the
# output.

# %%
plot_at_time = datetime.datetime(2021, 4, 19, 0, 0, tzinfo=datetime.timezone.utc)
historical_timedelta = datetime.timedelta(hours=24 * 5)
plot_horizon_forecast(
    targets,
    named_predictions,
    plot_at_time,
    historical_timedelta,
    target_column_name_pattern,
).skb.preview()

# %% [markdown]
#
# On this curve, the red line corresponds to the observed values past to the the date
# for which we would like to forecast. The orange line corresponds to the observed
# values for the next 24 hours and the blue line corresponds to the predicted values
# for the next 24 hours.
#
# Since we are using a strong model and very few training data to check the validity
# we observe that our model perfectly fits the training data.
#
# So, we are now ready to assess the performance of this multi-output model and we need
# to cross-validate it. Since we do not want to aggregate the metrics for the different
# horizons, we need to create a scikit-learn scorer in which we set
# `multioutput="raw_values"` to get the scores for each horizon.
#
# Passing this scorer to the `cross_validate` function returns all horizons scores.

# %%
from sklearn.metrics import r2_score


def multioutput_scorer(regressor, X, y, score_func, score_name):
    y_pred = regressor.predict(X)
    return {
        f"{score_name}_horizon_{h}h": score
        for h, score in enumerate(
            score_func(y, y_pred, multioutput="raw_values"), start=1
        )
    }


def scoring(regressor, X, y):
    return {
        **multioutput_scorer(regressor, X, y, mean_absolute_percentage_error, "mape"),
        **multioutput_scorer(regressor, X, y, r2_score, "r2"),
    }


multioutput_cv_results = multioutput_predictions.skb.cross_validate(
    cv=ts_cv_5,
    scoring=scoring,
    return_train_score=True,
    verbose=1,
    n_jobs=-1,
)

# %% [markdown]
#
# One thing that we observe is that training such multi-output model is expensive. It is
# expected since each horizon involves a different model and thus a training.

# %%
multioutput_cv_results.round(3)

# %% [markdown]
#
# Instead of reading the results in the table, we can plot the scores depending on the
# type of data and the metric.

# %%
import itertools
from IPython.display import display

for metric_name, dataset_type in itertools.product(["mape", "r2"], ["train", "test"]):
    columns = multioutput_cv_results.columns[
        multioutput_cv_results.columns.str.startswith(f"{dataset_type}_{metric_name}")
    ]
    data_to_plot = multioutput_cv_results[columns]
    data_to_plot.columns = [
        col.replace(f"{dataset_type}_", "")
        .replace(f"{metric_name}_", "")
        .replace("_", " ")
        for col in columns
    ]

    data_long = data_to_plot.melt(var_name="horizon", value_name="score")
    chart = (
        altair.Chart(
            data_long,
            title=f"{dataset_type.title()} {metric_name.upper()} Scores by Horizon",
        )
        .mark_boxplot(extent="min-max")
        .encode(
            x=altair.X(
                "horizon:N",
                title="Horizon",
                sort=altair.Sort(
                    [f"horizon {h}h" for h in range(1, data_to_plot.shape[1])]
                ),
            ),
            y=altair.Y("score:Q", title=f"{metric_name.upper()} Score"),
            color=altair.Color("horizon:N", legend=None),
        )
    )

    display(chart)

# %% [markdown]
#
# An interesting and unexpected observation is that the MAPE error on the test
# data is first increases and then decreases once past the horizon 18h. We
# would not necessarily expect this behaviour.
#
# ## Native multi-output handling using `RandomForestRegressor`
#
# In the previous section, we showed how to wrap a `HistGradientBoostingRegressor`
# in a `MultiOutputRegressor` to predict multiple horizons. With such a strategy, it
# means that we trained independent `HistGradientBoostingRegressor`, one for each
# horizon.
#
# `RandomForestRegressor` natively supports multi-output regression: instead of
# independently training a model per horizon, it will train a joint model that
# predicts all horizons at once.
#
# Repeat the previous analysis using a `RandomForestRegressor`. Fix the parameter
# `min_samples_leaf` to 5.
#
# Once you created the model, plot the horizon forecast for a given date and time.
# In addition, compute the cross-validated predictions and plot the R2 and MAPE
# scores for each horizon.
#
# Does this model perform better or worse than the previous model?

# %%
from sklearn.ensemble import RandomForestRegressor

# %%
# Write your code here.
#
#
#
#
#
#
#
#
#
#
#

# %%
multioutput_predictions_rf = features_with_dropped_cols.skb.apply(
    RandomForestRegressor(min_samples_leaf=5, random_state=0, n_jobs=-1),
    y=targets.skb.drop(cols=["prediction_time", "load_mw"]).skb.mark_as_y(),
).skb.set_name("multioutput_rf")

# %%
named_predictions_rf = multioutput_predictions_rf.rename(
    {k: v for k, v in zip(target_column_names, predicted_target_column_names)}
)

# %%
plot_at_time = datetime.datetime(2021, 4, 24, 0, 0, tzinfo=datetime.timezone.utc)
historical_timedelta = datetime.timedelta(hours=24 * 5)
plot_horizon_forecast(
    targets,
    named_predictions_rf,
    plot_at_time,
    historical_timedelta,
    target_column_name_pattern,
).skb.preview()

# %%
plot_at_time = datetime.datetime(2021, 4, 25, 0, 0, tzinfo=datetime.timezone.utc)
plot_horizon_forecast(
    targets,
    named_predictions_rf,
    plot_at_time,
    historical_timedelta,
    target_column_name_pattern,
).skb.preview()

# %%
multioutput_cv_results_rf = multioutput_predictions_rf.skb.cross_validate(
    cv=ts_cv_5,
    scoring=scoring,
    return_train_score=True,
    verbose=1,
    n_jobs=-1,
)

# %%
multioutput_cv_results_rf.round(3)

# %%
import itertools
from IPython.display import display

for metric_name, dataset_type in itertools.product(["mape", "r2"], ["train", "test"]):
    columns = multioutput_cv_results_rf.columns[
        multioutput_cv_results.columns.str.startswith(f"{dataset_type}_{metric_name}")
    ]
    data_to_plot = multioutput_cv_results_rf[columns]
    data_to_plot.columns = [
        col.replace(f"{dataset_type}_", "")
        .replace(f"{metric_name}_", "")
        .replace("_", " ")
        for col in columns
    ]

    data_long = data_to_plot.melt(var_name="horizon", value_name="score")
    chart = (
        altair.Chart(
            data_long,
            title=f"{dataset_type.title()} {metric_name.upper()} Scores by Horizon",
        )
        .mark_boxplot(extent="min-max")
        .encode(
            x=altair.X(
                "horizon:N",
                title="Horizon",
                sort=altair.Sort(
                    [f"horizon {h}h" for h in range(1, data_to_plot.shape[1])]
                ),
            ),
            y=altair.Y("score:Q", title=f"{metric_name.upper()} Score"),
            color=altair.Color("horizon:N", legend=None),
        )
    )

    display(chart)

# %% [markdown]
#
# We observe that the performance of the `RandomForestRegressor` is not better in terms
# of scores or computational cost. The trend of the scores along the horizon is also
# different from the `HistGradientBoostingRegressor`: the scores worsen as the horizon
# increases.

# %% [markdown]
#
# ## Uncertainty quantification using quantile regression
#
# In this section, we show how one can use a gradient boosting but modify the loss
# function to predict different quantiles and thus obtain an uncertainty quantification
# of the predictions.

# %%
from sklearn.metrics import d2_pinball_score

scoring = {
    "r2": get_scorer("r2"),
    "mape": make_scorer(mean_absolute_percentage_error),
    "d2_pinball_05": make_scorer(d2_pinball_score, alpha=0.05),
    "d2_pinball_50": make_scorer(d2_pinball_score, alpha=0.50),
    "d2_pinball_95": make_scorer(d2_pinball_score, alpha=0.95),
}

# %%
common_params = dict(
    loss="quantile", learning_rate=0.1, max_leaf_nodes=100, random_state=0
)
predictions_hgbr_05 = features_with_dropped_cols.skb.apply(
    HistGradientBoostingRegressor(**common_params, quantile=0.05),
    y=target,
)
predictions_hgbr_50 = features_with_dropped_cols.skb.apply(
    HistGradientBoostingRegressor(**common_params, quantile=0.5),
    y=target,
)
predictions_hgbr_95 = features_with_dropped_cols.skb.apply(
    HistGradientBoostingRegressor(**common_params, quantile=0.95),
    y=target,
)

# %%
cv_results_hgbr_05 = predictions_hgbr_05.skb.cross_validate(
    cv=ts_cv_5,
    scoring=scoring,
    return_pipeline=True,
    verbose=1,
    n_jobs=-1,
)
cv_results_hgbr_50 = predictions_hgbr_50.skb.cross_validate(
    cv=ts_cv_5,
    scoring=scoring,
    return_pipeline=True,
    verbose=1,
    n_jobs=-1,
)
cv_results_hgbr_95 = predictions_hgbr_95.skb.cross_validate(
    cv=ts_cv_5,
    scoring=scoring,
    return_pipeline=True,
    verbose=1,
    n_jobs=-1,
)

# %%
cv_results_hgbr_05[
    [col for col in cv_results_hgbr_05.columns if col.startswith("test_")]
].mean(axis=0).round(3)

# %%
cv_results_hgbr_50[
    [col for col in cv_results_hgbr_50.columns if col.startswith("test_")]
].mean(axis=0).round(3)

# %%
cv_results_hgbr_95[
    [col for col in cv_results_hgbr_95.columns if col.startswith("test_")]
].mean(axis=0).round(3)

# %%
results = pl.concat(
    [
        targets.skb.select(cols=["prediction_time", target_column_name]).skb.preview(),
        predictions_hgbr_05.rename({target_column_name: "quantile_05"}).skb.preview(),
        predictions_hgbr_50.rename({target_column_name: "median"}).skb.preview(),
        predictions_hgbr_95.rename({target_column_name: "quantile_95"}).skb.preview(),
    ],
    how="horizontal",
).tail(24 * 7)

# %%
median_chart = (
    altair.Chart(results)
    .transform_fold([target_column_name, "median"])
    .mark_line(tooltip=True)
    .encode(x="prediction_time:T", y="value:Q", color="key:N")
)

quantile_band_chart = (
    altair.Chart(results)
    .mark_area(opacity=0.4, tooltip=True)
    .encode(
        x="prediction_time:T",
        y="quantile_05:Q",
        y2="quantile_95:Q",
        color=altair.value("lightgreen"),
    )
)

combined_chart = quantile_band_chart + median_chart
combined_chart.interactive()

# %%
cv_predictions_hgbr_05 = collect_cv_predictions(
    cv_results_hgbr_05["pipeline"], ts_cv_5, predictions_hgbr_05, prediction_time
)
cv_predictions_hgbr_50 = collect_cv_predictions(
    cv_results_hgbr_50["pipeline"], ts_cv_5, predictions_hgbr_50, prediction_time
)
cv_predictions_hgbr_95 = collect_cv_predictions(
    cv_results_hgbr_95["pipeline"], ts_cv_5, predictions_hgbr_95, prediction_time
)

# %%
plot_residuals_vs_predicted(cv_predictions_hgbr_05).interactive().properties(
    title=(
        "Residuals vs Predicted Values from cross-validation predictions"
        " for quantile 0.05"
    )
)

# %%
plot_residuals_vs_predicted(cv_predictions_hgbr_50).interactive().properties(
    title=(
        "Residuals vs Predicted Values from cross-validation predictions" " for median"
    )
)

# %%
plot_residuals_vs_predicted(cv_predictions_hgbr_95).interactive().properties(
    title=(
        "Residuals vs Predicted Values from cross-validation predictions"
        " for quantile 0.95"
    )
)

# %%
cv_predictions_hgbr_05_concat = pl.concat(cv_predictions_hgbr_05, how="vertical")
cv_predictions_hgbr_50_concat = pl.concat(cv_predictions_hgbr_50, how="vertical")
cv_predictions_hgbr_95_concat = pl.concat(cv_predictions_hgbr_95, how="vertical")

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import PredictionErrorDisplay


for kind in ["actual_vs_predicted", "residual_vs_predicted"]:
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    PredictionErrorDisplay.from_predictions(
        y_true=cv_predictions_hgbr_05_concat["load_mw"].to_numpy(),
        y_pred=cv_predictions_hgbr_05_concat["predicted_load_mw"].to_numpy(),
        kind=kind,
        ax=axs[0],
    )
    axs[0].set_title("0.05 quantile regression")

    PredictionErrorDisplay.from_predictions(
        y_true=cv_predictions_hgbr_50_concat["load_mw"].to_numpy(),
        y_pred=cv_predictions_hgbr_50_concat["predicted_load_mw"].to_numpy(),
        kind=kind,
        ax=axs[1],
    )
    axs[1].set_title("Median regression")

    PredictionErrorDisplay.from_predictions(
        y_true=cv_predictions_hgbr_95_concat["load_mw"].to_numpy(),
        y_pred=cv_predictions_hgbr_95_concat["predicted_load_mw"].to_numpy(),
        kind=kind,
        ax=axs[2],
    )
    axs[2].set_title("0.95 quantile regression")

    fig.suptitle(f"{kind} for GBRT minimzing different quantile losses")


# %%
def coverage(y_true, y_quantile_low, y_quantile_high):
    y_true = np.asarray(y_true)
    y_quantile_low = np.asarray(y_quantile_low)
    y_quantile_high = np.asarray(y_quantile_high)
    return float(
        np.logical_and(y_true >= y_quantile_low, y_true <= y_quantile_high)
        .mean()
        .round(4)
    )


def mean_width(y_true, y_quantile_low, y_quantile_high):
    y_true = np.asarray(y_true)
    y_quantile_low = np.asarray(y_quantile_low)
    y_quantile_high = np.asarray(y_quantile_high)
    return float(np.abs(y_quantile_high - y_quantile_low).mean().round(1))


# %%
coverage(
    cv_predictions_hgbr_50_concat["load_mw"].to_numpy(),
    cv_predictions_hgbr_05_concat["predicted_load_mw"].to_numpy(),
    cv_predictions_hgbr_95_concat["predicted_load_mw"].to_numpy(),
)

mean_width(
    cv_predictions_hgbr_50_concat["load_mw"].to_numpy(),
    cv_predictions_hgbr_05_concat["predicted_load_mw"].to_numpy(),
    cv_predictions_hgbr_95_concat["predicted_load_mw"].to_numpy(),
)

# Compute binned coverage scores
binned_coverage_results = binned_coverage(
    [df["load_mw"].to_numpy() for df in cv_predictions_hgbr_50],
    [df["predicted_load_mw"].to_numpy() for df in cv_predictions_hgbr_05],
    [df["predicted_load_mw"].to_numpy() for df in cv_predictions_hgbr_95],
    n_bins=10,
)

binned_coverage_results

# %%
coverage_by_bin = binned_coverage_results.copy()
coverage_by_bin["bin_label"] = coverage_by_bin.apply(
    lambda row: f"[{row.bin_left:.0f}, {row.bin_right:.0f}]", axis=1
)

# %%
ax = coverage_by_bin.boxplot(
    column="coverage", by="bin_label", figsize=(12, 6), vert=False, whis=1000
)
ax.axvline(x=0.9, color="red", linestyle="--", label="Target coverage (0.9)")
ax.set_xlabel("Load bins (MW)")
ax.set_ylabel("Coverage")
ax.set_title("Coverage Distribution by Load Bins")
ax.legend()
plt.suptitle("")  # Remove automatic suptitle from boxplot
plt.xticks(rotation=45)
plt.tight_layout()

# %% [markdown]
#
# ## Reliability diagrams and Lorenz curves for quantile regression

# %%
plot_reliability_diagram(
    cv_predictions_hgbr_50, kind="quantile", quantile_level=0.50
).interactive().properties(
    title="Reliability diagram for quantile 0.50 from cross-validation predictions"
)

# %%
plot_reliability_diagram(
    cv_predictions_hgbr_05, kind="quantile", quantile_level=0.05
).interactive().properties(
    title="Reliability diagram for quantile 0.05 from cross-validation predictions"
)

# %%
plot_reliability_diagram(
    cv_predictions_hgbr_95, kind="quantile", quantile_level=0.95
).interactive().properties(
    title="Reliability diagram for quantile 0.95 from cross-validation predictions"
)

# %%
plot_lorenz_curve(cv_predictions_hgbr_50).interactive().properties(
    title="Lorenz curve for quantile 0.50 from cross-validation predictions"
)

# %%
plot_lorenz_curve(cv_predictions_hgbr_05).interactive().properties(
    title="Lorenz curve for quantile 0.05 from cross-validation predictions"
)

# %%
plot_lorenz_curve(cv_predictions_hgbr_95).interactive().properties(
    title="Lorenz curve for quantile 0.95 from cross-validation predictions"
)


# %% [markdown]
#
# ## Quantile regression as classification
#
# In the following, we turn a quantile regression problem for all possible
# quantile levels into a multiclass classification problem by discretizing the
# target variable into bins and interpolating the cumulative sum of the bin
# membership probability to estimate the CDF of the distribution of the
# continuous target variable conditioned on the features.
#
# Ideally, the classifier should be efficient when trained on a large number of
# classes (induced by the number of bins). Therefore we use a Random Forest
# classifier as the default base estimator.
#
# There are several advantages to this approach:
# - a single model is trained and can jointly estimate quantiles for all
#   quantile levels (assuming a well tuned number of bins);
# - the quantile levels can be chosen at prediction time, which allows for a
#   flexible quantile regression model;
# - in practice, the resulting predictions are often reasonably well calibrated
#   as we will see in the reliability diagrams below.
#
# One possible drawback is that current implementations of gradient boosting
# models tend to be very slow to train with a large number of classes. Random
# Forests are much more efficient in this case, but they do not always provide
# the best predictive performance. It could be the case that combining this
# approach with tabular neural networks can lead to competitive results.
#
# However, the current scikit-learn API is not expressive enough to to handle
# the output shape of the quantile prediction function. We therefore cannot
# make it fit into a skrub pipeline.

# %%
from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils.validation import check_consistent_length
from sklearn.utils import check_random_state
import numpy as np


class BinnedQuantileRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        estimator=None,
        n_bins=100,
        quantile=0.5,
        random_state=None,
    ):
        self.n_bins = n_bins
        self.estimator = estimator
        self.quantile = quantile
        self.random_state = random_state

    def fit(self, X, y):
        # Lightweight input validation: most of the input validation will be
        # handled by the sub estimators.
        random_state = check_random_state(self.random_state)
        check_consistent_length(X, y)
        self.target_binner_ = KBinsDiscretizer(
            n_bins=self.n_bins,
            strategy="quantile",
            subsample=200_000,
            encode="ordinal",
            quantile_method="averaged_inverted_cdf",
            random_state=random_state,
        )

        y_binned = (
            self.target_binner_.fit_transform(np.asarray(y).reshape(-1, 1))
            .ravel()
            .astype(np.int32)
        )

        # Fit the multiclass classifier to predict the binned targets from the
        # training set.
        if self.estimator is None:
            estimator = RandomForestClassifier(random_state=random_state)
        else:
            estimator = clone(self.estimator)
        self.estimator_ = estimator.fit(X, y_binned)
        return self

    def predict_quantiles(self, X, quantiles=(0.05, 0.5, 0.95)):
        check_is_fitted(self, "estimator_")
        edges = self.target_binner_.bin_edges_[0]
        n_bins = edges.shape[0] - 1
        expected_shape = (X.shape[0], n_bins)
        y_proba_raw = self.estimator_.predict_proba(X)

        # Some might stay empty on the training set. Typically, classifiers do
        # not learn to predict an explicit 0 probability for unobserved classes
        # so we have to post process their output:
        if y_proba_raw.shape != expected_shape:
            y_proba = np.zeros(shape=expected_shape)
            y_proba[:, self.estimator_.classes_] = y_proba_raw
        else:
            y_proba = y_proba_raw

        # Build the mapper for inverse CDF mapping, from cumulated
        # probabilities to continuous prediction.
        y_cdf = np.zeros(shape=(X.shape[0], edges.shape[0]))
        y_cdf[:, 1:] = np.cumsum(y_proba, axis=1)
        return np.asarray([interp1d(y_cdf_i, edges)(quantiles) for y_cdf_i in y_cdf])

    def predict(self, X):
        return self.predict_quantiles(X, quantiles=(self.quantile,)).ravel()


# %%
quantiles = (0.05, 0.5, 0.95)
bqr = BinnedQuantileRegressor(
    RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        max_features=0.2,
        n_jobs=-1,
        random_state=0,
    ),
    n_bins=30,
)
bqr

# %%
from sklearn.model_selection import cross_validate

X, y = features_with_dropped_cols.skb.eval(), target.skb.eval()

cv_results_bqr = cross_validate(
    bqr,
    X,
    y,
    cv=ts_cv_5,
    scoring={
        "d2_pinball_50": make_scorer(d2_pinball_score, alpha=0.5),
    },
    return_estimator=True,
    return_indices=True,
    verbose=1,
    n_jobs=-1,
)

# %%
cv_predictions_bqr_all = [
    cv_predictions_bqr_05 := [],
    cv_predictions_bqr_50 := [],
    cv_predictions_bqr_95 := [],
]
for fold_ix, (qreg, test_idx) in enumerate(
    zip(cv_results_bqr["estimator"], cv_results_bqr["indices"]["test"])
):
    print(f"CV iteration #{fold_ix}")
    print(f"Test set size: {test_idx.shape[0]} rows")
    print(
        f"Test time range: {prediction_time.skb.eval()[test_idx][0, 0]} to "
        f"{prediction_time.skb.eval()[test_idx][-1, 0]} "
    )
    y_pred_all_quantiles = qreg.predict_quantiles(X[test_idx], quantiles=quantiles)

    coverage_score = coverage(
        y[test_idx],
        y_pred_all_quantiles[:, 0],
        y_pred_all_quantiles[:, 2],
    )
    print(f"Coverage: {coverage_score:.3f}")

    mean_width_score = mean_width(
        y[test_idx],
        y_pred_all_quantiles[:, 0],
        y_pred_all_quantiles[:, 2],
    )
    print(f"Mean prediction interval width: " f"{mean_width_score:.1f} MW")

    for q_idx, (quantile, predictions) in enumerate(
        zip(quantiles, cv_predictions_bqr_all)
    ):
        observed = y[test_idx]
        predicted = y_pred_all_quantiles[:, q_idx]
        predictions.append(
            pl.DataFrame(
                {
                    "prediction_time": prediction_time.skb.eval()[test_idx],
                    "load_mw": observed,
                    "predicted_load_mw": predicted,
                }
            )
        )
        print(f"d2_pinball score: {d2_pinball_score(observed, predicted):.3f}")
    print()

# %% [markdown
# Let's assess the calibration of the quantile regression model:

# %%
plot_reliability_diagram(
    cv_predictions_bqr_50, kind="quantile", quantile_level=0.50
).interactive().properties(
    title="Reliability diagram for quantile 0.50 from cross-validation predictions"
)

# %%
plot_reliability_diagram(
    cv_predictions_bqr_05, kind="quantile", quantile_level=0.05
).interactive().properties(
    title="Reliability diagram for quantile 0.05 from cross-validation predictions"
)

# %%
plot_reliability_diagram(
    cv_predictions_bqr_95, kind="quantile", quantile_level=0.95
).interactive().properties(
    title="Reliability diagram for quantile 0.95 from cross-validation predictions"
)

# %% [markdown]
#
# We can complement this assessment with the Lorenz curves, which only assess
# the ranking power of the predictions, irrespective of their absolute values.

# %%
plot_lorenz_curve(cv_predictions_bqr_50).interactive().properties(
    title="Lorenz curve for quantile 0.50 from cross-validation predictions"
)

# %%
plot_lorenz_curve(cv_predictions_bqr_05).interactive().properties(
    title="Lorenz curve for quantile 0.05 from cross-validation predictions"
)

# %%
plot_lorenz_curve(cv_predictions_bqr_95).interactive().properties(
    title="Lorenz curve for quantile 0.95 from cross-validation predictions"
)
