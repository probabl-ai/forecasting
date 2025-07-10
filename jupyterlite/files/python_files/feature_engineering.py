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
import tzdata  # noqa: F401
import pandas as pd
from pyarrow.parquet import read_table

import altair
import polars as pl
import skrub
from pathlib import Path
import holidays


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
data_source_folder = skrub.var("data_source_folder", "../datasets")

for data_file in sorted(Path(data_source_folder.skb.eval()).iterdir()):
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
        for data_file in sorted(Path(data_source_folder).iterdir())
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
#
# Let's serialize this data loading and feature engineering pipeline for
# reuse in later notebooks.

# %%
import cloudpickle

if not Path("feature_engineering_pipeline.pkl").exists():
    with open("feature_engineering_pipeline.pkl", "wb") as f:
        cloudpickle.dump(
            {
                "features": features,
                "targets": targets,
                "prediction_time": prediction_time,
                "horizons": horizons,
                "target_column_name_pattern": target_column_name_pattern,
            },
            f,
        )
