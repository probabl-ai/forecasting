# %% [markdown]
# # Feature engineering for electricity load forecasting
#
# The purpose of this notebook is to demonstrate how to use `skrub` and
# `polars` to perform feature engineering for electricity load forecasting.

# We build a pipeline that for a given prediction time, predicts the future
# electricity load. We start by doing it for 1 horizon, then we extend to
# predicting multiple horizons in the same pipeline.
# This means that for each prediction time, it outputs predicted loads 
# for 1 or several horizons.

# Features (and targets) from different data sources are used:

# - Historical weather data for 10 medium to large urban areas in France;
# - Historical electricity load data for the whole of France;
# - Holidays and standard calendar features for France.

# All these data sources cover a time range from March 23, 2021 to May 31,
# 2025.

# Exogenous features derived from the weather and calendar data can
# be used to engineer "future covariates". Since the load data is our prediction target, 
# we can also use it to engineer  "past covariates" such as lagged features and rolling 
# aggregations. 

# The future values of the load data (with respect to the prediction time) are 
# used as targets for the forecasting model. 

#
# ## Environment setup
#
# We need to install some extra dependencies for this notebook if needed (when
# running jupyterlite).

# %%
# %pip install -q skrub altair holidays plotly nbformat polars pydot graphviz

# %% [markdown]
#
# The following 3 imports are only needed to workaround some limitations when
# using polars in a pyodide/jupyterlite notebook.
#
# TODO: remove those workarounds once pyodide enables again the package:
# xref: https://github.com/pyodide/pyodide-recipes/blob/0.29.X/packages/polars/meta.yaml

 # %%
import tzdata  # noqa: F401
import pandas as pd
import datetime
from pyarrow.parquet import read_table

import altair
import polars as pl
import skrub


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
# operations in this notebook are chained together in a directed
# acyclic graph that is automatically tracked by `skrub`. This allows us to
# extract the resulting pipeline and apply it to new data later on, exactly
# like a trained scikit-learn pipeline. The main difference is that we do so
# incrementally and while eagerly executing and inspecting the results of each
# step as is customary when working with dataframe libraries such as polars and
# pandas in Jupyter notebooks.


# %%

def time_range(start, end=None):
    """
    Build a 1-hour-spaced datetime range from start to end.

    Times are truncated to the nearest full hour.

    If end is None, we get a time range containing only the start time.
    """
    if end is None:
        end = start
    if isinstance(start, str):
        start = datetime.datetime.fromisoformat(start)
    if isinstance(end, str):
        end = datetime.datetime.fromisoformat(end)
    return pl.DataFrame().with_columns(
        pl.datetime_range(
            start=start,
            end=end,
            time_zone="UTC",
            interval="1h",
        )
        .dt.truncate("1h")
        .alias("time"),
    )

range_start = skrub.var("start", "2021-03-23")
range_end = skrub.var("end", "2025-05-31")

prediction_time = skrub.deferred(time_range)(range_start, range_end)
prediction_time

# %% [markdown]
#
# If you run the above locally with pydot and graphviz installed, you can
# visualize the expression graph of the `time` variable by expanding the "Show
# graph" button.
#
# Let's now load the data records for the time range defined above.
#
# To avoid network issues when running this notebook, the necessary data files
# have already been downloaded and saved in the `datasets` folder. 

# %%
from pathlib import Path
def get_data_dir():
    return Path(".").resolve().parent / "datasets"


# %%
for data_file in sorted(get_data_dir().iterdir()):
    print(data_file)

# %% [markdown]
#
# ## Electricity load data
#
# We load the electricity load data. This data will both be used as a
# target variable but also to craft the data pipeline. We build a pipeline that 
# for a given prediction time, predicts the future electricity load. 
# We start by doing it for 1 horizon, then we extend to
# predicting multiple horizons in the same pipeline.
#
# The historical data is sampled irregularly, sometimes every hour, sometimes
# every 15 min, and with missing rows. We define a function to resample it on a
# regular 1h-spaced grid.
#
# As this will serve as the basis for our lagged features, we add a buffer of
# empty rows beyond the range of our data. We do not have the actual load for
# those rows, but lagged loads can be defined for them and joined onto the
# feature set we are building.
#
# %%

def load_electricity_history_data(data_dir=get_data_dir()):
    """Load and aggregate historical load data from the raw CSV files."""
    return (
        pl.read_csv(get_data_dir() / "Total Load - Day Ahead*.csv", null_values=["N/A", "-"])
        .drop_nulls()
        .select(
            pl.col("Time (UTC)")
            .str.split(by=" - ")
            .list.first()
            .str.to_datetime("%d.%m.%Y %H:%M", time_zone="UTC")
            .alias("time"),
            pl.col("Actual Total Load [MW] - BZN|FR").alias("load_mw"),
        )
    )

def resample(electricity_history_data):
    """
    Resample the load history on a regular time grid to have exactly 1 row every hour.

    Parts where sampling was finer (eg every 15 minutes) are averaged over 1h
    intervals, and if some hours are missing a corresponding row is inserted
    containing explicit NULL values (rather than a missing row).

    We add an extra empty 48h at the end to receive lags that can be used to
    predict beyond the range of the available data.
    """
    averaged = electricity_history_data.group_by(pl.col("time").dt.truncate("1h")).agg(
        pl.col("load_mw").mean()
    )
    all_times = averaged["time"]
    return time_range(
        all_times.min(), (all_times.max() + datetime.timedelta(hours=48))
    ).join(averaged, on="time", how="left", maintain_order="left")

# %%

raw_electricity_load_history = skrub.as_data_op(load_electricity_history_data).skb.set_name(
    "electricity_history_data"
)()
raw_electricity_load_history

# %%
electricity_load_history = raw_electricity_load_history.skb.apply_func(resample)
electricity_load_history

# %% [markdown]
#
# ## Building the training dataset
# The prediction time range we built above is the input query to our system.
# For each row, it outputs a prediction.
#
# We use it to build the ground truth y, by shifting the historical load by the
# horizon. To account for missing data in the ground truth, we restrict the data to
# timestamps for which we have a ground truth. At inference, when making a
# prediction we keep all the query timestamps.
#
# This function is almost the same for handling single or multiple horizons so
# we anticipate a little bit the need for multiple horizons and make it general
# enough to accomodate both.


# %%
def get_X_y(prediction_time, electricity_load_history, horizons, mode=skrub.eval_mode()):
    """
    Compute input and target variables.

    For fitting (and validation), this builds the targets y by applying
    appropriate shifts to the historical data. The targets y and prediction
    times X are aligned, and rows with missing ground truth are dropped.
    Returns a dictionary with keys X and y, ready to be split for
    cross-validation or used to fit a model.

    For prediction, simply returns `target_time` in a dictionary with a
    single key X.
    """
    if isinstance(horizons, int):
        single_horizon = True
        horizons = (horizons,)
    else:
        single_horizon = False
    prediction_time = prediction_time.rename({"time": "prediction_time"})
    if mode in ("fit", "fit_transform", "preview"):
        # For those modes we need the ground truth; restrict to rows for which
        # there is y
        load = electricity_load_history.select(
            pl.col("time"),
            *[pl.col("load_mw").shift(-h).alias(f"{h}h") for h in horizons],
        ).drop_nulls()
        X_y = prediction_time.join(
            load,
            left_on="prediction_time",
            right_on="time",
            how="inner",
            maintain_order="left",
        )
        return {
            "X": X_y.select(pl.col("prediction_time")),
            "y": (
                X_y[f"{horizons[0]}h"] if single_horizon else X_y.drop("prediction_time")
            ),
        }
    else:
        # In predict mode there is no y and we return unmodified query
        return {"X": prediction_time}

# Example output for 1 hours
EXAMPLE_TIME_HORIZON = 1
X_y = prediction_time.skb.apply_func(get_X_y, electricity_load_history, EXAMPLE_TIME_HORIZON)
X = X_y["X"].skb.mark_as_X()
y = X_y["y"].skb.mark_as_y()
X

# %%
y

# %% [markdown]
#
# ## Feature engineering
#
# Now that we have our query and the ground-truth answers for it, we can start
# building the rest of our predictive pipeline: creating the features and
# adding a supervised predictor.
#
# We already marked X and y with `.skb.mark_as_X()` and `.skb.mark_as_y()`.
# Doing this early lets us reuse the same expressions later with
# `train_test_split` or `cross_validate` without extra wiring.
#
# Feature engineering takes _target time_ into account. In X we have the
# prediction time, the time at which we make the prediction. We also want to
# take into account the target time, i.e., the time about which we make a
# prediction. For example if we are predicting what the load will be on Tuesday
# at 3pm, we want to know what the weather will be, whether Tuesday is a
# holiday, and what the load was on Monday at 3pm and the previous Tuesday at
# 3pm. Those features are driven by the target time. So our first step is to
# add it to the dataframe of features we are building up.

# %%
from polars import selectors as cs
import holidays 

def add_target_time(df, horizon):
    return df.with_columns(
        (pl.col("prediction_time") + pl.duration(hours=horizon)).alias("target_time")
    )
def add_lagged_features(df, electricity_load_history, horizon):
    """
    Build lagged features for the given horizon.

    horizon must be <= 24 (hours). Only features that would be available at
    prediction time, ie that require data at least horizon hours in the past,
    are created.
    """
    assert horizon <= 24
    lags = (
        pl.col("load_mw").shift(lag).alias(f"lag_{lag}")
        for lag in list(range(horizon, 24)) + [24, 24 * 2, 24 * 7]
    )

    rolling_lags = sorted(set((horizon, 24)))
    rolling_widths = (24, 24 * 7)

    def rolling(e, name):
        return [
            e.rolling(
                index_column="time", period=f"{width}h", offset=f"{-width -lag}h"
            ).alias(f"lag_{lag}_width_{width}_{name}")
            for lag in rolling_lags
            for width in rolling_widths
        ]

    medians = rolling(pl.col("load_mw").median(), "median")
    iqr = rolling(
        (pl.col("load_mw").quantile(0.75) - pl.col("load_mw").quantile(0.25)), "iqr"
    )
    features = electricity_load_history.select(pl.col("time"), *lags, *medians, *iqr)
    return df.join(
        features,
        left_on="target_time",
        right_on="time",
        how="left",
        maintain_order="left",
    )
def fetch_city_weather(city, data_dir=get_data_dir()):
    return pl.read_parquet(get_data_dir() / f"weather_{city}.parquet")

def add_weather(
    df,
    horizon,
    cities="all",
    temperature_only=True,
    city_weather_fetcher=fetch_city_weather,
):
    """Add weather information for the required cities."""
    # NOTE: here ideally we should retrieve the exact weather forecast
    # corresponding to the horizon. But we do not have it available in the
    # historical data. Therefore we just take the only forecast we have and
    # ignore the horizon.
    del horizon
    if isinstance(cities, str):
        assert cities == "all"
        cities =  (
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
                )
    with_weather = df
    for city in cities:
        with_weather = with_weather.join(
            city_weather_fetcher(city)
            .with_columns(pl.col("time").dt.cast_time_unit("us"))
            .select(
                (pl.col("time"), cs.matches(".*temperature.*"))
                if temperature_only
                else pl.all()
            )
            .select(
                pl.col("time"),
                (~cs.by_name("time")).as_expr().name.map(f"weather_{{}}_{city}".format),
            ),
            left_on="target_time",
            right_on="time",
            how="left",
            maintain_order="left",
        )
    return with_weather

def add_calendar_and_holidays(target_time):
    """Add calendar features and holiday information."""
    fr_time = pl.col("target_time").dt.convert_time_zone("Europe/Paris")
    fr_year_min = target_time.select(fr_time.dt.year().min()).item()
    fr_year_max = target_time.select(fr_time.dt.year().max()).item()
    holidays_fr = holidays.country_holidays(
        "FR", years=range(fr_year_min, fr_year_max + 1)
    )
    return target_time.with_columns(
        fr_time.dt.hour().alias("cal_hour_of_day"),
        fr_time.dt.weekday().alias("cal_day_of_week"),
        fr_time.dt.ordinal_day().alias("cal_day_of_year"),
        fr_time.dt.year().alias("cal_year"),
        fr_time.dt.date().is_in(holidays_fr.keys()).alias("cal_is_holiday"),
    )
    
# %%    

with_target_time = X.skb.apply_func(add_target_time, EXAMPLE_TIME_HORIZON)
with_target_time

# %% [markdown]
#
# ## Lagged features
#
# Next we have a function for adding lagged features (such as load on the same
# day of the previous week). It needs the input dataframe (which so far only
# contains prediction and target time), the historical data that will be used
# to build the lagged features and join them to the input. The horizon
# (difference between target and prediction time) is also needed to ensure that
# we do not include lags that would not be available after deployment: for
# example if we are creating a pipeline for a 12 h horizon we cannot include
# the 3-hour lagged load (because it would only become available 9 hours after
# the deadline for our prediction).

# %%
with_lags = with_target_time.skb.apply_func(
    add_lagged_features, electricity_load_history, EXAMPLE_TIME_HORIZON
)
with_lags

# %% [markdown]
#
# ## Weather Data

# %% 
fetch_city_weather("paris")

# %% [markdown]
#
# We are not sure if it is best to use all cities or only a few big ones. Also,
# we don't know which features to use, temperature is probably the most
# important one so we may want to try using all features or the temperature
# only. Therefore the function we define has parameters for controlling that.
#
# Skrub lets us create "choice" objects, nodes in our pipeline that can take
# different values for hyperparameter search. We use this for the choice of
# city names and of temperature only vs all features.

# %%
city_weather_fetcher = skrub.as_data_op(fetch_city_weather).skb.set_name(
    "city_weather_fetcher"
)

temperature_only = skrub.choose_bool(name="temperature_only", default=True)
cities = skrub.choose_from(["all", ["paris", "lyon", "marseille"]], name="cities")


with_weather = with_lags.skb.apply_func(
    add_weather,
    EXAMPLE_TIME_HORIZON,
    cities=cities,
    temperature_only=temperature_only,
    city_weather_fetcher=city_weather_fetcher,
)
with_weather

# %%
lag_window = with_lags.filter(
    (pl.col("target_time") > pl.datetime(2021, 12, 1, time_zone="UTC"))
    & (pl.col("target_time") < pl.datetime(2021, 12, 31, time_zone="UTC"))
).skb.eval()

altair.Chart(lag_window).transform_fold(
    [
        "lag_1",
        "lag_24",
        "lag_1_width_24_median",
        "lag_1_width_168_iqr",
    ],
    as_=["key", "value"],
).mark_line(
    tooltip=True
).encode(
    x="target_time:T", y="value:Q", color="key:N"
).interactive()

# %%
weather_window = with_weather.filter(
    (pl.col("target_time") > pl.datetime(2021, 12, 1, time_zone="UTC"))
    & (pl.col("target_time") < pl.datetime(2021, 12, 10, time_zone="UTC"))
).skb.eval()

weather_cols = [
    c for c in weather_window.columns if c.startswith("weather_") and "temperature" in c
][:6]

altair.Chart(weather_window).transform_fold(
    weather_cols,
    as_=["key", "value"],
).mark_line(
    tooltip=True
).encode(
    x="target_time:T", y="value:Q", color="key:N"
).interactive()

# %% [markdown]
#
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
with_calendar = with_weather.skb.apply_func(add_calendar_and_holidays)
with_calendar

# %%
altair.Chart(with_calendar.tail(100).skb.preview()).transform_fold(
    [
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_1_width_24_median",
        "lag_1_width_168_median",
        "lag_24_width_24_median",
        "lag_24_width_168_median",
        "lag_1_width_24_iqr",
        "lag_24_width_24_iqr",
    ],
    as_=["key", "load_mw"],
).mark_line(tooltip=True).encode(x="target_time:T", y="load_mw:Q", color="key:N").interactive()

# %% [markdown]
#
# ## Final dataset
#
# Now we are done with all the feature engineering steps. For later reuse we
# group the steps we just created into one function:

# %%  

def add_features(df, horizon, electricity_load_history, cities, temperature_only, city_weather_fetcher):
    df = add_target_time(df, horizon=horizon)
    df = add_lagged_features(df, electricity_load_history, horizon=horizon)
    df = add_weather(
    df,
    horizon,
    cities=cities,
    temperature_only=temperature_only,
    city_weather_fetcher=city_weather_fetcher,
    )
    df = add_calendar_and_holidays(df)
    return df
    