# %% [markdown]
# # Feature engineering for electricity load forecasting
#
# The purpose of this notebook is to demonstrate how to use `skrub` and
# `polars` to perform feature engineering for electricity load forecasting.
#
# We will build a set of features (and targets) from different data sources:
#
# - Historical weather data for 10 medium to large urban areas in France;
# - Historical electricity load data for the whole of France;
# - Holidays and standard calendar features for France.
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
# be used to engineer "future covariates". Since the load (demand) data is our
# prediction target, we can also use it to engineer "past covariates" such
# as lagged features and rolling aggregations. The future values of the load
# data (with respect to the prediction time) are used as targets for the
# forecasting model.
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
import datetime
from pathlib import Path

import altair
import pandas as pd
import polars as pl
import skrub
import tzdata  # noqa: F401
from polars import selectors as cs

# %% [markdown]
#
# To avoid network issues when running this notebook, the necessary data files
# have already been downloaded and saved in the `datasets` folder.


# %%
def get_data_dir():
    return Path(".").resolve().parent / "datasets"


# %%
for data_file in sorted(get_data_dir().iterdir()):
    print(data_file)

# %% [markdown]
#
# ## Electricity demand data
#
# We fetch the electricity demand data from our local data folder. This data
# will both be used as a target variable but also to craft the data pipeline.
#
# All the operations we perform, from data loading to the final prediction,
# will be tracked in a skrub DataOp graph, rather than executed immediately.
# This will allow us to fit the whole pipeline and apply it to unseen data, and
# also to cross-validate it and tune hyperparameters.
#
# Inputs to the computation graph are declared with `skrub.var()`. As we may
# want to change the data source when using a fitted pipeline, we make the
# function that loads the historical data a variable, so that another fetcher
# can be passed instead if needed.


# %%
def fetch_demand_history():
    """Load historical electricity demand from the raw CSV files."""
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


history_fetcher = skrub.var("history_fetcher", fetch_demand_history, becomes_default=True)
raw_demand_history = history_fetcher()

# %% [markdown]
#
# Our pipeline has 2 nodes so far, which load the historical data. Skrub shows
# a preview of each intermediate result as we build the pipeline, so we can
# check our work as we go.

# %% [markdown]
#
# The historical data is sampled irregularly, sometimes every hour, sometimes
# every 15 min, and with missing rows. We define a function to resample it on a
# regular 1h-spaced grid.
#
# We start by defining the function that builds our grid of prediction times
# given a start and end date.
#
# Let's define a hourly time range from March 23, 2021 to May 31, 2025 that
# will be used to join the electricity load data and the weather data. The time
# range is in UTC timezone to avoid any ambiguity when joining with the weather
# data that is also in UTC.


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
# Now we define the function that resamples the historical data to this regular
# grid. As this will serve as the basis for our lagged features, we add a
# buffer of empty rows beyond the range of our data. We do not have the actual
# electricity demand for those rows, but lagged loads can be defined for them
# and joined onto the feature set we are building.


# %%
def resample(demand_history):
    """
    Resample the load history on a regular time grid to have exactly 1 row every hour.

    Parts where sampling was finer (eg every 15 minutes) are averaged over 1h
    intervals, and if some hours are missing a corresponding row is inserted
    containing explicit NULL values (rather than a missing row).

    We add an extra empty 48h at the end to receive lags that can be used to
    predict beyond the range of the available data.
    """
    averaged = demand_history.group_by(pl.col("time").dt.truncate("1h")).agg(
        pl.col("load_mw").mean()
    )
    all_times = averaged["time"]
    return time_range(all_times.min(), (all_times.max() + datetime.timedelta(hours=48))).join(
        averaged, on="time", how="left", maintain_order="left"
    )


# %%
demand_history = raw_demand_history.skb.apply_func(resample)
demand_history

# %% [markdown]
#
# ## Building the training dataset
#
# The prediction time range we built above is the input query to our system:
# for each row, the final pipeline outputs a prediction.
#
# We build the ground truth y by shifting the historical demand by the horizon,
# and keep only the timestamps for which a ground truth exists. At inference,
# we keep all the query timestamps.
#
# The same function handles a single horizon or several, as we will need the
# latter in later parts of the tutorial.


# %%
def get_X_y(prediction_time, demand_history, horizons, mode=skrub.eval_mode()):
    """
    Compute input and target variables.

    For fitting (and validation), this builds the targets y by applying
    appropriate shifts to the historical data. The targets y and prediction
    times X are aligned, and rows with missing ground truth are dropped.
    Returns a dictionary with keys X and y, ready to be split for
    cross-validation or used to fit a model.

    For prediction, simply returns the query (a dataframe with a single
    `prediction_time` column) in a dictionary with a single key X.
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
        demand = demand_history.select(
            pl.col("time"),
            *[pl.col("load_mw").shift(-h).alias(f"{h}h") for h in horizons],
        ).drop_nulls()
        X_y = prediction_time.join(
            demand,
            left_on="prediction_time",
            right_on="time",
            how="inner",
            maintain_order="left",
        )
        return {
            "X": X_y.select(pl.col("prediction_time")),
            "y": (X_y[f"{horizons[0]}h"] if single_horizon else X_y.drop("prediction_time")),
        }
    else:
        # In predict mode there is no y and we return unmodified query
        return {"X": prediction_time}


# Example output for 12-hour horizon
EXAMPLE_TIME_HORIZON = 12

X_y = prediction_time.skb.apply_func(get_X_y, demand_history, EXAMPLE_TIME_HORIZON)
X = X_y["X"].skb.mark_as_X()
y = X_y["y"].skb.mark_as_y()
X

# %%
y

# %% [markdown]
#
# ## Feature engineering
#
# With our query and its ground truth in place, we can build the rest of the
# pipeline: the features (this notebook) and a supervised predictor (later
# parts of the tutorial).
#
# X contains the _prediction time_, when the prediction is made. Features,
# however, are driven by the _target time_: the time being predicted. To
# predict demand on Monday at 22:00, we want the weather forecast for that
# time, whether Monday is a holiday, and the demand on the previous Monday at
# 22:00. Our first step is therefore to add the target time to the dataframe
# of features we are building.
#
# ![](horizons.svg)


# %%
def add_target_time(df, horizon):
    return df.with_columns(
        (pl.col("prediction_time") + pl.duration(hours=horizon)).alias("target_time")
    )


# %%
with_target_time = X.skb.apply_func(add_target_time, EXAMPLE_TIME_HORIZON)
with_target_time

# %% [markdown]
#
# ## Lagged features
#
# Next, a function that adds lagged features (such as the demand on the same
# day of the previous week). It takes the input dataframe (so far only
# prediction and target time), the historical demand used to build the lags,
# and the horizon (the difference between target and prediction time).
#
# The horizon ensures we only use lags that would be available at deployment.
# For a 12 h horizon, for example, we cannot use the 3-hour lagged demand: it
# would only become available 9 hours after the prediction time.


# %%
def add_lagged_features(df, demand_history, horizon):
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
            e.rolling(index_column="time", period=f"{width}h", offset=f"{-width -lag}h").alias(
                f"lag_{lag}_width_{width}_{name}"
            )
            for lag in rolling_lags
            for width in rolling_widths
        ]

    medians = rolling(pl.col("load_mw").median(), "median")
    iqr = rolling((pl.col("load_mw").quantile(0.75) - pl.col("load_mw").quantile(0.25)), "iqr")
    features = demand_history.select(pl.col("time"), *lags, *medians, *iqr)
    return df.join(
        features,
        left_on="target_time",
        right_on="time",
        how="left",
        maintain_order="left",
    )


# %%
with_lags = with_target_time.skb.apply_func(
    add_lagged_features, demand_history, EXAMPLE_TIME_HORIZON
)
with_lags

# %% [markdown]
#
# Let us plot some of the features we just created.
# To obtain the preview result displayed by skrub (as a python object) we can use .skb.preview()
#
# TODO: move this code into a function in tutorial_helpers?
# TODO: both plots seems redundant, keep only 1?

# %%
lag_window = with_lags.skb.preview().filter(
    (pl.col("target_time") > pl.datetime(2021, 12, 1, time_zone="UTC"))
    & (pl.col("target_time") < pl.datetime(2021, 12, 31, time_zone="UTC"))
)

altair.Chart(lag_window).transform_fold(
    [
        "lag_12",
        "lag_24",
        "lag_12_width_24_median",
        "lag_12_width_168_iqr",
    ],
    as_=["key", "value"],
).mark_line(tooltip=True).encode(x="target_time:T", y="value:Q", color="key:N").interactive()

# %%
altair.Chart(with_lags.tail(100).skb.preview()).transform_fold(
    [
        "lag_12",
        "lag_13",
        "lag_14",
        "lag_12_width_24_median",
        "lag_12_width_168_median",
        "lag_24_width_24_median",
        "lag_24_width_168_median",
        "lag_12_width_24_iqr",
        "lag_24_width_24_iqr",
    ],
    as_=["key", "load_mw"],
).mark_line(tooltip=True).encode(x="target_time:T", y="load_mw:Q", color="key:N").interactive()


# %% [markdown]
#
# ## Weather Data
#
# As the weather has a strong influence on electricity demand, we add it to our
# feature set.
#
# We define a list of 10 medium to large urban areas to approximately cover
# most regions in France with a slight focus on most populated regions that are
# likely to drive electricity demand.
#
# As for the historical data, we make the exact function that loads this data
# an input to our pipeline so we can change it after fitting if needed.


# %%
def fetch_weather(city):
    return pl.read_parquet(get_data_dir() / f"weather_{city}.parquet")


weather_fetcher = skrub.var("weather_fetcher", fetch_weather, becomes_default=True)
weather_fetcher("paris")

# %% [markdown]
#
# Now we define the function that actually adds those features to the dataframe
# we are building up.
#
# We are not sure if it is best to use all cities or only a few big ones. Also,
# we don't know which features to use, temperature is probably the most
# important one so we may want to try using all features or the temperature
# only. Therefore the function we define has parameters for controlling that.


# %%
def add_weather(
    df,
    *,
    cities,
    temperature_only,
    weather_fetcher,
):
    """Add weather information for the required cities."""
    if isinstance(cities, str):
        assert cities == "all"
        cities = (
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
            weather_fetcher(city)
            .with_columns(pl.col("time").dt.cast_time_unit("us"))
            .select(
                (pl.col("time"), cs.matches(".*temperature.*")) if temperature_only else pl.all()
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


# %% [markdown]
#
# Skrub lets us create "choice" objects, nodes in our pipeline that can take
# different values for hyperparameter search. We use this for the choice of
# city names and of temperature only vs all features.

# %%
temperature_only = skrub.choose_bool(name="temperature_only", default=True)
cities = skrub.choose_from(["all", ["paris", "lyon", "marseille"]], name="cities")


with_weather = with_lags.skb.apply_func(
    add_weather,
    cities=cities,
    temperature_only=temperature_only,
    weather_fetcher=weather_fetcher,
)
with_weather


# %%
weather_window = with_weather.skb.preview().filter(
    (pl.col("target_time") > pl.datetime(2021, 12, 1, time_zone="UTC"))
    & (pl.col("target_time") < pl.datetime(2021, 12, 10, time_zone="UTC"))
)

weather_cols = [
    c for c in weather_window.columns if c.startswith("weather_") and "temperature" in c
][:6]

altair.Chart(weather_window).transform_fold(
    weather_cols,
    as_=["key", "value"],
).mark_line(
    tooltip=True
).encode(x="target_time:T", y="value:Q", color="key:N").interactive()

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
import holidays


def fetch_holidays(years):
    return holidays.country_holidays("FR", years=years)


holidays_fetcher = skrub.var("holidays_fetcher", fetch_holidays, becomes_default=True)
holidays_fetcher([2024, 2025])


# %%
def add_calendar_and_holidays(df, *, holidays_fetcher):
    fr_time = pl.col("target_time").dt.convert_time_zone("Europe/Paris")
    fr_year_min = df.select(fr_time.dt.year().min()).item()
    fr_year_max = df.select(fr_time.dt.year().max()).item()
    holidays_fr = holidays_fetcher(years=range(fr_year_min, fr_year_max + 1))
    return df.with_columns(
        fr_time.dt.hour().alias("cal_hour_of_day"),
        fr_time.dt.weekday().alias("cal_day_of_week"),
        fr_time.dt.ordinal_day().alias("cal_day_of_year"),
        fr_time.dt.year().alias("cal_year"),
        fr_time.dt.date().is_in(holidays_fr.keys()).alias("cal_is_holiday"),
    )


# %%
with_calendar = with_weather.skb.apply_func(
    add_calendar_and_holidays, holidays_fetcher=holidays_fetcher
)
with_calendar

# %% [markdown]
#
# ## Final dataset
#
# Now we are done with all the feature engineering steps. For later reuse we
# group the steps we just created into one function:


# %%
def add_features(
    df,
    *,
    horizon,
    demand_history,
    cities,
    temperature_only,
    weather_fetcher,
    holidays_fetcher,
):
    df = add_target_time(df, horizon=horizon)
    df = add_lagged_features(df, demand_history, horizon=horizon)
    df = add_weather(
        df,
        cities=cities,
        temperature_only=temperature_only,
        weather_fetcher=weather_fetcher,
    )
    df = add_calendar_and_holidays(df, holidays_fetcher)
    return df


# %%
def feature_engineering_outputs(horizons, cv_splitter=None):
    range_start = skrub.var("start", "2021-03-23")
    range_end = skrub.var("end", "2025-05-31")

    prediction_time = skrub.deferred(time_range)(range_start, range_end)
    demand_history = skrub.var(
        "history_fetcher",
        fetch_demand_history,
        becomes_default=True,
    )().skb.apply_func(resample)
    X_y = prediction_time.skb.apply_func(get_X_y, demand_history, horizons)
    X = X_y["X"].skb.mark_as_X(cv=cv_splitter)
    y = X_y["y"].skb.mark_as_y()
    temperature_only = skrub.choose_bool(name="temperature_only", default=True)
    cities = skrub.choose_from(["all", ["paris", "lyon", "marseille"]], name="cities")
    weather_fetcher = skrub.var("weather_fetcher", fetch_weather, becomes_default=True)
    holidays_fetcher = skrub.var("holidays_fetcher", fetch_holidays, becomes_default=True)
    if isinstance(horizons, int):
        single_horizon = True
        horizons = (horizons,)
    else:
        single_horizon = False
    all_features = {}
    for h in horizons:
        all_features[h] = X.skb.apply_func(
            add_features,
            horizon=h,
            temperature_only=temperature_only,
            cities=cities,
            demand_history=demand_history,
            weather_fetcher=weather_fetcher,
            holidays_fetcher=holidays_fetcher,
        ).skb.set_name(f"feat_{h}h")
    return all_features[horizons[0]] if single_horizon else all_features, y
