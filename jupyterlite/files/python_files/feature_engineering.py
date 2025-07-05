# %% [markdown]
# # Feature engineering for electricity load forecasting
#
# The purpose of this notebook is to demonstrate how to use `skrub` and `polars`
# to perform feature engineering for electricity load forecasting.
#
# We will build a set of features from different sources:
#
# - Historical weather data for 10 medium to large urban areas in France;
# - Holidays and calendar features for France;
# - Historical electricity load data for the whole of France.
#
# All these data sources cover a time range from March 23, 2021 to May 31, 2025.
#
# Since our maximum forecasting horizon is 24 hours, we consider that the
# future weather data is known at a chosen prediction time. Similarly, the
# holidays and calendar features are known at prediction time for any point in
# the future.
#
# Therefore, features derived from the weather and calendar data can be used to
# engineer "future covariates". Since the load data is our prediction target,
# we will can also use it to engineer "past covariates" such as lagged features
# and rolling aggregations.
#
# ## Environment setup
#
# We need to install some extra dependencies for this notebook if needed (when running
# jupyterlite). We need the development version of skrub to be able to use the
# skrub expressions.
# %%
# %pip install -q https://pypi.anaconda.org/ogrisel/simple/polars/1.24.0/polars-1.24.0-cp39-abi3-emscripten_3_1_58_wasm32.whl
# %pip install -q altair holidays https://pypi.anaconda.org/ogrisel/simple/skrub/0.6.dev0/skrub-0.6.dev0-py3-none-any.whl
# %%
# The following 3 imports are only needed to workaround some limitations
# when using polars in a pyodide/jupyterlite notebook.
import tzdata  # noqa: F401
import pandas as pd
from pyarrow.parquet import read_table

import polars as pl
import skrub
from pathlib import Path
import holidays


# %% [markdown]
# ## Time range
#
# Let's define a hourly time range from March 23, 2021 to May 31, 2025 that
# will be used to join the electricity load data and the weather data. The time
# range is in UTC timezone to avoid any ambiguity when joining with the weather
# data that is also in UTC.
#
# We wrap the polars dataframe in a skrub variable to benefit from the
# built-in TableReport display in the notebook. Using the skrub expression
# system will also be useful later.

# %%
time_range_start = pl.datetime(2021, 3, 23, hour=0, time_zone="UTC")
time_range_end = pl.datetime(2025, 5, 31, hour=23, time_zone="UTC")
time = skrub.var(
    "time",
    pl.DataFrame().with_columns(
        pl.datetime_range(
            start=time_range_start,
            end=time_range_end,
            time_zone="UTC",
            interval="1h",
        ).alias("time"),
    ),
)
time

# %% [markdown]
#
# To avoid network issues when running this notebook, the necessary data
# files have already been downloaded and saved in the `datasets` folder.
# See the README.md file for instructions to download the data manually
# if you want to re-run this notebook with more recent data.

# %%
data_source_folder = Path("../datasets")
for data_file in sorted(data_source_folder.iterdir()):
    print(data_file)

# %% [markdown]
#
# List of 10 medium to large urban areas to approximately cover most regions in
# France with a slight focus on most populated regions that are likely to drive
# electricity demand.

# %%
city_names = [
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
]

# %%
all_city_weather_raw = {}
for city_name in city_names:
    # all_city_weather_raw[city_name] = skrub.var(
        # f"{city_name}_weather_raw",
    all_city_weather_raw[city_name] = (
        pl.from_arrow(read_table(f"../datasets/weather_{city_name}.parquet"))
    ).with_columns(
        [
            pl.col("time").dt.cast_time_unit(
                "us"
            ),  # Ensure time column has the same type
        ]
    )

# %%
all_city_weather_raw["brest"]

# %%
all_city_weather_raw["brest"].drop_nulls(subset=["temperature_2m"])


# %%
all_city_weather = time.skb.eval()
for city_name, city_weather_raw in all_city_weather_raw.items():
    all_city_weather = all_city_weather.join(
        city_weather_raw.rename(lambda x: x if x == "time" else x + "_" + city_name),
        on="time",
        how="inner",
    )

all_city_weather = skrub.var(
    "all_city_weather",
    all_city_weather,
)
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
# the time in the French timezone.
# %%
holidays_fr = holidays.country_holidays("FR", years=range(2019, 2026))

fr_time = pl.col("time").dt.convert_time_zone("Europe/Paris")
calendar = time.with_columns(
    [
        fr_time.dt.date().is_in(holidays_fr.keys()).alias("is_holiday_fr"),
        fr_time.dt.weekday().alias("day_of_week_fr"),
        fr_time.dt.ordinal_day().alias("day_of_year_fr"),
        fr_time.dt.hour().alias("hour_of_day_fr"),
    ],
)
calendar

# %% [markdown]
#
# ## Electricity load data
#
# Finally we load the electricity load data. This data will both be used as a
# target variable but also to craft some lagged and window-aggregated features.
# %%
load_data_files = [
    data_file
    for data_file in sorted(data_source_folder.iterdir())
    if data_file.name.startswith("Total Load - Day Ahead")
    and data_file.name.endswith(".csv")
]
# %%
electricity_raw = skrub.var(
    "electricity_raw",
    pl.concat(
        [
            pl.from_pandas(pd.read_csv(data_file, na_values=["N/A", "-"])).drop(
                ["Day-ahead Total Load Forecast [MW] - BZN|FR"]
            )
            for data_file in load_data_files
        ],
        how="vertical",
    ),
)
electricity_raw

# %%
electricity = (
    electricity_raw.with_columns(
        [
            pl.col("Time (UTC)")
            .str.split(by=" - ")
            .list.first()
            .str.to_datetime("%d.%m.%Y %H:%M", time_zone="UTC")
            .alias("time"),
        ]
    )
    .drop(["Time (UTC)"])
    .rename({"Actual Total Load [MW] - BZN|FR": "load_mw"})
    .filter(pl.col("time").dt.minute().eq(0))
    .filter(pl.col("time") >= time_range_start)
    .filter(pl.col("time") <= time_range_end)
    .select(["time", "load_mw"])
)
electricity

# %%
electricity.filter(pl.col("load_mw").is_null())

# %%

electricity.filter(
    (pl.col("time") > pl.datetime(2021, 10, 30, hour=10, time_zone="UTC"))
    & (pl.col("time") < pl.datetime(2021, 10, 31, hour=10, time_zone="UTC"))
).skb.eval().plot.line(x="time:T", y="load_mw:Q")

# %%
electricity = electricity.with_columns([pl.col("load_mw").interpolate()])
electricity.filter(
    (pl.col("time") > pl.datetime(2021, 10, 30, hour=10, time_zone="UTC"))
    & (pl.col("time") < pl.datetime(2021, 10, 31, hour=10, time_zone="UTC"))
).skb.eval().plot.line(x="time:T", y="load_mw:Q")

# %% [markdown]
#
# Check that the number of rows matches our expectations based on
# the number of hours that separate the first and the last dates. We can do
# that by joining with the time range dataframe and checking that the number of
# rows stays the same.

# %%
assert (
    time.join(electricity, on="time", how="inner").shape[0] == time.shape[0]
).skb.eval()

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
import altair


altair.Chart(electricity_lagged.tail(100).skb.eval()).transform_fold(
    [
        "load_mw",
        "load_mw_lag_1h",
        "load_mw_lag_2h",
        "load_mw_lag_3h",
        "load_mw_lag_1d",
        "load_mw_lag_1w",
        "load_mw_rolling_median_24h",
        "load_mw_rolling_median_7d",
        "load_mw_iqr_24h",
        "load_mw_iqr_7d",
    ],
    as_=["key", "load_mw"],
).mark_line(tooltip=True).encode(x="time:T", y="load_mw:Q", color="key:N").interactive()

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
    [f"temperature_2m_{city_name}" for city_name in city_names],
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
prediction_time = time = skrub.var(
    "prediction_time",
    pl.DataFrame().with_columns(
        pl.datetime_range(
            start=time_range_start + pl.duration(days=7),
            end=time_range_end - pl.duration(hours=24),
            time_zone="UTC",
            interval="1h",
        ).alias("prediction_time"),
    ),
)
prediction_time

features = (
    (
        prediction_time.join(
            electricity_lagged, left_on="prediction_time", right_on="time"
        )
        .join(all_city_weather, left_on="prediction_time", right_on="time")
        .join(calendar, left_on="prediction_time", right_on="time")
    )
    .drop("prediction_time")
    .skb.mark_as_X()
)
features

# %%
horizon = 1
target_column_name = f"load_mw_horizon_{horizon}h"
target_df = prediction_time.join(
    electricity.with_columns(
        [pl.col("load_mw").shift(-horizon).alias(target_column_name)]
    ),
    left_on="prediction_time",
    right_on="time",
)
target = target_df[target_column_name].skb.mark_as_y()

# %%
from sklearn.ensemble import HistGradientBoostingRegressor
import threadpoolctl


with threadpoolctl.threadpool_limits(limits=1):
    predictions = features.skb.apply(
        HistGradientBoostingRegressor(
            random_state=0,
            learning_rate=skrub.choose_float(
                0.01, 0.9, default=0.1, log=True, name="learning_rate"
            ),
        ),
        y=target,
    )
predictions

# %%
altair.Chart(
    pl.concat(
        [
            target_df.skb.eval(),
            predictions.rename({"load_mw_horizon_1h": "predicted_load_mw_horizon_1h"}).skb.eval(),
        ],
        how="horizontal",
    ).tail(1000)
).transform_fold(
    [target_column_name, "predicted_load_mw_horizon_1h"],
).mark_line(
    tooltip=True
).encode(
    x="prediction_time:T", y="value:Q", color="key:N"
).interactive()


# %%
from sklearn.model_selection import TimeSeriesSplit

ts_cv = TimeSeriesSplit(n_splits=5, max_train_size=None, gap=0)
predictions.skb.cross_validate(
    cv=ts_cv,
    scoring=[
        "neg_mean_absolute_error",
        "neg_mean_squared_error",
        "r2",
        "neg_mean_absolute_percentage_error",
    ],
    verbose=1,
    n_jobs=-1,
)
# %%
