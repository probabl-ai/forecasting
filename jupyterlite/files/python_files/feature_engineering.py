# %% [markdown]
# # Feature engineering for electricity load forecasting
#
# ## Environment setup
#
# We need to install some extra dependencies for this notebook if needed (when running
# jupyterlite). We need the development version of skrub to be able to use the
# skrub expressions.
# %%
# %pip install -q https://pypi.anaconda.org/ogrisel/simple/polars/1.24.0/polars-1.24.0-cp39-abi3-emscripten_3_1_58_wasm32.whl
# %pip install -q holidays https://pypi.anaconda.org/ogrisel/simple/skrub/0.6.dev0/skrub-0.6.dev0-py3-none-any.whl
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
    all_city_weather_raw[city_name] = skrub.var(
        f"{city_name}_weather_raw",
        pl.from_arrow(read_table(f"../datasets/weather_{city_name}.parquet")),
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
time.join(all_city_weather_raw["brest"], on="time", how="inner")


# %%
all_city_weather = time
for city_name, city_weather_raw in all_city_weather_raw.items():
    all_city_weather = all_city_weather.join(
        city_weather_raw.rename(lambda x: x if x == "time" else x + "_" + city_name),
        on="time",
        how="inner",
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

# %%
# ## Lagged features
#
# We can now create some lagged features from the electricity load data. We

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
time.join(electricity_lagged, on="time", how="inner")


# %% [markdown]
# %%
import altair


altair.Chart(
    electricity_lagged.tail(100).skb.eval()
).transform_fold(
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
).mark_line(
    tooltip=True
).encode(
    x="time:T", y="load_mw:Q", color="key:N"
).interactive()

# %%
from skrub import TableReport

TableReport(electricity_lagged.skb.eval())
# %%
electricity_lagged.filter(pl.col("load_mw_iqr_7d") > 15_000)[
    "time"
].dt.date().unique().sort().to_list().skb.eval()

# %%
all_city_weather.filter(
        (pl.col("time") > pl.datetime(2021, 12, 1, time_zone="UTC"))
        & (pl.col("time") < pl.datetime(2021, 12, 31, time_zone="UTC"))
).skb.eval().plot.line(
    x="time:T",
    y="temperature_2m_paris:Q",
)

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
