from pathlib import Path
from polars import selectors as cs
from pyarrow.parquet import read_table
import altair
import datetime
import holidays
import pandas as pd
import polars as pl
import skrub
import tzdata  # noqa: F401


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


def get_data_dir():
    return Path(".").resolve().parent / "datasets"


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
    return time_range(all_times.min(), (all_times.max() + datetime.timedelta(hours=48))).join(
        averaged, on="time", how="left", maintain_order="left"
    )


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
            "y": (X_y[f"{horizons[0]}h"] if single_horizon else X_y.drop("prediction_time")),
        }
    else:
        # In predict mode there is no y and we return unmodified query
        return {"X": prediction_time}


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
            e.rolling(index_column="time", period=f"{width}h", offset=f"{-width -lag}h").alias(
                f"lag_{lag}_width_{width}_{name}"
            )
            for lag in rolling_lags
            for width in rolling_widths
        ]

    medians = rolling(pl.col("load_mw").median(), "median")
    iqr = rolling((pl.col("load_mw").quantile(0.75) - pl.col("load_mw").quantile(0.25)), "iqr")
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
            city_weather_fetcher(city)
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


def add_calendar_and_holidays(target_time):
    """Add calendar features and holiday information."""
    fr_time = pl.col("target_time").dt.convert_time_zone("Europe/Paris")
    fr_year_min = target_time.select(fr_time.dt.year().min()).item()
    fr_year_max = target_time.select(fr_time.dt.year().max()).item()
    holidays_fr = holidays.country_holidays("FR", years=range(fr_year_min, fr_year_max + 1))
    return target_time.with_columns(
        fr_time.dt.hour().alias("cal_hour_of_day"),
        fr_time.dt.weekday().alias("cal_day_of_week"),
        fr_time.dt.ordinal_day().alias("cal_day_of_year"),
        fr_time.dt.year().alias("cal_year"),
        fr_time.dt.date().is_in(holidays_fr.keys()).alias("cal_is_holiday"),
    )


def add_features(
    df,
    horizon,
    electricity_load_history,
    cities,
    temperature_only,
    city_weather_fetcher,
):
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


def feature_engineering_outputs(horizons, cv_splitter=None):
    range_start = skrub.var("start", "2021-03-23")
    range_end = skrub.var("end", "2025-05-31")

    prediction_time = skrub.deferred(time_range)(range_start, range_end)
    resampled_history = skrub.var(
        "electricity_history_loader",
        load_electricity_history_data,
        becomes_default=True,
    )().skb.apply_func(resample)
    X_y = prediction_time.skb.apply_func(get_X_y, resampled_history, horizons)
    X = X_y["X"].skb.mark_as_X(cv=cv_splitter)
    y = X_y["y"].skb.mark_as_y()
    temperature_only = skrub.choose_bool(name="temperature_only", default=True)
    cities = skrub.choose_from(["all", ["paris", "lyon", "marseille"]], name="cities")
    city_weather_fetcher = skrub.var(
        "city_weather_fetcher", fetch_city_weather, becomes_default=True
    )
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
            electricity_load_history=resampled_history,
            city_weather_fetcher=city_weather_fetcher,
        ).skb.set_name(f"feat_{h}h")
    return all_features[horizons[0]] if single_horizon else all_features, y
