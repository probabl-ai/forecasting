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
# Therefore, features derived from the weather and calendar data can be used to
# engineer "future covariates". Since the load data is our prediction target,
# we will can also use it to engineer "past covariates" such as lagged features
# and rolling aggregations. The future values of the load data (with respect to
# the prediction time) are used as targets for the forecasting model.
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
import numpy as np
import polars as pl
import skrub
from pathlib import Path
import holidays
import warnings

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


electricity = load_electricity_load_data(time, data_source_folder)
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
        "load_mw_rolling_iqr_24h",
        "load_mw_rolling_iqr_7d",
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
)
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
            targets.skb.eval(),
            hgbr_predictions.rename(
                {target_column_name: predicted_target_column_name}
            ).skb.eval(),
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

# %%
from sklearn.model_selection import TimeSeriesSplit


max_train_size = 2 * 52 * 24 * 7  # max ~2 years of training data
test_size = 24 * 7 * 24  # 24 weeks of test data
gap = 7 * 24  # 1 week gap between train and test sets
ts_cv_5 = TimeSeriesSplit(
    n_splits=5, max_train_size=max_train_size, test_size=test_size, gap=gap
)

for cv_idx, (train_idx, test_idx) in enumerate(
    ts_cv_5.split(prediction_time.skb.eval())
):
    print(f"CV iteration #{cv_idx}")
    train_datetimes = prediction_time.skb.eval()[train_idx]
    test_datetimes = prediction_time.skb.eval()[test_idx]
    print(
        f"Train: {train_datetimes.shape[0]} rows, "
        f"Test: {test_datetimes.shape[0]} rows"
    )
    print(f"Train time range: {train_datetimes[0, 0]} to " f"{train_datetimes[-1, 0]} ")
    print(f"Test time range: {test_datetimes[0, 0]} to " f"{test_datetimes[-1, 0]} ")
    print()

# %%
from sklearn.metrics import make_scorer, mean_absolute_percentage_error, get_scorer
from sklearn.metrics import d2_tweedie_score


cv_results = hgbr_predictions.skb.cross_validate(
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
cv_results.round(3)


# %%
def collect_cv_predictions(pipelines, cv_splitter, predictions, prediction_time):
    index_generator = cv_splitter.split(prediction_time.skb.eval())

    def splitter(X, y, index_generator):
        """Workaround to transform a scikit-learn splitter into a function understood
        by `skrub.train_test_split`."""
        train_idx, test_idx = next(index_generator)
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    results = []
    for (_, test_idx), pipeline in zip(
        cv_splitter.split(prediction_time.skb.eval()), pipelines
    ):
        split = predictions.skb.train_test_split(
            predictions.skb.get_data(),
            splitter=splitter,
            index_generator=index_generator,
        )
        results.append(
            pl.DataFrame(
                {
                    "prediction_time": prediction_time.skb.eval()[test_idx],
                    "load_mw": split["y_test"],
                    "predicted_load_mw": pipeline.predict(split["test"]),
                }
            )
        )
    return results


# %%
cv_predictions = collect_cv_predictions(
    cv_results["pipeline"], ts_cv_5, hgbr_predictions, prediction_time
)
cv_predictions[0]

# %%


def lorenz_curve(observed_value, predicted_value, n_samples=1_000):
    """Compute the Lorenz curve for a given true and predicted values."""

    def gini_index(cum_proportion_population, cum_proportion_y_true):
        from sklearn.metrics import auc

        return 1 - 2 * auc(cum_proportion_population, cum_proportion_y_true)

    observed_value = np.asarray(observed_value)
    predicted_value = np.asarray(predicted_value)

    sort_idx = np.argsort(predicted_value)
    observed_value_sorted = observed_value[sort_idx]

    original_n_samples = observed_value_sorted.shape[0]
    cum_proportion_population = np.cumsum(np.ones(original_n_samples))
    cum_proportion_population /= cum_proportion_population[-1]

    cum_proportion_y_true = np.cumsum(observed_value_sorted)
    cum_proportion_y_true /= cum_proportion_y_true[-1]

    gini_model = gini_index(cum_proportion_population, cum_proportion_y_true)

    cum_proportion_population_interpolated = np.linspace(0, 1, n_samples)
    cum_proportion_y_true_interpolated = np.interp(
        cum_proportion_population_interpolated,
        cum_proportion_population,
        cum_proportion_y_true,
    )

    return pl.DataFrame(
        {
            "cum_population": cum_proportion_population_interpolated,
            "cum_observed": cum_proportion_y_true_interpolated,
        }
    ).with_columns(
        pl.lit(gini_model).alias("gini_index"),
    )


def plot_lorenz_curve(cv_predictions, n_samples=1_000):
    """Plot the Lorenz curve for a given true and predicted values."""

    results = []
    for cv_idx, predictions in enumerate(cv_predictions):
        results.append(
            lorenz_curve(
                observed_value=predictions["load_mw"],
                predicted_value=predictions["predicted_load_mw"],
                n_samples=n_samples,
            ).with_columns(
                pl.lit(cv_idx).alias("cv_idx"),
                pl.lit("model").alias("model"),
            )
        )

        results.append(
            lorenz_curve(
                observed_value=predictions["load_mw"],
                predicted_value=predictions["load_mw"],
                n_samples=n_samples,
            ).with_columns(
                pl.lit(cv_idx).alias("cv_idx"),
                pl.lit("oracle").alias("model"),
            )
        )

    results = pl.concat(results)

    gini_stats = results.group_by("model").agg(
        [
            pl.col("gini_index")
            .mean()
            .map_elements(lambda x: f"{x:.4f}", return_dtype=pl.String)
            .alias("gini_mean"),
            pl.col("gini_index")
            .std()
            .map_elements(lambda x: f"{x:.4f}", return_dtype=pl.String)
            .alias("gini_std_dev"),
        ]
    )

    results = results.join(gini_stats, on="model").with_columns(
        pl.format("{} ({} +/- {})", "model", "gini_mean", "gini_std_dev").alias(
            "model_label"
        )
    )

    model_chart = (
        altair.Chart(results)
        .mark_line(strokeDash=[4, 2, 4, 2], opacity=0.8, tooltip=True)
        .encode(
            x=altair.X(
                "cum_population:Q",
                title="Fraction of observations sorted by predicted label",
            ),
            y=altair.Y("cum_observed:Q", title="Cumulative observed load proportion"),
            color=altair.Color(
                "model_label:N", legend=altair.Legend(title="Models"), sort=None
            ),
            detail="cv_idx:N",
        )
    )

    diagonal_chart = (
        altair.Chart(
            pl.DataFrame(
                {
                    "cum_population": [0, 1],
                    "cum_observed": [0, 1],
                    "model_label": "Non-informative model",
                }
            )
        )
        .mark_line(strokeDash=[4, 4], opacity=0.5, tooltip=True)
        .encode(
            x=altair.X(
                "cum_population:Q",
                title="Fraction of observations sorted by predicted label",
            ),
            y=altair.Y("cum_observed:Q", title="Cumulative observed load proportion"),
            color=altair.Color(
                "model_label:N", legend=altair.Legend(title="Models"), sort=None
            ),
        )
    )

    return model_chart + diagonal_chart


plot_lorenz_curve(cv_predictions, n_samples=500).interactive()


# %%
def plot_reliability_diagram(cv_predictions, n_bins=10):
    # min and max load over all predictions and observations for any folds:
    all_loads = pl.concat(
        [
            cv_prediction.select(["load_mw", "predicted_load_mw"])
            for cv_prediction in cv_predictions
        ]
    )
    all_loads = pl.concat(all_loads["load_mw", "predicted_load_mw"])
    min_load, max_load = all_loads.min(), all_loads.max()
    scale = altair.Scale(domain=[min_load, max_load])

    # Create the perfect line
    chart = (
        altair.Chart(
            pl.DataFrame(
                {
                    "mean_predicted_load_mw": [min_load, max_load],
                    "mean_load_mw": [min_load, max_load],
                    "label": ["Perfect"] * 2,
                }
            )
        )
        .mark_line(tooltip=True, opacity=0.8, strokeDash=[5, 5])
        .encode(
            x=altair.X("mean_predicted_load_mw:Q", scale=scale),
            y=altair.Y("mean_load_mw:Q", scale=scale),
            color=altair.Color(
                "label:N",
                scale=altair.Scale(range=["black"]),
                legend=altair.Legend(title="Legend"),
            ),
        )
    )

    # Add lines for each CV fold with date labels
    for i, cv_predictions_i in enumerate(cv_predictions):
        # Get date range for this CV fold
        min_date = cv_predictions_i["prediction_time"].min().strftime("%Y-%m-%d")
        max_date = cv_predictions_i["prediction_time"].max().strftime("%Y-%m-%d")
        fold_label = f"#{i+1} - {min_date} to {max_date}"

        mean_per_bins = (
            cv_predictions_i.group_by(
                pl.col("predicted_load_mw").qcut(np.linspace(0, 1, n_bins))
            )
            .agg(
                [
                    pl.col("load_mw").mean().alias("mean_load_mw"),
                    pl.col("predicted_load_mw").mean().alias("mean_predicted_load_mw"),
                ]
            )
            .sort("predicted_load_mw")
            .with_columns(pl.lit(fold_label).alias("fold"))
        )

        chart += (
            altair.Chart(mean_per_bins)
            .mark_line(tooltip=True, point=True, opacity=0.8)
            .encode(
                x=altair.X("mean_predicted_load_mw:Q", scale=scale),
                y=altair.Y("mean_load_mw:Q", scale=scale),
                color=altair.Color(
                    "fold:N",
                    legend=altair.Legend(title=None),
                ),
                detail=altair.Detail("fold:N"),
            )
        )
    return chart.resolve_scale(color="independent")


plot_reliability_diagram(cv_predictions).interactive().properties(
    title="Reliability diagram from cross-validation predictions"
)

# %%
def plot_residuals_vs_predicted(cv_predictions):
    """Plot residuals vs predicted values scatter plot for all CV folds."""
    all_scatter_plots = []

    for i, cv_prediction in enumerate(cv_predictions):
        # Get date range for this CV fold
        min_date = cv_prediction["prediction_time"].min().strftime("%Y-%m-%d")
        max_date = cv_prediction["prediction_time"].max().strftime("%Y-%m-%d")
        fold_label = f"#{i+1} - {min_date} to {max_date}"

        # Calculate residuals
        residuals_data = cv_prediction.with_columns(
            [(pl.col("predicted_load_mw") - pl.col("load_mw")).alias("residual")]
        ).with_columns([pl.lit(fold_label).alias("fold")])

        # Create scatter plot for this CV fold
        scatter_plot = (
            altair.Chart(residuals_data)
            .mark_circle(opacity=0.6, size=20)
            .encode(
                x=altair.X(
                    "predicted_load_mw:Q",
                    title="Predicted Load (MW)",
                    scale=altair.Scale(zero=False),
                ),
                y=altair.Y("residual:Q", title="Residual (MW)"),
                color=altair.Color("fold:N", legend=None),
                tooltip=[
                    "prediction_time:T",
                    "load_mw:Q",
                    "predicted_load_mw:Q",
                    "residual:Q",
                    "fold:N",
                ],
            )
        )

        all_scatter_plots.append(scatter_plot)

    # Get the range of predicted values for the perfect line
    all_predictions = pl.concat(
        [cv_pred["predicted_load_mw"] for cv_pred in cv_predictions]
    )
    min_pred, max_pred = all_predictions.min(), all_predictions.max()

    # Create perfect residuals line at y=0
    perfect_line = (
        altair.Chart(
            pl.DataFrame(
                {
                    "predicted_load_mw": [min_pred, max_pred],
                    "perfect_residual": [0, 0],
                    "label": ["Perfect"] * 2,
                }
            )
        )
        .mark_line(strokeDash=[5, 5], opacity=0.8, color="black")
        .encode(
            x=altair.X("predicted_load_mw:Q", title="Predicted Load (MW)"),
            y=altair.Y("perfect_residual:Q", title="Residual (MW)"),
            color=altair.Color(
                "label:N",
                scale=altair.Scale(range=["black"]),
                legend=None,
            ),
        )
    )

    # Combine all scatter plots
    combined_scatter = all_scatter_plots[0]
    for plot in all_scatter_plots[1:]:
        combined_scatter += plot

    # Layer the scatter plots with the perfect line
    return (combined_scatter + perfect_line).resolve_scale(color="independent")


plot_residuals_vs_predicted(cv_predictions).interactive().properties(
    title="Residuals vs Predicted Values from cross-validation predictions"
)


# %%
def plot_binned_residuals(cv_predictions, by="hour"):
    """Plot the average residuals binned by time period, one line per CV fold."""
    # Configure binning based on the 'by' parameter
    if by == "hour":
        time_column = "hour_of_day"
        time_extractor = pl.col("prediction_time").dt.hour().alias(time_column)
        x_title = "Hour of day"
    elif by == "month":
        time_column = "month_of_year"
        time_extractor = pl.col("prediction_time").dt.month().alias(time_column)
        x_title = "Month of year"
    else:
        raise ValueError(f"Unsupported binning method: {by}. Use 'hour' or 'month'.")

    all_iqr_bands = []
    all_mean_lines = []
    time_range = None  # Will store the min/max time values for the perfect line

    for i, cv_prediction in enumerate(cv_predictions):
        # Get date range for this CV fold
        min_date = cv_prediction["prediction_time"].min().strftime("%Y-%m-%d")
        max_date = cv_prediction["prediction_time"].max().strftime("%Y-%m-%d")
        fold_label = f"#{i+1} - {min_date} to {max_date}"

        # Create residuals and time binning columns
        residuals_detailed = cv_prediction.with_columns(
            [
                (pl.col("predicted_load_mw") - pl.col("load_mw")).alias("residual"),
                time_extractor,
            ]
        )

        # Calculate statistics for this CV fold
        residuals_stats = (
            residuals_detailed.group_by(time_column)
            .agg(
                [
                    pl.col("residual").mean().round(1).alias("mean_residual"),
                    pl.col("residual").quantile(0.25).round(1).alias("q25_residual"),
                    pl.col("residual").quantile(0.75).round(1).alias("q75_residual"),
                ]
            )
            .sort(time_column)
            .with_columns(pl.lit(fold_label).alias("fold"))
        )

        # Store time range for perfect line (use the first CV fold)
        if time_range is None:
            time_range = (
                residuals_stats[time_column].min(),
                residuals_stats[time_column].max(),
            )
        else:
            time_range = (
                min(time_range[0], residuals_stats[time_column].min()),
                max(time_range[1], residuals_stats[time_column].max()),
            )
        # Create IQR band for this CV fold
        iqr_band = (
            altair.Chart(residuals_stats)
            .mark_area(opacity=0.15)
            .encode(
                x=altair.X(f"{time_column}:O", title=x_title),
                y=altair.Y("q25_residual:Q"),
                y2=altair.Y2("q75_residual:Q"),
            )
        )

        # Create mean line for this CV fold
        mean_line = (
            altair.Chart(residuals_stats)
            .mark_line(tooltip=True, point=True, opacity=0.8)
            .encode(
                x=altair.X(f"{time_column}:O", title=x_title),
                y=altair.Y("mean_residual:Q", title="Mean residual (MW)"),
                color=altair.Color("fold:N", legend=None),
                detail="fold:N",
            )
        )

        all_iqr_bands.append(iqr_band)
        all_mean_lines.append(mean_line)

    # Create perfect residuals line at y=0
    perfect_line = (
        altair.Chart(
            pl.DataFrame(
                {
                    time_column: [time_range[0], time_range[1]],
                    "perfect_residual": [0, 0],
                    "label": ["Perfect"] * 2,
                }
            )
        )
        .mark_line(strokeDash=[5, 5], opacity=0.8, color="black")
        .encode(
            x=altair.X(f"{time_column}:O", title=x_title),
            y=altair.Y("perfect_residual:Q", title="Mean residual (MW)"),
            color=altair.Color(
                "label:N",
                scale=altair.Scale(range=["black"]),
                legend=None,
            ),
        )
    )

    # Combine all IQR bands
    combined_iqr = all_iqr_bands[0]
    for band in all_iqr_bands[1:]:
        combined_iqr += band

    # Combine all mean lines
    combined_lines = all_mean_lines[0]
    for line in all_mean_lines[1:]:
        combined_lines += line

    # Layer the IQR bands behind the mean lines, with perfect line on top
    return (combined_iqr + combined_lines + perfect_line).resolve_scale(
        color="independent"
    )


plot_binned_residuals(cv_predictions, by="hour").interactive().properties(
    title="Residuals by hour of the day from cross-validation predictions"
)


# %%
plot_binned_residuals(cv_predictions, by="month").interactive().properties(
    title="Residuals by hour of the day from cross-validation predictions"
)

# %%
ts_cv_2 = TimeSeriesSplit(
    n_splits=2, test_size=test_size, max_train_size=max_train_size, gap=24
)
randomized_search = hgbr_predictions.skb.get_randomized_search(
    cv=ts_cv_2,
    scoring="r2",
    n_iter=100,
    fitted=True,
    verbose=1,
    n_jobs=-1,
)
# %%
randomized_search.results_.round(3)

# %%
randomized_search.plot_results().update_layout(margin=dict(l=150))

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
# for outer_cv_idx in range(len(nested_cv_results)):
#     print(
#         nested_cv_results.loc[outer_cv_idx, "pipeline"]
#         .results_.loc[0]
#         .round(3)
#         .to_dict()
#     )

# %%
# TODO: Exercise applying a a linear model with some additional feature engineering
from sklearn.linear_model import Ridge
from sklearn.kernel_approximation import Nystroem

model = skrub.tabular_learner(
    estimator=Ridge(alpha=skrub.choose_float(1e-6, 1e6, log=True, default=1e-3))
)
model.steps.insert(
    -1,
    (
        "nystroem",
        Nystroem(n_components=skrub.choose_int(10, 200, log=True, default=150)),
    ),
)

predictions_ridge = features_with_dropped_cols.skb.apply(model, y=target)
predictions_ridge

# %%
altair.Chart(
    pl.concat(
        [
            targets.skb.eval(),
            predictions_ridge.rename(
                {target_column_name: predicted_target_column_name}
            ).skb.eval(),
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

# %%
ts_cv_2 = TimeSeriesSplit(
    n_splits=2, test_size=test_size, max_train_size=max_train_size, gap=24
)
randomized_search = predictions_ridge.skb.get_randomized_search(
    cv=ts_cv_2,
    scoring="r2",
    n_iter=100,
    fitted=True,
    verbose=1,
    n_jobs=-1,
)

# %%
randomized_search.plot_results().update_layout(margin=dict(l=200))

# %%
nested_cv_results = skrub.cross_validate(
    environment=predictions_ridge.skb.get_data(),
    pipeline=randomized_search,
    cv=ts_cv_5,
    scoring={
        "r2": get_scorer("r2"),
        "mape": make_scorer(mean_absolute_percentage_error),
    },
    n_jobs=-1,
    return_pipeline=True,
).round(3)

# %%
nested_cv_results.round(3)

# %%
cv_predictions_ridge = collect_cv_predictions(
    nested_cv_results["pipeline"], ts_cv_5, predictions_ridge, prediction_time
)

# %%
plot_lorenz_curve(cv_predictions_ridge, n_samples=500).interactive()

# %%
plot_reliability_diagram(cv_predictions_ridge).interactive().properties(
    title="Reliability diagram from cross-validation predictions"
)

# %%
from sklearn.multioutput import MultiOutputRegressor

multioutput_predictions = features_with_dropped_cols.skb.apply(
    MultiOutputRegressor(
        estimator=HistGradientBoostingRegressor(random_state=0), n_jobs=-1
    ),
    y=targets.skb.drop(cols=["prediction_time", "load_mw"]).skb.mark_as_y(),
).skb.set_name("multioutput_gbdt")

# %%
target_column_names = [target_column_name_pattern.format(horizon=h) for h in horizons]
predicted_target_column_names = [
    f"predicted_{target_column_name}" for target_column_name in target_column_names
]
named_predictions = multioutput_predictions.rename(
    {k: v for k, v in zip(target_column_names, predicted_target_column_names)}
)

# %%
import datetime


def plot_horizon_forecast(
    targets, named_predictions, plot_at_time, historical_timedelta
):
    """Plot the true target and the forecast values."""
    merged_data = targets.skb.select(cols=["prediction_time", "load_mw"]).skb.concat(
        [named_predictions], axis=1
    )
    start_time = plot_at_time - historical_timedelta
    end_time = plot_at_time + datetime.timedelta(
        hours=named_predictions.skb.eval().shape[1]
    )
    true_values_past = merged_data.filter(
        pl.col("prediction_time").is_between(start_time, plot_at_time, closed="both")
    ).rename({"load_mw": "Past true load"})
    true_values_future = merged_data.filter(
        pl.col("prediction_time").is_between(plot_at_time, end_time, closed="both")
    ).rename({"load_mw": "Future true load"})
    predicted_record = (
        merged_data.skb.select(
            cols=skrub.selectors.filter_names(str.startswith, "predict")
        )
        .row(by_predicate=pl.col("prediction_time") == plot_at_time, named=True)
        .skb.eval()
    )
    forecast_values = pl.DataFrame(
        {
            "prediction_time": predicted_record["prediction_time"]
            + datetime.timedelta(hours=horizon),
            "Forecast load": predicted_record[
                "predicted_" + target_column_name_pattern.format(horizon=horizon)
            ],
        }
        for horizon in range(1, len(predicted_record))
    )

    true_values_past_chart = (
        altair.Chart(true_values_past.skb.eval())
        .transform_fold(["Past true load"])
        .mark_line(tooltip=True)
        .encode(x="prediction_time:T", y="Past true load:Q", color="key:N")
    )
    true_values_future_chart = (
        altair.Chart(true_values_future.skb.eval())
        .transform_fold(["Future true load"])
        .mark_line(tooltip=True)
        .encode(x="prediction_time:T", y="Future true load:Q", color="key:N")
    )
    forecast_values_chart = (
        altair.Chart(forecast_values)
        .transform_fold(["Forecast load"])
        .mark_line(tooltip=True)
        .encode(x="prediction_time:T", y="Forecast load:Q", color="key:N")
    )
    return (
        true_values_past_chart + true_values_future_chart + forecast_values_chart
    ).interactive()


# %%
plot_at_time = datetime.datetime(2025, 5, 24, 0, 0, tzinfo=datetime.timezone.utc)
historical_timedelta = datetime.timedelta(hours=24 * 5)
plot_horizon_forecast(targets, named_predictions, plot_at_time, historical_timedelta)

# %%
plot_at_time = datetime.datetime(2025, 5, 25, 0, 0, tzinfo=datetime.timezone.utc)
plot_horizon_forecast(targets, named_predictions, plot_at_time, historical_timedelta)

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
).round(3)

# %%
multioutput_cv_results

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

# %%
# TODO: Exercise using RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

multioutput_predictions_rf = features_with_dropped_cols.skb.apply(
    RandomForestRegressor(min_samples_leaf=30, random_state=0, n_jobs=-1),
    y=targets.skb.drop(cols=["prediction_time", "load_mw"]).skb.mark_as_y(),
).skb.set_name("random_forest")

# %%
named_predictions_rf = multioutput_predictions_rf.rename(
    {k: v for k, v in zip(target_column_names, predicted_target_column_names)}
)

# %%
plot_at_time = datetime.datetime(2025, 5, 24, 0, 0, tzinfo=datetime.timezone.utc)
historical_timedelta = datetime.timedelta(hours=24 * 5)
plot_horizon_forecast(targets, named_predictions_rf, plot_at_time, historical_timedelta)

# %%
plot_at_time = datetime.datetime(2025, 5, 25, 0, 0, tzinfo=datetime.timezone.utc)
plot_horizon_forecast(targets, named_predictions_rf, plot_at_time, historical_timedelta)

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

# %%
from sklearn.metrics import mean_pinball_loss

scoring = {
    "r2": get_scorer("r2"),
    "mape": make_scorer(mean_absolute_percentage_error),
    "mean_pinball_05_loss": make_scorer(mean_pinball_loss, alpha=0.05),
    "mean_pinball_50_loss": make_scorer(mean_pinball_loss, alpha=0.5),
    "mean_pinball_95_loss": make_scorer(mean_pinball_loss, alpha=0.95),
}

# %%
common_params = dict(
    loss="quantile", learning_rate=0.1, max_leaf_nodes=100, random_state=0
)
predictions_gbrt_05 = features_with_dropped_cols.skb.apply(
    HistGradientBoostingRegressor(**common_params, quantile=0.05),
    y=target,
)
predictions_gbrt_50 = features_with_dropped_cols.skb.apply(
    HistGradientBoostingRegressor(**common_params, quantile=0.5),
    y=target,
)
predictions_gbrt_95 = features_with_dropped_cols.skb.apply(
    HistGradientBoostingRegressor(**common_params, quantile=0.95),
    y=target,
)

# %%
cv_results_05 = predictions_gbrt_05.skb.cross_validate(
    cv=ts_cv_5,
    scoring=scoring,
    verbose=1,
    n_jobs=-1,
)
cv_results_50 = predictions_gbrt_50.skb.cross_validate(
    cv=ts_cv_5,
    scoring=scoring,
    verbose=1,
    n_jobs=-1,
)
cv_results_95 = predictions_gbrt_95.skb.cross_validate(
    cv=ts_cv_5,
    scoring=scoring,
    verbose=1,
    n_jobs=-1,
)

# %%
cv_results_05.mean(axis=0).round(3)

# %%
cv_results_50.mean(axis=0).round(3)

# %%
cv_results_95.mean(axis=0).round(3)

# %%
results = pl.concat(
    [
        targets.skb.select(cols=["prediction_time", target_column_name]).skb.eval(),
        predictions_gbrt_05.rename({target_column_name: "quantile_05"}).skb.eval(),
        predictions_gbrt_50.rename({target_column_name: "median"}).skb.eval(),
        predictions_gbrt_95.rename({target_column_name: "quantile_95"}).skb.eval(),
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
        color=altair.value("green"),
    )
)

combined_chart = quantile_band_chart + median_chart
combined_chart.interactive()

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import PredictionErrorDisplay

fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

PredictionErrorDisplay.from_predictions(
    y_true=targets["load_mw_horizon_24h"].skb.eval().to_numpy(),
    y_pred=predictions_gbrt_05["load_mw_horizon_24h"].skb.eval().to_numpy(),
    kind="actual_vs_predicted",
    ax=axs[0],
)
axs[0].set_title("0.05 Quantile")

PredictionErrorDisplay.from_predictions(
    y_true=targets["load_mw_horizon_24h"].skb.eval().to_numpy(),
    y_pred=predictions_gbrt_50["load_mw_horizon_24h"].skb.eval().to_numpy(),
    kind="actual_vs_predicted",
    ax=axs[1],
)
axs[1].set_title("Median")

PredictionErrorDisplay.from_predictions(
    y_true=targets["load_mw_horizon_24h"].skb.eval().to_numpy(),
    y_pred=predictions_gbrt_95["load_mw_horizon_24h"].skb.eval().to_numpy(),
    kind="actual_vs_predicted",
    ax=axs[2],
)
axs[2].set_title("0.95 Quantile")
_ = fig.suptitle("GBRT Predictions for different quantiles")

# %%
