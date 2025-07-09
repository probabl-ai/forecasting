import datetime

import numpy as np
import polars as pl
import altair
import skrub


def lorenz_curve(observed_value, predicted_value, n_samples=1_000):
    """Compute the Lorenz curve for a given true and predicted values.

    Parameters
    ----------
    observed_value : array-like
        The true values.
    predicted_value : array-like
        The predicted values.
    n_samples : int, default=1_000
        The number of samples to use to compute the Lorenz curve.

    Returns
    -------
    pl.DataFrame
        A DataFrame with the Lorenz curve. There is three columns in this DataFrame:
        - `cum_population`: The cumulative proportion of the population sorted
          by predicted label.
        - `cum_observed`: The cumulative proportion of the observed values
          sorted by predicted label.
        - `gini_index`: The Gini index of the Lorenz curve.
    """

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
    """Plot the Lorenz curve for a given cross-validation results containing
    observed and predicted values.

    Parameters
    ----------
    cv_predictions : list of polars.DataFrame
        A list of polars DataFrames, each containing the observed and predicted values
        for a given fold. It is the output of the `collect_cv_predictions` function.
    n_samples : int, default=1_000
        The number of samples to use to compute the Lorenz curve.

    Returns
    -------
    altair.Chart
        A chart with the Lorenz curve.
    """

    results = []
    for fold_idx, predictions in enumerate(cv_predictions):
        results.append(
            lorenz_curve(
                observed_value=predictions["load_mw"],
                predicted_value=predictions["predicted_load_mw"],
                n_samples=n_samples,
            ).with_columns(
                pl.lit(fold_idx).alias("fold_idx"),
                pl.lit("Model").alias("model"),
            )
        )

        results.append(
            lorenz_curve(
                observed_value=predictions["load_mw"],
                predicted_value=predictions["load_mw"],
                n_samples=n_samples,
            ).with_columns(
                pl.lit(fold_idx).alias("fold_idx"),
                pl.lit("Oracle").alias("model"),
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
        pl.format("{} (Gini: {} +/- {})", "model", "gini_mean", "gini_std_dev").alias(
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
            detail="fold_idx:N",
        )
    )

    diagonal_chart = (
        altair.Chart(
            pl.DataFrame(
                {
                    "cum_population": [0, 1],
                    "cum_observed": [0, 1],
                    "model_label": "Non-informative model (Gini = 0.0)",
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


def plot_reliability_diagram(
    cv_predictions, kind="mean", quantile_level=0.5, n_bins=10
):
    """Plot the reliability diagram given cross-validation results containing
    observed and predicted values.

    Parameters
    ----------
    cv_predictions : list of polars.DataFrame
        A list of polars DataFrames, each containing the observed and predicted values
        for a given fold. It is the output of the `collect_cv_predictions` function.
    kind : str, default="mean"
        The kind of reliability diagram to plot. Can be "mean" or "quantile".
    quantile_level : float, default=0.5
        The quantile level to use for the quantile-based reliability diagram.
    n_bins : int, default=10
        The number of bins to use for the binned reliability diagram.

    Returns
    -------
    altair.Chart
        A chart with the reliability diagram.
    """
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

    for fold_idx, cv_predictions_i in enumerate(cv_predictions):
        min_date = cv_predictions_i["prediction_time"].min().strftime("%Y-%m-%d")
        max_date = cv_predictions_i["prediction_time"].max().strftime("%Y-%m-%d")
        fold_label = f"#{fold_idx} - {min_date} to {max_date}"

        if kind == "mean":
            y_name = "mean_load_mw"
            agg_expr = pl.col("load_mw").mean()
        elif kind == "quantile":
            y_name = "quantile_of_load_mw"
            agg_expr = pl.col("load_mw").quantile(quantile_level)
        else:
            raise ValueError(f"Unknown kind: {kind}. Use 'mean' or 'quantile'.")

        mean_per_bins = (
            cv_predictions_i.group_by(
                pl.col("predicted_load_mw").qcut(np.linspace(0, 1, n_bins))
            )
            .agg(
                [
                    agg_expr.alias(y_name),
                    pl.col("predicted_load_mw").mean().alias("mean_predicted_load_mw"),
                ]
            )
            .sort("predicted_load_mw")
            .with_columns(pl.lit(fold_label).alias("fold_label"))
        )

        chart += (
            altair.Chart(mean_per_bins)
            .mark_line(tooltip=True, point=True, opacity=0.8)
            .encode(
                x=altair.X("mean_predicted_load_mw:Q", scale=scale),
                y=altair.Y(f"{y_name}:Q", scale=scale),
                color=altair.Color(
                    "fold_label:N",
                    legend=altair.Legend(title=None),
                ),
                detail=altair.Detail("fold_label:N"),
            )
        )
    return chart.resolve_scale(color="independent")


def plot_residuals_vs_predicted(cv_predictions):
    """Plot residuals vs predicted values scatter plot for all CV folds.

    Parameters
    ----------
    cv_predictions : list of polars.DataFrame
        A list of polars DataFrames, each containing the observed and predicted values
        for a given fold. It is the output of the `collect_cv_predictions` function.

    Returns
    -------
    altair.Chart
        A chart with the residuals vs predicted values scatter plot.
    """
    all_scatter_plots = []

    x_title = "Predicted Load (MW)"
    y_title = "Residual load (MW): predicted - actual"

    for i, cv_prediction in enumerate(cv_predictions):
        # Get date range for this CV fold
        min_date = cv_prediction["prediction_time"].min().strftime("%Y-%m-%d")
        max_date = cv_prediction["prediction_time"].max().strftime("%Y-%m-%d")
        fold_label = f"#{i+1} - {min_date} to {max_date}"

        # Calculate residuals
        residuals_data = cv_prediction.with_columns(
            [(pl.col("predicted_load_mw") - pl.col("load_mw")).alias("residual")]
        ).with_columns([pl.lit(fold_label).alias("fold_label")])

        # Create scatter plot for this CV fold
        scatter_plot = (
            altair.Chart(residuals_data)
            .mark_circle(opacity=0.6, size=20)
            .encode(
                x=altair.X(
                    "predicted_load_mw:Q",
                    title=x_title,
                    scale=altair.Scale(zero=False),
                ),
                y=altair.Y("residual:Q", title=y_title),
                color=altair.Color("fold_label:N", legend=None),
                tooltip=[
                    "prediction_time:T",
                    "load_mw:Q",
                    "predicted_load_mw:Q",
                    "residual:Q",
                    "fold_label:N",
                ],
            )
        )

        all_scatter_plots.append(scatter_plot)

    all_predictions = pl.concat(
        [cv_pred["predicted_load_mw"] for cv_pred in cv_predictions]
    )
    min_pred, max_pred = all_predictions.min(), all_predictions.max()

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
            x=altair.X("predicted_load_mw:Q", title=x_title),
            y=altair.Y("perfect_residual:Q", title=y_title),
            color=altair.Color(
                "label:N",
                scale=altair.Scale(range=["black"]),
                legend=None,
            ),
        )
    )
    combined_scatter = all_scatter_plots[0]
    for plot in all_scatter_plots[1:]:
        combined_scatter += plot

    return (combined_scatter + perfect_line).resolve_scale(color="independent")


def plot_binned_residuals(cv_predictions, by="hour"):
    """Plot the average residuals binned by time period, one line per CV fold.

    Parameters
    ----------
    cv_predictions : list of polars.DataFrame
        A list of polars DataFrames, each containing the observed and predicted values
        for a given fold. It is the output of the `collect_cv_predictions` function.
    by : str, default="hour"
        The time period to bin by. Can be "hour" or "month".

    Returns
    -------
    altair.Chart
        A chart with the binned residuals.
    """
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
        min_date = cv_prediction["prediction_time"].min().strftime("%Y-%m-%d")
        max_date = cv_prediction["prediction_time"].max().strftime("%Y-%m-%d")
        fold_label = f"#{i+1} - {min_date} to {max_date}"

        residuals_detailed = cv_prediction.with_columns(
            [
                (pl.col("predicted_load_mw") - pl.col("load_mw")).alias("residual"),
                time_extractor,
            ]
        )

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
            .with_columns(pl.lit(fold_label).alias("fold_label"))
        )

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
        iqr_band = (
            altair.Chart(residuals_stats)
            .mark_area(opacity=0.15)
            .encode(
                x=altair.X(f"{time_column}:O", title=x_title),
                y=altair.Y("q25_residual:Q"),
                y2=altair.Y2("q75_residual:Q"),
            )
        )

        mean_line = (
            altair.Chart(residuals_stats)
            .mark_line(tooltip=True, point=True, opacity=0.8)
            .encode(
                x=altair.X(f"{time_column}:O", title=x_title),
                y=altair.Y("mean_residual:Q", title="Mean residual (MW)"),
                color=altair.Color("fold_label:N", legend=None),
                detail="fold_label:N",
            )
        )

        all_iqr_bands.append(iqr_band)
        all_mean_lines.append(mean_line)

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

    combined_iqr = all_iqr_bands[0]
    for band in all_iqr_bands[1:]:
        combined_iqr += band

    combined_lines = all_mean_lines[0]
    for line in all_mean_lines[1:]:
        combined_lines += line

    return (combined_iqr + combined_lines + perfect_line).resolve_scale(
        color="independent"
    )


def plot_horizon_forecast(
    targets,
    named_predictions,
    plot_at_time,
    historical_timedelta,
    target_column_name_pattern,
):
    """Plot the true target and the forecast values for different horizons.

    Parameters
    ----------
    targets : polars.DataFrame
        A DataFrame containing the true target values.
    named_predictions : polars.DataFrame
        A DataFrame containing the predicted values.
    plot_at_time : datetime.datetime
        The time at which to plot the forecast.
    historical_timedelta : datetime.timedelta
        The historical timedelta to use for the plot.
    target_column_name_pattern : str
        The pattern to use for the target column names.

    Returns
    -------
    altair.Chart
        A chart with the true target and the forecast values for different horizons.
    """
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
