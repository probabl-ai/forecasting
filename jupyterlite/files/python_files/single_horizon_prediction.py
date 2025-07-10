# %% [markdown]
#
# # Single horizon predictive modeling
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

# %%
import warnings

import altair
import cloudpickle
import pyarrow  # noqa: F401
import skrub
from plotly.io import write_json, read_json  # noqa: F401

from tutorial_helpers import (
    plot_lorenz_curve,
    plot_reliability_diagram,
    plot_residuals_vs_predicted,
    plot_binned_residuals,
    collect_cv_predictions,
)


# Ignore warnings from pkg_resources triggered by Python 3.13's multiprocessing.
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")


# %%
with open("feature_engineering_pipeline.pkl", "rb") as f:
    feature_engineering_pipeline = cloudpickle.load(f)


features = feature_engineering_pipeline["features"]
targets = feature_engineering_pipeline["targets"]
prediction_time = feature_engineering_pipeline["prediction_time"]
horizons = feature_engineering_pipeline["horizons"]
target_column_name_pattern = feature_engineering_pipeline["target_column_name_pattern"]

# %% [markdown]
#
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
# Once the cross-validation strategy is defined, we pass it to the
# `cross_validate` function provided by `skrub` to compute the cross-validated
# scores. Here, we compute the mean absolute percentage error that is easily
# interpretable and customary for regression tasks with a strictly positive
# target variable such as electricity load forecasting.
#
# We can also look at the R2 score and the Poisson and Gamma deviance which are
# all strictly proper scoring rules for estimation of E[y|X]: in the large
# sample limit, minimizers of those metrics all identify the conditional
# expectation of the target variable given the features for strictly positive
# target variables. All those metrics follow the higher is better convention,
# 1.0 is the maximum reachable score and 0.0 is the score of a model that
# predicts the mean of the target variable for all observations, irrespective
# of the features.
#
# No that in general, a deviance score of 1.0 is not reachable since it
# corresponds to a model that always predicts the target value exactly
# for all observations. In practice, because there is always a fraction of the
# variability in the target variable that is not explained by the information
# available to construct the features.

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
# Those results show very good performance of the model: less than 3% of mean
# absolute percentage error (MAPE) on the test folds. Similarly, all the
# deviance scores are close to 1.0.
#
# We observe a bit of variability in the scores across the different folds: in
# particular the test performance on the first fold seems to be worse than the
# other folds. This is likely due to the fact that the first fold contains
# training data from 2021 and 2022 and the test data mostly from 2023.
#
# The invasion in Ukraine and a sharp drop in nuclear electricity production
# due to safety problems strongly impacted the distribution of the electricity
# prices in 2022, with unprecedented high prices, which can in turn cause a
# shift in the electricity load demand. This could explain a higher than usual
# distribution shift between the train and test folds of the first CV
# iteration.
#
# We can further refine the analysis of the performance of our model by
# collecting the predictions on each cross-validation split.


# %%
hgbr_cv_predictions = collect_cv_predictions(
    hgbr_cv_results["pipeline"], ts_cv_5, hgbr_predictions, prediction_time
)
hgbr_cv_predictions[0]

# %% [markdown]
#
# As a sanity check, we will take a look at the predictions on the first fold and plot
# the observed values and the prediction values from the model. We limit the
# visualization to the last 7 days of the fold.

# %%
altair.Chart(
    hgbr_cv_predictions[0].tail(24 * 7)
).transform_fold(
    ["load_mw", "predicted_load_mw"],
).mark_line(
    tooltip=True
).encode(
    x="prediction_time:T", y="value:Q", color="key:N"
).interactive()

# %% [markdown]
#
# Now, let's check the performance of our models.
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
# Let's evaluate the performance of the model using cross-validation. Use the
# time-based cross-validation splitter `ts_cv_5` defined earlier. Make sure to compute
# the R2 score and the mean absolute percentage error. Return the training scores as
# well as the fitted pipeline such that we can make additional analysis.

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
# Do a sanity check by plotting the observed values and predictions for the first fold
# as we did earlier.
#
# Then, make an analysis of the cross-validated metrics.
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
cv_results_ridge.round(3)

# %%
cv_predictions_ridge = collect_cv_predictions(
    cv_results_ridge["pipeline"], ts_cv_5, predictions_ridge, prediction_time
)

# %%
altair.Chart(cv_predictions_ridge[0].tail(24 * 7)).transform_fold(
    ["load_mw", "predicted_load_mw"],
).mark_line(
    tooltip=True
).encode(
    x="prediction_time:T", y="value:Q", color="key:N"
).interactive()

# %% [markdown]
#
# Compute the Lorenz curve and the reliability diagram for this pipeline.

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
