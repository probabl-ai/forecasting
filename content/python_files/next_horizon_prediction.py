# %% [markdown]
#
# # Next horizon predictive modeling
#
# ## Environment setup
#
# We need to install some extra dependencies for this notebook if needed (when
# running jupyterlite).

# %%
# %pip install -q skrub altair holidays plotly nbformat polars

# %%
import warnings
from pathlib import Path

import altair
import numpy as np
import cloudpickle
import pyarrow  # noqa: F401
import skrub
import tzdata  # noqa: F401
from plotly.io import write_json, read_json  # noqa: F401
import polars as pl

from sklearn.ensemble import HistGradientBoostingRegressor


from tutorial_helpers import (
    plot_lorenz_curve,
    plot_reliability_diagram,
    plot_residuals_vs_predicted,
    plot_binned_residuals,
    collect_cv_predictions,
)
from feature_engineering_lib import feature_engineering_outputs

# Ignore warnings from pkg_resources triggered by Python 3.13's multiprocessing.
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

# %% [markdown]
#
# For now, let's focus on the last horizon (1 hour) to train a model
# predicting the electricity load at the next 1 hour.

# %% [markdown]
#
# ## Cross-validation splitter
#
# The first thing we need to do in our pipeline, now that we have X and y, is
# to define how they are split into training and testing sets. For this we
# define a custom time-based cross-validation splitter.
#
# We do not use scikit-learn's TimeSeriesSplit because it is based on
# positional indices, while here we can have an irregular grid after dropping
# rows with missing ground truth. It is also easier to inspect and debug splits
# based on actual dates and a datetime column than on row positions.
#
# In this implementation, each fold starts after an initial training period of
# two years, keeps a 7-day gap between the end of training and the start of the
# test window, and evaluates on 3-month blocks that move forward by 3 months at
# each iteration. This is an example of a [Backtesting with intermittent refit including gap](https://skforecast.org/latest/introduction-forecasting/introduction-forecasting#backtesting-with-intermittent-refit)
# setup that is common in forecasting problems, where split boundaries must mimic operational
# constraints in deployment.
#
# ![Backtesting with intermittent refit](https://skforecast.org/latest/img/time-series-backtesting-forecasting-with-gap.gif)
#
# When we want an actual value to inspect, experiment with, or debug, we can
# call .skb.preview(). It gives us the output of the pipeline for the preview
# example data we set on the variables. Getting it is cheap because it is
# precomputed eagerly when we define the dataop so it is readily available.
# Here, for example, we grab the value of X (a dataframe) and use it to test
# and debug our splitter.

# %%
import datetime
from dateutil.relativedelta import relativedelta


def _split_indices(X, test_start_date, test_end_date, gap_days=7):
    train = (
        X.with_row_index()
        .filter(
            pl.col("prediction_time") < test_start_date - datetime.timedelta(days=gap_days)
        )["index"]
        .to_numpy()
    )
    test = (
        X.with_row_index()
        .filter(
            (pl.col("prediction_time") >= test_start_date)
            & (pl.col("prediction_time") < test_end_date)
        )["index"]
        .to_numpy()
    )
    return train, test
    
class TimeSeriesSplitter:
    train_test_gap_days = 7
    test_blocks = 3

    def split(self, X, y=None, groups=None, blocks=None):
        if blocks is None:
            blocks = self.test_blocks
        min_train_days = 365 * 2  # Initial train period: 2 years
        min_date = X["prediction_time"].min()
        max_date = X["prediction_time"].max()

        first_allowed = min_date + relativedelta(days=min_train_days) + datetime.timedelta(days=self.train_test_gap_days)

        # Align to the first day of the first full month available.
        start_date = first_allowed.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if start_date < first_allowed:
            start_date = start_date + relativedelta(months=1)
        
        test_start_dates = []
        current_test_start = start_date

        while current_test_start < max_date:
            test_start_dates.append(current_test_start)
            # advance by 3 months (quarter)
            # Using relativedelta for correct month arithmetic:
            current_test_start = current_test_start + relativedelta(months=blocks)

        for test_start in test_start_dates:
            test_end = test_start + relativedelta(months=blocks)  
            train, test = _split_indices(X, test_start, test_end, gap_days=self.train_test_gap_days)
            if len(train) and len(test):
                yield train, test

    def get_n_splits(self, X, y=None, groups=None):
        return len(list(self.split(X, y)))

# %%
TIME_HORIZON = 1  # Focus on next step prediction
features, y = feature_engineering_outputs(horizons=TIME_HORIZON, cv_splitter=TimeSeriesSplitter())

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
# Let's check those statistics by iterating over the different folds provided by the
# splitter..

# %%
def get_regressor():
    loss = skrub.choose_from(["squared_error", "poisson", "gamma"], name="loss")

    return HistGradientBoostingRegressor(
        random_state=0,
        loss=loss,
        learning_rate=skrub.choose_float(
            0.01, 0.7, default=0.1, log=True, name="learning_rate"
        ),
        max_leaf_nodes=skrub.choose_int(3, 300, default=30, log=True, name="max_leaf_nodes"),
    )

pred = features.skb.apply(get_regressor(), y=y).skb.with_scoring(
    ["neg_mean_absolute_percentage_error", "r2"]
)
pred.skb.preview()

# %%

pred.skb.cross_validate()

# %% [markdown]
#
# For further inspection of predictions, we will collect the cross-validated
# prediction into a dataframe. To easily inspect the output of the pipeline and
# debug our cross-validation loop, we perform one train/test split to have an
# example to work with.

# %%
split = pred.skb.train_test_split()
split["X_test"]

# %% 
split["y_test"]

# %%
pred.skb.make_learner().fit(split["train"]).predict(split["test"])

# %% [markdown]
#
# Now we can collect predictions for all splits and plot them.

# %%
def get_cv_results(pred, return_train_score=False):
    predictions = []
    scores = []
    for i, split in enumerate(pred.skb.iter_cv_splits()):
        learner = pred.skb.make_learner().fit(split["train"])

        split_scores, split_predictions = learner.score(
            split["test"], return_predictions=True
        )
        if return_train_score:
            split_scores.update(
                {f"train_{k}": v for k, v in learner.score(split["train"]).items()}
            )
        scores.append(split_scores | {"split": i})
        y_test = pl.DataFrame(split["y_test"])
        pred_values = np.asarray(split_predictions["predict"])
        if pred_values.ndim == 1:
            pred_values = pred_values[:, None]
        pred_columns = pl.DataFrame(
            {
                f"pred_{column}": pred_values[:, idx]
                for idx, column in enumerate(y_test.columns)
            }
        )
        predictions.append(
            pl.concat(
                [
                    split["X_test"],
                    y_test,
                    pred_columns,
                ], how="horizontal"
            ).with_columns(split=pl.lit(i))
        )
        print(f"split {i}:", split["X_test"]["prediction_time"].min().isoformat())
        print(split_scores)

    return pl.concat(predictions, how="vertical"), pl.DataFrame(scores)


cv_predictions, cv_scores = get_cv_results(pred)


# %% [markdown]
#
# As a sanity check, we will take a look at the predictions on the first fold and plot
# the observed values and the prediction values from the model. We limit the
# visualization to the last 7 days of the fold.

# %%
altair.Chart(
    cv_predictions.tail(100)
).transform_fold(
    ["1h", "pred_1h"],
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
plot_lorenz_curve(cv_predictions, TIME_HORIZON).interactive()

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
# For our usecase, we observe that each oracle model is not far from the diagonal. It
# means that the observed values do not contain a couple of large values with high
# variability. Therefore, it informs us that the complexity of our problem at hand is
# not too high. Looking at the Lorenz curve of each model, we observe that it is quite
# close to the oracle model. Therefore, the gradient boosting regressor is
# discriminative for our task.
#
# Then, we have a look at the reliability diagram. This diagram shows on the x-axis the
# mean predicted load and on the y-axis the mean observed load.

# %%
plot_reliability_diagram(cv_predictions, TIME_HORIZON).interactive().properties(
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
plot_residuals_vs_predicted(cv_predictions, TIME_HORIZON).interactive().properties(
    title="Residuals vs Predicted Values from cross-validation predictions"
) 

# %%
plot_binned_residuals(cv_predictions, TIME_HORIZON, by="hour").interactive().properties(
    title="Residuals by hour of the day from cross-validation predictions"
)

# %%

plot_binned_residuals(cv_predictions, TIME_HORIZON, by="month").interactive().properties(
    title="Residuals by hour of the day from cross-validation predictions"
)

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
# Chaining several `.skb.apply(...)` calls can be useful to inspect intermediate
# outputs in reports, but for this exercise we keep a single pipeline. With the
# current behavior of `SplineTransformer(sparse_output=True)`, step-by-step
# `.skb.apply(...)` would require `no_wrap=True` to avoid wrapping sparse output
# into a dataframe.
#
# Once the predictive model is defined, apply it on `X` and pass `y` as the
# target.


# %%
# Here we provide all the imports for creating the predictive model.
from functools import partial
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.metrics import get_scorer, make_scorer, mean_absolute_percentage_error

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
predictions_ridge = features.skb.apply(
    make_pipeline(
        SimpleImputer(add_indicator=True),
        SplineTransformer(sparse_output=True),
        VarianceThreshold(threshold=1e-6),
        SelectKBest(
            score_func=partial(f_regression, force_finite=True),
            k=skrub.choose_int(100, 400, log=True, name="n_selected_splines"),
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
    y=y,
).skb.with_scoring(["neg_mean_absolute_percentage_error", "r2"])
predictions_ridge.skb.preview()

# %% [markdown]
#
# Now that you defined the predictive model, let's evaluate the performance of
# the model using cross-validation. Use the time-based cross-validation
# splitter defined earlier in `TimeSeriesSplitter`. Make sure to compute the R2 score and the
# mean absolute percentage error. Return the training scores as well as the
# fitted pipeline such that we can make additional analysis.

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
cv_predictions_ridge, cv_scores_ridge = get_cv_results(predictions_ridge, return_train_score=True)


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
cv_scores_ridge


# %%
altair.Chart(cv_predictions_ridge.tail(24 * 7)).transform_fold(
    ["1h", "pred_1h"],
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
plot_lorenz_curve(cv_predictions_ridge, TIME_HORIZON).interactive()

# %%
plot_reliability_diagram(cv_predictions_ridge, TIME_HORIZON).interactive().properties(
    title="Reliability diagram from cross-validation predictions"
)

# %% [markdown]
#
# Now, let's perform a randomized search on the hyper-parameters of the model. The code
# to perform the search is shown below. Since it will be pretty computationally
# expensive, we are reloading the results of the parallel coordinates plot.

# %%
randomized_search_ridge = predictions_ridge.skb.make_randomized_search(
     refit="r2",
     n_iter=50,
     fitted=True,
     verbose=1,
     n_jobs=-1,
 )

# %%
randomized_search_ridge.plot_results().update_layout(margin=dict(l=200))

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
#      environment=predictions_ridge.skb.get_data(),
#      learner=randomized_search_ridge,
#      cv=TimeSeriesSplitter(),
#      scoring={
#          "r2": get_scorer("r2"),
#          "mape": make_scorer(mean_absolute_percentage_error),
#      },
#      n_jobs=-1,
#      return_learner=True,
#  ).round(3)

# %%
# nested_cv_results_ridge.round(3)