from dateutil.relativedelta import relativedelta
from feature_engineering_lib import feature_engineering_outputs
from functools import partial
from pathlib import Path
from plotly.io import read_json, write_json  # noqa: F401
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.metrics import get_scorer, make_scorer, mean_absolute_percentage_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from tutorial_helpers import (
    collect_cv_predictions,
    plot_binned_residuals,
    plot_lorenz_curve,
    plot_reliability_diagram,
    plot_residuals_vs_predicted,
)
import altair
import cloudpickle
import datetime
import numpy as np
import polars as pl
import pyarrow  # noqa: F401
import skrub
import tzdata  # noqa: F401
import warnings


def _split_indices(X, test_start_date, test_end_date, gap_days=7):
    train = (
        X.with_row_index()
        .filter(pl.col("prediction_time") < test_start_date - datetime.timedelta(days=gap_days))[
            "index"
        ]
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

        first_allowed = (
            min_date
            + relativedelta(days=min_train_days)
            + datetime.timedelta(days=self.train_test_gap_days)
        )

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


def get_hgb_params(max_iter=100):
    return dict(
        random_state=0,
        loss=skrub.choose_from(["squared_error", "poisson", "gamma"], name="loss"),
        early_stopping=True,
        n_iter_no_change=50,
        max_depth=None,
        learning_rate=skrub.choose_float(0.01, 0.5, default=0.1, log=True, name="learning_rate"),
        max_leaf_nodes=skrub.choose_int(2, 128, default=31, log=True, name="max_leaf_nodes"),
        min_samples_leaf=skrub.choose_int(2, 128, default=20, log=True, name="min_samples_leaf"),
    )


def get_regressor(max_iter=100):
    # max_iter = 100 for fast notebook but should set to 1000 or more for best results
    return HistGradientBoostingRegressor(**get_hgb_params(max_iter=max_iter))


def get_cv_results(pred, return_train_score=False):
    """Collect cross-validated predictions and scores.

    The predictions can be a numpy array (one column per target in ``y_test``)
    or a polars dataframe (e.g. several quantiles per horizon, indicated by the
    column names). In both cases they are returned as ``pred_<column>`` columns
    next to ``X_test`` and ``y_test``.
    """
    predictions = []
    scores = []
    for i, split in enumerate(pred.skb.iter_cv_splits()):
        learner = pred.skb.make_learner().fit(split["train"])

        split_scores, split_predictions = learner.score(split["test"], return_predictions=True)
        if return_train_score:
            split_scores.update({f"train_{k}": v for k, v in learner.score(split["train"]).items()})
        scores.append(split_scores | {"split": i})
        y_test = pl.DataFrame(split["y_test"])
        pred_values = split_predictions["predict"]
        if not isinstance(pred_values, pl.DataFrame):
            pred_values = np.asarray(pred_values)
            if pred_values.ndim == 1:
                pred_values = pred_values[:, None]
            pred_values = pl.DataFrame(pred_values, schema=y_test.columns)
        predictions.append(
            pl.concat(
                [
                    split["X_test"],
                    y_test,
                    pred_values.rename("pred_{}".format),
                ],
                how="horizontal",
            ).with_columns(split=pl.lit(i))
        )
        print(f"split {i}:", split["X_test"]["prediction_time"].min().isoformat())
        print(split_scores)

    return pl.concat(predictions, how="vertical"), pl.DataFrame(scores)


def make_ridge_predictions(features, y):
    return features.skb.apply(
        make_pipeline(
            SimpleImputer(add_indicator=True),
            SplineTransformer(sparse_output=True),
            VarianceThreshold(threshold=1e-6),
            SelectKBest(
                score_func=partial(f_regression, force_finite=True),
                k=skrub.choose_int(100, 400, log=True, name="n_selected_splines"),
            ),
            Nystroem(
                n_components=skrub.choose_int(10, 200, log=True, name="n_components", default=150)
            ),
            Ridge(alpha=skrub.choose_float(1e-6, 1e3, log=True, name="alpha", default=1e-2)),
        ),
        y=y,
    ).skb.with_scoring(["neg_mean_absolute_percentage_error", "r2"])
