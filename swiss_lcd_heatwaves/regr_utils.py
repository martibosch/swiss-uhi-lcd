"""Regression utils."""

import copy
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sktime.forecasting import auto_reg, compose


class BestScaleRadiationTransformer(BaseEstimator, TransformerMixin):
    """Select best radiation rolling-sum window and apply it.

    During ``fit``, evaluates each candidate window size in *window_minutes*
    by Pearson correlation with the target and stores the best one as
    ``best_scale_``.  During ``transform``, applies a rolling sum of that
    window to the radiation column and returns a single-column DataFrame.

    Parameters
    ----------
    window_minutes : sequence of int
        Candidate window sizes (in minutes) to evaluate.
    time_col : str, default "time"
        Column of X containing datetime values.
    radiation_col : str, default "radiation_shortwave"
        Column of X containing raw shortwave radiation values.
    """

    def __init__(
        self,
        window_minutes,
        time_col="time",
        radiation_col="radiation_shortwave",
    ):
        self.window_minutes = window_minutes
        self.time_col = time_col
        self.radiation_col = radiation_col

    def _apply_rolling(self, X, window_minutes):
        rad_ser = pd.Series(
            X[self.radiation_col].values,
            index=pd.DatetimeIndex(X[self.time_col].values),
        ).sort_index()
        rolled = rad_ser.rolling(
            pd.Timedelta(minutes=window_minutes), min_periods=1
        ).sum()
        return X[self.time_col].map(rolled)

    def fit(self, X, y):
        """Fit the model to the given data."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        y_arr = np.asarray(y, dtype=float)

        best_r = -np.inf
        best_scale = list(self.window_minutes)[0]
        for w in self.window_minutes:
            x_vals = self._apply_rolling(X, w)
            mask = x_vals.notna() & np.isfinite(y_arr)
            if mask.sum() < 2:
                continue
            r = stats.pearsonr(x_vals[mask].values, y_arr[mask]).statistic
            if r > best_r:
                best_r = r
                best_scale = w

        self.best_scale_ = best_scale
        return self

    def transform(self, X):
        """Apply the rolling sum with the best window size found during fit."""
        check_is_fitted(self)
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return self._apply_rolling(X, self.best_scale_).to_frame(self.radiation_col)


def _compare_models(station_ts_df, model_dict, y_col, x_cols):
    station_X_df = station_ts_df[x_cols].reset_index(drop=True)
    y_ser = station_ts_df[y_col].reset_index(drop=True)
    # station_index = station_ts_df.index
    _fit_dict = {}
    _fit_model_dict = {}
    for model_label in model_dict:
        _pipeline = copy.deepcopy(model_dict[model_label])
        # model = _pipeline.steps[-1][1]
        if isinstance(_pipeline, compose.ForecastingPipeline):
            _pipeline.fit(y_ser, X=station_X_df)
            yhat = _pipeline.predict(
                fh=list(range(1, len(y_ser) + 1)),
                # fh=base.ForecastingHorizon(
                #     np.arange(1, len(station_X_df.index) + 1)
                # ),
                # fh=station_X_df.index,
                X=station_X_df,
            )
        else:
            _pipeline.fit(station_X_df, y=y_ser)
            yhat = _pipeline.predict(station_X_df)
        _fit_dict[model_label] = stats.pearsonr(y_ser, yhat).statistic ** 2
        _fit_model_dict[model_label] = _pipeline

    return _fit_dict, _fit_model_dict, yhat


def compare_models(regr_df, model_dict, y_col, x_cols):
    """Compare models."""
    # p_val, should_diff = pm.arima.stationarity.ADFTest(alpha=0.05).should_diff(
    #     station_ts_df[y_col]
    # )
    # if lam is None:
    #     lam = np.logspace(-3, 5, 5)
    fit_dict = {}
    fit_model_dict = {}
    # res_df = regr_df[["station_id", y_col] + x_cols].copy()
    res_df = pd.DataFrame(index=regr_df.index)
    for station_id, station_ts_df in regr_df.groupby("station_id"):
        station_X_df = station_ts_df[x_cols].reset_index(drop=True)
        y_ser = station_ts_df[y_col].reset_index(drop=True)
        station_index = station_ts_df.index
        _fit_dict = {}
        _fit_model_dict = {}
        for model_label in model_dict:
            _pipeline = copy.deepcopy(model_dict[model_label])
            model = _pipeline.steps[-1][1]
            # if isinstance(_pipeline, compose.ForecastingPipeline):
            #     _pipeline.fit(y_ser, X=station_X_df)
            #     yhat = _pipeline.predict(
            #         fh=list(range(1, len(y_ser) + 1)),
            #         # fh=base.ForecastingHorizon(
            #         #     np.arange(1, len(station_X_df.index) + 1)
            #         # ),
            #         # fh=station_X_df.index,
            #         X=station_X_df,
            #     )
            # else:
            #     _pipeline.fit(station_X_df, y=y_ser)
            #     yhat = _pipeline.predict(station_X_df)
            if isinstance(model, auto_reg.AutoREG):
                _pipeline.fit(y_ser, X=station_X_df)
                yhat = _pipeline.predict(
                    fh=list(range(1, len(y_ser) + 1)),
                    # fh=base.ForecastingHorizon(
                    #     np.arange(1, len(station_X_df.index) + 1)
                    # ),
                    # fh=station_X_df.index,
                    X=station_X_df,
                )
            else:
                _pipeline.fit(station_X_df, y=y_ser)
                yhat = _pipeline.predict(station_X_df)

            _fit_dict[model_label] = stats.pearsonr(y_ser, yhat).statistic ** 2
            _fit_model_dict[model_label] = _pipeline

            res_df.loc[station_index, model_label] = yhat
        fit_dict[station_id] = _fit_dict
        fit_model_dict[station_id] = _fit_model_dict

    return (res_df, pd.DataFrame(fit_dict).T.rename_axis("station_id"), fit_model_dict)


class MultiScaleRegression:
    """Multi-scale regression."""

    def __init__(self, ref_ts_df: pd.DataFrame):
        self.ref_ts_df = ref_ts_df
        self.ref_freq = ref_ts_df.index.inferred_freq

    def compute_variable_at_scale(self, X_df, variable, window):
        """Compute variable at the given scale."""
        try:
            min_periods = int(window / self.ref_freq)
        except ValueError:
            # ValueError: unit abbreviation w/o a number
            min_periods = int(window / pd.Timedelta(f"1 {self.ref_freq}"))
        return (
            X_df["time"]
            .map(
                self.ref_ts_df[variable].rolling(window, min_periods=min_periods).sum()
            )
            .rename(variable)
        )

    def eval_time_scales(
        self,
        X_df: pd.DataFrame,
        y_ser: pd.Series,
        window_minutes: Sequence[float],
        *,
        variables: Sequence | None = None,
        eval_func: str | None = None,
        **eval_func_kwargs: Mapping,
    ):
        """Evaluate the given time scales."""
        if variables is None:
            variables = self.ref_ts_df.columns
        if eval_func is None:
            # TODO: use settings
            eval_func = "pearsonr"
        _eval_func = getattr(stats, eval_func)
        _eval_dfs = []
        for _window_minutes in window_minutes:
            window = pd.Timedelta(minutes=_window_minutes)
            # TODO: support evaluating all variables together
            for variable in variables:
                x_ser = self.compute_variable_at_scale(X_df, variable, window)
                nan_ser = x_ser.isna() | y_ser.isna()
                _eval_dfs.append(
                    (
                        variable,
                        _window_minutes,
                        _eval_func(
                            x_ser[~nan_ser].values,
                            y_ser[~nan_ser].values,
                            **eval_func_kwargs,
                        ).statistic,
                    )
                )
        return pd.DataFrame(_eval_dfs, columns=["variable", "scale", eval_func])

    def get_regr_df(
        self,
        long_ts_df,
        y_col,
        window_minutes,
        *,
        variables=None,
        eval_func="pearsonr",
        add_scale_to_col_name=False,
        rescale=True,
    ):
        """Get regression data frame."""
        # evaluate scales
        eval_df = (
            self.eval_time_scales(
                long_ts_df.drop(columns=y_col),
                long_ts_df[y_col],  # .shift(periods=-time_lag_dict[station_id]),
                window_minutes,
                variables=variables,
                eval_func=eval_func,
            )
            .set_index("scale")
            .groupby("variable")[eval_func]
            .idxmax()
        ).fillna(0)
        # compute features at the scale of maximum influence
        if add_scale_to_col_name:

            def compute_variable_at_scale(long_ts_df, variable, scale):
                return self.compute_variable_at_scale(
                    long_ts_df, variable, scale
                ).rename(f"{variable}_{int(scale.seconds / 60)}")
        else:
            compute_variable_at_scale = self.compute_variable_at_scale
        regr_df = pd.concat(
            [
                compute_variable_at_scale(
                    long_ts_df, variable, pd.Timedelta(minutes=scale)
                )
                for variable, scale in eval_df.items()
            ],
            axis="columns",
        )
        if rescale:
            # rescale
            regr_df = pd.DataFrame(
                preprocessing.StandardScaler().fit_transform(regr_df),
                columns=regr_df.columns,
                index=regr_df.index,
            )
        # assign time and response cols and drop nan
        regr_df = regr_df.assign(
            **{col: long_ts_df[col] for col in ["time", y_col]}
        ).dropna()

        # return
        return regr_df
