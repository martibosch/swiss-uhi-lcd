"""Heat warning utils."""

import datetime as dt
from collections.abc import Callable

import pandas as pd
from meteora import settings, utils

# data types
AggFuncType = str | Callable | None

# MeteoSwiss warnings
LEVEL_4_KWARGS = dict(
    heatwave_t_threshold=27,
    heatwave_n_consecutive_days=3,
    station_agg_func="mean",
    inter_station_agg_func="max",
)
LEVEL_3_KWARGS = dict(
    heatwave_t_threshold=25,
    heatwave_n_consecutive_days=3,
    station_agg_func="mean",
    inter_station_agg_func="max",
)
LEVEL_2_KWARGS = dict(
    heatwave_t_threshold=25,
    heatwave_n_consecutive_days=1,
    station_agg_func="mean",
    inter_station_agg_func="max",
)


def get_heatwave_periods(
    ts_df: pd.DataFrame,
    *,
    heatwave_t_threshold: float | None = None,
    heatwave_n_consecutive_days: int | None = None,
    station_agg_func: AggFuncType = None,
    inter_station_agg_func: AggFuncType = None,
) -> list[tuple[dt.date, dt.date]]:
    """Get the heatwave periods from a time series of temperature measurements.

    A heatwave is defined as a period of at least `heatwave_n_consecutive_days` days
    with a temperature above `heatwave_t_threshold`.

    Parameters
    ----------
    ts_df : pd.DataFrame
        Data frame with a time series of temperature measurements at each station, in
        long or wide format.
    heatwave_t_threshold : float, optional
        The temperature threshold for a heatwave, in units of `ts_df`. If not provided,
        the value from `settings.HEATWAVE_T_THRESHOLD` is used.
    heatwave_n_consecutive_days : int, optional
        The number of consecutive days above the mean temperature threshold for the
        corresponding period to be  considered a heatwave. If not provided, the value
        from `settings.HEATWAVE_N_CONSECUTIVE_DAYS` is used.
    station_agg_func, inter_station_agg_func : str or function, optional
        How to respectively aggregate the daily temperature measurements at each station
        and the aggregated daily temperature measurements across all stations. Must be a
        string function name or a callable function, which will be passed as the `func`
        argument of `pandas.core.groupby.DataFrameGroupBy.agg`. If not provided, the
        respective values from `settings.HEATWAVE_STATION_AGG_FUNC` and
        `settings.HEATWAVE_INTER_STATION_AGG_FUNC` are used.

    Returns
    -------
    heatwave_range_df : pd.DataFrame
        Data frame with the heatwave start and end dates as columns, indexed by the
        heatwave event identifier.
    """
    # if using a multi-index, assume that it is a long-form data frame so transform it
    if isinstance(ts_df.index, pd.MultiIndex):
        # ACHTUNG: we are assuming that there is a single column with the temperature
        ts_df = utils.long_to_wide(ts_df, variables=ts_df.columns[0])

    # process arguments
    if heatwave_t_threshold is None:
        heatwave_t_threshold = settings.HEATWAVE_T_THRESHOLD
    if heatwave_n_consecutive_days is None:
        heatwave_n_consecutive_days = settings.HEATWAVE_N_CONSECUTIVE_DAYS
    if station_agg_func is None:
        station_agg_func = settings.HEATWAVE_STATION_AGG_FUNC
    if inter_station_agg_func is None:
        inter_station_agg_func = settings.HEATWAVE_INTER_STATION_AGG_FUNC

    # find consecutive days above threshold
    # day_agg_ts_ser = getattr(
    #     getattr(ts_df.groupby(ts_df.index.date), station_agg_func)(),
    #     inter_station_agg_func,
    # )(axis="columns")
    day_agg_ts_ser = (
        ts_df.groupby(ts_df.index.date)
        .agg(station_agg_func)
        .agg(inter_station_agg_func, axis="columns")
    )
    # idx = (
    #     day_agg_ts_ser.ge(heatwave_t_threshold)
    #     .rolling(window=heatwave_n_consecutive_days, center=False)
    #     .sum()
    #     .ge(heatwave_n_consecutive_days)
    # )

    ge_sel_ser = day_agg_ts_ser.ge(heatwave_t_threshold)
    consecutive_ge_ser = (
        day_agg_ts_ser[ge_sel_ser]
        .index.to_series()
        .groupby((~ge_sel_ser).cumsum())
        .agg(["first", "last", "count"])
    )

    return [
        (
            dt.datetime.combine(row["first"], dt.time.min),
            dt.datetime.combine(row["last"], dt.time.max),
        )
        for i, row in consecutive_ge_ser[
            consecutive_ge_ser["count"].ge(heatwave_n_consecutive_days)
        ].iterrows()
    ]


def get_heatwave_periods_dict(ts_df_dict: dict) -> dict:
    """Get a dictionary of heatwave periods for each station type and warning level."""
    return {
        level: {
            station_type: {
                year: get_heatwave_periods(year_ts_df, **level_kwargs)
                for year, year_ts_df in ts_df.groupby(ts_df.index.year)
            }
            for station_type, ts_df in ts_df_dict.items()
        }
        for level, level_kwargs in {
            "Level 4": LEVEL_4_KWARGS,
            "Level 3": LEVEL_3_KWARGS,
            "Level 2": LEVEL_2_KWARGS,
        }.items()
    }


def get_heat_days_df(heatwave_periods_dict: dict) -> pd.DataFrame:
    """Get heat days data frame for each station type and warning level."""

    def _days_in_periods(periods):
        return sum(
            [pd.Timedelta(end - start).round("d").days for start, end in periods]
        )

    records = []
    for level, station_type_dict in heatwave_periods_dict.items():
        for station_type, station_heatwave_dict in station_type_dict.items():
            for year, heatwave_periods in station_heatwave_dict.items():
                records.append(
                    (
                        level,
                        station_type,
                        year,
                        _days_in_periods(heatwave_periods),
                    )
                )
    return pd.DataFrame(records, columns=["level", "station_type", "year", "n. days"])
