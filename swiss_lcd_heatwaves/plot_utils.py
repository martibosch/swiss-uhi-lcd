"""Plot utils."""

from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats


def r2_annotate(
    data, x=None, y=None, annot_x=0.05, annot_y=0.8, label=None, labels=None, **kws
):
    """Annotate an axis with the coefficient of determination."""
    r, _ = stats.pearsonr(data[x], data[y])
    # slope, intercept, r, p, se = stats.linregress(data[x], data[y])
    ax = plt.gca()
    if labels is not None:
        annot_y = {_label: annot_y + 0.05 * i for i, _label in enumerate(labels)}.get(
            label
        )
    ax.text(
        annot_x,
        annot_y,
        f"R$^2$={r**2:.2f}",
        color=kws.get("color", None),
        transform=ax.transAxes,
    )


def facet_twinx_lineplot(
    daily_ts_df: pd.DataFrame,
    ref_ts_df: pd.DataFrame,
    variable: str,
    y_col: str,
    *,
    col_wrap: int | None = None,
) -> sns.FacetGrid:
    """Facet grid line plots, each with a twin axes sharing the xaxis."""
    g = sns.FacetGrid(
        daily_ts_df.assign(
            **{
                variable: daily_ts_df["time"].map(
                    ref_ts_df.resample("h").mean()[variable]
                )
            }
        ),
        col="station_id",
        col_wrap=col_wrap,
    )
    # for ax in g.axes.flat:
    #     # add horizontal line at 0
    #     ax.axhline(0, color="gray", linestyle="--")
    g.refline(y=0)

    def plot_variable_twinx(variable, *, data=None, **kwargs):
        """Plot a given variable on a second y axis."""
        # add radiation at a second y axis
        _ax = plt.gca().twinx()
        _ax.grid(False)
        # _ax = ax.twinx()
        sns.lineplot(data, x="hour", y=variable, **kwargs)

    g.map(sns.lineplot, "hour", y_col)
    # g.map(foo, "hour", y_col)
    g.map_dataframe(plot_variable_twinx, variable, color=sns.color_palette()[1])

    # treat last axis separately (because it needs labels on the right even if it is not
    # the last column
    for i, ax in enumerate(g.axes[:-1], start=1):
        res = i % col_wrap
        # all columns but the first: remove left ticks
        if not res == 1:
            # ax.spines["left"].set_visible(False)
            # ax.yaxis.set_ticklabels([])
            for y_ticklabel in ax.yaxis.get_ticklabels():
                y_ticklabel.set_visible(False)
        # all columns but the last: remove right ticks
        if not res == 0:
            twin_ax = ax.get_shared_x_axes().get_siblings(ax)[i + len(g.axes) - 1]
            # twin_ax.spines["left"].set_visible(False)
            twin_ax.set_ylabel(None)
            twin_ax.yaxis.set_ticklabels([])

    # if the last axis is not in the first column, remove the labels on the left
    if len(g.axes) % col_wrap != 1:
        for y_ticklabel in g.axes[-1].yaxis.get_ticklabels():
            y_ticklabel.set_visible(False)

    g.set(xticks=range(0, 26, 6))
    g.set_xlabels("hour")
    # g.figure.savefig("../reports/delta-t-daily-cycle.pdf")
    # g.add_legend()
    # sns.move_legend(g, "center", bbox_to_anchor=(.75, .3))

    g.tight_layout()

    return g


def enforce_min_ax_range(
    ax_row: Sequence[plt.Axes],
    min_range: float,
    axis: Literal["x", "y"],
) -> None:
    """
    Ensure that every Axes in `ax_row` spans at least `min_range` of the axis units.

    Parameters
    ----------
    ax_row : list-like
        Sequence of containing the row of Axes objects.
    min_range : numeric
        Desired minimum span (e.g. `4e3`).
    axis :
        Which axis to adjust.
    """
    # get the current limits from the *first* Axes in the row (all axes in the same row
    # are assumed to share the same data extent)
    low, high = getattr(ax_row[0], f"get_{axis}lim")()
    cur_span = high - low

    # if the current span is already big enough, do nothing.
    if cur_span >= min_range:
        return

    # compute new centred limits that give exactly `min_range` span.
    midpoint = (low + high) / 2.0
    new_low = midpoint - min_range / 2.0
    new_high = midpoint + min_range / 2.0

    # apply the new limits to every Axes in the row.
    set_func = f"set_{axis}lim"
    for ax in ax_row:
        getattr(ax, set_func)(new_low, new_high)
