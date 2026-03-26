"""Plot utils."""

from collections.abc import Sequence
from typing import Literal

import contextily as cx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xvec  # noqa: F401
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats

from swiss_uhi_lcd import plot_utils


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


def bland_altman_plot(
    m1,
    m2,
    sd_limit=1.96,
    ax=None,
    scatter_kwds=None,
    mean_line_kwds=None,
    limit_lines_kwds=None,
    ylim_factor=2,
    annot_fontsize=None,
    axlabel_fontsize=None,
    tick_fontsize=None,
    axes_margin=0.05,
    annot_top_pad=0.12,
    annot_bottom_pad=0.12,
    show_annotations=False,
):
    """
    Bland-Altman plot.

    Mostly hardcopied from `statsmodels.graphics.agreement.mean_diff_plot`.
    """
    # fig, ax = utils.create_mpl_ax(ax)
    if ax is None:
        _, ax = plt.subplots()

    if len(m1) != len(m2):
        raise ValueError("m1 does not have the same length as m2.")
    if sd_limit < 0:
        raise ValueError(f"sd_limit ({sd_limit}) is less than 0.")
    if axes_margin < 0:
        raise ValueError(f"axes_margin ({axes_margin}) is less than 0.")
    if not (0 <= annot_top_pad <= 1):
        raise ValueError(f"annot_top_pad ({annot_top_pad}) must be in [0, 1].")
    if not (0 <= annot_bottom_pad <= 1):
        raise ValueError(f"annot_bottom_pad ({annot_bottom_pad}) must be in [0, 1].")
    if annot_top_pad + annot_bottom_pad >= 1:
        raise ValueError(
            "annot_top_pad + annot_bottom_pad must be < 1 to keep annotations visible."
        )

    means = np.asarray(np.mean([m1, m2], axis=0), dtype=float)
    diffs = np.asarray(m1 - m2, dtype=float)

    finite_mask = np.isfinite(means) & np.isfinite(diffs)
    if not np.any(finite_mask):
        raise ValueError(
            "No finite paired values available to build Bland-Altman plot."
        )

    means = means[finite_mask]
    diffs = diffs[finite_mask]
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, axis=0)

    scatter_kwds = scatter_kwds or {}
    # if "s" not in scatter_kwds:
    #     scatter_kwds["s"] = 20
    mean_line_kwds = mean_line_kwds or {}
    limit_lines_kwds = limit_lines_kwds or {}
    # for kwds in [mean_line_kwds, limit_lines_kwds]:
    #     if "color" not in kwds:
    #         kwds["color"] = "gray"
    #     if "linewidth" not in kwds:
    #         kwds["linewidth"] = 1
    if "linestyle" not in mean_line_kwds:
        mean_line_kwds["linestyle"] = "--"
    if "linestyle" not in limit_lines_kwds:
        limit_lines_kwds["linestyle"] = ":"
    ax.scatter(means, diffs, **scatter_kwds)  # Plot the means against the diffs.
    mean_line = ax.axhline(mean_diff, **mean_line_kwds)  # draw mean line.
    mean_line_color = mean_line.get_color()

    # Annotate mean line with mean difference.
    if show_annotations:
        ax.annotate(
            rf"$\bar{{\Delta T}}={mean_diff:0.2f}$",
            xy=(0.01, 0.5),
            horizontalalignment="left",
            verticalalignment="center",
            rotation=90,
            color=mean_line_color,
            fontsize=annot_fontsize,
            xycoords="axes fraction",
            fontweight="bold",
        )

    y_candidates = [diffs, np.array([mean_diff])]
    if sd_limit > 0:
        half_ylim = (ylim_factor * sd_limit) * std_diff
        # ax.set_ylim(mean_diff - half_ylim, mean_diff + half_ylim)
        limit_of_agreement = sd_limit * std_diff
        lower = mean_diff - limit_of_agreement
        upper = mean_diff + limit_of_agreement
        limit_line_color = None
        for _, lim in enumerate([lower, upper]):
            limit_line = ax.axhline(lim, **limit_lines_kwds)
            limit_line_color = limit_line.get_color()
        y_candidates.append(np.array([lower, upper]))
        if show_annotations:
            ax.annotate(
                f"-{sd_limit} SD: {lower:0.2g}",
                xy=(0.01, annot_bottom_pad),
                horizontalalignment="left",
                verticalalignment="bottom",
                color=limit_line_color,
                fontsize=annot_fontsize,
                xycoords="axes fraction",
            )
            ax.annotate(
                f"+{sd_limit} SD: {upper:0.2g}",
                xy=(0.01, 1 - annot_top_pad),
                horizontalalignment="left",
                verticalalignment="top",
                color=limit_line_color,
                fontsize=annot_fontsize,
                xycoords="axes fraction",
            )
    elif sd_limit == 0:
        half_ylim = 3 * std_diff
        y_candidates.append(np.array([mean_diff - half_ylim, mean_diff + half_ylim]))

    x_min, x_max = np.min(means), np.max(means)
    y_all = np.concatenate(y_candidates)
    y_min, y_max = np.min(y_all), np.max(y_all)

    x_span = x_max - x_min
    y_span = y_max - y_min
    x_pad = axes_margin * (x_span if x_span > 0 else 1.0)
    y_pad = axes_margin * (y_span if y_span > 0 else 1.0)

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_ylabel(r"$T_{LCD} - T_{ref}$", fontsize=axlabel_fontsize)
    ax.set_xlabel("Means", fontsize=axlabel_fontsize)
    ax.tick_params(labelsize=tick_fontsize)
    # fig.tight_layout()
    # return fig
    return ax


def plot_station_map_grid_separate(
    plot_gdf,
    value_col,
    vmin_dict,
    vmax_dict,
    *,
    cmap,
    value_label=None,
    min_width=10e3,
    min_height=6e3,
    cbar_height=1.6,
    fig_width=12,
    left=0.06,
    right=0.88,
    top_with_title=0.85,
    top_no_title=0.99,
    bottom=0.01,
    wspace=0.05,
    pad_frac=0.05,
):
    """Plot station maps on a single city (single-row) grid."""
    if value_label is None:
        value_label = value_col

    station_types = list(plot_gdf["Station type"].unique())
    ncols = len(station_types)
    agglom_names = list(plot_gdf["agglom_name"].unique())

    # subplot width in inches — same for every figure
    ax_w = fig_width * (right - left) / (ncols + (ncols - 1) * wspace)

    figs = {}
    for i_row, agglom_name in enumerate(agglom_names):
        is_first_row = i_row == 0
        top = top_with_title if is_first_row else top_no_title

        agglom_gdf = plot_gdf[plot_gdf["agglom_name"] == agglom_name]
        bounds = agglom_gdf.total_bounds  # minx, miny, maxx, maxy

        # apply padding and min extent
        raw_w = bounds[2] - bounds[0]
        raw_h = bounds[3] - bounds[1]
        pad_w = pad_frac * raw_w
        pad_h = pad_frac * raw_h
        data_w = max(raw_w + 2 * pad_w, min_width)
        data_h = max(raw_h + 2 * pad_h, min_height)
        cx_mid = (bounds[0] + bounds[2]) / 2
        cy_mid = (bounds[1] + bounds[3]) / 2
        xlim = (cx_mid - data_w / 2, cx_mid + data_w / 2)
        ylim = (cy_mid - data_h / 2, cy_mid + data_h / 2)

        # figure height from geographic aspect ratio
        ax_h = ax_w * data_h / data_w
        fig_height = ax_h / (top - bottom)

        with sns.axes_style("white"):
            fig, axes = plt.subplots(
                1, ncols, figsize=(fig_width, fig_height), sharex=True, sharey=True
            )
        fig.subplots_adjust(
            left=left, right=right, top=top, bottom=bottom, wspace=wspace
        )

        for ax, station_type in zip(axes, station_types):
            data = agglom_gdf[agglom_gdf["Station type"] == station_type]
            data.plot(
                value_col,
                ax=ax,
                vmin=vmin_dict[agglom_name],
                vmax=vmax_dict[agglom_name],
                cmap=cmap,
                edgecolor="black",
                aspect=None,
            )

        # set limits derived from geographic extent
        axes[0].set_xlim(*xlim)
        axes[0].set_ylim(*ylim)

        for ax in axes:
            cx.add_basemap(ax, crs=plot_gdf.crs, attribution=False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_ticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        if is_first_row:
            for ax, st in zip(axes, station_types):
                ax.set_title(st)

        axes[0].set_ylabel(agglom_name)

        im = axes[-1].collections[-1]
        cax = inset_axes(
            axes[-1],
            width="3%",
            height=cbar_height,
            loc="center left",
            bbox_to_anchor=(1.05, 0, 1, 1),
            bbox_transform=axes[-1].transAxes,
            borderpad=0,
        )
        fig.colorbar(im, cax=cax, label=value_label)

        figs[agglom_name] = fig

    return figs


def plot_station_map_grid(
    plot_gdf,
    value_col,
    vmin_dict,
    vmax_dict,
    *,
    cmap,
    value_label=None,
    min_width=10e3,
    min_height=6e3,
    cbar_height=1.6,
):
    """Plot station maps on a multi-city (multi-row) grid."""
    if value_label is None:
        value_label = value_col
    with sns.axes_style("white"):
        g = sns.FacetGrid(
            plot_gdf,
            row="agglom_name",
            col="Station type",
            sharex="row",
            sharey="row",
        )

    def _gdf_plot(data, **kwargs):
        ax = plt.gca()
        agglom_name = data["agglom_name"].iloc[0]
        data.plot(
            value_col,
            ax=ax,
            vmin=vmin_dict[agglom_name],
            vmax=vmax_dict[agglom_name],
            cmap=cmap,
            edgecolor="black",
        )

    g.map_dataframe(_gdf_plot)

    for ax_row in g.axes:
        for min_range, axis in zip([min_width, min_height], ["x", "y"]):
            plot_utils.enforce_min_ax_range(ax_row, min_range, axis)

    for ax in g.axes.flat:
        cx.add_basemap(ax, crs=plot_gdf.crs, attribution=False)

    for ax in g.axes.flat:
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_ticks([])
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_visible(False)

    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    for ax_row in g.axes:
        ax_row[0].set_ylabel(ax_row[0].get_title().split(" | ")[0])
    for ax in g.axes[0]:
        ax.set_title(ax.get_title().split(" | ")[1])
    for ax_row in g.axes[1:]:
        for ax in ax_row:
            ax.set_title("")

    g.tight_layout()

    for ax_row in g.axes:
        ax = ax_row[-1]
        im = ax.get_children()[0]
        cax = inset_axes(
            ax,
            width="3%",
            height=cbar_height,
            loc="center left",
            bbox_to_anchor=(1.05, 0, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        g.figure.colorbar(im, cax=cax, label=value_label)

    return g
