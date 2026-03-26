"""Statistical utility functions."""

import numpy as np
import pandas as pd


def barplot_summary(
    df,
    *,
    x=None,
    y,
    hue=None,
    col=None,
    row=None,
    errorbar=("ci", 95),
    n_boot=1000,
    seed=42,
):
    """Compute summary statistics matching seaborn barplot/catplot.

    Parameters mirror the corresponding seaborn catplot keyword arguments.

    Returns
    -------
    pd.DataFrame
        One row per group with columns for each grouping variable plus
        ``Mean <y>``, ``CI low``, ``CI high``, and ``N``.
    """
    group_cols = [c for c in [row, col, x, hue] if c is not None]

    records = []
    for group_vals, grp in df.groupby(group_cols, sort=True):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        vals = grp[y].dropna().to_numpy(dtype=float)
        n = len(vals)
        mean_val = float(np.mean(vals)) if n > 0 else np.nan

        ci_low = ci_high = np.nan
        if n > 1 and errorbar[0] == "ci":
            rng = np.random.RandomState(seed)
            boot_means = np.array(
                [np.mean(rng.choice(vals, size=n, replace=True)) for _ in range(n_boot)]
            )
            alpha = (100 - errorbar[1]) / 2
            ci_low, ci_high = np.percentile(boot_means, [alpha, 100 - alpha]).tolist()
        elif n == 1:
            ci_low = ci_high = mean_val

        record = dict(zip(group_cols, group_vals))
        record[f"Mean {y}"] = mean_val
        record["CI low"] = ci_low
        record["CI high"] = ci_high
        record["N"] = n
        records.append(record)

    return pd.DataFrame(records)
