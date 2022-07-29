"""
Functions for plotting things

"""
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def projections(
    mc: np.ndarray, ag: np.ndarray, mc_wt: np.ndarray = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot projections of 6d points

    :param mc: (Nx6) shape array of monte carlo (i.e. the original distribution) points
    :param ag: (Nx6) shape array of AmpGen (i.e. the target distribution) points
    :param mc_wt: weights to apply to the MC points
    :param mc_wt: weights to apply to the AmpGen points

    :returns: the figure and axes that we have plotted on

    """
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))

    hist_kw = {"density": True, "histtype": "step"}
    for axis, ag_x, mc_x in zip(ax.ravel(), ag.T, mc.T):
        _, bins, _ = axis.hist(ag_x, bins=100, label="AG", **hist_kw)
        axis.hist(mc_x, bins=bins, label="MC", **hist_kw, alpha=0.5)
        if mc_wt is not None:
            axis.hist(mc_x, bins=bins, label="Reweighted", **hist_kw, weights=mc_wt)

    ax[0, 0].legend()

    return fig, ax


def _counts(t: np.ndarray, wt: np.ndarray, bins: np.ndarray) -> Tuple:
    """
    Returns the counts in each bin and their errors

    :param t: times
    :param wt: weights
    :param bins: bins

    :return: counts in each bin
    :return: errors on counts

    """
    indices = np.digitize(t, bins) - 1

    # Underflow
    if -1 in indices:
        raise ValueError(f"Underflow: bins from {bins[0]}; {np.min(t)=}")

    # Overflow
    if len(bins) - 1 in indices:
        raise ValueError("Overflow")

    n_bins = len(bins) - 1

    # Init with NaN so its obvious if something has gone wrong
    counts = np.ones(n_bins) * np.nan
    errs = np.ones(n_bins) * np.nan

    for i in range(n_bins):
        mask = indices == i

        counts[i] = np.sum(wt[mask])
        errs[i] = np.sqrt(np.sum(wt[mask] ** 2))

    return counts, errs


def _plot_shaded_area(
    ax: plt.Axes, opt: np.ndarray, cov: np.ndarray, pts: np.ndarray
) -> None:
    """ Plot shaded area around the best fit on the axis """
    err = np.sqrt(np.diag(cov))
    a_max, b_max = opt[0] + err[0], opt[1] + err[1]
    a_min, b_min = opt[0] - err[0], opt[1] - err[1]

    ax.fill_between(
        pts,
        [a_max * x + b_max for x in pts],
        [a_min * x + b_min for x in pts],
        color="r",
        alpha=0.2,
    )

    # Display gradient on the plot
    ax.text(2.0, 0.85, rf"{opt[0]:.4f}$\pm${err[0]:.4f}")


def _plot_ratio(
    ax: plt.Axes,
    numerator: np.ndarray,
    denominator: np.ndarray,
    num_wt: np.ndarray,
    denom_wt: np.ndarray,
    bins: np.ndarray,
) -> float:
    """
    Plot the ratio of two histograms on an axis

    Returns chi2

    """
    # Just use the centre of each bin for plotting
    centres = (bins[:-1] + bins[1:]) / 2
    widths = (bins[:-1] - bins[1:]) / 2

    num_counts, num_errs = _counts(numerator, num_wt, bins)
    denom_counts, denom_errs = _counts(denominator, denom_wt, bins)

    ratio = num_counts / denom_counts
    err = ratio * np.sqrt(
        (num_errs / num_counts) ** 2 + (denom_errs / denom_counts) ** 2
    )

    # Plot
    ax.errorbar(
        centres,
        ratio,
        xerr=widths,
        yerr=err,
        fmt="k+",
    )
    ax.set_ylim(0.8, 1.2)

    # Plot ideal
    pts = np.linspace(*ax.get_xlim())
    ax.plot(pts, np.ones_like(pts), "k--")

    # Do fit, plot best
    opt, cov = curve_fit(
        lambda x, a, b: a * x + b,
        centres,
        ratio,
        p0=(0, 1),
        sigma=err,
        absolute_sigma=True,
    )

    def best_fit(t: float) -> float:
        """ Best fit fcn """
        return opt[0] * t + opt[1]

    ax.plot(pts, [best_fit(x) for x in pts], "r--")

    # Plot also shaded area
    _plot_shaded_area(ax, opt, cov, pts)

    # Find chi2
    n_dof = (len(bins) - 1) - 2  # n_bins - 2 since we have 2 fit params
    return np.sum(((ratio - best_fit(centres)) ** 2) / (n_dof * err))


def plot_ratios(
    rs_mc_t, ws_mc_t, rs_ag_t, ws_ag_t, rs_mc_wt, ws_mc_wt, bins: np.ndarray
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot WS/RS ratios for mc, ampgen, reweighted

    """
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    mc_chi2 = _plot_ratio(
        ax[0], ws_mc_t, rs_mc_t, np.ones_like(ws_mc_t), np.ones_like(rs_mc_t), bins
    )
    ag_chi2 = _plot_ratio(
        ax[1], ws_ag_t, rs_ag_t, np.ones_like(ws_ag_t), np.ones_like(rs_ag_t), bins
    )
    reweighted_chi2 = _plot_ratio(ax[2], ws_mc_t, rs_mc_t, ws_mc_wt, rs_mc_wt, bins)

    ax[0].set_title(fr"MC, $\chi^2=${mc_chi2:.3f}")
    ax[1].set_title(fr"AmpGen, $\chi^2=${ag_chi2:.3f}")
    ax[2].set_title(fr"Reweighted, $\chi^2=${reweighted_chi2:.3f}")

    ax[0].set_ylabel(r"$\frac{WS}{RS}$ ratio")
    for a in ax:
        a.set_xlabel("t/ ps")
    ax[2].text(0.2, 0.92, "Errors are approximate")

    fig.tight_layout()
    plt.subplots_adjust(wspace=0)

    return fig, ax
