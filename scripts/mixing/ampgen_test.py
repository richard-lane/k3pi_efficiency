"""
Introduce some mixing to the AmpGen dataframes via weighting

"""
import sys
import pathlib
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fourbody.param import helicity_param

sys.path.append(str(pathlib.Path(__file__).resolve().parents[0]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
import common
import pdg_params
from lib_efficiency import efficiency_util, mixing, metrics
from lib_efficiency.plotting import phsp_labels
from lib_efficiency.phsp_binning import coherence_factor


def _weight_hist(weights: np.ndarray) -> None:
    """ Show histogram of weights """
    plt.hist(weights, bins=100)
    plt.show()


def _bc_params(
    dataframe: pd.DataFrame, weights: np.ndarray, params: mixing.MixingParams
) -> Tuple[float, float]:
    """
    Time ratio params, assuming rD = 1 and times in lifetimes

    """
    z_mag, z_phase = coherence_factor(*efficiency_util.k_3pi(dataframe), weights)
    z_re = -z_mag * np.cos(z_phase)
    z_im = -z_mag * np.sin(z_phase)

    return (params.mixing_x * z_im + params.mixing_y * z_re), (
        params.mixing_x ** 2 + params.mixing_y ** 2
    ) / 4


def _time_plot(
    params: mixing.MixingParams,
    cf_df: pd.DataFrame,
    dcs_df: pd.DataFrame,
    dcs_wt: np.ndarray,
) -> None:
    """
    Plot ratio of WS/RS decay times

    """
    cf_t = cf_df["time"]
    dcs_t = dcs_df["time"]

    fig, ax = plt.subplots()

    bins = np.linspace(0, 8, 15)
    bins = np.append(bins, 12.6)
    centres = (bins[1:] + bins[:-1]) / 2
    widths = (bins[1:] - bins[:-1]) / 2

    cf_counts, cf_errs = metrics._counts(cf_t, np.ones(len(cf_t)), bins)
    dcs_counts, dcs_errs = metrics._counts(dcs_t, np.ones(len(dcs_t)), bins)
    weighted_counts, weighted_errs = metrics._counts(dcs_t, dcs_wt, bins)

    ratio = dcs_counts / cf_counts
    err = ratio * np.sqrt((dcs_errs / dcs_counts) ** 2 + (cf_errs / cf_counts) ** 2)

    weighted_ratio = weighted_counts / cf_counts
    weighted_err = weighted_ratio * np.sqrt(
        (weighted_errs / weighted_counts) ** 2 + (cf_errs / cf_counts) ** 2
    )

    ax.errorbar(centres, ratio, yerr=err, xerr=widths, label="Unweighted", fmt="k+")
    ax.errorbar(
        centres,
        weighted_ratio,
        yerr=weighted_err,
        xerr=widths,
        label="Weighted",
        fmt="r+",
    )

    # Ideal
    a, (b, c) = 1, _bc_params(dcs_df, dcs_wt, params)
    points = np.linspace(*ax.get_xlim())
    ax.plot(points, [1 for _ in points], "k--")
    ax.plot(points, [a + b * x + c * x ** 2 for x in points], "r--")

    ax.set_xlabel(r"$\frac{t}{\tau}$")
    ax.set_ylabel(r"$\frac{WS}{RS}$")
    ax.legend()
    plt.savefig("ampgen_mixed_times.png")

    plt.show()


def _hists(cf_df: pd.DataFrame, dcs_df: pd.DataFrame, weights: np.ndarray) -> None:
    """
    Show phase space histograms

    And time ratio plots

    """
    cf_pts = np.column_stack(
        (helicity_param(*efficiency_util.k_3pi(cf_df)), cf_df["time"])
    )
    dcs_pts = np.column_stack(
        (helicity_param(*efficiency_util.k_3pi(dcs_df)), dcs_df["time"])
    )

    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    hist_kw = {"histtype": "step", "density": True}
    for a, cf, dcs, label in zip(ax.ravel(), cf_pts.T, dcs_pts.T, phsp_labels()):
        _, bins, _ = a.hist(cf, bins=100, label="CF", **hist_kw)
        a.hist(dcs, bins=bins, label="DCS", **hist_kw)
        a.hist(dcs, bins=bins, label="weighted", weights=weights, **hist_kw)
        a.set_xlabel(label)

    ax[0, 0].legend()
    ax.ravel()[-1].legend()

    fig.tight_layout()
    plt.savefig("ampgen_mixed_hists.png")

    plt.show()


def main():
    """
    Read AmpGen dataframes, add some mixing to the DCS frame, plot the ratio of the decay times

    """
    # Read AmpGen dataframes
    cf_df = common.ampgen_df("cf", "k_plus")
    dcs_df = common.ampgen_df("dcs", "k_plus")

    # Introduce mixing
    params = mixing.MixingParams(
        d_mass=pdg_params.d_mass(),
        d_width=pdg_params.d_width(),
        mixing_x=5 * pdg_params.mixing_x(),
        mixing_y=5 * pdg_params.mixing_y(),
    )
    dcs_k3pi = efficiency_util.k_3pi(dcs_df)
    dcs_lifetimes = dcs_df["time"]
    q_p = [1 / np.sqrt(2) for _ in range(2)]

    mixing_weights = mixing.ws_mixing_weights(dcs_k3pi, dcs_lifetimes, params, +1, q_p)

    _weight_hist(mixing_weights)
    _hists(cf_df, dcs_df, mixing_weights)

    _time_plot(params, cf_df, dcs_df, mixing_weights)


if __name__ == "__main__":
    main()