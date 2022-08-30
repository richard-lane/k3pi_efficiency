"""
Make plots of the ratio of decay times before/after reweighting

Test data

"""
import sys
import pathlib
import argparse
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[0]))

import common
from lib_data import get, util

from lib_efficiency import (
    efficiency_util,
    plotting,
    efficiency_model,
    efficiency_definitions,
)


def _times_and_weights(
    year: str,
    magnetisation: str,
    sign: str,
    data_sign: str,
    weighter_sign: str,
    fit: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get testing times and weights

    """
    ampgen_df = common.ampgen_df(sign, data_sign)
    pgun_df = common.pgun_df(sign, data_sign)

    # Just pass the arrays into the efficiency function and it should find the right weights
    mc_k, mc_pi1, mc_pi2, mc_pi3 = efficiency_util.k_3pi(pgun_df)

    ag_t, mc_t = ampgen_df["time"], pgun_df["time"]

    # Open the reweighter according to weighter_sign
    weights = efficiency_model.weights(
        mc_k,
        mc_pi1,
        mc_pi2,
        mc_pi3,
        mc_t,
        weighter_sign,
        year,
        sign,
        magnetisation,
        fit,
        verbose=True,
    )

    return ag_t, mc_t, weights


def _allowed_times(
    times: np.ndarray, min_t: float, max_t: float, wt: np.ndarray = None
):
    mask = (min_t < times) & (times < max_t)
    if wt is None:
        return times[mask]
    return times[mask], wt[mask]


def main(year: str, magnetisation: str, data_sign: str, weighter_sign: str, fit: bool):
    """
    Create a plot

    """
    bins = np.array(
        [
            efficiency_definitions.MIN_TIME,
            1.0,
            1.2,
            1.35,
            1.5,
            1.7,
            2.0,
            2.4,
            2.8,
            6.0,
            10.0,
        ]
    )
    rs_ag_t, rs_mc_t, rs_wt = _times_and_weights(
        year, magnetisation, "cf", data_sign, weighter_sign, fit
    )
    ws_ag_t, ws_mc_t, ws_wt = _times_and_weights(
        year, magnetisation, "dcs", data_sign, weighter_sign, fit
    )

    # Keep only times in the allowed range
    rs_ag_t = _allowed_times(rs_ag_t, bins[0], bins[-1])
    ws_ag_t = _allowed_times(ws_ag_t, bins[0], bins[-1])

    rs_mc_t, rs_wt = _allowed_times(rs_mc_t, bins[0], bins[-1], rs_wt)
    ws_mc_t, ws_wt = _allowed_times(ws_mc_t, bins[0], bins[-1], ws_wt)

    plotting.plot_ratios(rs_mc_t, ws_mc_t, rs_ag_t, ws_ag_t, rs_wt, ws_wt, bins)

    fit_suffix = "_fit" if fit else ""
    plt.savefig(
        f"ratio_{year}_{magnetisation}_data_{data_sign}_weighter_{weighter_sign}{fit_suffix}.png"
    )

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make plots of ampgen + MC phase space variables, but don't do the reweighting."
    )
    parser.add_argument("year", type=str, choices={"2018"})
    parser.add_argument("magnetisation", type=str, choices={"magdown"})

    parser.add_argument(
        "data_sign",
        type=str,
        choices={"k_plus", "k_minus", "both"},
        help="whether to read K+ or K- data",
    )
    parser.add_argument(
        "weighter_sign",
        type=str,
        choices={"k_plus", "k_minus"},
        help="whether to open the reweighter trained on K+ or K-",
    )
    parser.add_argument("--fit", action="store_true")

    args = parser.parse_args()
    main(args.year, args.magnetisation, args.data_sign, args.weighter_sign, args.fit)
