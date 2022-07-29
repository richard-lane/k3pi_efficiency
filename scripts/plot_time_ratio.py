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
from fourbody.param import helicity_param

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_signal_cuts"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from lib_cuts.read_data import momentum_order
from lib_efficiency import (
    efficiency_util,
    plotting,
    efficiency_model,
    efficiency_definitions,
)


def _times_and_weights(
    year: str, magnetisation: str, sign: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get testing times and weights

    """
    ampgen_df = efficiency_util.ampgen_dump(sign)
    mc_df = efficiency_util.mc_dump(year, sign, magnetisation)

    # We only want test data here
    ampgen_df = ampgen_df[~ampgen_df["train"]]
    mc_df = mc_df[~mc_df["train"]]

    # Just pass the arrays into the efficiency function and it should find the right weights
    ag_k, ag_pi1, ag_pi2, ag_pi3 = efficiency_util.k_3pi(ampgen_df)
    mc_k, mc_pi1, mc_pi2, mc_pi3 = efficiency_util.k_3pi(mc_df)

    ag_t, mc_t = ampgen_df["time"], mc_df["time"]

    weights = efficiency_model.weights(
        mc_k, mc_pi1, mc_pi2, mc_pi3, mc_t, year, sign, magnetisation, verbose=True
    )

    return ag_t, mc_t, weights


def _allowed_times(
    times: np.ndarray, min_t: float, max_t: float, wt: np.ndarray = None
):
    mask = (min_t < times) & (times < max_t)
    if wt is None:
        return times[mask]
    return times[mask], wt[mask]


def main(year, magnetisation):
    """
    Create a plot

    """
    bins = np.array(
        [
            0.3854,
            0.48585,
            0.574,
            0.6642,
            0.7585,
            0.8733,
            1.0045,
            1.1767,
            1.435,
            3.28,
            7.7899,
        ]
    )
    rs_ag_t, rs_mc_t, rs_wt = _times_and_weights(year, magnetisation, "RS")
    ws_ag_t, ws_mc_t, ws_wt = _times_and_weights(year, magnetisation, "WS")

    # Keep only times in the allowed range
    rs_ag_t = _allowed_times(rs_ag_t, bins[0], bins[-1])
    ws_ag_t = _allowed_times(ws_ag_t, bins[0], bins[-1])

    rs_mc_t, rs_wt = _allowed_times(rs_mc_t, bins[0], bins[-1], rs_wt)
    ws_mc_t, ws_wt = _allowed_times(ws_mc_t, bins[0], bins[-1], ws_wt)

    plotting.plot_ratios(rs_mc_t, ws_mc_t, rs_ag_t, ws_ag_t, rs_wt, ws_wt, bins)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make plots of ampgen + MC phase space variables, but don't do the reweighting."
    )
    parser.add_argument("year", type=str, choices={"2018"})
    parser.add_argument("magnetisation", type=str, choices={"magdown"})

    args = parser.parse_args()
    main(args.year, args.magnetisation)
