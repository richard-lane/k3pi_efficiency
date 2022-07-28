"""
Make plots of the variables that will be used for the reweighting

This script doesn't require a reweighter to exist - it's intended to be used to check that you've created
the dataframes correctly.

"""
import sys
import pathlib
import pickle
from typing import Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fourbody.param import helicity_param

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_signal_cuts"))

from lib_cuts.read_data import momentum_order


def _k_3pi(
    dataframe: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the kaon and 3 pions as 4xN numpy arrays of (px, py, pz, E)

    """
    # TODO make this nicer by having consistently named column headings in the dataframes
    try:
        suffixes = "P_X", "P_Y", "P_Z", "P_E"
        particles = [
            (f"D0_P0_TRUE{s}" for s in suffixes),
            (f"D0_P1_TRUE{s}" for s in suffixes),
            (f"D0_P3_TRUE{s}" for s in suffixes),
            (f"D0_P2_TRUE{s}" for s in suffixes),
        ]
        return tuple(
            np.row_stack([dataframe[x] for x in labels]) for labels in particles
        )

    except KeyError:
        suffixes = "Px", "Py", "Pz", "E"
        particles = [
            (f"_1_K~_{s}" for s in suffixes),
            (f"_2_pi#_{s}" for s in suffixes),
            (f"_3_pi#_{s}" for s in suffixes),
            (f"_4_pi~_{s}" for s in suffixes),
        ]
        return tuple(
            np.row_stack([dataframe[x] for x in labels]) for labels in particles
        )


def main():
    """
    Create a plot

    """
    sign = "RS"
    with open("ampgen.pkl", "rb") as f:
        ampgen_df = pickle.load(f)
    with open(f"mc_{sign}.pkl", "rb") as f:
        mc_df = pickle.load(f)

    ag_k, ag_pi1, ag_pi2, ag_pi3 = _k_3pi(ampgen_df)
    mc_k, mc_pi1, mc_pi2, mc_pi3 = _k_3pi(mc_df)

    # TODO should momentum order when creating dataframes
    ag_pi1, ag_pi2 = momentum_order(ag_k, ag_pi1, ag_pi2)
    mc_pi1, mc_pi2 = momentum_order(mc_k, mc_pi1, mc_pi2)

    ag = np.column_stack(
        (helicity_param(ag_k, ag_pi1, ag_pi2, ag_pi3), ampgen_df["time"])
    )
    mc = np.column_stack((helicity_param(mc_k, mc_pi1, mc_pi2, mc_pi3), mc_df["time"]))

    fig, ax = plt.subplots(2, 3, figsize=(12, 8))

    hist_kw = {"density": True, "histtype": "step"}
    for axis, ag_x, mc_x in zip(ax.ravel(), ag.T, mc.T):
        _, bins, _ = axis.hist(ag_x, bins=100, label="AG", **hist_kw)
        axis.hist(mc_x, bins=bins, label="MC", **hist_kw)

    ax[0, 0].legend()

    plt.show()


if __name__ == "__main__":
    main()
