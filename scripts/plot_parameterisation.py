"""
Make plots of the variables that will be used for the reweighting

This script doesn't require a reweighter to exist - it's intended to be used to check that you've created
the dataframes correctly.

"""
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from fourbody.param import helicity_param

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_signal_cuts"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from lib_cuts.read_data import momentum_order
from lib_efficiency import efficiency_util


def main():
    """
    Create a plot

    """
    sign = "RS"
    year, magnetisation = "2018", "magdown"

    ampgen_df = efficiency_util.ampgen_dump(sign)
    mc_df = efficiency_util.mc_dump(year, sign, magnetisation)

    ag_k, ag_pi1, ag_pi2, ag_pi3 = efficiency_util.k_3pi(ampgen_df)
    mc_k, mc_pi1, mc_pi2, mc_pi3 = efficiency_util.k_3pi(mc_df)

    ag_pi1, ag_pi2 = momentum_order(ag_k, ag_pi1, ag_pi2)
    mc_pi1, mc_pi2 = momentum_order(mc_k, mc_pi1, mc_pi2)

    ag = np.column_stack(
        (helicity_param(ag_k, ag_pi1, ag_pi2, ag_pi3), ampgen_df["time"])
    )
    mc = np.column_stack((helicity_param(mc_k, mc_pi1, mc_pi2, mc_pi3), mc_df["time"]))

    _, ax = plt.subplots(2, 3, figsize=(12, 8))

    hist_kw = {"density": True, "histtype": "step"}
    for axis, ag_x, mc_x in zip(ax.ravel(), ag.T, mc.T):
        _, bins, _ = axis.hist(ag_x, bins=100, label="AG", **hist_kw)
        axis.hist(mc_x, bins=bins, label="MC", **hist_kw)

    ax[0, 0].legend()

    plt.show()


if __name__ == "__main__":
    main()
