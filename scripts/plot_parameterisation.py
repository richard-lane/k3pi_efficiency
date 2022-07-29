"""
Make plots of the variables that will be used for the reweighting

This script doesn't require a reweighter to exist - it's intended to be used to check that you've created
the dataframes correctly.

"""
import sys
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
from fourbody.param import helicity_param

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi_signal_cuts"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from lib_cuts.read_data import momentum_order
from lib_efficiency import efficiency_util, plotting


def main(year: str, sign: str, magnetisation: str):
    """
    Create a plot

    """
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

    plotting.projections(mc, ag)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make plots of ampgen + MC phase space variables, but don't do the reweighting."
    )
    parser.add_argument("sign", type=str, choices={"RS", "WS"})
    parser.add_argument("year", type=str, choices={"2018"})
    parser.add_argument("magnetisation", type=str, choices={"magdown"})

    args = parser.parse_args()
    main(args.year, args.sign, args.magnetisation)
