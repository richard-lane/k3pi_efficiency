"""
Make plots of the variables used in the reweighting, before and after the reweighting

Test data

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
from lib_efficiency import (
    efficiency_util,
    plotting,
    efficiency_model,
    efficiency_definitions,
)


def main(year, sign, magnetisation):
    """
    Create a plot

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

    # For plotting we will want to momentum order, maybe TODO
    # ag_pi1, ag_pi2 = momentum_order(ag_k, ag_pi1, ag_pi2)
    # mc_pi1, mc_pi2 = momentum_order(mc_k, mc_pi1, mc_pi2)

    ag = np.column_stack((helicity_param(ag_k, ag_pi1, ag_pi2, ag_pi3), ag_t))
    mc = np.column_stack((helicity_param(mc_k, mc_pi1, mc_pi2, mc_pi3), mc_t))

    ag = ag[ag[:, -1] > efficiency_definitions.MIN_TIME]

    plotting.projections(mc, ag, mc_wt=weights)

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
