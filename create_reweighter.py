"""
Create a reweighter

"""
import sys
import pickle
import pathlib
import numpy as np
from fourbody.param import helicity_param
from lib_efficiency import efficiency_definitions, efficiency_util
from lib_efficiency.reweighter import Binned_Reweighter

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "k3pi_signal_cuts"))

from lib_cuts.read_data import momentum_order


def main(year: str, sign: str, magnetisation: str):
    """
    Read the right data, use it to create a reweighter, pickle the reweighter

    """
    # Read the right stuff
    ag_df = efficiency_util.ampgen_dump(sign)
    mc_df = efficiency_util.mc_dump(year, sign, magnetisation)

    # Get the right arrays
    ag_k, ag_pi1, ag_pi2, ag_pi3 = efficiency_util.k_3pi(ag_df)
    mc_k, mc_pi1, mc_pi2, mc_pi3 = efficiency_util.k_3pi(mc_df)

    # Momentum order
    ag_pi1, ag_pi2 = momentum_order(ag_k, ag_pi1, ag_pi2)
    mc_pi1, mc_pi2 = momentum_order(mc_k, mc_pi1, mc_pi2)

    # Parameterise points
    ag = np.column_stack((helicity_param(ag_k, ag_pi1, ag_pi2, ag_pi3), ag_df["time"]))
    mc = np.column_stack((helicity_param(mc_k, mc_pi1, mc_pi2, mc_pi3), mc_df["time"]))

    # Choose time bins
    time_bins = np.quantile(mc[:, -1], [0.2, 0.4, 0.6, 0.8])
    time_bins = np.concatenate(([0], time_bins, [10.0]))

    # Create reweighter
    reweighter = Binned_Reweighter(time_bins, mc, ag)

    # Plot the times in the bins used
    reweighter.hist()

    # Dump it
    with open(
        efficiency_definitions.reweighter_path(year, sign, magnetisation), "wb"
    ) as f:
        pickle.dump(reweighter, f)


if __name__ == "__main__":
    # TODO argparse to pass these in
    main("2018", "RS", "magdown")
