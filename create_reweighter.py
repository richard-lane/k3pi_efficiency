"""
Create a reweighter

"""
import os
import sys
import pickle
import pathlib
import argparse
import numpy as np
from fourbody.param import helicity_param
from lib_efficiency import efficiency_definitions, efficiency_util
from lib_efficiency.reweighter import EfficiencyWeighter

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "k3pi-data"))

from lib_data.util import momentum_order
from lib_data import get


def main(year: str, sign: str, magnetisation: str, fit: bool):
    """
    Read the right data, use it to create a reweighter, pickle the reweighter

    """
    if not efficiency_definitions.REWEIGHTER_DIR.is_dir():
        os.mkdir(efficiency_definitions.REWEIGHTER_DIR)

    # Read the right stuff
    ag_df = get.ampgen(sign)
    pgun_df = get.particle_gun(sign, show_progress=True)

    # We only want to train on training data
    ag_df = ag_df[ag_df["train"]]
    pgun_df = pgun_df[pgun_df["train"]]

    # We want to swap the 3 momentum of any K- type particle gun candidate
    pgun_df = efficiency_util.efficiency_df(pgun_df)

    # Get the right arrays
    ag_k, ag_pi1, ag_pi2, ag_pi3 = efficiency_util.k_3pi(ag_df)
    mc_k, mc_pi1, mc_pi2, mc_pi3 = efficiency_util.k_3pi(pgun_df)

    # Momentum order
    ag_pi1, ag_pi2 = momentum_order(ag_k, ag_pi1, ag_pi2)
    mc_pi1, mc_pi2 = momentum_order(mc_k, mc_pi1, mc_pi2)

    # Parameterise points
    ag = np.column_stack((helicity_param(ag_k, ag_pi1, ag_pi2, ag_pi3), ag_df["time"]))
    mc = np.column_stack(
        (helicity_param(mc_k, mc_pi1, mc_pi2, mc_pi3), pgun_df["time"])
    )

    # Create reweighter
    reweighter = EfficiencyWeighter(ag, mc, fit, efficiency_definitions.MIN_TIME)

    # Dump it
    with open(
        efficiency_definitions.reweighter_path(year, sign, magnetisation, fit), "wb"
    ) as f:
        pickle.dump(reweighter, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create efficiency reweighters")
    parser.add_argument("sign", type=str, choices={"cf", "dcs"})
    parser.add_argument("year", type=str, choices={"2018"})
    parser.add_argument("magnetisation", type=str, choices={"magdown"})
    parser.add_argument("--fit", action="store_true")

    args = parser.parse_args()

    main(args.year, args.sign, args.magnetisation, args.fit)
