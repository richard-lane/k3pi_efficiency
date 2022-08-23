"""
Make scatter plots of the numerically evaluated coherence factor

Test data

"""
import sys
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from lib_data import get, util
from lib_efficiency import (
    efficiency_util,
    plotting,
    efficiency_model,
)


def main(
    year: str,
    sign: str,
    magnetisation: str,
    data_sign: str,
    weighter_sign: str,
    fit: bool,
):
    """
    Create a plot

    """
    ampgen_df = get.ampgen(sign)
    pgun_df = get.particle_gun(sign, show_progress=True)

    # We only want test data here
    pgun_df = pgun_df[~pgun_df["train"]]
    ampgen_df = ampgen_df[~ampgen_df["train"]]

    # Deal with getting rid of evts/flipping momenta if we need to
    pgun_df = efficiency_util.k_sign_cut(pgun_df, data_sign)
    if data_sign == "k_minus":
        ampgen_df = util.flip_momenta(
            ampgen_df, np.ones(len(ampgen_df), dtype=np.bool_)
        )

    # Just pass the arrays into the efficiency function and it should find the right weights
    ag_k, ag_pi1, ag_pi2, ag_pi3 = efficiency_util.k_3pi(ampgen_df)
    mc_k, mc_pi1, mc_pi2, mc_pi3 = (
        a.astype(np.float64) for a in efficiency_util.k_3pi(pgun_df)
    )

    mc_t = pgun_df["time"]

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

    plotting.z_scatter(
        ag_k, ag_pi1, ag_pi2, ag_pi3, mc_k, mc_pi1, mc_pi2, mc_pi3, weights, 8
    )

    fit_suffix = "_fit" if fit else ""
    plt.savefig(
        f"z_{year}_{magnetisation}_{sign}_data_{data_sign}_weighter_{weighter_sign}{fit_suffix}.png"
    )

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Using the AmpGen models, plot the measured coherence factor before"
        "and after the reweighting. Splits the data into chunks for better visualisation."
    )
    parser.add_argument("year", type=str, choices={"2018"})
    parser.add_argument("sign", type=str, choices={"cf", "dcs"})
    parser.add_argument("magnetisation", type=str, choices={"magdown"})
    parser.add_argument(
        "data_sign",
        type=str,
        choices={"k_plus", "k_minus"},
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
    main(
        args.year,
        args.sign,
        args.magnetisation,
        args.data_sign,
        args.weighter_sign,
        args.fit,
    )
