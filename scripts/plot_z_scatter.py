"""
Make scatter plots of the numerically evaluated coherence factor

Test data

"""
import sys
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[0]))

import common
from lib_efficiency import (
    efficiency_util,
    plotting,
    efficiency_model,
)


def main(args: argparse.Namespace):
    """
    Create a plot

    """
    pgun_df = common.pgun_df(args.data_sign, args.data_k_sign)
    ag_sign = "dcs" if args.data_sign == "false" else args.data_sign
    ampgen_df = common.ampgen_df(ag_sign, args.data_k_sign)

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
        args.weighter_k_sign,
        args.year,
        args.weighter_sign,
        args.magnetisation,
        args.fit,
        verbose=True,
    )

    plotting.z_scatter(
        ag_k, ag_pi1, ag_pi2, ag_pi3, mc_k, mc_pi1, mc_pi2, mc_pi3, weights, 8
    )

    fit_suffix = "_fit" if args.fit else ""
    plt.savefig(
        f"z_{args.year}_{args.magnetisation}_data_{args.data_sign}_{args.data_k_sign}"
        f"_weighter_{args.weighter_sign}_{args.weighter_k_sign}{fit_suffix}.png"
    )

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Using the AmpGen models, plot the measured coherence factor before"
        "and after the reweighting. Splits the data into chunks for better visualisation."
    )
    parser.add_argument("year", type=str, choices={"2018"})
    parser.add_argument(
        "data_sign",
        type=str,
        choices={"cf", "dcs", "false"},
        help="CF, DCS or false sign (WS amplitude but RS charges) data",
    )
    parser.add_argument(
        "weighter_sign",
        type=str,
        choices={"cf", "dcs"},
        help="CF or DCS reweighter",
    )
    parser.add_argument("magnetisation", type=str, choices={"magdown"})
    parser.add_argument(
        "data_k_sign",
        type=str,
        choices={"k_plus", "k_minus", "both"},
        help="whether to read K+ or K- data",
    )
    parser.add_argument(
        "weighter_k_sign",
        type=str,
        choices={"k_plus", "k_minus"},
        help="whether to open the reweighter trained on K+ or K-",
    )
    parser.add_argument("--fit", action="store_true")

    main(parser.parse_args())
