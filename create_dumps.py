"""
Read momenta and decay times into an array for LHCb monte carlo (signal MC)
and AmpGen

For now, uses the same straight cuts to isolate the signal as the background
classifier - TODO find a better solution for this, or something

"""
import sys
import pickle
import pathlib
import argparse
from multiprocessing import Process
import pandas as pd
import numpy as np
import uproot

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "k3pi_signal_cuts"))

from lib_cuts import read_data, definitions, cuts
from lib_efficiency import efficiency_definitions


def _ampgen_df(tree, sign: str) -> pd.DataFrame:
    """
    Populate a pandas dataframe with momenta and time arrays from the provided tree

    """
    df = pd.DataFrame()

    t_branch = "Dbar0_decayTime" if sign == "RS" else "D0_decayTime"
    df["time"] = tree[t_branch].array() * 1000  # Convert to ps

    suffixes = "Px", "Py", "Pz", "E"

    branches = [
        *(f"_1_K~_{s}" for s in suffixes),
        *(f"_2_pi#_{s}" for s in suffixes),
        *(f"_3_pi#_{s}" for s in suffixes),
        *(f"_4_pi~_{s}" for s in suffixes),
    ]

    for branch in branches:
        df[branch] = tree[branch].array() * 1000  # Convert to MeV

    return df


def _flip_momenta(d_ids: np.ndarray, sign: str, df: pd.DataFrame, mc_branches: list):
    """
    Flip 3 momenta of the right D meson

    We want both types of decay to be to K+ (I think, TODO) so we need to flip RS D0s and WS Dbar0s

    Modifies the dataframe in place

    """
    # TODO make this function much nicer
    # Flip 3 momenta of D0 mesons
    # Do this by creating an array of 1s and -1s and multiplying the 3-momenta by these
    # -1 for D0 meson, 1 for Dbar0
    correct_id = -421 if sign == "RS" else 421

    flip_momenta = np.ones(len(df))
    flip_momenta[d_ids != correct_id] = -1

    for branch in (
        *mc_branches[0:3],
        *mc_branches[4:7],
        *mc_branches[8:11],
        *mc_branches[12:15],
    ):
        df[branch] *= flip_momenta


def _mc_df(tree, sign: str) -> pd.DataFrame:
    """
    Populate a pandas dataframe with momenta and time arrays from the provided tree

    """
    df = pd.DataFrame()
    keep = cuts.keep(tree)

    df["time"] = read_data.decay_times(tree)[keep]
    suffixes = "P_X", "P_Y", "P_Z", "P_E"

    # Use truth value momenta, instead of refit TODO see if that's right
    mc_branches = [
        *(f"D0_P0_TRUE{s}" for s in suffixes),
        *(f"D0_P1_TRUE{s}" for s in suffixes),
        *(f"D0_P2_TRUE{s}" for s in suffixes),
        *(f"D0_P3_TRUE{s}" for s in suffixes),
    ]

    for branch in mc_branches:
        # Take the first value from each entry in the jagged array - this is the best fit value
        df[branch] = tree[branch].array()[keep]

    _flip_momenta(tree["D0_TRUEID"].array()[keep], sign, df, mc_branches)

    return df


def main(sign: str, year: str = None, magnetisation: str = None) -> None:
    """
    Create pickle dump of the momenta and decay times for D -> K3pi events.
    AmpGen evts are all K+3pi, I think

    :param sign: "RS" or "WS". Tells us which tree to read from the MC file,
                 or tells us which AmpGen file to use.
    :param year: data taking year - only needs to be specified for MC

    """
    # Need to either specify both year + mag for MC, or dont need either for AmpGen
    assert (year is None and magnetisation is None) or (
        year is not None and magnetisation is not None
    )

    # Make it more readable - we're dealing with MC if the year has been specified
    is_mc = year is not None

    files = (
        definitions.mc_files(year, "magdown", sign)
        if is_mc
        else definitions.ampgen_files(sign)
    )

    # TODO maybe different seed
    gen = np.random.default_rng(seed=0)

    df_fcn = _mc_df if year else _ampgen_df
    tree_name = definitions.tree_name(sign) if is_mc else "DalitzEventList"

    dfs = []
    for root_file in files:
        with uproot.open(root_file) as f:
            df = df_fcn(f[tree_name], sign)
            df["train"] = gen.random(len(df)) < 0.5  # Half for test half for train

            dfs.append(df)

    # Dump it
    path = (
        efficiency_definitions.mc_dump_path(year, sign, magnetisation)
        if is_mc
        else efficiency_definitions.ampgen_dump_path(sign)
    )
    with open(path, "wb") as f:
        print(f"dumping {path}")
        pickle.dump(pd.concat(dfs), f)


def _parse_args() -> argparse.Namespace:
    """
    Read arguments from CLI, return the namespace

    """
    parser = argparse.ArgumentParser(
        description="Create DataFrame pickle dumps containing the right info for"
        "the efficiency reweighting. Reads ROOT files from the data "
        "directory, which might live somewhere else"
    )
    parser.add_argument("sign", type=str, choices={"RS", "WS"})
    parser.add_argument("year", type=str, choices={"2018"})
    parser.add_argument("magnetisation", type=str, choices={"magdown"})

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    procs = [
        Process(target=main, args=(args.sign, args.year, args.magnetisation)),  # MC
        Process(target=main, args=(args.sign,)),  # AmpGen
    ]

    for p in procs:
        p.start()
    for p in procs:
        p.join()
