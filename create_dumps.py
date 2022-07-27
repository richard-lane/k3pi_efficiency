"""
Read momenta and decay times into an array for LHCb monte carlo (signal MC)
and AmpGen

For now, uses the same straight cuts to isolate the signal as the background
classifier - TODO find a better solution for this, or something

"""
import sys
import pickle
import pathlib
from multiprocessing import Process
from typing import List
import pandas as pd
import numpy as np
import uproot

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "k3pi_signal_cuts"))

from lib_cuts import read_data, definitions, cuts


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


def _mc_df(tree) -> pd.DataFrame:
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

    # Flip 3 momenta of Dbar mesons
    # Do this by creating an array of 1s and -1s and multiplying the 3-momenta by these
    # 1 for D meson, -1 for Dbar
    flip_momenta = np.ones(len(df))
    dbars = tree["D0_TRUEID"].array()[keep] == -421
    flip_momenta[dbars] = -1

    for branch in (
        *mc_branches[0:3],
        *mc_branches[4:7],
        *mc_branches[8:11],
        *mc_branches[12:15],
    ):
        df[branch] *= flip_momenta

    return df


def _create_dump(
    gen: np.random.Generator,
    files: List[str],
    sign: str = None,
    mc: bool = False,
    train_fraction: float = 0.5,
) -> None:
    """
    Create pickle dump of the momenta and decay times for D -> K3pi events.
    AmpGen evts are all K+3pi, I think

    :param gen: random number generator for train/test split
    :param sign: "RS" or "WS" - must be specified for MC. tells us which tree to read from the file
    :param files: iterable of filepaths to read from
    :param background: bool flag telling us whether this data is MC or AmpGen - we need to apply cuts to MC
    :param train_fraction: how much data to set aside for testing/training.

    """
    if sign is None and mc:
        raise ValueError("Must provide sign for reading MC")

    df_fcn = _mc_df if mc else lambda tree: _ampgen_df(tree, sign)
    tree_name = definitions.tree_name(sign) if mc else "DalitzEventList"

    dfs = []
    for root_file in files:
        with uproot.open(root_file) as f:
            # TODO get the right tree for ampgen
            df = df_fcn(f[tree_name])
            df["train"] = gen.random(len(df)) < train_fraction

            dfs.append(df)

    # Dump it
    path = (
        "mc.pkl" if mc else "ampgen.pkl"
    )  # TODO nicer. Could use the definitions fcn, extend it to take ampgen
    with open(path, "wb") as f:
        print(f"dumping {path}")
        pickle.dump(pd.concat(dfs), f)


def main(year: str, sign: str, mc: bool) -> None:
    """
    Create a pickle dump

    """
    files = (
        definitions.mc_files(year, "magdown") if mc else definitions.ampgen_files(sign)
    )
    gen = np.random.default_rng(seed=0)
    _create_dump(gen, files, sign, mc)


if __name__ == "__main__":
    procs = [Process(target=main, args=("2018", "RS", mc)) for mc in (True, False)]

    for p in procs:
        p.start()
    for p in procs:
        p.join()
