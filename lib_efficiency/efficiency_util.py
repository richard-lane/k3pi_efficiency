"""
Utility functions for efficiency stuff

"""
from typing import Tuple
import pickle
import numpy as np
import pandas as pd

from . import efficiency_definitions


def ampgen_dump(sign: str) -> pd.DataFrame:
    """
    Get the ampgen DataFrame

    """
    with open(efficiency_definitions.ampgen_dump_path(sign), "rb") as f:
        return pickle.load(f)


def mc_dump(year: str, sign: str, magnetisation: str) -> pd.DataFrame:
    """
    Get a MC dataframe

    """
    with open(
        efficiency_definitions.mc_dump_path(year, sign, magnetisation), "rb"
    ) as f:
        return pickle.load(f)


def k_3pi(
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
