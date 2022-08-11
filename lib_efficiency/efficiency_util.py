"""
Utility functions for efficiency stuff

"""
import sys
import pathlib
from typing import Tuple
import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))

from lib_data import definitions


def k_3pi(
    dataframe: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the kaon and 3 pions as 4xN numpy arrays of (px, py, pz, E)

    """
    particles = [
        definitions.MOMENTUM_COLUMNS[0:4],
        definitions.MOMENTUM_COLUMNS[4:8],
        definitions.MOMENTUM_COLUMNS[8:12],
        definitions.MOMENTUM_COLUMNS[12:16],
    ]
    return tuple(np.row_stack([dataframe[x] for x in labels]) for labels in particles)
