"""
Utility functions for efficiency stuff

"""
import sys
import pathlib
from typing import Tuple
import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))

from lib_data import definitions, util


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


def efficiency_df(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Transform a dataframe into what we need to do the efficiency reweighting
    Basically this just gets rid of all the unnecessary columns, and uses the
    "K ID" column to flip the 3-momenta of any D->K-3pi type candidates (since
    the AmpGen events were generated as K+3pi).

    :param dataframe: dataframe to transform
    :returns: copy of the dataframe with only the momentum and time columns

    """
    keep_columns = [*definitions.MOMENTUM_COLUMNS, "time", "K ID"]
    df_slice = dataframe[keep_columns]

    return util.flip_momenta(df_slice)


def k_sign_cut(dataframe: pd.DataFrame, k_sign: str):
    """
    Choose the right kaons - modifies the dataframe in place

    """
    assert k_sign in {"k_minus", "k_plus"}

    k_ids = dataframe["K ID"].to_numpy()
    drop = k_ids > 0 if k_sign == "k_minus" else k_ids < 0

    return dataframe[~drop]
