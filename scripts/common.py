"""
Common utilities for running scripts

"""
import sys
import pathlib
import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from lib_data import get, util
from lib_efficiency import efficiency_util


def ampgen_df(sign: str, k_sign: str) -> pd.DataFrame:
    """
    AmpGen dataframe - testing data

    """
    assert sign in {"dcs", "cf"}
    assert k_sign in {"k_plus", "k_minus", "both"}

    dataframe = get.ampgen("dcs" if sign == "false" else sign)
    dataframe = dataframe[~dataframe["train"]]

    if k_sign == "k_plus":
        # Don't flip any momenta
        return dataframe

    if k_sign == "k_minus":
        # Flip all the momenta
        mask = np.ones(len(dataframe), dtype=np.bool_)

    elif k_sign == "both":
        # Flip half of the momenta randomly
        mask = np.random.random(len(dataframe)) < 0.5

    dataframe = util.flip_momenta(dataframe, mask)
    return dataframe


def pgun_df(sign: str, k_sign: str) -> pd.DataFrame:
    """
    Particle gun dataframe - testing data, unless false sign
    (since we didnt train on it)

    """
    assert sign in {"cf", "dcs", "false"}

    if sign == "false":
        dataframe = get.false_sign(show_progress=True)

        # False sign is flipped - if we asked for K+ we want K- false sign
        dataframe = efficiency_util.k_sign_cut(
            dataframe, "k_minus" if k_sign == "k_plus" else "k_plus"
        )

    else:
        dataframe = get.particle_gun(sign, show_progress=True)

        # We only want test data here
        dataframe = dataframe[~dataframe["train"]]

        dataframe = efficiency_util.k_sign_cut(dataframe, k_sign)

    return dataframe
