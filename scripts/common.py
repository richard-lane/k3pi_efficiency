"""
Common utilities for running scripts

"""
import sys
import pathlib
import argparse
import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "k3pi-data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from lib_data import get, util
from lib_efficiency import efficiency_util


def parser(description: str) -> argparse.ArgumentParser:
    """
    Create an argument parser for all the right arguments

    :param description: description of your script's functionality
    :returns: argument parser object with all arguments added

    """
    retval = argparse.ArgumentParser(description=description)

    retval.add_argument(
        "year",
        type=str,
        choices={"2018"},
        help="Data taking year, so we know which reweighter to open",
    )
    retval.add_argument(
        "decay_type",
        type=str,
        choices={"cf", "dcs", "false"},
        help="CF, DCS or false sign (WS amplitude but RS charges) data",
    )
    retval.add_argument(
        "weighter_type",
        type=str,
        choices={"cf", "dcs"},
        help="Whether to open the reweighter trained on DCS or CF data",
    )
    retval.add_argument(
        "magnetisation",
        type=str,
        choices={"magdown"},
        help="So we know which reweighter to open",
    )
    retval.add_argument(
        "data_k_charge",
        type=str,
        choices={"k_plus", "k_minus", "both"},
        help="whether to use D->K+ or K- 3pi data (or both)",
    )
    retval.add_argument(
        "weighter_k_charge",
        type=str,
        choices={"k_plus", "k_minus", "both"},
        help="whether to open the reweighter trained on D0->K+ or K- 3pi",
    )
    retval.add_argument(
        "--fit", action="store_true", help="So we know which reweighter to open"
    )

    return retval


def remove_arg(argparser: argparse.ArgumentParser, arg: str):
    """
    Remove an argument from an ArgumentParser instance

    """
    for action in argparser._actions:
        opts = action.option_strings
        if (opts and opts[0] == arg) or action.dest == arg:
            argparser._remove_action(action)
            break

    for action in argparser._action_groups:
        for group_action in action._group_actions:
            if group_action.dest == arg:
                action._group_actions.remove(group_action)
                return


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
