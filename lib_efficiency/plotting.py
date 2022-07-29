"""
Functions for plotting things

"""
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt


def projections(
    mc: np.ndarray, ag: np.ndarray, mc_wt: np.ndarray = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot projections of 6d points

    :param mc: (Nx6) shape array of monte carlo (i.e. the original distribution) points
    :param ag: (Nx6) shape array of AmpGen (i.e. the target distribution) points
    :param mc_wt: weights to apply to the MC points
    :param mc_wt: weights to apply to the AmpGen points

    :returns: the figure and axes that we have plotted on

    """
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))

    hist_kw = {"density": True, "histtype": "step"}
    for axis, ag_x, mc_x in zip(ax.ravel(), ag.T, mc.T):
        _, bins, _ = axis.hist(ag_x, bins=100, label="AG", **hist_kw)
        axis.hist(mc_x, bins=bins, label="MC", **hist_kw, alpha=0.5)
        if mc_wt is not None:
            axis.hist(mc_x, bins=bins, label="Reweighted", **hist_kw, weights=mc_wt)

    ax[0, 0].legend()

    return fig, ax
