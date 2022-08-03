"""
Plot the MC decay times and perform a fit to them

"""
import sys
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from lib_efficiency import time_fitter, efficiency_util


def main(year: str, sign: str, magnetisation: str):
    """
    Create a plot

    """
    times = efficiency_util.mc_dump(year, sign, magnetisation)["time"] / 0.41

    fitter = time_fitter.fit(times, (0.21, 1.0, 2.0, 1.0, 2.0, 1.0))

    def fitted_pdf(x: np.ndarray) -> np.ndarray:
        return time_fitter.normalised_pdf(x, *fitter.values)[1]

    pts = np.linspace(0, np.max(times), 250)

    fig, ax = plt.subplot_mosaic("AAA\nAAA\nAAA\nBBB")

    counts, bins, _ = ax["A"].hist(times, bins=100, histtype="step", color="r")
    centres = (bins[1:] + bins[:-1]) / 2.0

    expected_count = fitted_pdf(centres) * len(times) / np.sum(fitted_pdf(centres))
    ax["A"].plot(centres, expected_count, "k--")

    diff = counts - expected_count
    err = np.sqrt(counts)

    ax["B"].errorbar(centres, diff, yerr=err, fmt="k.")
    ax["B"].plot(centres, [0 for _ in centres], "r--")

    fig.suptitle(f"MC {sign}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make plots of ampgen + MC phase space variables, but don't do the reweighting."
    )
    parser.add_argument("sign", type=str, choices={"RS", "WS"})
    parser.add_argument("year", type=str, choices={"2018"})
    parser.add_argument("magnetisation", type=str, choices={"magdown"})

    args = parser.parse_args()
    main(args.year, args.sign, args.magnetisation)
