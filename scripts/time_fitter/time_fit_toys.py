"""
Generate toys from the time fitter PDF

"""
import sys
import pathlib
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from lib_efficiency import time_fitter


def _gen(
    rng: np.random.Generator,
    pdf: Callable[[np.ndarray], np.ndarray],
    n_gen: int,
    plot=False,
) -> np.ndarray:
    """
    Generate samples from a pdf

    """
    if not n_gen:
        return np.array([])

    max_x = 10
    x = max_x * rng.random(size=n_gen)

    a, b = 0.1, 0.5
    y = a * np.exp(-b * x) * rng.random(n_gen)

    f_eval = pdf(x)

    # Explicitly check that all the generating PDF is strictly
    # greater than the function
    pts = np.linspace(0, max_x, 1000)
    assert np.all(a * np.exp(-b * pts) > pdf(pts))

    keep = y < f_eval

    if plot:
        _, ax = plt.subplots()

        pts = np.linspace(0, 10, 1000)
        ax.plot(pts, pdf(pts))
        ax.scatter(x[keep], y[keep], c="k", marker=".")
        ax.scatter(x[~keep], y[~keep], c="r", alpha=0.4, marker=".")
        plt.show()

    return x[keep]


def main():
    """
    TODO for now, just generate 1 thing and plot it

    Then do a fit to it

    Then do a pull study

    """
    n_gen = 1000000
    rng = np.random.default_rng()

    args = (0.5, 1, 1, 1, 1, 1)
    points = _gen(rng, lambda x: time_fitter.pdf(x, *args), n_gen, plot=False)

    m = time_fitter.fit(points)

    print(m)

    def fitted_pdf(x: np.ndarray) -> np.ndarray:
        return time_fitter.normalised_pdf(x, *m.values)[1]

    pts = np.linspace(0, 12, 500)
    plt.plot(pts, fitted_pdf(pts), "r--")
    plt.hist(points, bins=250, density=True)
    plt.show()


if __name__ == "__main__":
    main()
