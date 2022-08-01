"""
Fitter for Monte Carlo decay times

"""
from typing import Tuple
import numpy as np
from scipy.integrate import quad
from iminuit import Minuit
from iminuit.cost import ExtendedUnbinnedNLL


def _pdf_no_min_t(
    t: np.ndarray, t0: float, n: float, m: float, a: float, b: float, beta: float
):
    """
    PDF without accounting for the minimum time

    """
    delta_t = t - t0
    numerator = (delta_t ** (n + m)) * np.exp(-beta * t)

    denominator = (1 + (a * delta_t) ** n) * (1 + (b * delta_t) ** m)

    d_lifetime = 0.41
    return (numerator / denominator) * np.exp(-d_lifetime * t)


def pdf(
    t: np.ndarray, t0: float, n: float, m: float, a: float, b: float, beta: float
) -> np.ndarray:
    """
    PDF model for decay time distribution

    """
    retval = _pdf_no_min_t(t, t0, n, m, a, b, beta)

    # Any times below a threshhold get set to 0
    retval[t < t0] = 0

    return retval


def integral(t0: float, n: float, m: float, a: float, b: float, beta: float) -> float:
    """
    Integral of the PDF

    """
    return quad(_pdf_no_min_t, t0, np.inf, args=(t0, n, m, a, b, beta))[0]


def normalised_pdf(
    t: np.ndarray, t0: float, n: float, m: float, a: float, b: float, beta: float
) -> Tuple[int, np.ndarray]:
    """
    Normalised PDF model for decay time distribution

    Returns length of the t array and the pdf values

    """
    return len(t), pdf(t, t0, n, m, a, b, beta) / integral(t0, n, m, a, b, beta)


def fit(times: np.ndarray) -> Minuit:
    """
    Perform a fit to some decay times

    """
    nll = ExtendedUnbinnedNLL(times, normalised_pdf, verbose=1)

    m = Minuit(nll, t0=0.5, n=1.0, m=1.0, a=1.0, b=1.0, beta=1.0)

    m.limits["t0"] = (0.0, None)
    m.limits["beta"] = (0.0, None)
    m.limits["n"] = (0.0, None)
    m.limits["m"] = (0.0, None)
    m.limits["a"] = (0.0, None)
    m.limits["b"] = (0.0, None)

    m.migrad()

    return m
