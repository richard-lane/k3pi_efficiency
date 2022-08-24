r"""
Introduce mixing via simulation

Mass and flavour eigenstates are related via the relations:
    $ D_1^0 = pD^0 + q \overline{D}^0 $
    $ D_2^0 = pD^0 - q \overline{D}^0 $

where p and q are complex numbers satisfying $|q|^2 + |p|^2 = 1$.

The mass and width difference between these mass states gives D mixing

"""
from typing import Tuple
from collections import namedtuple
import numpy as np

MixingParams = namedtuple(
    "Params",
    ["d_mass", "d_width", "mixing_x", "mixing_y"],
)


def _propagate(times: np.ndarray, d_mass: float, d_width: float) -> np.ndarray:
    """
    Time propagation part of the mixing functions

    """
    return np.exp(-d_mass * 1j * times) * np.exp(-d_width * times / 2)


def _phase_arg(
    times: np.ndarray, d_width: float, mixing_x: complex, mixing_y: complex
) -> np.ndarray:
    """
    Part that goes inside the sin/cos

    """
    return d_width * times * (mixing_x - mixing_y * 1j) / 2


def _g_plus(times: np.ndarray, params: MixingParams) -> np.ndarray:
    """
    Time-dependent mixing function

    :param times: array of times to evaluate fcn at
    :param d_mass: average mass of D mass eigenstates
    :param d_width: average width of D mass eigenstates
    :param mixing_x: mixing parameter: mass difference / width
    :param mixing_y: mixing parameter: width difference / 2 * width

    :returns: array of complex

    """
    return _propagate(times, params.d_mass, params.d_width) * np.cos(
        _phase_arg(times, params.d_width, params.mixing_x, params.mixing_y)
    )


def _g_minus(times: np.ndarray, params: MixingParams) -> np.ndarray:
    """
    Time-dependent mixing function

    :param times: array of times to evaluate fcn at
    :param d_mass: average mass of D mass eigenstates
    :param d_width: average width of D mass eigenstates
    :param mixing_x: mixing parameter: mass difference / width
    :param mixing_y: mixing parameter: width difference / 2 * width

    :returns: array of complex

    """
    return (
        _propagate(times, params.d_mass, params.d_width)
        * 1j
        * np.sin(_phase_arg(times, params.d_width, params.mixing_x, params.mixing_y))
    )


def _good_p_q(p: complex, q: complex) -> bool:
    """
    Check mag of p+q is 1

    """
    mag = p * p.conjugate() + q * q.conjugate()
    return np.isclose(mag, 1.0)


def mixed_d0_coeffs(
    times: np.ndarray,
    p: complex,
    q: complex,
    params: MixingParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Coefficients for D0 and Dbar0 for a particle that was created as a D0, after mixing

    :param times: array of times to evaluate coefficents at
    :param p: complex number relating flavour and mass eigenstates
    :param q: complex number relating flavour and mass eigenstates
    :param d_mass: average mass of D mass eigenstates
    :param d_width: average width of D mass eigenstates
    :param mixing_x: mixing parameter: mass difference / width
    :param mixing_y: mixing parameter: width difference / 2 * width

    :returns: D0 coefficient at the times provided
    :returns: Dbar0 coefficient at the times provided

    """
    assert _good_p_q(p, q)
    return (
        _g_plus(times, params),
        q * _g_minus(times, params) / p,
    )


def mixed_dbar0_coeffs(
    times: np.ndarray,
    p: complex,
    q: complex,
    params: MixingParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Coefficients for D0 and Dbar0 for a particle that was created as a Dbar0, after mixing

    :param times: array of times to evaluate coefficeents at
    :param p: complex number relating flavour and mass eigenstates
    :param q: complex number relating flavour and mass eigenstates
    :param d_mass: average mass of D mass eigenstates
    :param d_width: average width of D mass eigenstates
    :param mixing_x: mixing parameter: mass difference / width
    :param mixing_y: mixing parameter: width difference / 2 * width

    :returns: D0 coefficient at the times provided
    :returns: Dbar0 coefficient at the times provided

    """
    assert _good_p_q(p, q)
    return p * _g_minus(times, params) / q, _g_plus(times, params)
