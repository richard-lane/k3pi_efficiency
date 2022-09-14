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

from .amplitude_models import amplitudes

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


def _lifetimes2invmev(lifetimes: np.ndarray) -> np.ndarray:
    """
    Convert from D0 lifetimes to inverse MeV

    """
    decay_times_ps = lifetimes * 0.4103
    return decay_times_ps * (10 ** 10) / 6.58


def _ws_weights(
    times: np.ndarray,
    d0_amplitudes: np.ndarray,
    dbar0_amplitudes: np.ndarray,
    params: MixingParams,
    q_p: Tuple,
) -> np.ndarray:
    """
    Weights from amplitudes + times

    Only works for WS D0->K+3pi decays

    :param times: decay times in lifetimes
    :param d0_amplitudes: DCS amplitudes
    :param dbar0_amplitudes: CF amplitudes
    :param params: mixing parameters
    :param q_p: values of q and p.
    :returns: weights

    """
    # Convert lifetimes to inverse MeV
    t_invmev = _lifetimes2invmev(times)

    g_plus = _g_plus(t_invmev, params)
    g_minus = _g_minus(t_invmev, params)

    q, p = q_p
    num = g_plus * d0_amplitudes + q * g_minus * dbar0_amplitudes / p
    num = np.abs(num) ** 2

    denom = np.exp(-times) * np.abs(d0_amplitudes) ** 2

    return num / denom


def ws_mixing_weights(
    k3pi: Tuple,
    t_lifetimes: np.ndarray,
    mixing_params: MixingParams,
    k_charge: int,
    q_p: Tuple = None,
) -> np.ndarray:
    """
    Weights to apply to WS events to add mixing as defined by the mixing parameters provided

    :param k3pi: tuple of (k, pi1, pi2, pi3) parameters as returned by efficiency_util.k3pi
    :param t: decay times in lifetimes
    :param mixing_params: mixing parameters to use.
                          The dimensionful parameters (D mass and width) should be in MeV.
    :param k_charge: +1 or -1;
    :param q_p: values of q and p. If not provided defaults to no CPV

    """
    if not q_p:
        q_p = 1 / np.sqrt(2), 1 / np.sqrt(2)

    # For now only deal with K+
    # TODO deal with K- as well
    assert k_charge == 1

    # Evaluate amplitudes
    cf_amplitudes = amplitudes.cf_amplitudes(*k3pi, k_charge) / amplitudes.CF_AVG
    dcs_amplitudes = amplitudes.dcs_amplitudes(*k3pi, k_charge) / amplitudes.DCS_AVG

    return _ws_weights(t_lifetimes, dcs_amplitudes, cf_amplitudes, mixing_params, q_p)
