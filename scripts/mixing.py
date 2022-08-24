"""
Plot things that show mixing

"""
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from lib_efficiency import mixing


def _pdg_d_width():
    """ D width from the PDG in MeV^-1 """
    d_lifetime_s = 410.3  # x10^-15
    sec_to_inv_mev = 1 / 6.58  # x10^22

    return 1 / (d_lifetime_s * sec_to_inv_mev * 10 ** 7)


def _pdg_x():
    """ mixing x from pdg, dimensionless """
    return 410.3 * 0.997 * 10 ** -5


def _pdg_y():
    """ mixing y from pdg, dimensionless """
    return 1.23 * 0.01 / 2


def _plot(params: mixing.MixingParams, path: str, scale: bool = False) -> None:
    """
    Plot the probability of detecting a particle as a D0 or Dbar0 with time

    :param params: mixing parameters
    :path: where to save to
    :scale: whether to scale by dividing probabilities by exponential

    """
    d_lifetime_ps = 0.4103
    times_ps = np.linspace(0, 20 * d_lifetime_ps, 100)
    times_inv_mev = times_ps * (10 ** 10) / 6.58

    # Assume no CPV in mixing
    p_q = 1 / np.sqrt(2)

    d0_prob, dbar0_prob = (
        np.abs(amplitude) ** 2
        for amplitude in mixing.mixed_d0_coeffs(times_inv_mev, p_q, p_q, params)
    )

    if scale:
        d0_prob /= np.exp(-times_ps / d_lifetime_ps)
        dbar0_prob /= np.exp(-times_ps / d_lifetime_ps)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        times_ps,
        d0_prob,
        label=r"$\frac{p(D^0)}{e^{-t/\tau}}$" if scale else r"$p(D^0)$",
    )
    ax.plot(
        times_ps,
        dbar0_prob,
        label=r"$\frac{p(\overline{D}^0)}{e^{-t/\tau}}$"
        if scale
        else r"$p(\overline{D}^0)$",
    )

    fig.suptitle(f"x={params.mixing_x:.4f}, y={params.mixing_y:.4f}")

    ax.legend()
    ax.set_ylabel("probability")
    ax.set_xlabel("time /ps")

    fig.tight_layout()

    fig.savefig(path)
    plt.clf()


def main():
    """
    Find the D0 and Dbar0 coefficients for particles that begin life as a D

    """
    params = mixing.MixingParams(
        d_mass=1864.84, d_width=_pdg_d_width(), mixing_x=_pdg_x(), mixing_y=_pdg_y()
    )
    _plot(params, "mixing.png", scale=False)
    _plot(params, "scaled_mixing.png", scale=True)

    params = mixing.MixingParams(
        d_mass=1864.84,
        d_width=_pdg_d_width(),
        mixing_x=100 * _pdg_x(),
        mixing_y=_pdg_y(),
    )
    _plot(params, "more_mixing.png", scale=True)


if __name__ == "__main__":
    main()
