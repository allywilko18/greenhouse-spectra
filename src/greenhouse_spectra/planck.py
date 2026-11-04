import numpy as np
from scipy.constants import h, c, k


def planck_nu(nu, T):
    """Spectral radiance B_nu, W m^-2 sr^-1 Hz^-1. nu in Hz, T in K."""
    with np.errstate(over='ignore'):
        return 2 * h * nu**3 / c**2 / np.expm1(h * nu / (k * T))


def planck_lambda(lam, T):
    """Spectral radiance B_lambda, W m^-2 sr^-1 m^-1. lam in m, T in K."""
    with np.errstate(over='ignore'):
        return 2 * h * c**2 / lam**5 / np.expm1(h * c / (lam * k * T))