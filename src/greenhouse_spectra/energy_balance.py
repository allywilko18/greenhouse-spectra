import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import brentq
from scipy.constants import sigma

from .planck import planck_nu, planck_lambda
from .constants import T_SUN, R_SUN, AU

DILUTION = (R_SUN / AU) ** 2


def solar_irradiance_nu(nu, T_sun=T_SUN):
    """Top-of-atmosphere solar irradiance, W m^-2 Hz^-1."""
    return np.pi * planck_nu(nu, T_sun) * DILUTION


def solar_irradiance_lambda(lam, T_sun=T_SUN):
    """Top-of-atmosphere solar irradiance, W m^-2 m^-1."""
    return np.pi * planck_lambda(lam, T_sun) * DILUTION


def earth_emission_nu(nu, T):
    """Flux emitted by a blackbody Earth, W m^-2 Hz^-1."""
    return np.pi * planck_nu(nu, T)


def earth_emission_lambda(lam, T):
    """Flux emitted by a blackbody Earth, W m^-2 m^-1."""
    return np.pi * planck_lambda(lam, T)


def integrate(y, x):
    """Area under a curve; works whether x is increasing or decreasing."""
    return abs(trapezoid(y, x))


def absorbed_flux(S0, albedo):
    """Globally averaged absorbed solar flux, W m^-2."""
    return (1 - albedo) * S0 / 4


def balance_temperature(S0, albedo, outgoing=None, bracket=(150.0, 400.0)):
    """Temperature where absorbed solar flux equals outgoing(T)."""
    if outgoing is None:
        outgoing = lambda T: sigma * T**4
    target = absorbed_flux(S0, albedo)
    return brentq(lambda T: outgoing(T) - target, *bracket)