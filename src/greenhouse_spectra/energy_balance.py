import numpy as np
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
    """Earth's emitted flux, W m^-2 Hz^-1."""
    return np.pi * planck_nu(nu, T)

def earth_emission_lambda(lam, T):
    """Earth's emitted flux, W m^-2 m^-1."""
    return np.pi * planck_lambda(lam, T)