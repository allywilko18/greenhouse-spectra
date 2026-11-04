import numpy as np
from scipy.constants import k, c, g, sigma as SB

from .planck import planck_nu
from .energy_balance import integrate
from .constants import P_SURFACE, M_AIR, GAMMA_LR, ETA


def scale_height(Ts):
    """z0 = k_B Ts / (M_air g), m."""
    return k * Ts / (M_AIR * g)


def tropopause_temperature(Ts):
    """T_trop = Ts + Gamma_LR z0 ln(1 - eta)."""
    return Ts + GAMMA_LR * scale_height(Ts) * np.log(1 - ETA)


def column_density(x_gas):
    """N0 z0 = x p0 / (M_air g), in cm^-2."""
    return x_gas * P_SURFACE / (M_AIR * g) * 1e-4


def optical_depth(sig_cm2, x_gas):
    """OD_s = N0 sigma z0 (dimensionless)."""
    return column_density(x_gas) * sig_cm2


def wn_to_hz(wn):
    """Wavenumber (cm^-1) to frequency (Hz)."""
    return 100 * c * wn


def outgoing_radiance(wn, Ts, sig_cm2, x_gas):
    """I+ at top of atmosphere, eqn (15.6). W m^-2 sr^-1 Hz^-1."""
    nu = wn_to_hz(wn)
    trans = np.exp(-optical_depth(sig_cm2, x_gas))
    return (planck_nu(nu, Ts) * trans
            + planck_nu(nu, tropopause_temperature(Ts)) * (1 - trans))


def outgoing_flux(Ts, wn_band, sig_band, x_gas):
    """Total outgoing flux, W m^-2: sigma Ts^4 minus the CO2 notch."""
    nu = wn_to_hz(wn_band)
    Ttrop = tropopause_temperature(Ts)
    absorbed_frac = -np.expm1(-optical_depth(sig_band, x_gas))
    notch = np.pi * integrate((planck_nu(nu, Ts) - planck_nu(nu, Ttrop))
                              * absorbed_frac, nu)
    return SB * Ts**4 - notch