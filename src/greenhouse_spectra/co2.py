import numpy as np
import pandas as pd


def load_hitran(path):
    """Load HITRAN lines: nu (cm^-1), sw (cm/molecule), gamma_air (cm^-1/atm)."""
    df = pd.read_csv(path)
    if {'nu', 'sw', 'gamma_air'}.issubset(df.columns):
        df = df[['nu', 'sw', 'gamma_air']]
    else:
        df = df.iloc[:, :3]
        df.columns = ['nu', 'sw', 'gamma_air']
    return df.sort_values('nu').reset_index(drop=True)


def lorentzian_cross_section(wn, nu_i, S_i, gamma_i, chunk=200):
    """Sum of Lorentzians, eqn (15.4). wn in cm^-1, returns sigma in cm^2."""
    wn = np.asarray(wn, dtype=float)
    sig = np.zeros_like(wn)
    for j in range(0, len(nu_i), chunk):
        n = nu_i[j:j + chunk, None]
        S = S_i[j:j + chunk, None]
        g = gamma_i[j:j + chunk, None]
        sig += np.sum(S / np.pi * g / (g**2 + (wn[None, :] - n) ** 2), axis=0)
    return sig


def fit_effective_cross_section(wn, sig, nu0=667.5, bin_width=2.0,
                                stat='mean', exclude_core=3.0):
    """Fit sigma0*exp(-|wn-nu0|/delta) to binned sigma. Returns (sigma0, delta)."""
    edges = np.arange(wn.min(), wn.max() + bin_width, bin_width)
    idx = np.digitize(wn, edges)
    centres, vals = [], []
    for i in np.unique(idx):
        m = idx == i
        centres.append(wn[m].mean())
        vals.append(sig[m].max() if stat == 'max' else sig[m].mean())
    centres, vals = np.array(centres), np.array(vals)

    x = np.abs(centres - nu0)
    keep = (x > exclude_core) & (vals > 0)        # skip the Q-branch spike
    slope, intercept = np.polyfit(x[keep], np.log(vals[keep]), 1)
    return np.exp(intercept), -1.0 / slope


def effective_cross_section(wn, sigma0, delta, nu0=667.5):
    """Single effective cross-section (yellow line in Fig. 15.1), cm^2."""
    return sigma0 * np.exp(-np.abs(wn - nu0) / delta)