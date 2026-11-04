from scipy.constants import h, c, k, sigma, g, atomic_mass

# Sun / orbit
T_SUN = 5772.0              # K
R_SUN = 6.957e8             # m
AU    = 1.496e11            # m

# Earth / atmosphere
ALBEDO    = 0.30
P_SURFACE = 101325.0        # Pa
M_AIR     = 28.97 * atomic_mass   # kg, mean mass of an air molecule
GAMMA_LR  = 0.00649         # K/m, lapse rate
ETA       = 0.75            # Ref. [4] parameter for T_trop
X_CO2     = 420e-6          # CO2 mixing ratio