"""
forward.py -- MAST-U free-boundary forward model with RICH (high-DOF) current profiles.

The lesson from experiment 02: a 2-parameter interior (alpha_m, alpha_n) is fully
observable by magnetics, so it shows no degeneracy. The genuine ill-posedness of
equilibrium reconstruction lives in a richly parameterized current profile, where the
magnetics constrain only a few low-order moments and leave the rest unmeasured.

This module provides:
  - a fixed MAST-U double-null configuration (boundary + grid + synthetic sensors),
  - a reference profile (alpha-family) used only to anchor the amplitude/normalisation,
  - RICH profiles p'(psi_n), FF'(psi_n) = reference * (1 + sum_k theta_k * Legendre_k),
  - the forward map theta -> equilibrium -> synthetic magnetic observables.

Everything downstream (degeneracy analysis, SBI training data, MCMC baseline) is built
on top of `forward_observables`.
"""
import numpy as np
import freegs
import freegs.machine as machine
from freegs.jtor import ConstrainPaxisIp, ProfilesPprimeFfprime

MU0 = 4.0e-7 * np.pi

# --- Fixed experimental configuration (shared by every solve) --------------
IP_TARGET = 6.0e5      # A
PAXIS = 8.4e3          # Pa  (sets reference amplitude)
FVAC = 0.40            # T m
ALPHA_M, ALPHA_N = 1.2, 2.0
RAXIS = 0.9            # m   (R used in the p'/ff' split)
GRID = dict(Rmin=0.06, Rmax=2.0, Zmin=-2.2, Zmax=2.2, nx=65, ny=65)

XPOINTS = [(0.60, -1.10), (0.60, 1.10)]
ISOFLUX = [
    (0.60, -1.10, 1.35, 0.0),
    (0.60,  1.10, 1.35, 0.0),
    (0.60, -1.10, 0.30, 0.0),
    (0.60,  1.10, 0.30, 0.0),
    (1.35,  0.00, 0.30, 0.0),
]

# Synthetic poloidal magnetic-probe array (fixed positions).
def build_sensor_array():
    zline = np.linspace(-1.2, 1.2, 9)
    outboard = [(1.55, z) for z in zline]
    inboard = [(0.22, z) for z in zline]
    rline = np.linspace(0.4, 1.3, 5)
    top = [(r, 1.45) for r in rline]
    bottom = [(r, -1.45) for r in rline]
    return np.array(outboard + inboard + top + bottom)

SENSORS = build_sensor_array()
SIGMA_PSI = 1.0e-3     # Wb/rad
SIGMA_B = 2.0e-3       # T
# Noise vector for observable = [psi(N), Br(N), Bz(N)]  (coil currents excluded:
# that makes the interior look MORE observable, so any null space we find is conservative).
SIGMA = np.concatenate([
    np.full(len(SENSORS), SIGMA_PSI),
    np.full(len(SENSORS), SIGMA_B),
    np.full(len(SENSORS), SIGMA_B),
])


def make_eq():
    return freegs.Equilibrium(tokamak=machine.MASTU_simple(), **GRID)


def make_constrain():
    return freegs.control.constrain(xpoints=XPOINTS, isoflux=ISOFLUX)


# --- Legendre shape basis on psi_norm in [0,1] -----------------------------
def legendre_basis(pn, K):
    """Return array (K, ...) of shifted Legendre polynomials P_1..P_K on [0,1] (GLOBAL smooth)."""
    x = 2.0 * np.clip(pn, 0.0, 1.0) - 1.0
    P0 = np.ones_like(x)
    P1 = x
    Ps = [P1]
    Pm1, Pm = P0, P1
    for n in range(1, K):
        Pn = ((2 * n + 1) * x * Pm - n * Pm1) / (n + 1)
        Ps.append(Pn)
        Pm1, Pm = Pm, Pn
    return np.array(Ps[:K])


def local_basis(pn, K):
    """Return array (K, ...) of LOCAL hat (triangular) bumps on [0,1].

    Each bump is centred at node t_k=(k+0.5)/K with half-width = node spacing, so a
    coefficient perturbs the profile only near one flux surface -- a free-form / local
    current representation (PLATO-style independent current elements), the opposite of the
    global-smooth Legendre basis.
    """
    x = np.clip(pn, 0.0, 1.0)
    h = 1.0 / K
    nodes = (np.arange(K) + 0.5) * h
    return np.array([np.maximum(0.0, 1.0 - np.abs(x - t) / h) for t in nodes])


def _basis(pn, K, kind):
    return local_basis(pn, K) if kind == "local" else legendre_basis(pn, K)


def reference_state():
    """Solve the alpha-family reference ONCE (with shape control to find good coils).

    Returns a dict with the reference profile callables AND the coil currents. Downstream
    forward solves FIX these coil currents (coils are known inputs in reconstruction) and
    let the boundary float -- the physically correct forward map theta -> magnetics. The
    earlier boundary-constrained map was non-smooth because re-solving the (ill-conditioned)
    coil least-squares for every theta injected coil-current jumps into the observables.
    """
    eq = make_eq()
    prof = ConstrainPaxisIp(eq, PAXIS, IP_TARGET, FVAC,
                            alpha_m=ALPHA_M, alpha_n=ALPHA_N, Raxis=RAXIS)
    freegs.solve(eq, prof, make_constrain(), show=False)
    L, Beta0 = prof.L, prof.Beta0
    coils = {lab: c.current for lab, c in eq.tokamak.coils}

    def ref_pp(pn):
        shape = (1.0 - np.clip(pn, 0, 1) ** ALPHA_M) ** ALPHA_N
        return L * Beta0 / RAXIS * shape

    def ref_ff(pn):
        shape = (1.0 - np.clip(pn, 0, 1) ** ALPHA_M) ** ALPHA_N
        return MU0 * L * (1 - Beta0) * RAXIS * shape

    return dict(ref_pp=ref_pp, ref_ff=ref_ff, coils=coils, eq=eq)


def reference_callables():
    """Back-compat shim. Prefer reference_state()."""
    s = reference_state()
    return s["ref_pp"], s["ref_ff"], s["eq"]


def rich_profiles(theta, ref_pp, ref_ff, K, basis="legendre"):
    """Profiles with shape perturbations: p' = ref_pp*(1+sum a_k phi_k), ff' likewise.

    theta = [a_1..a_K, b_1..b_K] (2K params). theta=0 recovers the reference.
    basis = "legendre" (global smooth) or "local" (free-form hat bumps).
    """
    a = np.asarray(theta[:K], float)
    b = np.asarray(theta[K:2 * K], float)

    def pp(pn):
        phi = _basis(pn, K, basis)
        return ref_pp(pn) * (1.0 + np.tensordot(a, phi, axes=([0], [0])))

    def ff(pn):
        phi = _basis(pn, K, basis)
        return ref_ff(pn) * (1.0 + np.tensordot(b, phi, axes=([0], [0])))

    return ProfilesPprimeFfprime(pp, ff, FVAC)


def synthetic_magnetics(eq):
    """Observable vector m = [psi, Br, Bz] at the fixed sensor array."""
    psi = np.array([float(eq.psiRZ(R, Z)) for (R, Z) in SENSORS])
    Br = np.array([float(eq.Br(R, Z)) for (R, Z) in SENSORS])
    Bz = np.array([float(eq.Bz(R, Z)) for (R, Z) in SENSORS])
    return np.concatenate([psi, Br, Bz])


def forward_observables(theta, ref, K, return_eq=False, basis="legendre"):
    """Full forward map: profile coefficients theta -> magnetic observables.

    `ref` is the dict from reference_state(). Coil currents are FIXED to the reference
    (known inputs); the boundary floats. This is smooth in theta, unlike the
    boundary-constrained map. Returns (m, info) with info = dict(Ip, li, R0).
    """
    eq = make_eq()
    for lab, c in eq.tokamak.coils:
        c.current = ref["coils"][lab]
    profiles = rich_profiles(theta, ref["ref_pp"], ref["ref_ff"], K, basis=basis)
    # Fixed-coil free-boundary Picard solve. Free-boundary solves can be stiff under
    # profile perturbations, so escalate damping (blend) on failure before giving up.
    last_err = None
    for blend in (0.0, 0.5, 0.8):
        try:
            eq = make_eq()
            for lab, c in eq.tokamak.coils:
                c.current = ref["coils"][lab]
            freegs.solve(eq, profiles, None, show=False, blend=blend, maxits=100)
            last_err = None
            break
        except RuntimeError as e:
            last_err = e
    if last_err is not None:
        raise last_err
    m = synthetic_magnetics(eq)
    info = dict(
        Ip=float(eq.plasmaCurrent()),
        li=float(eq.internalInductance()),
        R0=float(eq.magneticAxis()[0]),
    )
    if return_eq:
        return m, info, eq
    return m, info
