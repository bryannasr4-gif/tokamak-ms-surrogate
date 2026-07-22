"""
tier1_lib.py -- FAIR-MAST real-EFIT data layer for Tier-1 (real-data anchoring).

Reads original-MAST (campaigns M5-M9) EFIT reconstructions from the public FAIR-MAST S3 Zarr
(anon, https://s3.echo.stfc.ac.uk) via xarray, and exposes per-slice REAL shape descriptors +
the stored last-closed-flux-surface (LCFS) + real profiles, for:
  (a) shape-realism / coverage of the synthetic MAST-U training cloud,
  (b) surrogate-vs-solver emulator self-consistency on shape-anchored re-solves (see tier1_resolve.py),
  (c) same-code descriptor fidelity.

HONEST SCOPE (see TIER1_PREREG.md): this is ORIGINAL MAST (a different machine from MAST-U), EFIT is a
fit not a measurement, and any re-solved m_s is model-vs-model. NOT an experimental validation.

The `efm` group (opened with xarray) gives dims (time, profile_z=nZ, profile_r=nR) for psirz and clean
per-slice fields: lcfs_r/lcfs_z (stored LCFS), elongation, triang_upper/lower, li, betap, betan, q_95,
minor_radius, geom_axis_rc/zc, magnetic_axis_r/z, plasma_area, plasma_current_x (Ip, A),
bvac_r/bvac_val (fvac=|R*Btor|), pprime/ffprime (on psi_norm), xpoint1/2_rc/zc, final_chisq, cnvrgd_times.

BLAS threads must be pinned to 1 before importing numpy elsewhere; this module itself is solver-free.
"""
import os
import json
import numpy as np

ENDPOINT = "https://s3.echo.stfc.ac.uk"
BUCKET = "mast"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, "data", "tier1_efm_cache")

# The scalar per-slice fields we read from efm (real EFIT-reported quantities).
SCALAR_FIELDS = [
    "time", "plasma_current_x", "final_chisq", "cnvrgd_times",
    "elongation", "triang_upper", "triang_lower", "li", "betap", "betan",
    "q_95", "minor_radius", "geom_axis_rc", "geom_axis_zc",
    "magnetic_axis_r", "magnetic_axis_z", "plasma_area", "bvac_r", "bvac_val",
    "xpoint1_rc", "xpoint1_zc", "xpoint2_rc", "xpoint2_zc",
]
# Fields required for a shot to be "complete" enough for Tier-1.
REQUIRED = ["lcfs_r", "lcfs_z", "elongation", "li", "betap", "pprime", "ffprime", "plasma_current_x"]


def patch_freegs4e_profile_bug():
    """freegs4e 0.13.1 has a typo: GeneralPprimeFFprime.pressure()/fpol() call
    `super(GeneralPprimeFfprime, self)` (misspelled class -> NameError) whenever p_func/f_func is
    None (our case: we pass pprime_data/ffprime_data arrays). This breaks eq.poloidalBeta2() (and
    thus descriptors()) for real-profile equilibria. Restore the intended behaviour by rebinding the
    two methods to call the PARENT Profile integrator correctly. Idempotent; call before solving."""
    from freegs4e import jtor as _jtor
    cls = _jtor.GeneralPprimeFFprime
    if getattr(cls, "_tier1_patched", False):
        return
    Parent = cls.__mro__[1]  # Profile

    def pressure(self, pn):
        pn_ = np.clip(np.array(pn), 0, 1)
        if getattr(self, "p_func", None) is not None:
            return self.p_func(pn_)
        return Parent.pressure(self, pn_)

    def fpol(self, pn):
        pn_ = np.clip(np.array(pn), 0, 1)
        if getattr(self, "f_func", None) is not None:
            return self.f_func(pn_)
        return Parent.fpol(self, pn_)

    cls.pressure = pressure
    cls.fpol = fpol
    cls._tier1_patched = True


def _fs():
    import s3fs
    return s3fs.S3FileSystem(anon=True, client_kwargs={"endpoint_url": ENDPOINT})


def open_efm(shot):
    """Open the efm group of a shot as an xarray Dataset (lazy, over S3)."""
    import s3fs
    import xarray as xr
    store = s3fs.S3Map(root=f"{BUCKET}/level1/shots/{shot}.zarr/efm", s3=_fs(), check=False)
    return xr.open_zarr(store)


def has_complete_efm(shot):
    """True if the shot's efm has all REQUIRED fields (proper physics shot, not commissioning)."""
    try:
        ds = open_efm(shot)
    except Exception:
        return False
    return all(v in ds.variables for v in REQUIRED)


def read_shot(shot, force=False):
    """Read + cache the per-slice arrays needed for Tier-1. Returns a dict of numpy arrays.

    Cached as a compressed npz in data/tier1_efm_cache/. Scalars are (nt,); lcfs_r/lcfs_z are
    (nt, n_lcfs); pprime/ffprime are (nt, n_psin); psi_norm is (n_psin,); limiter is (n_lim,).
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cf = os.path.join(CACHE_DIR, f"efm_{shot}.npz")
    if os.path.exists(cf) and not force:
        z = np.load(cf, allow_pickle=False)
        return {k: z[k] for k in z.files}
    ds = open_efm(shot)
    out = {"shot": np.array([shot])}
    for k in SCALAR_FIELDS:
        if k in ds:
            out[k] = np.asarray(ds[k].values, dtype=float)
    for k in ("lcfs_r", "lcfs_z", "pprime", "ffprime", "limiterr", "limiterz"):
        if k in ds:
            out[k] = np.asarray(ds[k].values, dtype=float)
    if "psi_norm" in ds.coords:
        out["psi_norm"] = np.asarray(ds.coords["psi_norm"].values, dtype=float)
    np.savez_compressed(cf, **out)
    return out


# ------------------------------------------------------------------ real descriptors
def lcfs_slice(shot_data, ti):
    """Return finite (R, Z) points of the stored EFIT LCFS for slice ti."""
    lr = np.asarray(shot_data["lcfs_r"][ti], float)
    lz = np.asarray(shot_data["lcfs_z"][ti], float)
    m = np.isfinite(lr) & np.isfinite(lz)
    return lr[m], lz[m]


def geom_descriptors(lr, lz):
    """Our-convention bounding-box geometric descriptors from a raw LCFS polygon. Definitions
    match freegs4e.equilibrium (triangularity via R at Zmax/Zmin; kappa/aspect via bounding box),
    so real and re-solved descriptors are comparable when computed with THIS same function."""
    Rmax, Rmin, Zmax, Zmin = float(lr.max()), float(lr.min()), float(lz.max()), float(lz.min())
    a = (Rmax - Rmin) / 2.0
    Rgeo = (Rmax + Rmin) / 2.0
    kappa = (Zmax - Zmin) / (Rmax - Rmin)
    R_Zmax = float(lr[np.argmax(lz)])
    R_Zmin = float(lr[np.argmin(lz)])
    du = (Rgeo - R_Zmax) / a
    dl = (Rgeo - R_Zmin) / a
    return dict(kappa=kappa, delta=(du + dl) / 2.0, delta_upper=du, delta_lower=dl,
                Rgeo=Rgeo, a=a, aspect=Rgeo / a, Rmin_lcfs=Rmin, Rmax_lcfs=Rmax,
                Zmax=Zmax, Zmin=Zmin)


def real_descriptor_row(shot_data, ti):
    """Full real-descriptor row for slice ti: geometry recomputed from the stored LCFS with our
    code + EFIT-reported li/betap/q95 + the EFIT-reported scalars (for definitional cross-check)."""
    lr, lz = lcfs_slice(shot_data, ti)
    g = geom_descriptors(lr, lz)
    def get(name):
        v = shot_data.get(name)
        return float(v[ti]) if v is not None and ti < len(v) else float("nan")
    g.update(dict(
        li=get("li"), betap=get("betap"), betan=get("betan"), q95=get("q_95"),
        Rmag=get("magnetic_axis_r"), Zaxis=get("magnetic_axis_z"),
        Ip=abs(get("plasma_current_x")), chisq=get("final_chisq"), t=get("time"),
        # EFIT-reported (native) shape scalars for the (c) definitional cross-check:
        efit_kappa=get("elongation"),
        efit_delta=0.5 * (get("triang_upper") + get("triang_lower")),
        efit_a=get("minor_radius"), efit_Rgeo=get("geom_axis_rc"),
        n_lcfs=len(lr),
    ))
    return g


# ------------------------------------------------------------------ slice selection
def flat_top_mask(shot_data, ip_frac=0.7, dip_frac=0.05):
    """Boolean per-slice mask: flat-top (|Ip|>=ip_frac*max, |dIp/dt| small), finite chisq."""
    ip = np.asarray(shot_data["plasma_current_x"], float)
    chi = np.asarray(shot_data["final_chisq"], float)
    absip = np.abs(ip)
    mx = np.nanmax(absip) if np.isfinite(absip).any() else 0.0
    dip = np.abs(np.gradient(ip)) if len(ip) > 2 else np.zeros_like(ip)
    return np.isfinite(ip) & np.isfinite(chi) & (absip >= ip_frac * mx) & (dip <= dip_frac * mx)


def quality_slices(shot_data, li_range=(0.5, 1.6), kappa_range=(1.2, 3.0), betap_min=0.0):
    """Indices of quality flat-top slices passing the pre-registered per-slice gate."""
    n = len(shot_data["plasma_current_x"])
    ft = flat_top_mask(shot_data)
    li = shot_data.get("li"); bp = shot_data.get("betap"); kap = shot_data.get("elongation")
    lr = shot_data.get("lcfs_r")
    keep = []
    for ti in range(n):
        if not ft[ti]:
            continue
        if li is None or not (li_range[0] <= li[ti] <= li_range[1]):
            continue
        if kap is None or not (kappa_range[0] <= kap[ti] <= kappa_range[1]):
            continue
        if bp is None or not (bp[ti] > betap_min):
            continue
        lrr = np.asarray(lr[ti], float)
        if np.isfinite(lrr).sum() < 30:
            continue
        keep.append(ti)
    # within a shot, keep the single best-chisq slice per small kappa bin to reduce autocorrelation
    return keep


def farthest_point_select(X, k, seed_idx=0):
    """Greedy farthest-point sampling on standardized rows X (n,d) -> list of k row indices."""
    X = np.asarray(X, float)
    mu = np.nanmean(X, 0); sd = np.nanstd(X, 0); sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    n = len(Z)
    if k >= n:
        return list(range(n))
    chosen = [seed_idx]
    d = np.linalg.norm(Z - Z[seed_idx], axis=1)
    while len(chosen) < k:
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(Z - Z[i], axis=1))
    return chosen
