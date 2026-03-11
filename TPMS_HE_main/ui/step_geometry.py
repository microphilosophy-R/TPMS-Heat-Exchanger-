"""ui.step_geometry -- Step 1: Geometry & Channels rendering and TPMS geometry estimator."""

import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from correlations.thermohydraulic_correlations import ThermoHydraulicCorrelations
from ui.components import _k
from ui.state import maybe_autosave


def _render_derived_channel_metrics(state, channel_name, channel_state):
    """Display read-only derived geometry metrics for one channel."""
    sk = channel_name
    geo = state["geometry"]
    ch_geo = channel_state.get("geometry", {}) or {}

    ch_L   = float(ch_geo.get("length") or geo.get("length", 0.94))
    ch_W   = float(ch_geo.get("width")  or geo.get("width",  0.25))
    ch_H   = float(ch_geo.get("height",   0.25))
    L_cell = float(ch_geo.get("unit_cell_size", 5e-3))
    struct = channel_state.get("structure", "Gyroid")

    if struct == "PlateFin":
        Hf = float(ch_geo.get("fin_height",    9.5e-3))
        sf = float(ch_geo.get("fin_spacing",   3.2e-3))
        tf = float(ch_geo.get("fin_thickness", 0.6e-3))
        eps = (sf - tf) / max(sf, 1e-12)
        Dh  = 2.0 * (Hf - tf) * (sf - tf) / max(Hf + sf - 2.0 * tf, 1e-12)
    elif struct == "SmoothPlateFin":
        eps = float(ch_geo.get("porosity", 0.65 if sk == "hot" else 0.70))
        Dh  = 4.0 * ch_W * ch_H * eps / max(2.0 * (ch_W + ch_H * eps), 1e-12)
    else:
        eps = float(ch_geo.get("porosity", 0.65 if sk == "hot" else 0.70))
        a   = L_cell / (2.0 * np.pi)
        Dh  = 4.0 * eps * a

    V_total = ch_L * ch_W * ch_H
    V_f     = eps * V_total
    V_s     = (1.0 - eps) * V_total
    SAD     = float(channel_state.get("surface_area_density", 0.0))
    A_HX    = SAD * V_total

    st.markdown("**Derived Channel Geometry**")
    cols = st.columns(6)
    cols[0].metric("α [1/m]",      f"{SAD:.1f}",  help="Surface area density — auto-computed by Geometry Estimator.")
    cols[1].metric("Dh [mm]",      f"{Dh * 1e3:.3f}")
    cols[2].metric("ε [-]",        f"{eps:.3f}",  help="Structural (void) porosity of the TPMS lattice.")
    cols[3].metric("A_HX [m²]",    f"{A_HX:.4f}")
    cols[4].metric("V_fluid [L]",  f"{V_f * 1e3:.4f}")
    cols[5].metric("V_solid [L]",  f"{V_s * 1e3:.4f}")


_TPMS_EQUATIONS = {
    "Gyroid": (
        r"f(x,y,z) = \sin x \cos y + \sin y \cos z + \sin z \cos x = 0",
        r"x = 2\pi X/L,\quad y = 2\pi Y/L,\quad z = 2\pi Z/L",
    ),
    "Diamond": (
        r"f = \sin x \sin y \sin z + \sin x \cos y \cos z"
        r" + \cos x \sin y \cos z + \cos x \cos y \sin z = 0",
        None,
    ),
    "Primitive": (
        r"f(x,y,z) = \cos x + \cos y + \cos z = 0",
        None,
    ),
    "Neovius": (
        r"f(x,y,z) = 3(\cos x + \cos y + \cos z) + 4\cos x\cos y\cos z = 0",
        None,
    ),
    "FRD": (
        r"f = 4(\cos x\cos y + \cos y\cos z + \cos z\cos x)"
        r" - (\cos 2x + \cos 2y + \cos 2z) = 0",
        None,
    ),
    "FKS": (
        r"f = 2(\cos x + \cos y + \cos z)"
        r" - \bigl(\cos(x{+}y)+\cos(x{-}y)+\cos(y{+}z)+\cos(y{-}z)+\cos(z{+}x)+\cos(z{-}x)\bigr) = 0",
        None,
    ),
}


def _render_tpms_equation(tpms_type: str):
    """Show the implicit surface equation for a TPMS type in a collapsible expander."""
    eq = _TPMS_EQUATIONS.get(tpms_type)
    if eq is None:
        return
    with st.expander(f"{tpms_type} implicit surface equation", expanded=False):
        st.latex(eq[0])
        if eq[1]:
            st.caption(eq[1])


def _render_per_channel_geo_col(state, sk):
    """Render all per-channel geometry inputs inside one column (hot or cold)."""
    from correlations.thermohydraulic_correlations import ThermoHydraulicCorrelations
    color_icon = "🔴" if sk == "hot" else "🔵"
    label = sk.capitalize()
    ch = state["channels"][sk]
    ch_geo = ch.setdefault("geometry", {})

    st.markdown(f"#### {color_icon} {label} Channel")

    # ── 1. Structure ──────────────────────────────────────────────────────────
    st.markdown("**🔷 Structure**")
    structures = list(ThermoHydraulicCorrelations.get_supported_tpms_types())
    current_struct = ch.get("structure", structures[0])
    struct_idx = structures.index(current_struct) if current_struct in structures else 0
    ch["structure"] = st.selectbox(
        f"{label} structure type",
        options=structures,
        index=struct_idx,
        key=_k(f"geo_{sk}_structure"),
        help="Select the channel geometry type. Parameter fields below update accordingly.",
    )
    struct = ch["structure"]
    _render_tpms_thumbnail(struct)
    _render_tpms_equation(struct)

    st.divider()

    # ── 2. Shape Parameters ───────────────────────────────────────────────────
    st.markdown("**📐 Shape Parameters**")

    r1a, r1b = st.columns(2)
    ch_geo["height"] = r1a.number_input(
        f"{label} height [m]",
        min_value=1e-4,
        value=float(ch_geo.get("height", 0.25)),
        format="%.5f",
        key=_k(f"geo_{sk}_H"),
        help=f"Stack height of the {label.lower()} channel cross-section.",
    )
    if struct != "PlateFin":
        ch_geo["porosity"] = r1b.slider(
            f"{label} porosity ε [-]",
            min_value=0.05, max_value=0.95,
            value=float(ch_geo.get("porosity", 0.65 if sk == "hot" else 0.70)),
            step=0.01,
            key=_k(f"geo_{sk}_eps"),
            help="Structural (void) porosity of the TPMS lattice.",
        )
    else:
        sf = float(ch_geo.get("fin_spacing",   3.2e-3))
        tf = float(ch_geo.get("fin_thickness", 0.6e-3))
        eps_pf = (sf - tf) / max(sf, 1e-12)
        ch_geo["porosity"] = eps_pf
        r1b.metric(f"{label} porosity ε [-]", f"{eps_pf:.3f}",
                   help="Auto-calculated from fin params: (sf − tf) / sf")

    if struct != "PlateFin":
        r2a, r2b = st.columns(2)
        ch_geo["unit_cell_size"] = r2a.number_input(
            f"{label} unit cell size [m]",
            min_value=1e-5,
            value=float(ch_geo.get("unit_cell_size", 5e-3)),
            format="%.6f",
            key=_k(f"geo_{sk}_cell"),
            help="TPMS periodic unit cell side length a.",
        )
        ch_geo["wall_thickness"] = r2b.number_input(
            f"{label} skeleton thickness [m]",
            min_value=1e-7,
            value=float(ch_geo.get("wall_thickness", 5e-4)),
            format="%.6f",
            key=_k(f"geo_{sk}_tw"),
            help="TPMS solid ligament / fin thickness.",
        )
    else:
        ch_geo["unit_cell_size"] = ch_geo.get("unit_cell_size", 5e-3)
        ch_geo["wall_thickness"] = ch_geo.get("wall_thickness", 5e-4)

    if struct == "PlateFin":
        st.markdown(f"**Plate-Fin Dimensions**")
        pf1, pf2, pf3 = st.columns(3)
        ch_geo["fin_height"] = pf1.number_input(
            "Fin height Hf [m]",
            min_value=1e-4,
            value=float(ch_geo.get("fin_height", 9.5e-3)),
            format="%.5f",
            key=_k(f"geo_{sk}_fin_H"),
            help="Distance between the two plates (fin height).",
        )
        ch_geo["fin_spacing"] = pf2.number_input(
            "Fin spacing sf [m]",
            min_value=1e-5,
            value=float(ch_geo.get("fin_spacing", 3.2e-3)),
            format="%.5f",
            key=_k(f"geo_{sk}_fin_s"),
            help="Centre-to-centre fin pitch. Porosity ε = (sf−tf)/sf.",
        )
        ch_geo["fin_thickness"] = pf3.number_input(
            "Fin thickness tf [m]",
            min_value=1e-6,
            value=float(ch_geo.get("fin_thickness", 0.6e-3)),
            format="%.6f",
            key=_k(f"geo_{sk}_fin_t"),
            help="Fin wall thickness. Used in Dh and fin efficiency.",
        )
        pp1, pp2 = st.columns(2)
        ch_geo["perf_density"] = pp1.number_input(
            "Perforation density n [1/m²]",
            min_value=0.0,
            value=float(ch_geo.get("perf_density", 0.0)),
            format="%.1f",
            key=_k(f"geo_{sk}_perf_n"),
            help="Number of perforations per m² of fin face area. Set 0 to disable.",
        )
        ch_geo["perf_radius"] = pp2.number_input(
            "Perforation radius r [m]",
            min_value=0.0,
            value=float(ch_geo.get("perf_radius", 0.0)),
            format="%.5f",
            key=_k(f"geo_{sk}_perf_r"),
            help="Radius of each perforation hole [m]. Active only when perf_density > 0.",
        )
    else:
        ch_geo.setdefault("fin_height",   9.5e-3)
        ch_geo.setdefault("fin_spacing",  3.2e-3)
        ch_geo.setdefault("fin_thickness", 0.6e-3)
        ch_geo.setdefault("perf_density", 0.0)
        ch_geo.setdefault("perf_radius",  0.0)

    # Always mirror shared length/width from global geometry
    ch_geo["length"] = float(state["geometry"]["length"])
    ch_geo["width"]  = float(state["geometry"]["width"])

    st.divider()

    # ── 3. Geometry Estimator (computes & auto-applies SAD) ───────────────────
    st.markdown("**🔬 Geometry Estimator**")
    st.caption("Expand below to inspect surface area density and fluid cross-section profile. "
               "SAD is auto-computed from current parameters on every page update.")
    _auto_apply_sad(state, sk)
    _render_channel_estimator(state, sk)

    st.divider()

    # ── 4. Derived Geometry (result — depends on SAD from estimator above) ────
    st.markdown("**📊 Derived Geometry**")
    _render_derived_channel_metrics(state, sk, ch)


def _auto_apply_sad(state, sk):
    """Compute SAD for channel sk from current geometry state and write to channel state.
    Called unconditionally on every render pass so the value is always up to date."""
    geo = state["geometry"]
    ch = state["channels"][sk]
    ch_geo = ch.get("geometry", {}) or {}
    struct  = ch.get("structure", "Gyroid")
    ch_L    = float(ch_geo.get("length") or geo.get("length", 0.94))
    ch_W    = float(ch_geo.get("width")  or geo.get("width",  0.25))
    ch_H    = float(ch_geo.get("height", 0.25))
    L_cell  = float(ch_geo.get("unit_cell_size", 5e-3))

    if struct == "PlateFin":
        sf = float(ch_geo.get("fin_spacing",   3.2e-3))
        tf = float(ch_geo.get("fin_thickness", 0.6e-3))
        Hf = float(ch_geo.get("fin_height",    9.5e-3))
        n_d = float(ch_geo.get("perf_density", 0.0))
        r_p = float(ch_geo.get("perf_radius",  0.0))
        Ah_factor = (2 * Hf - tf) + 2 * (sf - tf)
        if n_d > 0 and r_p > 0:
            perf_corr = 2 * n_d * np.pi * r_p**2 - 2 * n_d * np.pi * (2 * r_p) * tf
            Ah_factor = max(Ah_factor - perf_corr, 1e-12)
        sad = Ah_factor / max(sf * Hf, 1e-12)
    elif struct == "SmoothPlateFin":
        sad = 2.0 / max(ch_H, 1e-12)
    else:
        eps = float(ch_geo.get("porosity", 0.65 if sk == "hot" else 0.70))
        MC_TYPES = ("Gyroid", "Diamond", "Primitive", "Neovius", "FRD", "FKS")
        if struct in MC_TYPES:
            try:
                sad = _mc_surface_area_density(struct, L_cell, eps, grid_n=40)
            except (ImportError, Exception):
                sad = _empirical_sad_fallback(struct, L_cell, eps)
        else:
            sad = _empirical_sad(struct, L_cell, eps)

    ch["surface_area_density"] = float(sad)
    state["geometry"]["surface_area_density"] = float(sad)
    # Push value into the Streamlit widget's session-state key so the
    # number_input in the Channels card reflects it without a button click.
    widget_key = _k(f"{sk}_sad")
    if widget_key in st.session_state:
        st.session_state[widget_key] = float(sad)


def render_step_geometry(state):
    from ui.step_channels import render_step_channels
    st.subheader("Geometry Settings")
    st.caption(
        "📌 **Workflow:** Set shared dimensions → configure each channel's structure and shape "
        "→ open the Geometry Estimator to compute surface area density → "
        "review derived metrics → proceed to Channel Modeling below."
    )

    # ── 🌐 Shared Dimensions ───────────────────────────────────────────────────
    st.markdown("**🌐 Shared Dimensions**")
    s1, s2, s3, s4 = st.columns(4)
    state["geometry"]["length"] = s1.number_input(
        "Length [m]",
        min_value=1e-4,
        value=float(state["geometry"].get("length", 0.94)),
        key=_k("geom_length"),
        help="Axial length of the heat exchanger core (both channels).",
    )
    state["geometry"]["width"] = s2.number_input(
        "Width [m]",
        min_value=1e-4,
        value=float(state["geometry"].get("width", 0.25)),
        key=_k("geom_width"),
        help="Cross-flow width of the core (both channels).",
    )
    state["geometry"]["plate_thickness"] = s3.number_input(
        "Plate thickness [m]",
        min_value=1e-6,
        value=float(state["geometry"].get("plate_thickness", 1e-3)),
        format="%.6f",
        key=_k("geom_plate"),
        help="Thickness of the solid L×W dividing plate (conduction resistance).",
    )
    # k_wall lives in material; display it here for convenience
    state.setdefault("material", {})["k_wall"] = s4.number_input(
        "Wall conductivity [W/m·K]",
        min_value=0.1,
        value=float(state.get("material", {}).get("k_wall", 237.0)),
        key=_k("geom_kwall"),
        help="Thermal conductivity of the dividing plate material.",
    )

    st.divider()

    # ── 🏗️ Per-Channel Geometry ────────────────────────────────────────────────
    st.markdown("**🏗️ Per-Channel Geometry**")
    col_hot, col_cold = st.columns(2)
    with col_hot:
        _render_per_channel_geo_col(state, "hot")
    with col_cold:
        _render_per_channel_geo_col(state, "cold")

    st.divider()

    # ── ⚙️ Channel Modeling ────────────────────────────────────────────────────
    render_step_channels(state)



# =============================================================================
# TPMS Geometry Estimator -- helper functions
# =============================================================================

def _eval_tpms_normalized(tpms_type: str, Xa, Ya, Za):
    """Evaluate TPMS implicit function with normalized coordinates (already divided by a).
    Returns an array with the same shape as the inputs. Fluid domain: {f < t_iso}."""
    if tpms_type == "Gyroid":
        return (np.sin(Xa) * np.cos(Ya)
                + np.sin(Ya) * np.cos(Za)
                + np.sin(Za) * np.cos(Xa))
    elif tpms_type == "Diamond":
        return (np.sin(Xa) * np.sin(Ya) * np.sin(Za)
                + np.sin(Xa) * np.cos(Ya) * np.cos(Za)
                + np.cos(Xa) * np.sin(Ya) * np.cos(Za)
                + np.cos(Xa) * np.cos(Ya) * np.sin(Za))
    elif tpms_type == "Primitive":
        return np.cos(Xa) + np.cos(Ya) + np.cos(Za)
    elif tpms_type == "Neovius":
        # Neovius: 3(cos x + cos y + cos z) + 4 cos x cos y cos z = 0
        return (3 * (np.cos(Xa) + np.cos(Ya) + np.cos(Za))
                + 4 * np.cos(Xa) * np.cos(Ya) * np.cos(Za))
    elif tpms_type == "FRD":
        # F-RD (Fischer-Koch S surface):
        # 4(cos x cos y + cos y cos z + cos z cos x) - (cos 2x + cos 2y + cos 2z) = 0
        return (4 * (np.cos(Xa) * np.cos(Ya)
                     + np.cos(Ya) * np.cos(Za)
                     + np.cos(Za) * np.cos(Xa))
                - (np.cos(2 * Xa) + np.cos(2 * Ya) + np.cos(2 * Za)))
    elif tpms_type == "FKS":
        # FKS (Schoen I-WP approximation):
        # 2(cos x + cos y + cos z) - (cos(x+y) + cos(x-y) + cos(y+z) + cos(y-z) + cos(z+x) + cos(z-x)) = 0
        return (2 * (np.cos(Xa) + np.cos(Ya) + np.cos(Za))
                - (np.cos(Xa + Ya) + np.cos(Xa - Ya)
                   + np.cos(Ya + Za) + np.cos(Ya - Za)
                   + np.cos(Za + Xa) + np.cos(Za - Xa)))
    else:
        return np.zeros_like(Xa, dtype=float)


# ── TPMS thumbnail ─────────────────────────────────────────────────────────────

_TPMS_DESCRIPTIONS = {
    "PlateFin": (
        "Perforated plate-fin array. Fins of height Hf and pitch sf are aligned "
        "axially, forming rectangular passages separated by thin walls of thickness tf."
    ),
    "SmoothPlateFin": (
        "Smooth parallel-plate duct. Two flat walls of spacing H form a rectangular "
        "channel. Nu and f follow Dittus-Boelter / Petukhov-Filonenko correlations."
    ),
    "FRD": (
        "F-RD (Fischer-Koch) surface: four-connected minimal surface with cubic symmetry. "
        "Higher tortuosity than Gyroid; validated for Re = 35–290 (water)."
    ),
    "FKS": (
        "FKS (Faces-Karcher-Schwarz) surface: schoen-like minimal surface. "
        "High surface area density; validated for Re = 730–10230 (gas)."
    ),
    "Neovius": (
        "Neovius surface: triply-periodic minimal surface with cubic symmetry. "
        "Very high surface area density; validated for Re = 10–75 (water)."
    ),
}


@st.cache_data(show_spinner=False)
def _tpms_thumbnail_bytes(tpms_type: str, grid_n: int = 40, porosity: float = 0.5) -> bytes | None:
    """Render a 3D isosurface of the TPMS implicit function using Marching Cubes.
    Returns PNG bytes, or None for types without an implicit function."""
    _HAS_FUNC = ("Gyroid", "Diamond", "Primitive", "Neovius", "FRD", "FKS")
    if tpms_type not in _HAS_FUNC:
        return None

    try:
        from skimage.measure import marching_cubes
    except ImportError:
        return None

    # Build 3D grid over one unit cell [0, 2π]³
    coords = np.linspace(0, 2 * np.pi, grid_n, endpoint=False)
    Xa, Ya, Za = np.meshgrid(coords, coords, coords, indexing='ij')
    f_grid = _eval_tpms_normalized(tpms_type, Xa, Ya, Za)
    # Wrap periodic boundary so MC sees a closed surface at each face
    f_grid = np.concatenate([f_grid, f_grid[0:1]], axis=0)
    f_grid = np.concatenate([f_grid, f_grid[:, 0:1]], axis=1)
    f_grid = np.concatenate([f_grid, f_grid[:, :, 0:1]], axis=2)

    t_iso = _find_iso_value(tpms_type, porosity, grid_n=grid_n)

    try:
        verts, faces, _, _ = marching_cubes(f_grid, level=t_iso)
    except (ValueError, RuntimeError):
        return None

    fig = plt.figure(figsize=(3, 3), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(
        verts[:, 0], verts[:, 1], faces, verts[:, 2],
        cmap='coolwarm', alpha=0.88, linewidth=0, antialiased=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(tpms_type, fontsize=10, pad=4)
    ax.view_init(elev=25, azim=45)
    fig.tight_layout(pad=0.2)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _render_tpms_thumbnail(tpms_type: str):
    """Show a small TPMS preview or description text."""
    img_bytes = _tpms_thumbnail_bytes(tpms_type)
    if img_bytes is not None:
        st.image(img_bytes, caption=f"{tpms_type} cross-section (z=0 plane)", width=260)
    else:
        desc = _TPMS_DESCRIPTIONS.get(tpms_type, "")
        if desc:
            st.caption(desc)


def _eval_tpms(tpms_type: str, X, Y, Z, a: float):
    """Evaluate TPMS implicit function with physical coordinates.
    a = unit_cell_size / (2*pi)."""
    return _eval_tpms_normalized(tpms_type, X / a, Y / a, Z / a)


@st.cache_data(show_spinner=False)
def _find_iso_value(tpms_type: str, porosity: float, grid_n: int = 40):
    """Find iso-value t such that volume fraction of {f < t} equals porosity.
    Works in normalized [0, 2π] space — independent of unit_cell_size."""
    from scipy.optimize import brentq

    coords = np.linspace(0, 2.0 * np.pi, grid_n, endpoint=False)
    Xa, Ya, Za = np.meshgrid(coords, coords, coords, indexing='ij')
    f_grid = _eval_tpms_normalized(tpms_type, Xa, Ya, Za)

    f_min, f_max = float(f_grid.min()), float(f_grid.max())

    def vol_frac(t):
        return float(np.mean(f_grid < t)) - porosity

    # Guard degenerate porosity values
    if vol_frac(f_min + 1e-6) >= 0:
        return f_min + 1e-6
    if vol_frac(f_max - 1e-6) <= 0:
        return f_max - 1e-6

    return float(brentq(vol_frac, f_min + 1e-6, f_max - 1e-6,
                        xtol=1e-4, rtol=1e-4, maxiter=60))


def _empirical_sad(tpms_type: str, unit_cell_size: float, porosity: float) -> float:
    """Empirical specific surface area for Neovius, FRD, FKS [1/m]."""
    L, eps = unit_cell_size, porosity
    if tpms_type == "Neovius":
        return 5.0 * (1.0 - eps) / L
    else:  # FRD, FKS
        return 4.2 * ((1.0 - eps) ** 0.5) / L


def _empirical_sad_fallback(tpms_type: str, unit_cell_size: float, porosity: float) -> float:
    """Fallback empirical estimate for MC-capable types when scikit-image is absent.
    Uses known analytic constants near ε=0.5, scaled linearly with porosity deviation."""
    c0 = {"Gyroid": 3.091, "Diamond": 3.840, "Primitive": 2.350}.get(tpms_type, 3.0)
    return c0 / unit_cell_size * (1.0 + 0.3 * abs(porosity - 0.5))


def _mc_surface_area_density(tpms_type: str, unit_cell_size: float,
                              porosity: float, grid_n: int = 60) -> float:
    """Compute surface area density [1/m] via Marching Cubes over one unit cell.
    Requires scikit-image. Falls back to empirical on any failure."""
    from skimage.measure import marching_cubes  # raises ImportError if absent

    L = unit_cell_size
    a = L / (2.0 * np.pi)
    spacing = L / grid_n

    coords = np.linspace(0, L, grid_n, endpoint=False)
    XX, YY, ZZ = np.meshgrid(coords, coords, coords, indexing='ij')
    f_grid = _eval_tpms(tpms_type, XX, YY, ZZ, a)
    # Wrap periodic boundary: append first plane in each axis so MC sees the full unit cell
    f_grid = np.concatenate([f_grid, f_grid[0:1, :, :]], axis=0)
    f_grid = np.concatenate([f_grid, f_grid[:, 0:1, :]], axis=1)
    f_grid = np.concatenate([f_grid, f_grid[:, :, 0:1]], axis=2)
    t_iso = _find_iso_value(tpms_type, porosity, grid_n=40)

    try:
        verts, faces, _, _ = marching_cubes(
            f_grid, level=t_iso, spacing=(spacing, spacing, spacing)
        )
    except (ValueError, RuntimeError):
        return _empirical_sad_fallback(tpms_type, unit_cell_size, porosity)

    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    total_area = float(0.5 * np.linalg.norm(cross, axis=1).sum())
    return total_area / (L ** 3)


@st.cache_data(show_spinner="Estimating surface area...")
def _compute_tpms_surface_area(tpms_type: str, unit_cell_size: float,
                                porosity: float, grid_n: int = 60):
    """Dispatcher: Marching Cubes for all TPMS types with implicit functions; empirical for rest.
    Returns (sad [1/m], method_label, warning_str_or_None)."""
    MC_TYPES = ("Gyroid", "Diamond", "Primitive", "Neovius", "FRD", "FKS")
    if tpms_type in MC_TYPES:
        try:
            sad = _mc_surface_area_density(tpms_type, unit_cell_size, porosity, grid_n)
            return sad, "Marching Cubes", None
        except ImportError:
            sad = _empirical_sad_fallback(tpms_type, unit_cell_size, porosity)
            return sad, "Empirical (fallback)", (
                "scikit-image not installed — using empirical approximation. "
                "Install scikit-image for Marching Cubes accuracy."
            )
    else:
        sad = _empirical_sad(tpms_type, unit_cell_size, porosity)
        return sad, "Empirical", None


@st.cache_data(show_spinner="Computing cross-section profile...")
def _compute_tpms_cross_section(tpms_type: str, unit_cell_size: float,
                                 porosity: float, n_slices: int = 30):
    """Voxel-counting cross-section profile over one unit cell.
    Returns (x_positions [m], areas [m²], t_iso) where areas[i] = fluid_fraction * L²."""
    L = unit_cell_size
    a = L / (2.0 * np.pi)
    t_iso = _find_iso_value(tpms_type, porosity, grid_n=40)

    n_yz = 80
    h_yz = L / n_yz
    y_vals = np.arange(n_yz) * h_yz + 0.5 * h_yz   # cell-centred midpoints, no boundary nodes
    z_vals = np.arange(n_yz) * h_yz + 0.5 * h_yz
    YY, ZZ = np.meshgrid(y_vals, z_vals, indexing='ij')
    x_positions = np.linspace(0, L, n_slices, endpoint=True)

    areas = np.empty(n_slices)
    for i, x0 in enumerate(x_positions):
        XX = np.full_like(YY, x0)
        f_vals = _eval_tpms(tpms_type, XX, YY, ZZ, a)
        areas[i] = float(np.mean(f_vals < t_iso)) * (L ** 2)

    return x_positions, areas, t_iso


def _render_surface_area_tab(state, tpms_type, L_cell, ch_L, ch_W, ch_H, eps, ch_label, grid_n):
    """Render surface area metrics and 'Use this value' button for one channel."""
    geo = state["geometry"]
    if tpms_type == "SmoothPlateFin":
        sad = 2.0 / ch_H
        method = "Analytical (2/H)"
        warn = None
        st.info(
            f"SmoothPlateFin baseline: SAD = 2/H = **{sad:.1f} 1/m** "
            f"(two flat walls, channel height H = {ch_H*1e3:.1f} mm). "
            "Set surface_area_density to this value for a fair PEC comparison."
        )
    elif tpms_type == "PlateFin":
        # SAD from heat transfer area (Eq. 9) per total volume
        sk = ch_label.lower()
        ch_geo_ov = state["channels"][sk].get("geometry", {}) or {}
        Hf = float(ch_geo_ov.get("fin_height")    or geo.get("fin_height",    9.5e-3))
        sf = float(ch_geo_ov.get("fin_spacing")   or geo.get("fin_spacing",   3.2e-3))
        tf = float(ch_geo_ov.get("fin_thickness") or geo.get("fin_thickness", 0.6e-3))
        n_d = float(geo.get("perf_density", 0.0))
        r_p = float(geo.get("perf_radius",  0.0))
        Ah_factor = (2 * Hf - tf) + 2 * (sf - tf)
        if n_d > 0 and r_p > 0:
            perf_corr = 2 * n_d * np.pi * r_p**2 - 2 * n_d * np.pi * (2 * r_p) * tf
            Ah_factor = max(Ah_factor - perf_corr, 1e-12)
        sad = Ah_factor / (sf * Hf)
        method = "Analytical (Eq. 9, Wang et al. 2024)"
        warn = None
        eps_pf = (sf - tf) / sf
        Dh_pf = 2.0 * (Hf - tf) * (sf - tf) / max(Hf + sf - 2.0 * tf, 1e-12)
        st.info(
            f"PlateFin: SAD = **{sad:.1f} 1/m**  "
            f"(Hf = {Hf*1e3:.2f} mm, sf = {sf*1e3:.2f} mm, tf = {tf*1e3:.2f} mm)  \n"
            f"Porosity ε = {eps_pf:.3f},  Dh = {Dh_pf*1e3:.3f} mm"
        )
    else:
        sad, method, warn = _compute_tpms_surface_area(tpms_type, L_cell, eps, grid_n)

    total_area = sad * ch_W * ch_H * ch_L

    if warn:
        st.warning(warn)

    c1, c2 = st.columns(2)
    c1.metric(f"Surface area density [1/m]  ({method})", f"{sad:.1f}")
    c2.metric("Total HX area [m²]", f"{total_area:.4f}")

    sk = ch_label.lower()
    if st.button(
        f"📥 Apply {sad:.1f} 1/m → {ch_label.capitalize()} α",
        key=_k(f"use_sad_{ch_label}"),
        help="Manually push the computed SAD value to the channel surface area density field.",
    ):
        state["channels"][sk]["surface_area_density"] = float(sad)
        state["geometry"]["surface_area_density"] = float(sad)
        widget_key = _k(f"{sk}_sad")
        if widget_key in st.session_state:
            st.session_state[widget_key] = float(sad)
        st.session_state.ui_version += 1
        maybe_autosave(force=True)
        st.rerun()


def _render_cross_section_plot(hot_struct, cold_struct, L, W, H,
                                eps_hot, eps_cold, n_slices):
    """Matplotlib plot of fluid cross-section area vs. axial position."""
    import matplotlib.pyplot as plt

    MC_TYPES = ("Gyroid", "Diamond", "Primitive", "Neovius", "FRD", "FKS")

    def get_profile(struct, eps):
        if struct in ("SmoothPlateFin", "PlateFin"):
            x_pos = np.linspace(0, L, n_slices, endpoint=False)
            areas = np.full(n_slices, eps * W * H)
            return x_pos, areas, None, None
        if struct in MC_TYPES:
            try:
                x_pos, areas, t_iso = _compute_tpms_cross_section(
                    struct, L, eps, n_slices
                )
                scale = (W * H) / (L ** 2)
                return x_pos, areas * scale, t_iso, None
            except ImportError:
                pass
        # Fallback: constant profile at mean area
        x_pos = np.linspace(0, L, n_slices, endpoint=False)
        areas = np.full(n_slices, eps * W * H)
        msg = f"{struct}: no implicit function available — showing constant mean area."
        return x_pos, areas, None, msg

    x_hot,  A_hot,  t_hot,  w_hot  = get_profile(hot_struct,  eps_hot)
    x_cold, A_cold, t_cold, w_cold = get_profile(cold_struct, eps_cold)

    if w_hot:
        st.warning(w_hot)
    if w_cold:
        st.warning(w_cold)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(x_hot  * 1e3, A_hot  * 1e4, color="#d62728", lw=1.8,
            label=f"Hot ({hot_struct})")
    ax.plot(x_cold * 1e3, A_cold * 1e4, color="#1f77b4", lw=1.8,
            label=f"Cold ({cold_struct})")
    ax.axhline(np.mean(A_hot)  * 1e4, color="#d62728", ls="--", lw=0.9, alpha=0.6)
    ax.axhline(np.mean(A_cold) * 1e4, color="#1f77b4", ls="--", lw=0.9, alpha=0.6)
    ax.set_xlabel("Position in unit cell [mm]")
    ax.set_ylabel("Fluid cross-section area [cm²]")
    ax.set_title("Fluid channel cross-section vs. axial position (one unit cell)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    parts = []
    for label, A in (("Hot", A_hot), ("Cold", A_cold)):
        parts.append(
            f"{label}: min={A.min()*1e4:.3f} | mean={A.mean()*1e4:.3f} | "
            f"max={A.max()*1e4:.3f} cm²"
        )
    st.caption("  ·  ".join(parts))
    if t_hot is not None or t_cold is not None:
        iso_parts = []
        if t_hot  is not None: iso_parts.append(f"hot t = {t_hot:.4f}")
        if t_cold is not None: iso_parts.append(f"cold t = {t_cold:.4f}")
        st.caption("Iso-values: " + "   ".join(iso_parts))


def _render_cross_section_plot_single(struct, L_cell, W, H, eps, n_slices, sk):
    """Matplotlib plot of fluid cross-section area vs. axial position for one channel."""
    import matplotlib.pyplot as plt

    MC_TYPES = ("Gyroid", "Diamond", "Primitive", "Neovius", "FRD", "FKS")
    color = "#d62728" if sk == "hot" else "#1f77b4"
    label = f"{sk.capitalize()} ({struct})"

    if struct in ("SmoothPlateFin", "PlateFin"):
        x_pos = np.linspace(0, L_cell, n_slices, endpoint=False)
        areas = np.full(n_slices, eps * W * H)
        t_iso = None
        warn = None
    elif struct in MC_TYPES:
        try:
            x_pos, areas_norm, t_iso = _compute_tpms_cross_section(struct, L_cell, eps, n_slices)
            scale = (W * H) / (L_cell ** 2)
            areas = areas_norm * scale
            warn = None
        except ImportError:
            x_pos = np.linspace(0, L_cell, n_slices, endpoint=False)
            areas = np.full(n_slices, eps * W * H)
            t_iso = None
            warn = f"{struct}: scikit-image not installed — showing constant mean area."
    else:
        x_pos = np.linspace(0, L_cell, n_slices, endpoint=False)
        areas = np.full(n_slices, eps * W * H)
        t_iso = None
        warn = f"{struct}: no implicit function available — showing constant mean area."

    if warn:
        st.warning(warn)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(x_pos * 1e3, areas * 1e4, color=color, lw=1.8, label=label)
    ax.axhline(np.mean(areas) * 1e4, color=color, ls="--", lw=0.9, alpha=0.6,
               label=f"Mean = {np.mean(areas)*1e4:.3f} cm²")
    ax.set_xlabel("Position in unit cell [mm]")
    ax.set_ylabel("Fluid cross-section [cm²]")
    ax.set_title(f"Cross-section vs. axial position — one unit cell")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        f"min={areas.min()*1e4:.3f} | mean={areas.mean()*1e4:.3f} | max={areas.max()*1e4:.3f} cm²"
        + (f"   ·   iso-value t = {t_iso:.4f}" if t_iso is not None else "")
    )


def _render_channel_estimator(state, sk):
    """Per-channel geometry estimator expander shown inside each channel card."""
    geo = state["geometry"]
    ch_geo = state["channels"][sk].get("geometry", {}) or {}
    ch_L   = float(ch_geo.get("length")         or geo["length"])
    ch_W   = float(ch_geo.get("width")          or geo["width"])
    ch_H   = float(ch_geo.get("height",         0.25))
    L_cell = float(ch_geo.get("unit_cell_size", 5e-3))
    struct = state["channels"][sk]["structure"]
    # Porosity: use per-channel value (PlateFin computes its own from fin params)
    if struct == "PlateFin":
        sf = float(ch_geo.get("fin_spacing",   3.2e-3))
        tf = float(ch_geo.get("fin_thickness", 0.6e-3))
        eps = (sf - tf) / max(sf, 1e-12)
    else:
        eps = float(ch_geo.get("porosity", 0.65 if sk == "hot" else 0.70))
    GRID_N   = 80
    N_SLICES = 50

    title = "Plate-fin Geometry Estimator" if struct == "PlateFin" else "TPMS Geometry Estimator"
    with st.expander(f"{title} — {sk.capitalize()} channel", expanded=False):
        st.caption(
            f"**{struct}**  ·  ε = {eps:.3f}  ·  "
            f"Unit cell / fin pitch L = {L_cell*1e3:.3f} mm  ·  "
            f"Channel: {ch_L*1e3:.1f} \u00d7 {ch_W*1e3:.1f} \u00d7 {ch_H*1e3:.1f} mm"
        )
        st.markdown("##### Surface Area Density")
        _render_surface_area_tab(state, struct, L_cell, ch_L, ch_W, ch_H, eps, sk, GRID_N)
        st.markdown("##### Fluid Cross-Section vs. Axial Position")
        _render_cross_section_plot_single(struct, L_cell, ch_W, ch_H, eps, N_SLICES, sk)


