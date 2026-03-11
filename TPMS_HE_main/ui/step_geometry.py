"""ui.step_geometry -- Step 1: Geometry & Channels rendering and TPMS geometry estimator."""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from correlations.thermohydraulic_correlations import ThermoHydraulicCorrelations
from ui.components import _k
from ui.state import maybe_autosave

def render_step_geometry(state):
    from ui.step_channels import render_step_channels
    st.subheader("Geometry Settings")
    c1, c2 = st.columns(2)
    with c1:
        state["geometry"]["length"] = st.number_input(
            "Length [m]",
            min_value=1e-4,
            value=float(state["geometry"]["length"]),
            key=_k("geom_length"),
        )
        state["geometry"]["width"] = st.number_input(
            "Width [m]",
            min_value=1e-4,
            value=float(state["geometry"]["width"]),
            key=_k("geom_width"),
        )
        state["geometry"]["height"] = st.number_input(
            "Height [m]",
            min_value=1e-4,
            value=float(state["geometry"]["height"]),
            key=_k("geom_height"),
        )
        state["geometry"]["unit_cell_size"] = st.number_input(
            "Unit cell size [m]",
            min_value=1e-5,
            value=float(state["geometry"]["unit_cell_size"]),
            format="%.6f",
            key=_k("geom_cell"),
        )

    with c2:
        state["geometry"]["wall_thickness"] = st.number_input(
            "TPMS skeleton thickness [m]",
            min_value=1e-6,
            value=float(state["geometry"]["wall_thickness"]),
            format="%.6f",
            key=_k("geom_wall"),
            help="Thickness of the TPMS solid ligaments/sheets acting as fins. "
                 "Used for fin efficiency in packed-bed channels.",
        )
        state["geometry"]["plate_thickness"] = st.number_input(
            "Dividing plate thickness [m]",
            min_value=1e-6,
            value=float(state["geometry"].get("plate_thickness", 1e-3)),
            format="%.6f",
            key=_k("geom_plate"),
            help="Thickness of the solid L\u00d7W plate separating hot and cold streams. "
                 "Used for wall conduction resistance in the UA series chain.",
        )
        state["geometry"]["porosity_hot"] = st.slider(
            "Hot porosity [-]",
            min_value=0.05,
            max_value=0.95,
            value=float(state["geometry"]["porosity_hot"]),
            step=0.01,
            key=_k("geom_por_hot"),
        )
        state["geometry"]["porosity_cold"] = st.slider(
            "Cold porosity [-]",
            min_value=0.05,
            max_value=0.95,
            value=float(state["geometry"]["porosity_cold"]),
            step=0.01,
            key=_k("geom_por_cold"),
        )

    # --- PlateFin fin geometry (global defaults) ---
    with st.expander("Plate-fin geometry (used when structure = PlateFin)", expanded=False):
        pf1, pf2, pf3 = st.columns(3)
        state["geometry"]["fin_height"] = pf1.number_input(
            "Fin height Hf [m]",
            min_value=1e-4,
            value=float(state["geometry"].get("fin_height", 9.5e-3)),
            format="%.5f",
            key=_k("geom_fin_H"),
            help="Distance between the two plates (fin height). Used for plate-fin Dh and fin efficiency.",
        )
        state["geometry"]["fin_spacing"] = pf2.number_input(
            "Fin spacing sf [m]",
            min_value=1e-5,
            value=float(state["geometry"].get("fin_spacing", 3.2e-3)),
            format="%.5f",
            key=_k("geom_fin_s"),
            help="Center-to-center fin pitch. Determines porosity ε = (sf−tf)/sf.",
        )
        state["geometry"]["fin_thickness"] = pf3.number_input(
            "Fin thickness tf [m]",
            min_value=1e-6,
            value=float(state["geometry"].get("fin_thickness", 0.6e-3)),
            format="%.6f",
            key=_k("geom_fin_t"),
            help="Fin wall thickness. Used in hydraulic diameter Eq. 2 and fin efficiency Eqs. 11–12.",
        )
        pf4, pf5 = st.columns(2)
        state["geometry"]["perf_density"] = pf4.number_input(
            "Perforation density n [1/m²]",
            min_value=0.0,
            value=float(state["geometry"].get("perf_density", 0.0)),
            format="%.1f",
            key=_k("geom_perf_n"),
            help="Number of perforations per m² of fin face area (Eqs. 8–9). Set to 0 to ignore.",
        )
        state["geometry"]["perf_radius"] = pf5.number_input(
            "Perforation radius r [m]",
            min_value=0.0,
            value=float(state["geometry"].get("perf_radius", 0.0)),
            format="%.5f",
            key=_k("geom_perf_r"),
            help="Radius of each perforation hole [m] (Eqs. 8–9). Used only when perf_density > 0.",
        )

    # --- Per-channel geometry toggle ---
    identical = st.toggle(
        "Identical hot/cold geometry (same L, W, H, unit cell, skeleton)",
        value=bool(state["geometry"].get("identical_channels", True)),
        key=_k("geom_identical"),
        help="When ON, both channels share the global geometry above. "
             "Turn OFF to set independent values per channel.",
    )
    state["geometry"]["identical_channels"] = identical

    if not identical:
        st.markdown("#### Per-Channel Geometry Override")
        col_hot, col_cold = st.columns(2)
        geo_h = state["channels"]["hot"].setdefault("geometry", {})
        geo_c = state["channels"]["cold"].setdefault("geometry", {})
        hot_structure = state["channels"]["hot"].get("structure", "")
        cold_structure = state["channels"]["cold"].get("structure", "")
        with col_hot:
            st.markdown("**Hot channel**")
            geo_h["length"] = st.number_input(
                "Hot length [m]", min_value=1e-4,
                value=float(geo_h.get("length") or state["geometry"]["length"]),
                key=_k("ch_hot_L"),
            )
            geo_h["width"] = st.number_input(
                "Hot width [m]", min_value=1e-4,
                value=float(geo_h.get("width") or state["geometry"]["width"]),
                key=_k("ch_hot_W"),
            )
            geo_h["height"] = st.number_input(
                "Hot height [m]", min_value=1e-4,
                value=float(geo_h.get("height") or state["geometry"]["height"]),
                key=_k("ch_hot_H"),
            )
            if hot_structure == "PlateFin":
                geo_h["unit_cell_size"] = None
                geo_h["wall_thickness"] = None
            elif hot_structure != "SmoothPlateFin":
                geo_h["unit_cell_size"] = st.number_input(
                    "Hot unit cell size [m]", min_value=1e-6,
                    value=float(geo_h.get("unit_cell_size") or state["geometry"]["unit_cell_size"]),
                    format="%.6f", key=_k("ch_hot_cell"),
                )
                geo_h["wall_thickness"] = st.number_input(
                    "Hot skeleton thickness [m]", min_value=1e-7,
                    value=float(geo_h.get("wall_thickness") or state["geometry"]["wall_thickness"]),
                    format="%.6f", key=_k("ch_hot_tw"),
                    help="TPMS skeleton/fin thickness for the hot channel.",
                )
                geo_h["fin_height"] = None
                geo_h["fin_spacing"] = None
                geo_h["fin_thickness"] = None
            else:
                geo_h["unit_cell_size"] = None
                geo_h["wall_thickness"] = st.number_input(
                    "Hot skeleton thickness [m]", min_value=1e-7,
                    value=float(geo_h.get("wall_thickness") or state["geometry"]["wall_thickness"]),
                    format="%.6f", key=_k("ch_hot_tw"),
                    help="TPMS skeleton/fin thickness for the hot channel.",
                )
                geo_h["fin_height"] = None
                geo_h["fin_spacing"] = None
                geo_h["fin_thickness"] = None
        with col_cold:
            st.markdown("**Cold channel**")
            geo_c["length"] = st.number_input(
                "Cold length [m]", min_value=1e-4,
                value=float(geo_c.get("length") or state["geometry"]["length"]),
                key=_k("ch_cold_L"),
            )
            geo_c["width"] = st.number_input(
                "Cold width [m]", min_value=1e-4,
                value=float(geo_c.get("width") or state["geometry"]["width"]),
                key=_k("ch_cold_W"),
            )
            geo_c["height"] = st.number_input(
                "Cold height [m]", min_value=1e-4,
                value=float(geo_c.get("height") or state["geometry"]["height"]),
                key=_k("ch_cold_H"),
            )
            if cold_structure == "PlateFin":
                geo_c["unit_cell_size"] = None
                geo_c["wall_thickness"] = None
            elif cold_structure != "SmoothPlateFin":
                geo_c["unit_cell_size"] = st.number_input(
                    "Cold unit cell size [m]", min_value=1e-6,
                    value=float(geo_c.get("unit_cell_size") or state["geometry"]["unit_cell_size"]),
                    format="%.6f", key=_k("ch_cold_cell"),
                )
                geo_c["wall_thickness"] = st.number_input(
                    "Cold skeleton thickness [m]", min_value=1e-7,
                    value=float(geo_c.get("wall_thickness") or state["geometry"]["wall_thickness"]),
                    format="%.6f", key=_k("ch_cold_tw"),
                    help="TPMS skeleton/fin thickness for the cold channel.",
                )
                geo_c["fin_height"] = None
                geo_c["fin_spacing"] = None
                geo_c["fin_thickness"] = None
            else:
                geo_c["unit_cell_size"] = None
                geo_c["wall_thickness"] = st.number_input(
                    "Cold skeleton thickness [m]", min_value=1e-7,
                    value=float(geo_c.get("wall_thickness") or state["geometry"]["wall_thickness"]),
                    format="%.6f", key=_k("ch_cold_tw"),
                    help="TPMS skeleton/fin thickness for the cold channel.",
                )
                geo_c["fin_height"] = None
                geo_c["fin_spacing"] = None
                geo_c["fin_thickness"] = None
    else:
        # Identical: clear per-channel dimension overrides so solver uses global.
        # Fin params (fin_height/spacing/thickness) are owned by the channel card — leave them.
        for sk in ("hot", "cold"):
            geo = state["channels"][sk].setdefault("geometry", {})
            geo.update({"length": None, "width": None, "height": None,
                        "unit_cell_size": None, "wall_thickness": None})

    st.divider()
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
    else:
        return np.zeros_like(Xa, dtype=float)


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
    """Dispatcher: Marching Cubes for Gyroid/Diamond/Primitive; empirical for rest.
    Returns (sad [1/m], method_label, warning_str_or_None)."""
    MC_TYPES = ("Gyroid", "Diamond", "Primitive")
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

    if st.button(f"Use {sad:.1f} 1/m as surface_area_density",
                 key=_k(f"use_sad_{ch_label}")):
        # Write to per-channel SAD (ch_label is "Hot" or "Cold")
        sk = ch_label.lower()
        state["channels"][sk]["surface_area_density"] = float(sad)
        # Also keep global geometry SAD in sync if both channels use same structure
        state["geometry"]["surface_area_density"] = float(sad)
        st.session_state.ui_version += 1
        maybe_autosave(force=True)
        st.rerun()


def _render_cross_section_plot(hot_struct, cold_struct, L, W, H,
                                eps_hot, eps_cold, n_slices):
    """Matplotlib plot of fluid cross-section area vs. axial position."""
    import matplotlib.pyplot as plt

    MC_TYPES = ("Gyroid", "Diamond", "Primitive")

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

    MC_TYPES = ("Gyroid", "Diamond", "Primitive")
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
    ch_H   = float(ch_geo.get("height")         or geo["height"])
    L_cell = float(ch_geo.get("unit_cell_size") or geo["unit_cell_size"])
    struct = state["channels"][sk]["structure"]
    # For PlateFin use fin-geometry porosity; otherwise use global porosity slider
    if struct == "PlateFin":
        sf = float(ch_geo.get("fin_spacing")   or geo.get("fin_spacing",   3.2e-3))
        tf = float(ch_geo.get("fin_thickness") or geo.get("fin_thickness", 0.6e-3))
        eps = (sf - tf) / sf
    else:
        eps = float(geo[f"porosity_{sk}"])
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


