
import copy
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from packed_bed_model import SUPPORTED_PACKED_MODES
from tpms_correlations import TPMSCorrelations
from tpms_thermo_hydraulic_calculator import TPMSHeatExchanger, create_default_config

APP_DIR = Path(__file__).resolve().parent
AUTOSAVE_PATH = APP_DIR / ".streamlit" / "tpms_ui_state.json"

STEP_DEFS = [
    {"key": "geometry", "title": "1. Geometry & Channels", "sections": ["geometry", "channels"]},
    {"key": "operating", "title": "2. Operating", "sections": ["operating"]},
    {"key": "solver", "title": "3. Solver", "sections": ["solver"]},
    {"key": "output", "title": "4. Output", "sections": ["output"]},
    {
        "key": "confirm",
        "title": "5. Confirm & Run",
        "sections": ["geometry", "operating", "channels", "solver", "output"],
    },
    {"key": "results", "title": "6. Results", "sections": []},
]


def _extract_ui_state(cfg):
    return {
        "geometry": copy.deepcopy(cfg["geometry"]),
        "operating": copy.deepcopy(cfg["operating"]),
        "channels": copy.deepcopy(cfg["channels"]),
        "solver": copy.deepcopy(cfg["solver"]),
        "output": copy.deepcopy(cfg["output"]),
    }


def _deep_merge(base, override):
    merged = copy.deepcopy(base)
    if not isinstance(override, dict):
        return merged
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        elif key in merged:
            merged[key] = val
    return merged


def load_ui_state(path):
    if not path.exists():
        return None, []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None, [f"Autosave format invalid: {path}"]
        return data, []
    except Exception as exc:
        return None, [f"Autosave load failed ({path}): {exc}"]


def sanitize_ui_state(state, defaults):
    merged = _deep_merge(defaults, state)
    notices = []
    supported_tpms = set(TPMSCorrelations.get_supported_tpms_types())
    supported_modes = {"bare", "packed"}
    supported_packed_modes = set(SUPPORTED_PACKED_MODES)

    for ch in ("hot", "cold"):
        ch_cfg = merged["channels"][ch]
        if ch_cfg["mode"] not in supported_modes:
            ch_cfg["mode"] = defaults["channels"][ch]["mode"]
            notices.append(f"{ch} channel mode reset to default.")
        if ch_cfg["structure"] not in supported_tpms:
            ch_cfg["structure"] = defaults["channels"][ch]["structure"]
            notices.append(f"{ch} TPMS structure reset to default.")
        if ch_cfg["packed"]["mode"] not in supported_packed_modes:
            ch_cfg["packed"]["mode"] = defaults["channels"][ch]["packed"]["mode"]
            notices.append(f"{ch} packed mode reset to default.")

    return merged, notices


def save_ui_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(state, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def maybe_autosave(force=False):
    state_blob = json.dumps(
        st.session_state.ui_state, ensure_ascii=False, sort_keys=True
    )
    if force or state_blob != st.session_state.last_saved_blob:
        save_ui_state(AUTOSAVE_PATH, st.session_state.ui_state)
        st.session_state.last_saved_blob = state_blob
        st.session_state.last_saved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def init_ui_state():
    if st.session_state.get("ui_initialized"):
        return

    defaults = _extract_ui_state(create_default_config())
    loaded_state, load_msgs = load_ui_state(AUTOSAVE_PATH)
    if loaded_state is None:
        ui_state = defaults
        msgs = load_msgs
    else:
        ui_state, sanitize_msgs = sanitize_ui_state(loaded_state, defaults)
        msgs = load_msgs + sanitize_msgs

    st.session_state.ui_state = ui_state
    st.session_state.current_step = 0
    st.session_state.ui_version = 0
    st.session_state.run_result = None
    st.session_state.last_run_at = None
    st.session_state.init_messages = msgs
    st.session_state.last_saved_blob = json.dumps(
        ui_state, ensure_ascii=False, sort_keys=True
    )
    st.session_state.last_saved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.ui_initialized = True

    maybe_autosave(force=True)


def reset_to_defaults():
    st.session_state.ui_state = _extract_ui_state(create_default_config())
    st.session_state.current_step = 0
    st.session_state.ui_version += 1
    st.session_state.run_result = None
    maybe_autosave(force=True)
    st.rerun()


def reload_autosave():
    defaults = _extract_ui_state(create_default_config())
    loaded_state, load_msgs = load_ui_state(AUTOSAVE_PATH)
    if loaded_state is None:
        st.session_state.ui_state = defaults
        st.session_state.init_messages = load_msgs + [
            "Autosave unavailable; defaults loaded."
        ]
    else:
        merged, sanitize_msgs = sanitize_ui_state(loaded_state, defaults)
        st.session_state.ui_state = merged
        st.session_state.init_messages = load_msgs + sanitize_msgs + [
            "Autosave reloaded."
        ]

    st.session_state.current_step = 0
    st.session_state.ui_version += 1
    st.session_state.run_result = None
    maybe_autosave(force=True)
    st.rerun()


def _add_issue(issues, level, section, field, message):
    issues.append(
        {
            "level": level,
            "section": section,
            "field": field,
            "message": message,
        }
    )


def validate_ui_state(state):
    issues = []
    geo = state["geometry"]
    ops = state["operating"]
    solver = state["solver"]
    output = state["output"]
    channels = state["channels"]

    for key in ("length", "width", "height", "unit_cell_size", "wall_thickness"):
        if geo[key] <= 0:
            _add_issue(issues, "error", "geometry", key, f"{key} must be > 0.")
    if geo["surface_area_density"] <= 0:
        _add_issue(
            issues,
            "error",
            "geometry",
            "surface_area_density",
            "surface_area_density must be > 0.",
        )
    for key in ("porosity_hot", "porosity_cold"):
        if not (0.05 <= geo[key] <= 0.95):
            _add_issue(
                issues,
                "error",
                "geometry",
                key,
                f"{key} must be within [0.05, 0.95].",
            )
    # Per-channel geometry overrides (only when identical_channels is OFF)
    if not geo.get("identical_channels", True):
        for ch in ("hot", "cold"):
            ch_geo = channels[ch].get("geometry", {}) or {}
            for dim in ("length", "width", "height", "unit_cell_size", "wall_thickness"):
                val = ch_geo.get(dim)
                if val is not None and val <= 0:
                    _add_issue(issues, "error", "geometry",
                               f"{ch}_{dim}", f"{ch} {dim} must be > 0.")
    # Dividing plate thickness
    pt = geo.get("plate_thickness")
    if pt is not None and pt <= 0:
        _add_issue(issues, "error", "geometry", "plate_thickness",
                   "Plate thickness must be > 0.")

    for key in ("Th_in", "Tc_in"):
        if ops[key] <= 0:
            _add_issue(issues, "error", "operating", key, f"{key} must be > 0.")
    if ops["Tc_in"] >= ops["Th_in"]:
        _add_issue(
            issues,
            "error",
            "operating",
            "Tc_in",
            "Tc_in must be lower than Th_in.",
        )
    for key in ("Ph_in", "Pc_in", "mh", "mc"):
        if ops[key] <= 0:
            _add_issue(issues, "error", "operating", key, f"{key} must be > 0.")
    if not (0.0 <= ops["xh_in"] <= 1.0):
        _add_issue(
            issues,
            "error",
            "operating",
            "xh_in",
            "xh_in must be within [0, 1].",
        )
    if "hydrogen" not in ops["fluid_hot"].lower():
        _add_issue(
            issues,
            "warning",
            "operating",
            "fluid_hot",
            "Hot fluid is not hydrogen; xh_in/kinetics may be inactive.",
        )
    else:
        if channels["hot"]["mode"] == "packed":
            _add_issue(
                issues,
                "info",
                "channels",
                "hot.mode",
                "Ortho-para conversion enabled (hot channel packed).",
            )
        else:
            _add_issue(
                issues,
                "warning",
                "channels",
                "hot.mode",
                "Ortho-para conversion disabled (rate=0, hot channel bare).",
            )

    supported_tpms = set(TPMSCorrelations.get_supported_tpms_types())
    for ch in ("hot", "cold"):
        ch_cfg = channels[ch]
        if ch_cfg["mode"] not in ("bare", "packed"):
            _add_issue(
                issues,
                "error",
                "channels",
                f"{ch}.mode",
                f"{ch} mode must be 'bare' or 'packed'.",
            )
        if ch_cfg["structure"] not in supported_tpms:
            _add_issue(
                issues,
                "error",
                "channels",
                f"{ch}.structure",
                f"{ch} structure is unsupported.",
            )
        if ch_cfg["mode"] == "packed":
            packed = ch_cfg["packed"]
            if packed["mode"] not in SUPPORTED_PACKED_MODES:
                _add_issue(
                    issues,
                    "error",
                    "channels",
                    f"{ch}.packed.mode",
                    f"{ch} packed mode must be one of {SUPPORTED_PACKED_MODES}.",
                )
            if packed["particle_diameter"] <= 0:
                _add_issue(
                    issues,
                    "error",
                    "channels",
                    f"{ch}.packed.particle_diameter",
                    f"{ch} particle_diameter must be > 0.",
                )
            if not (0.05 <= packed["bed_porosity"] <= 0.95):
                _add_issue(
                    issues,
                    "error",
                    "channels",
                    f"{ch}.packed.bed_porosity",
                    f"{ch} bed_porosity must be within [0.05, 0.95].",
                )
            if packed["k_solid"] <= 0:
                _add_issue(
                    issues,
                    "error",
                    "channels",
                    f"{ch}.packed.k_solid",
                    f"{ch} k_solid must be > 0.",
                )
            if packed["shape_factor"] <= 0:
                _add_issue(
                    issues,
                    "error",
                    "channels",
                    f"{ch}.packed.shape_factor",
                    f"{ch} shape_factor must be > 0.",
                )

    if int(solver["n_elements"]) < 5:
        _add_issue(
            issues, "error", "solver", "n_elements", "n_elements must be >= 5."
        )
    if int(solver["max_iter"]) < 1:
        _add_issue(issues, "error", "solver", "max_iter", "max_iter must be >= 1.")
    if solver["tolerance"] <= 0:
        _add_issue(
            issues, "error", "solver", "tolerance", "tolerance must be > 0."
        )
    for key in ("relax_thermal", "relax_hydraulic", "relax_kinetics", "Q_damping"):
        if not (0.01 <= solver[key] <= 1.0):
            _add_issue(
                issues,
                "error",
                "solver",
                key,
                f"{key} must be within [0.01, 1.0].",
            )

    for key in ("results_csv", "convergence_csv", "performance_plot", "convergence_plot"):
        if not str(output[key]).strip():
            _add_issue(
                issues, "error", "output", key, f"{key} path cannot be empty."
            )

    return issues


def build_solver_config(state):
    cfg = create_default_config()
    cfg["geometry"].update(copy.deepcopy(state["geometry"]))
    cfg["operating"].update(copy.deepcopy(state["operating"]))
    cfg["channels"]["hot"] = copy.deepcopy(state["channels"]["hot"])
    cfg["channels"]["cold"] = copy.deepcopy(state["channels"]["cold"])
    cfg["solver"].update(copy.deepcopy(state["solver"]))
    cfg["output"].update(copy.deepcopy(state["output"]))

    # Keep legacy catalyst synced for compatibility code paths.
    cfg["catalyst"].update(copy.deepcopy(state["channels"]["hot"]["packed"]))
    return cfg


def render_channel_summary_strip(state):
    hot = state["channels"]["hot"]
    cold = state["channels"]["cold"]
    c1, c2 = st.columns(2)
    c1.info(
        f"Hot: {hot['mode']} / {hot['structure']} / packed-mode {hot['packed']['mode']}"
    )
    c2.info(
        f"Cold: {cold['mode']} / {cold['structure']} / packed-mode {cold['packed']['mode']}"
    )


def _k(name):
    return f"{name}_v{st.session_state.ui_version}"


def render_step_geometry(state):
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
            if hot_structure != "SmoothPlateFin":
                geo_h["unit_cell_size"] = st.number_input(
                    "Hot unit cell size [m]", min_value=1e-6,
                    value=float(geo_h.get("unit_cell_size") or state["geometry"]["unit_cell_size"]),
                    format="%.6f", key=_k("ch_hot_cell"),
                )
            else:
                geo_h["unit_cell_size"] = None
            geo_h["wall_thickness"] = st.number_input(
                "Hot skeleton thickness [m]", min_value=1e-7,
                value=float(geo_h.get("wall_thickness") or state["geometry"]["wall_thickness"]),
                format="%.6f", key=_k("ch_hot_tw"),
                help="TPMS skeleton/fin thickness for the hot channel.",
            )
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
            if cold_structure != "SmoothPlateFin":
                geo_c["unit_cell_size"] = st.number_input(
                    "Cold unit cell size [m]", min_value=1e-6,
                    value=float(geo_c.get("unit_cell_size") or state["geometry"]["unit_cell_size"]),
                    format="%.6f", key=_k("ch_cold_cell"),
                )
            else:
                geo_c["unit_cell_size"] = None
            geo_c["wall_thickness"] = st.number_input(
                "Cold skeleton thickness [m]", min_value=1e-7,
                value=float(geo_c.get("wall_thickness") or state["geometry"]["wall_thickness"]),
                format="%.6f", key=_k("ch_cold_tw"),
                help="TPMS skeleton/fin thickness for the cold channel.",
            )
    else:
        # Identical: clear per-channel overrides so solver uses global
        for sk in ("hot", "cold"):
            state["channels"][sk]["geometry"] = {
                "length": None, "width": None, "height": None,
                "unit_cell_size": None, "wall_thickness": None,
            }

    st.divider()
    render_step_channels(state)


# =============================================================================
# TPMS Geometry Estimator — helper functions and UI
# =============================================================================

def _eval_tpms_normalized(tpms_type: str, Xa, Ya, Za):
    """Evaluate TPMS implicit function with normalized coordinates (already divided by a).
    Returns an array with the same shape as the inputs. Fluid domain: {f < t_iso}."""
    import numpy as np
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
    import numpy as np
    return _eval_tpms_normalized(tpms_type, X / a, Y / a, Z / a)


@st.cache_data(show_spinner=False)
def _find_iso_value(tpms_type: str, porosity: float, grid_n: int = 40):
    """Find iso-value t such that volume fraction of {f < t} equals porosity.
    Works in normalized [0, 2π] space — independent of unit_cell_size."""
    import numpy as np
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
    import numpy as np
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
    import numpy as np
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
    if tpms_type == "SmoothPlateFin":
        sad = 2.0 / ch_H
        method = "Analytical (2/H)"
        warn = None
        st.info(
            f"SmoothPlateFin baseline: SAD = 2/H = **{sad:.1f} 1/m** "
            f"(two flat walls, channel height H = {ch_H*1e3:.1f} mm). "
            "Set surface_area_density to this value for a fair PEC comparison."
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
    import numpy as np
    import matplotlib.pyplot as plt

    MC_TYPES = ("Gyroid", "Diamond", "Primitive")

    def get_profile(struct, eps):
        if struct == "SmoothPlateFin":
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
    import numpy as np
    import matplotlib.pyplot as plt

    MC_TYPES = ("Gyroid", "Diamond", "Primitive")
    color = "#d62728" if sk == "hot" else "#1f77b4"
    label = f"{sk.capitalize()} ({struct})"

    if struct == "SmoothPlateFin":
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
    eps    = float(geo[f"porosity_{sk}"])
    struct = state["channels"][sk]["structure"]
    GRID_N   = 80
    N_SLICES = 50

    with st.expander(f"TPMS Geometry Estimator — {sk.capitalize()} channel", expanded=False):
        st.caption(
            f"**{struct}**  ·  ε = {eps:.2f}  ·  "
            f"Unit cell L = {L_cell*1e3:.3f} mm  ·  "
            f"Channel: {ch_L*1e3:.1f} \u00d7 {ch_W*1e3:.1f} \u00d7 {ch_H*1e3:.1f} mm"
        )
        st.markdown("##### Surface Area Density")
        _render_surface_area_tab(state, struct, L_cell, ch_L, ch_W, ch_H, eps, sk, GRID_N)
        st.markdown("##### Fluid Cross-Section vs. Axial Position")
        _render_cross_section_plot_single(struct, L_cell, ch_W, ch_H, eps, N_SLICES, sk)


def render_step_operating(state):
    st.subheader("Operating Conditions")
    c1, c2 = st.columns(2)
    with c1:
        hot_options = ["hydrogen mixture", "normal hydrogen", "helium", "argon"]
        idx_hot = hot_options.index(state["operating"]["fluid_hot"]) if state["operating"]["fluid_hot"] in hot_options else 0
        state["operating"]["fluid_hot"] = st.selectbox(
            "Hot fluid", options=hot_options, index=idx_hot, key=_k("op_hot_fluid")
        )
        state["operating"]["Th_in"] = st.number_input(
            "Th_in [K]",
            min_value=1.0,
            value=float(state["operating"]["Th_in"]),
            key=_k("op_Th"),
        )
        state["operating"]["Ph_in"] = st.number_input(
            "Ph_in [Pa]",
            min_value=1.0,
            value=float(state["operating"]["Ph_in"]),
            format="%.1f",
            key=_k("op_Ph"),
        )
        state["operating"]["mh"] = st.number_input(
            "m_hot [kg/s]",
            min_value=1e-6,
            value=float(state["operating"]["mh"]),
            format="%.6f",
            key=_k("op_mh"),
        )

    with c2:
        cold_options = ["helium", "argon", "hydrogen mixture", "normal hydrogen"]
        idx_cold = cold_options.index(state["operating"]["fluid_cold"]) if state["operating"]["fluid_cold"] in cold_options else 0
        state["operating"]["fluid_cold"] = st.selectbox(
            "Cold fluid",
            options=cold_options,
            index=idx_cold,
            key=_k("op_cold_fluid"),
        )
        state["operating"]["Tc_in"] = st.number_input(
            "Tc_in [K]",
            min_value=1.0,
            value=float(state["operating"]["Tc_in"]),
            key=_k("op_Tc"),
        )
        state["operating"]["Pc_in"] = st.number_input(
            "Pc_in [Pa]",
            min_value=1.0,
            value=float(state["operating"]["Pc_in"]),
            format="%.1f",
            key=_k("op_Pc"),
        )
        state["operating"]["mc"] = st.number_input(
            "m_cold [kg/s]",
            min_value=1e-6,
            value=float(state["operating"]["mc"]),
            format="%.6f",
            key=_k("op_mc"),
        )

    state["operating"]["xh_in"] = st.slider(
        "Hot inlet para fraction xh_in [-]",
        min_value=0.0,
        max_value=1.0,
        value=float(state["operating"]["xh_in"]),
        step=0.001,
        key=_k("op_xh"),
    )
    state["operating"]["T_ambient"] = st.number_input(
        "Ambient (dead-state) temperature T₀ [K]",
        min_value=200.0,
        max_value=400.0,
        value=float(state["operating"].get("T_ambient", 298.0)),
        format="%.1f",
        key=_k("op_T0"),
        help="Reference temperature for exergy analysis. Use ambient (~298 K) for "
             "liquefaction applications. Both streams are sub-ambient, so cold exergy "
             "is measured relative to this value.",
    )


def _render_htc_model_equations(htc_model: str):
    """Show governing equations for the selected packed-bed HTC model."""
    if htc_model == 'dixon':
        with st.expander("Dixon model equations (Eqs 0.1 \u2013 0.6)", expanded=False):
            st.markdown("**Overall tube-side heat transfer coefficient [109]:**")
            st.latex(r"\frac{1}{h_i} = \frac{1}{h_w} + \frac{d_i}{6k_r}\,\frac{\mathrm{Bi}+3}{\mathrm{Bi}+4}")
            st.markdown("**Biot number:**")
            st.latex(r"\mathrm{Bi} = \frac{h_w\,d_i}{2k_r}")
            st.markdown("**Effective radial conductivity [110, 112]:**")
            st.latex(
                r"k_r = \lambda_h\!\left(\frac{\lambda_s}{\lambda_h}\right)^{\!0.28"
                r" - 0.757\log\varepsilon - 0.057\log\!\frac{\lambda_s}{\lambda_h}}"
                r" + \frac{\lambda_h}{\mathrm{Pe}_r}\,\mathrm{Re}\,\mathrm{Pr}"
            )
            st.markdown("**Wall heat transfer coefficient [111]:**")
            st.latex(r"h_w = \frac{\mathrm{Nu}_w\,\lambda_h}{d_p}")
            st.latex(
                r"\mathrm{Nu}_w = \mathrm{Nu}_{w,0} + 110.3\,\mathrm{Pr}^{1/3}\,\mathrm{Re}^{0.75}"
                r" + 10.054\,\mathrm{Re}\,\mathrm{Pr} \quad (\mathrm{Nu}_{w,0}=20)"
            )
            st.markdown("**Radial Peclet number [112]:**")
            st.latex(r"\mathrm{Pe}_r = \frac{1}{0.11 + \dfrac{20.64}{\mathrm{Re}}}")
            st.caption(
                r"$d_i$ = TPMS hydraulic diameter $D_h$; "
                r"$d_p$ = particle diameter; "
                r"$\lambda_h$ = fluid conductivity; "
                r"$\lambda_s$ = solid conductivity; "
                r"$\varepsilon$ = bed porosity; "
                r"$\mathrm{Re} = \rho u d_p / \mu$."
            )
    else:  # martin_nilles
        with st.expander("Martin-Nilles / ZBS model equations", expanded=False):
            st.markdown("**Overall heat transfer coefficient (two-resistance model):**")
            st.latex(
                r"\frac{1}{h_\mathrm{eff}} = \frac{1}{h_w}"
                r" + \frac{D_h}{C_\mathrm{shape}\,k_{r,\mathrm{eff}}}"
            )
            st.markdown("**Wall HTC \u2014 Martin-Nilles correlation:**")
            st.latex(
                r"\mathrm{Nu}_w = \left(1.3 + \frac{5}{D_h/d_p}\right)\frac{k_{r,\mathrm{eff}}}{k_f}"
                r" + 0.19\,\mathrm{Re}^{0.75}\,\mathrm{Pr}^{1/3}"
            )
            st.markdown("**Stagnant effective conductivity \u2014 ZBS model:**")
            st.latex(
                r"k_{r,\mathrm{eff}}^{0} = \bigl(1-\sqrt{1-\varepsilon}\bigr)\,k_f"
                r" + \sqrt{1-\varepsilon}\,k_\mathrm{cell}"
            )
            st.markdown("**Dispersion contribution (Wen-Fan):**")
            st.latex(r"k_\mathrm{disp} = 0.1\,\mathrm{Re}\,\mathrm{Pr}\,k_f")
            st.caption(
                r"$C_\mathrm{shape}$: geometric factor (lower = 8, nominal = 6, upper = 4). "
                "TPMS fin area enhancement applied on top."
            )


def _render_tpms_bare_equations(tpms_type: str):
    """Show Nu and f correlations for the selected TPMS structure (bare channel)."""
    corr_map = {
        'Gyroid': {
            'gas': (r"\mathrm{Nu} = 0.3250\,\mathrm{Re}^{0.7002}\,\mathrm{Pr}^{0.36}", r"f = 2.5\,\mathrm{Re}^{-0.2}", "Re: 2000–8170 (Gas/Air)"),
            'water': (r"\mathrm{Nu} = 0.471\,\mathrm{Re}^{0.627}\,\mathrm{Pr}^{1/3}", r"f = 2.577\,\mathrm{Re}^{-0.095}", "Re: 150–3000 (Water)"),
        },
        'Diamond': {
            'gas': (r"\mathrm{Nu} = 0.409\,\mathrm{Re}^{0.625}\,\mathrm{Pr}^{0.4}", r"f = 2.5892\,\mathrm{Re}^{-0.1940}", "Re: 800–9590 (Gas)"),
            'water': (r"\mathrm{Nu} = 0.12504\,\mathrm{Re}^{0.73143}\,\mathrm{Pr}^{1/3}", r"f = 2.74632\,\mathrm{Re}^{-0.36099}", "Re: 80–1500 (Water)"),
        },
        'Primitive': {
            'gas': (r"\mathrm{Nu} = 0.1\,\mathrm{Re}^{0.75}\,\mathrm{Pr}^{0.36}", r"f = 4.0\,\mathrm{Re}^{-0.25}", "Gas (estimated)"),
            'water': (r"\mathrm{Nu} = 0.05513\,\mathrm{Re}^{0.81370}\,\mathrm{Pr}^{1/3}", r"f = 3.96709\,\mathrm{Re}^{-0.23326}", "Re: 80–1500 (Water)"),
        },
        'Neovius': {
            'gas': (r"\mathrm{Nu} = 0.15\,\mathrm{Re}^{0.7}\,\mathrm{Pr}^{0.36}", r"f = 5.0\,\mathrm{Re}^{-0.3}", "Gas (estimated)"),
            'water': (r"\mathrm{Nu} = 2.48\,\mathrm{Re}^{0.45}", r"f = 59.2\,\mathrm{Re}^{-0.63}", "Re: 10–75 (Water)"),
        },
        'FRD': {
            'gas': (r"\mathrm{Nu} = 0.3\,\mathrm{Re}^{0.65}\,\mathrm{Pr}^{0.36}", r"f = 3.0\,\mathrm{Re}^{-0.2}", "Gas (estimated)"),
            'water': (r"\mathrm{Nu} = 1.74\,\mathrm{Re}^{0.54}", r"f = 11.5\,\mathrm{Re}^{-0.41}", "Re: 35–290 (Water)"),
        },
        'FKS': {
            'gas': (r"\mathrm{Nu} = 0.52\,\mathrm{Re}^{0.61}\,\mathrm{Pr}^{0.4}", r"f = 2.1335\,\mathrm{Re}^{-0.1334}", "Re: 730–10230 (Gas)"),
            'water': (r"\mathrm{Nu} = 3.02\,\mathrm{Re}^{0.40}", r"f = 25.0\,\mathrm{Re}^{-0.73}", "Re: 10–140 (Water)"),
        },
        'SmoothPlateFin': {
            'gas': (r"\mathrm{Nu} = 0.023\,\mathrm{Re}^{0.8}\,\mathrm{Pr}^{0.4}", r"f = \frac{(0.790\ln\mathrm{Re}-1.64)^{-2}}{4}", "Re > 2300 (Dittus-Boelter / Petukhov-Filonenko); laminar: Nu=3.66, f=16/Re"),
            'water': (r"\mathrm{Nu} = 0.023\,\mathrm{Re}^{0.8}\,\mathrm{Pr}^{0.4}", r"f = \frac{(0.790\ln\mathrm{Re}-1.64)^{-2}}{4}", "Re > 2300 (same correlation)"),
        },
    }
    corr = corr_map.get(tpms_type)
    if not corr:
        return
    title = ("SmoothPlateFin baseline correlations (Nu & f)"
             if tpms_type == "SmoothPlateFin"
             else f"{tpms_type} bare-channel correlations (Nu & f)")
    with st.expander(title, expanded=False):
        st.markdown(f"**Gas / Cryogenic** ({corr['gas'][2]}):")
        st.latex(corr['gas'][0])
        st.latex(corr['gas'][1])
        st.markdown(f"**Water** ({corr['water'][2]}):")
        st.latex(corr['water'][0])
        st.latex(corr['water'][1])
        st.markdown("**HTC from Nu:**")
        st.latex(r"h = \mathrm{Nu}\,k_f / D_h")
        if tpms_type == "SmoothPlateFin":
            st.caption(
                r"$f$ = Fanning friction factor; "
                r"$D_h = 4 A_c / P$ (rectangular duct); "
                r"SAD = $2/H$ (two flat walls); "
                r"$\mathrm{Re} = \rho u D_h / \mu$."
            )
        else:
            st.caption(
                r"$f$ = Fanning friction factor; "
                r"$D_h = 4\varepsilon L_\mathrm{cell}/(2\pi)$; "
                r"$\mathrm{Re} = \rho u D_h / \mu$."
            )


def _render_ergun_equations():
    """Show Ergun pressure drop and equivalent friction factor for packed bed."""
    with st.expander("Ergun pressure drop equations", expanded=False):
        st.markdown("**Ergun equation:**")
        st.latex(
            r"\frac{\Delta P}{L} = \frac{150\,\mu\,u_s\,(1-\varepsilon_b)^2}{\varepsilon_b^3\,d_p^2}"
            r" + \frac{1.75\,\rho\,u_s^2\,(1-\varepsilon_b)}{\varepsilon_b^3\,d_p}"
        )
        st.markdown("**Equivalent Fanning friction factor:**")
        st.latex(
            r"f_\mathrm{equiv} = \frac{D_h}{d_p}\,\frac{1-\varepsilon_b}{\varepsilon_b^3}"
            r"\!\left(\frac{300\,(1-\varepsilon_b)}{\mathrm{Re}_p} + 3.5\right)"
            r"\cdot \psi_\mathrm{TPMS}"
        )
        st.markdown("**TPMS pressure correction factors** $\\psi$:")
        st.markdown(
            "| Structure | ψ |\n"
            "|-----------|-----|\n"
            "| Gyroid | 1.15 |\n"
            "| Diamond | 1.20 |\n"
            "| Primitive | 1.10 |\n"
            "| Neovius | 1.30 |\n"
            "| FRD | 1.18 |\n"
            "| FKS | 1.12 |"
        )
        st.caption(
            r"$u_s$ = superficial velocity; $\varepsilon_b$ = bed porosity; "
            r"$d_p$ = particle diameter; $\mathrm{Re}_p = \rho u_s d_p / \mu$."
        )


def render_channel_card(channel_name, channel_state, state):
    st.markdown(f"#### {channel_name.capitalize()} Channel")
    mode_options = ["bare", "packed"]
    mode_idx = mode_options.index(channel_state["mode"]) if channel_state["mode"] in mode_options else 0
    channel_state["mode"] = st.selectbox(
        f"{channel_name} mode",
        options=mode_options,
        index=mode_idx,
        key=_k(f"{channel_name}_mode"),
    )
    structures = list(TPMSCorrelations.get_supported_tpms_types())
    structure_idx = structures.index(channel_state["structure"]) if channel_state["structure"] in structures else 0
    channel_state["structure"] = st.selectbox(
        f"{channel_name} TPMS structure",
        options=structures,
        index=structure_idx,
        key=_k(f"{channel_name}_structure"),
    )
    channel_state["surface_area_density"] = st.number_input(
        f"{channel_name} surface area density α [1/m]",
        min_value=1.0,
        value=float(channel_state.get("surface_area_density", 60.0)),
        help="Specific wetted area of this channel's TPMS structure. "
             "Use the Geometry Estimator in Step 1 to estimate this value.",
        key=_k(f"{channel_name}_sad"),
    )
    if channel_state["surface_area_density"] > 200:
        st.warning(
            f"**High SAD detected ({channel_state['surface_area_density']:.0f} 1/m).** "
            "A very large surface area density produces a stiff solver (high NTU), which "
            "can cause the heat-flux residual (dQ) to oscillate and prevent convergence. "
            "The solver will auto-reduce Q_damping, but if convergence fails you can also "
            "manually lower **relax_thermal** (< 0.10) and **Q_damping** (< 0.1) in the "
            "Solver Settings panel."
        )

    if channel_state["mode"] == "bare":
        _render_tpms_bare_equations(channel_state["structure"])

    if channel_state["mode"] == "packed":
        packed = channel_state["packed"]
        mode_idx = list(SUPPORTED_PACKED_MODES).index(packed["mode"]) if packed["mode"] in SUPPORTED_PACKED_MODES else 1
        packed["mode"] = st.selectbox(
            f"{channel_name} packed mode",
            options=list(SUPPORTED_PACKED_MODES),
            index=mode_idx,
            key=_k(f"{channel_name}_packed_mode"),
        )
        _HTC_MODELS = ('martin_nilles', 'dixon')
        _htc_default = packed.get("htc_model", "martin_nilles")
        _htc_idx = _HTC_MODELS.index(_htc_default) if _htc_default in _HTC_MODELS else 0
        packed["htc_model"] = st.selectbox(
            f"{channel_name} HTC model",
            options=list(_HTC_MODELS),
            index=_htc_idx,
            key=_k(f"{channel_name}_htc_model"),
        )
        _render_htc_model_equations(packed["htc_model"])
        _render_ergun_equations()
        packed["particle_diameter"] = st.number_input(
            f"{channel_name} particle diameter [m]",
            min_value=1e-6,
            value=float(packed["particle_diameter"]),
            format="%.6f",
            key=_k(f"{channel_name}_dp"),
        )
        packed["bed_porosity"] = st.slider(
            f"{channel_name} bed porosity [-]",
            min_value=0.05,
            max_value=0.95,
            value=float(packed["bed_porosity"]),
            step=0.01,
            key=_k(f"{channel_name}_bed_por"),
        )
        packed["k_solid"] = st.number_input(
            f"{channel_name} solid conductivity [W/m-K]",
            min_value=0.01,
            value=float(packed["k_solid"]),
            format="%.3f",
            key=_k(f"{channel_name}_ks"),
        )
        packed["shape_factor"] = st.slider(
            f"{channel_name} shape factor [-]",
            min_value=0.01,
            max_value=5.0,
            value=float(packed["shape_factor"]),
            step=0.01,
            key=_k(f"{channel_name}_shape"),
        )

    _render_channel_estimator(state, channel_name)


def render_step_channels(state):
    st.subheader("Channel Modeling")
    c1, c2 = st.columns(2)
    with c1:
        render_channel_card("hot", state["channels"]["hot"], state)
    with c2:
        render_channel_card("cold", state["channels"]["cold"], state)


def render_step_solver(state):
    st.subheader("Solver Controls")
    c1, c2 = st.columns(2)
    with c1:
        state["solver"]["n_elements"] = int(
            st.number_input(
                "n_elements",
                min_value=5,
                max_value=5000,
                value=int(state["solver"]["n_elements"]),
                key=_k("solver_n"),
            )
        )
        state["solver"]["max_iter"] = int(
            st.number_input(
                "max_iter",
                min_value=1,
                max_value=20000,
                value=int(state["solver"]["max_iter"]),
                key=_k("solver_max_iter"),
            )
        )
        state["solver"]["tolerance"] = st.number_input(
            "tolerance",
            min_value=1e-12,
            max_value=1.0,
            value=float(state["solver"]["tolerance"]),
            format="%.8f",
            key=_k("solver_tol"),
        )
    with c2:
        state["solver"]["relax_thermal"] = st.slider(
            "relax_thermal",
            min_value=0.01,
            max_value=1.0,
            value=float(state["solver"]["relax_thermal"]),
            step=0.01,
            key=_k("solver_rt"),
        )
        state["solver"]["relax_hydraulic"] = st.slider(
            "relax_hydraulic",
            min_value=0.01,
            max_value=1.0,
            value=float(state["solver"]["relax_hydraulic"]),
            step=0.01,
            key=_k("solver_rh"),
        )
        state["solver"]["relax_kinetics"] = st.slider(
            "relax_kinetics",
            min_value=0.01,
            max_value=1.0,
            value=float(state["solver"]["relax_kinetics"]),
            step=0.01,
            key=_k("solver_rk"),
        )
        state["solver"]["Q_damping"] = st.slider(
            "Q_damping",
            min_value=0.01,
            max_value=1.0,
            value=float(state["solver"]["Q_damping"]),
            step=0.01,
            key=_k("solver_qd"),
        )
    state["solver"]["relax"] = state["solver"]["relax_thermal"]


def render_step_output(state):
    st.subheader("Output Paths")
    state["output"]["results_csv"] = st.text_input(
        "results_csv", value=state["output"]["results_csv"], key=_k("out_results")
    )
    state["output"]["convergence_csv"] = st.text_input(
        "convergence_csv",
        value=state["output"]["convergence_csv"],
        key=_k("out_conv_csv"),
    )
    state["output"]["performance_plot"] = st.text_input(
        "performance_plot",
        value=state["output"]["performance_plot"],
        key=_k("out_perf"),
    )
    state["output"]["convergence_plot"] = st.text_input(
        "convergence_plot",
        value=state["output"]["convergence_plot"],
        key=_k("out_conv_plot"),
    )

def _summary_df(rows):
    return pd.DataFrame([(p, str(v)) for p, v in rows], columns=["Parameter", "Value"])


def render_summary_table(state, issues):
    st.subheader("Configuration Summary")
    st.markdown("Review all settings before execution.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Geometry**")
        st.table(
            _summary_df(
                [
                    ("length [m]", state["geometry"]["length"]),
                    ("width [m]", state["geometry"]["width"]),
                    ("height [m]", state["geometry"]["height"]),
                    ("unit_cell_size [m]", state["geometry"]["unit_cell_size"]),
                    ("wall_thickness [m]", state["geometry"]["wall_thickness"]),
                    ("α_hot [1/m]", state["channels"]["hot"].get("surface_area_density", state["geometry"]["surface_area_density"])),
                    ("α_cold [1/m]", state["channels"]["cold"].get("surface_area_density", state["geometry"]["surface_area_density"])),
                    ("porosity_hot", state["geometry"]["porosity_hot"]),
                    ("porosity_cold", state["geometry"]["porosity_cold"]),
                ]
            )
        )
        st.markdown("**Operating**")
        st.table(
            _summary_df(
                [
                    ("fluid_hot", state["operating"]["fluid_hot"]),
                    ("fluid_cold", state["operating"]["fluid_cold"]),
                    ("Th_in [K]", state["operating"]["Th_in"]),
                    ("Tc_in [K]", state["operating"]["Tc_in"]),
                    ("Ph_in [Pa]", state["operating"]["Ph_in"]),
                    ("Pc_in [Pa]", state["operating"]["Pc_in"]),
                    ("mh [kg/s]", state["operating"]["mh"]),
                    ("mc [kg/s]", state["operating"]["mc"]),
                    ("xh_in", state["operating"]["xh_in"]),
                ]
            )
        )

    with c2:
        for ch in ("hot", "cold"):
            ch_cfg = state["channels"][ch]
            st.markdown(f"**{ch.capitalize()} Channel**")
            rows = [
                ("mode", ch_cfg["mode"]),
                ("structure", ch_cfg["structure"]),
                ("packed.mode", ch_cfg["packed"]["mode"]),
            ]
            if ch_cfg["mode"] == "packed":
                rows.extend(
                    [
                        ("packed.particle_diameter [m]", ch_cfg["packed"]["particle_diameter"]),
                        ("packed.bed_porosity", ch_cfg["packed"]["bed_porosity"]),
                        ("packed.k_solid [W/m-K]", ch_cfg["packed"]["k_solid"]),
                        ("packed.shape_factor", ch_cfg["packed"]["shape_factor"]),
                    ]
                )
            st.table(_summary_df(rows))

        st.markdown("**Solver**")
        st.table(
            _summary_df(
                [
                    ("n_elements", state["solver"]["n_elements"]),
                    ("max_iter", state["solver"]["max_iter"]),
                    ("tolerance", state["solver"]["tolerance"]),
                    ("relax_thermal", state["solver"]["relax_thermal"]),
                    ("relax_hydraulic", state["solver"]["relax_hydraulic"]),
                    ("relax_kinetics", state["solver"]["relax_kinetics"]),
                    ("Q_damping", state["solver"]["Q_damping"]),
                ]
            )
        )
        st.markdown("**Output**")
        st.table(
            _summary_df(
                [
                    ("results_csv", state["output"]["results_csv"]),
                    ("convergence_csv", state["output"]["convergence_csv"]),
                    ("performance_plot", state["output"]["performance_plot"]),
                    ("convergence_plot", state["output"]["convergence_plot"]),
                ]
            )
        )

    with st.expander("Effective Config JSON"):
        st.json(build_solver_config(state))

    error_count = sum(1 for x in issues if x["level"] == "error")
    warning_count = sum(1 for x in issues if x["level"] == "warning")
    info_count = sum(1 for x in issues if x["level"] == "info")
    st.caption(
        f"Validation snapshot: {error_count} errors, {warning_count} warnings, {info_count} info"
    )


def render_validation_panel(issues, visible_sections):
    relevant = [x for x in issues if x["section"] in visible_sections]
    errors = [x for x in relevant if x["level"] == "error"]
    warnings_ = [x for x in relevant if x["level"] == "warning"]
    infos = [x for x in relevant if x["level"] == "info"]

    st.markdown("### Validation")
    if not errors and not warnings_ and not infos:
        st.success("No validation issues in this step.")
        return

    if errors:
        st.error(f"{len(errors)} blocking error(s) detected.")
        for item in errors:
            st.write(f"- `{item['section']}.{item['field']}`: {item['message']}")
    if warnings_:
        st.warning(f"{len(warnings_)} warning(s).")
        for item in warnings_:
            st.write(f"- `{item['section']}.{item['field']}`: {item['message']}")
    if infos:
        st.info(f"{len(infos)} info message(s).")
        for item in infos:
            st.write(f"- `{item['section']}.{item['field']}`: {item['message']}")


def has_blocking_issues(issues, sections):
    return any(x["level"] == "error" and x["section"] in sections for x in issues)


def run_simulation(state):
    cfg = build_solver_config(state)
    try:
        he = TPMSHeatExchanger(cfg)
        converged = he.solve(
            max_iter=cfg["solver"]["max_iter"],
            tolerance=cfg["solver"]["tolerance"],
        )
    except Exception as exc:
        return {"error": str(exc)}

    result = {
        "error": None,
        "converged": converged,
        "q_total": float(he.Q.sum()),
        "hot_out": float(he.Th[-1]),
        "cold_out": float(he.Tc[0]),
        "eta_ex":     he.perf.get('eta_ex',      None),
        "S_gen":      he.perf.get('S_gen_total',  None),
        "PEC_mean_h": he.perf.get('PEC_mean_h',  None),
        "PEC_mean_c": he.perf.get('PEC_mean_c',  None),
        "output": copy.deepcopy(cfg["output"]),
    }
    try:
        he.finalize_simulation()
        result["output"] = copy.deepcopy(he.config["output"])
    except Exception as exc:
        result["warning"] = f"Outputs partially failed: {exc}"
    return result


def render_run_result():
    result = st.session_state.get("run_result")
    if not result:
        st.info("No results yet. Complete the setup on pages 1–5 and press **Run Simulation**.")
        return

    last_run = st.session_state.get("last_run_at")
    if last_run:
        st.caption(f"Last run: {last_run}")

    st.divider()
    st.subheader("Latest Run Result")
    if result.get("error"):
        st.error(f"Run failed: {result['error']}")
        return

    st.success("Simulation finished")
    if result.get("warning"):
        st.warning(result["warning"])
    st.write(f"Converged: `{result['converged']}`")
    c1, c2, c3 = st.columns(3)
    c1.metric("Q_total [W]", f"{result['q_total']:.2f}")
    c2.metric("Hot outlet [K]", f"{result['hot_out']:.2f}")
    c3.metric("Cold outlet [K]", f"{result['cold_out']:.2f}")

    # Performance evaluation indicators row
    eta_ex = result.get('eta_ex')
    s_gen  = result.get('S_gen')
    pec_h  = result.get('PEC_mean_h')
    pec_c  = result.get('PEC_mean_c')
    if any(v is not None for v in (eta_ex, s_gen, pec_h, pec_c)):
        c4, c5, c6, c7 = st.columns(4)
        c4.metric("η_ex [%]",    f"{eta_ex*100:.1f}"  if eta_ex is not None else "—")
        c5.metric("S_gen [mW/K]", f"{s_gen*1e3:.3f}"  if s_gen  is not None else "—")
        c6.metric("PEC hot",     f"{pec_h:.4f}"        if pec_h  is not None else "—")
        c7.metric("PEC cold",    f"{pec_c:.4f}"        if pec_c  is not None else "—")

    out = result["output"]
    st.markdown("**Output Files**")
    for key in ("results_csv", "convergence_csv", "performance_plot", "convergence_plot",
                "resistance_pie", "performance_eval_plot"):
        p = Path(out.get(key, ""))
        if str(p):
            st.write(f"- `{p}` {'(found)' if p.exists() else '(missing)'}")

    for _img_key, _caption in [
        ("performance_plot",     "Performance Profile"),
        ("resistance_pie",       "Thermal Resistance Breakdown"),
        ("performance_eval_plot","Performance Evaluation — Exergy & PEC"),
        ("convergence_plot",     "Convergence Diagnostics"),
    ]:
        _p = out.get(_img_key, "")
        if _p and os.path.exists(_p):
            with open(_p, "rb") as _f:
                st.image(_f.read(), caption=_caption)
    if os.path.exists(out["results_csv"]):
        st.subheader("Results Preview")
        try:
            st.dataframe(pd.read_csv(out["results_csv"]).head(20))
        except Exception as exc:
            st.warning(f"Could not load CSV preview: {exc}")


def render_nav_bar(current: int, position: str = "top"):
    """Render a row of step-pill buttons. Active step is highlighted; Results pill
    is disabled until a simulation result exists.

    Args:
        current: Index of the active step.
        position: 'top' or 'bottom' — used to generate unique widget keys.
    """
    cols = st.columns(len(STEP_DEFS))
    for i, (col, sdef) in enumerate(zip(cols, STEP_DEFS)):
        is_active = (i == current)
        is_disabled = is_active or (
            sdef["key"] == "results" and not st.session_state.get("run_result")
        )
        if col.button(
            sdef["title"],
            key=f"navpill_{position}_{i}",
            type="primary" if is_active else "secondary",
            disabled=is_disabled,
            use_container_width=True,
        ):
            st.session_state.current_step = i
            st.rerun()


def main():
    st.set_page_config(page_title="TPMS HE Controller", layout="wide")
    init_ui_state()

    st.title("TPMS Heat Exchanger Wizard")
    st.caption("Step-by-step setup with autosave, validation, and channel-model visibility")

    for msg in st.session_state.get("init_messages", []):
        st.warning(msg)
    st.session_state.init_messages = []

    c_top1, c_top2, c_top3 = st.columns([2, 1, 1])
    c_top1.caption(f"Autosave file: `{AUTOSAVE_PATH}`")
    c_top1.caption(f"Last autosave: {st.session_state.last_saved_at}")
    if c_top2.button("Reload Autosave"):
        reload_autosave()
    if c_top3.button("Reset Defaults"):
        reset_to_defaults()

    state = st.session_state.ui_state
    render_channel_summary_strip(state)

    current = st.session_state.current_step
    step_def = STEP_DEFS[current]
    render_nav_bar(current, position="top")
    st.subheader(step_def["title"])

    if step_def["key"] == "geometry":
        render_step_geometry(state)
    elif step_def["key"] == "operating":
        render_step_operating(state)
    elif step_def["key"] == "solver":
        render_step_solver(state)
    elif step_def["key"] == "output":
        render_step_output(state)
    elif step_def["key"] == "results":
        render_run_result()

    maybe_autosave()
    issues = validate_ui_state(state)

    if step_def["key"] == "confirm":
        render_summary_table(state, issues)

    render_validation_panel(issues, step_def["sections"])

    has_error_current = has_blocking_issues(issues, step_def["sections"])
    has_error_global = has_blocking_issues(
        issues, ["geometry", "operating", "channels", "solver", "output"]
    )

    render_nav_bar(current, position="bottom")
    st.divider()
    _, c_nav1, c_nav2, _ = st.columns([2, 1, 1, 2])
    if current > 0 and c_nav1.button("← Back"):
        st.session_state.current_step -= 1
        st.rerun()
    if step_def["key"] == "confirm":
        if c_nav2.button("Run Simulation", type="primary", disabled=has_error_global):
            with st.spinner("Running solver..."):
                st.session_state.run_result = run_simulation(state)
            st.session_state.last_run_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            maybe_autosave(force=True)
            st.session_state.current_step = len(STEP_DEFS) - 1   # jump to Results page
            st.rerun()
    elif step_def["key"] != "results":
        if c_nav2.button("Next →", disabled=has_error_current):
            st.session_state.current_step += 1
            st.rerun()



if __name__ == "__main__":
    main()
