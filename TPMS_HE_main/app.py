
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
    {"key": "geometry", "title": "1. Geometry", "sections": ["geometry"]},
    {"key": "operating", "title": "2. Operating", "sections": ["operating"]},
    {"key": "channels", "title": "3. Channels", "sections": ["channels"]},
    {"key": "solver", "title": "4. Solver", "sections": ["solver"]},
    {"key": "output", "title": "5. Output", "sections": ["output"]},
    {
        "key": "confirm",
        "title": "6. Confirm & Run",
        "sections": ["geometry", "operating", "channels", "solver", "output"],
    },
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
            "Wall thickness [m]",
            min_value=1e-6,
            value=float(state["geometry"]["wall_thickness"]),
            format="%.6f",
            key=_k("geom_wall"),
        )
        state["geometry"]["surface_area_density"] = st.number_input(
            "Surface area density [1/m]",
            min_value=1.0,
            value=float(state["geometry"]["surface_area_density"]),
            key=_k("geom_sad"),
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


def render_channel_card(channel_name, channel_state):
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

    if channel_state["mode"] == "packed":
        packed = channel_state["packed"]
        mode_idx = list(SUPPORTED_PACKED_MODES).index(packed["mode"]) if packed["mode"] in SUPPORTED_PACKED_MODES else 1
        packed["mode"] = st.selectbox(
            f"{channel_name} packed mode",
            options=list(SUPPORTED_PACKED_MODES),
            index=mode_idx,
            key=_k(f"{channel_name}_packed_mode"),
        )
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


def render_step_channels(state):
    st.subheader("Channel Modeling")
    c1, c2 = st.columns(2)
    with c1:
        render_channel_card("hot", state["channels"]["hot"])
    with c2:
        render_channel_card("cold", state["channels"]["cold"])


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
    return pd.DataFrame(rows, columns=["Parameter", "Value"])


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
                    ("surface_area_density [1/m]", state["geometry"]["surface_area_density"]),
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
    st.caption(f"Validation snapshot: {error_count} errors, {warning_count} warnings")


def render_validation_panel(issues, visible_sections):
    relevant = [x for x in issues if x["section"] in visible_sections]
    errors = [x for x in relevant if x["level"] == "error"]
    warnings_ = [x for x in relevant if x["level"] == "warning"]

    st.markdown("### Validation")
    if not errors and not warnings_:
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
        he.finalize_simulation()
        return {
            "error": None,
            "converged": converged,
            "q_total": float(he.Q.sum()),
            "hot_out": float(he.Th[-1]),
            "cold_out": float(he.Tc[0]),
            "output": copy.deepcopy(cfg["output"]),
        }
    except Exception as exc:
        return {"error": str(exc)}


def render_run_result():
    result = st.session_state.get("run_result")
    if not result:
        return

    st.divider()
    st.subheader("Latest Run Result")
    if result.get("error"):
        st.error(f"Run failed: {result['error']}")
        return

    st.success("Simulation finished")
    st.write(f"Converged: `{result['converged']}`")
    c1, c2, c3 = st.columns(3)
    c1.metric("Q_total [W]", f"{result['q_total']:.2f}")
    c2.metric("Hot outlet [K]", f"{result['hot_out']:.2f}")
    c3.metric("Cold outlet [K]", f"{result['cold_out']:.2f}")

    out = result["output"]
    st.markdown("**Output Files**")
    for key in ("results_csv", "convergence_csv", "performance_plot", "convergence_plot"):
        p = Path(out[key])
        st.write(f"- `{p}` {'(found)' if p.exists() else '(missing)'}")

    if os.path.exists(out["performance_plot"]):
        st.image(out["performance_plot"], caption="Performance Profile")
    if os.path.exists(out["convergence_plot"]):
        st.image(out["convergence_plot"], caption="Convergence Diagnostics")
    if os.path.exists(out["results_csv"]):
        st.subheader("Results Preview")
        try:
            st.dataframe(pd.read_csv(out["results_csv"]).head(20))
        except Exception as exc:
            st.warning(f"Could not load CSV preview: {exc}")


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
    progress = (current + 1) / len(STEP_DEFS)
    st.progress(progress)
    st.subheader(step_def["title"])

    if step_def["key"] == "geometry":
        render_step_geometry(state)
    elif step_def["key"] == "operating":
        render_step_operating(state)
    elif step_def["key"] == "channels":
        render_step_channels(state)
    elif step_def["key"] == "solver":
        render_step_solver(state)
    elif step_def["key"] == "output":
        render_step_output(state)

    maybe_autosave()
    issues = validate_ui_state(state)

    if step_def["key"] == "confirm":
        render_summary_table(state, issues)

    render_validation_panel(issues, step_def["sections"])

    has_error_current = has_blocking_issues(issues, step_def["sections"])
    has_error_global = has_blocking_issues(
        issues, ["geometry", "operating", "channels", "solver", "output"]
    )

    st.divider()
    c_nav1, c_nav2, _ = st.columns([1, 1, 2])
    if current > 0 and c_nav1.button("Back"):
        st.session_state.current_step -= 1
        st.rerun()
    if current < len(STEP_DEFS) - 1:
        if c_nav2.button("Next", disabled=has_error_current):
            st.session_state.current_step += 1
            st.rerun()
    else:
        if c_nav2.button("Run Simulation", type="primary", disabled=has_error_global):
            with st.spinner("Running solver..."):
                st.session_state.run_result = run_simulation(state)
            maybe_autosave(force=True)
            st.rerun()

    render_run_result()


if __name__ == "__main__":
    main()
