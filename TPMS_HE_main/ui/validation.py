"""ui.validation -- Input validation and solver config builder for the TPMS HE Wizard."""

import copy

import streamlit as st

from correlations.thermohydraulic_correlations import ThermoHydraulicCorrelations
from models.packed_closures import (
    SUPPORTED_PACKED_MODES,
    SUPPORTED_HYDRAULIC_MODELS,
    SUPPORTED_PACKED_HEAT_TRANSFER_MODELS,
    SUPPORTED_WALL_ENHANCEMENT_MODELS,
    SUPPORTED_KINETIC_MODELS,
    get_allowed_channel_modes,
    get_allowed_hydraulic_models,
    get_allowed_packed_heat_transfer_models,
    get_allowed_wall_enhancement_models,
    get_allowed_kinetic_models,
)
from solver.config import create_default_config, normalize_config

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

    for key in ("length", "width"):
        if geo[key] <= 0:
            _add_issue(issues, "error", "geometry", key, f"{key} must be > 0.")
    # surface_area_density is per-channel; global key may not exist
    global_sad = geo.get("surface_area_density")
    if global_sad is not None and global_sad <= 0:
        _add_issue(
            issues,
            "error",
            "geometry",
            "surface_area_density",
            "surface_area_density must be > 0.",
        )
    for ch in ("hot", "cold"):
        ch_geo = channels[ch].get("geometry", {}) or {}
        eps = ch_geo.get("porosity")
        if eps is not None and not (0.05 <= eps <= 0.95):
            _add_issue(
                issues,
                "error",
                "geometry",
                f"porosity_{ch}",
                f"{ch} porosity must be within [0.05, 0.95].",
            )
    # Per-channel geometry validation
    for ch in ("hot", "cold"):
        ch_geo = channels[ch].get("geometry", {}) or {}
        for dim in ("height", "unit_cell_size", "wall_thickness"):
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

    supported_tpms = set(ThermoHydraulicCorrelations.get_supported_tpms_types())
    for ch in ("hot", "cold"):
        ch_cfg = channels[ch]
        allowed_modes = get_allowed_channel_modes(ch_cfg["structure"])
        if ch_cfg["mode"] not in ("bare", "packed"):
            _add_issue(
                issues,
                "error",
                "channels",
                f"{ch}.mode",
                f"{ch} mode must be 'bare' or 'packed'.",
            )
        elif ch_cfg["mode"] not in allowed_modes:
            _add_issue(
                issues,
                "error",
                "channels",
                f"{ch}.mode",
                f"{ch} structure {ch_cfg['structure']} only allows modes {allowed_modes}.",
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
            allowed_hydraulic = get_allowed_hydraulic_models(ch_cfg["structure"])
            allowed_htc_models = get_allowed_packed_heat_transfer_models(ch_cfg["structure"])
            allowed_enhancement = get_allowed_wall_enhancement_models(ch_cfg["structure"])
            allowed_kinetics = get_allowed_kinetic_models(ch_cfg["structure"])
            uncertainty_mode = packed.get("uncertainty_mode", packed.get("mode"))
            if uncertainty_mode not in SUPPORTED_PACKED_MODES:
                _add_issue(
                    issues,
                    "error",
                    "channels",
                    f"{ch}.packed.uncertainty_mode",
                    f"{ch} packed uncertainty_mode must be one of {SUPPORTED_PACKED_MODES}.",
                )
            if packed.get("hydraulic_model") not in SUPPORTED_HYDRAULIC_MODELS:
                _add_issue(
                    issues,
                    "error",
                    "channels",
                    f"{ch}.packed.hydraulic_model",
                    f"{ch} hydraulic_model must be one of {SUPPORTED_HYDRAULIC_MODELS}.",
                )
            elif packed.get("hydraulic_model") not in allowed_hydraulic:
                _add_issue(
                    issues,
                    "error",
                    "channels",
                    f"{ch}.packed.hydraulic_model",
                    f"{ch} structure {ch_cfg['structure']} only allows hydraulic models {allowed_hydraulic}.",
                )
            if packed.get("htc_model") not in SUPPORTED_PACKED_HEAT_TRANSFER_MODELS:
                _add_issue(
                    issues,
                    "error",
                    "channels",
                    f"{ch}.packed.htc_model",
                    f"{ch} htc_model must be one of {SUPPORTED_PACKED_HEAT_TRANSFER_MODELS}.",
                )
            elif packed.get("htc_model") not in allowed_htc_models:
                _add_issue(
                    issues,
                    "error",
                    "channels",
                    f"{ch}.packed.htc_model",
                    f"{ch} structure {ch_cfg['structure']} only allows HTC models {allowed_htc_models}.",
                )
            if packed.get("ht_enhancement_model") not in SUPPORTED_WALL_ENHANCEMENT_MODELS:
                _add_issue(
                    issues,
                    "error",
                    "channels",
                    f"{ch}.packed.ht_enhancement_model",
                    f"{ch} ht_enhancement_model must be one of {SUPPORTED_WALL_ENHANCEMENT_MODELS}.",
                )
            elif packed.get("ht_enhancement_model") not in allowed_enhancement:
                _add_issue(
                    issues,
                    "error",
                    "channels",
                    f"{ch}.packed.ht_enhancement_model",
                    f"{ch} structure {ch_cfg['structure']} only allows wall enhancement models {allowed_enhancement}.",
                )
            if packed.get("kinetic_model") not in SUPPORTED_KINETIC_MODELS:
                _add_issue(
                    issues,
                    "error",
                    "channels",
                    f"{ch}.packed.kinetic_model",
                    f"{ch} kinetic_model must be one of {SUPPORTED_KINETIC_MODELS}.",
                )
            elif packed.get("kinetic_model") not in allowed_kinetics:
                _add_issue(
                    issues,
                    "error",
                    "channels",
                    f"{ch}.packed.kinetic_model",
                    f"{ch} structure {ch_cfg['structure']} only allows kinetic models {allowed_kinetics}.",
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

    for ch in ("hot", "cold"):
        cfg["channels"][ch]["packed"].pop("mode", None)

    # Keep legacy catalyst synced for compatibility code paths.
    cfg["catalyst"].update(copy.deepcopy(state["channels"]["hot"]["packed"]))
    cfg["catalyst"].pop("mode", None)
    normalized = normalize_config(cfg)
    for ch in ("hot", "cold"):
        normalized["channels"][ch]["packed"].pop("mode", None)
    normalized["catalyst"].pop("mode", None)
    return normalized

