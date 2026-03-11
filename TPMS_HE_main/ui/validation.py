"""ui.validation -- Input validation and solver config builder for the TPMS HE Wizard."""

import copy

import streamlit as st

from correlations.thermohydraulic_correlations import ThermoHydraulicCorrelations
from models.packed_bed import SUPPORTED_PACKED_MODES
from solver.config import create_default_config

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

    supported_tpms = set(ThermoHydraulicCorrelations.get_supported_tpms_types())
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

