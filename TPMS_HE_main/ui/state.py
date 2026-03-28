"""ui.state -- UI state management: init, load, save, sanitize, presets."""

import copy
import json
import os
from datetime import datetime
from pathlib import Path

import streamlit as st

from solver.config import create_default_config, create_cpfhx_config
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
    normalize_hydraulic_model,
    normalize_packed_heat_transfer_model,
    normalize_wall_enhancement_model,
    normalize_kinetic_model,
)

APP_DIR = Path(__file__).resolve().parent.parent
AUTOSAVE_PATH = APP_DIR / ".streamlit" / "tpms_ui_state.json"

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


def _migrate_legacy_state(state):
    """Migrate old geometry schema (global porosity_hot/cold, etc.) to per-channel geometry."""
    geo = state.get("geometry", {})
    channels = state.get("channels", {})

    _field_map = {
        "hot":  {"porosity": geo.get("porosity_hot",  0.65)},
        "cold": {"porosity": geo.get("porosity_cold", 0.70)},
    }
    _shared_fields = {
        "unit_cell_size": geo.get("unit_cell_size",  5e-3),
        "wall_thickness": geo.get("wall_thickness",  5e-4),
        "height":         geo.get("height",          0.25),
        "fin_height":     geo.get("fin_height",      9.5e-3),
        "fin_spacing":    geo.get("fin_spacing",      3.2e-3),
        "fin_thickness":  geo.get("fin_thickness",    0.6e-3),
        "perf_density":   geo.get("perf_density",     0.0),
        "perf_radius":    geo.get("perf_radius",      0.0),
        "length":         geo.get("length",           0.94),
        "width":          geo.get("width",            0.25),
    }

    for sk in ("hot", "cold"):
        ch = channels.setdefault(sk, {})
        ch_geo = ch.setdefault("geometry", {})
        # Migrate shared fields
        for field, default in _shared_fields.items():
            if ch_geo.get(field) is None:
                ch_geo[field] = default
        # Migrate channel-specific porosity
        if ch_geo.get("porosity") is None:
            ch_geo["porosity"] = _field_map[sk]["porosity"]

    return state


def sanitize_ui_state(state, defaults):
    state = _migrate_legacy_state(state)
    merged = _deep_merge(defaults, state)
    notices = []
    supported_tpms = set(ThermoHydraulicCorrelations.get_supported_tpms_types())
    supported_modes = {"bare", "packed"}
    supported_packed_modes = set(SUPPORTED_PACKED_MODES)
    supported_hydraulic = set(SUPPORTED_HYDRAULIC_MODELS)
    supported_htc = set(SUPPORTED_PACKED_HEAT_TRANSFER_MODELS)
    supported_enhancement = set(SUPPORTED_WALL_ENHANCEMENT_MODELS)
    supported_kinetics = set(SUPPORTED_KINETIC_MODELS)

    for ch in ("hot", "cold"):
        ch_cfg = merged["channels"][ch]
        if ch_cfg["mode"] not in supported_modes:
            ch_cfg["mode"] = defaults["channels"][ch]["mode"]
            notices.append(f"{ch} channel mode reset to default.")
        if ch_cfg["structure"] not in supported_tpms:
            ch_cfg["structure"] = defaults["channels"][ch]["structure"]
            notices.append(f"{ch} TPMS structure reset to default.")
        allowed_modes = get_allowed_channel_modes(ch_cfg["structure"])
        if ch_cfg["mode"] not in allowed_modes:
            ch_cfg["mode"] = allowed_modes[0]
            notices.append(
                f"{ch} channel mode adjusted for structure {ch_cfg['structure']}."
            )
        packed_cfg = ch_cfg["packed"]
        if "uncertainty_mode" not in packed_cfg and "mode" in packed_cfg:
            packed_cfg["uncertainty_mode"] = packed_cfg["mode"]
            notices.append(f"{ch} packed.mode migrated to packed.uncertainty_mode.")
        packed_cfg.pop("mode", None)
        if packed_cfg.get("uncertainty_mode") not in supported_packed_modes:
            packed_cfg["uncertainty_mode"] = defaults["channels"][ch]["packed"]["uncertainty_mode"]
            notices.append(f"{ch} packed uncertainty mode reset to default.")

        allowed_hydraulic = set(get_allowed_hydraulic_models(ch_cfg["structure"]))
        allowed_htc_models = set(get_allowed_packed_heat_transfer_models(ch_cfg["structure"]))
        allowed_enhancement = set(get_allowed_wall_enhancement_models(ch_cfg["structure"]))
        allowed_kinetics = set(get_allowed_kinetic_models(ch_cfg["structure"]))

        try:
            packed_cfg["hydraulic_model"] = normalize_hydraulic_model(
                packed_cfg.get("hydraulic_model", defaults["channels"][ch]["packed"]["hydraulic_model"])
            )
        except ValueError:
            packed_cfg["hydraulic_model"] = defaults["channels"][ch]["packed"]["hydraulic_model"]
            notices.append(f"{ch} hydraulic model reset to default.")
        if packed_cfg["hydraulic_model"] not in supported_hydraulic or (
            allowed_hydraulic and packed_cfg["hydraulic_model"] not in allowed_hydraulic
        ):
            packed_cfg["hydraulic_model"] = defaults["channels"][ch]["packed"]["hydraulic_model"]
            if allowed_hydraulic and packed_cfg["hydraulic_model"] not in allowed_hydraulic:
                packed_cfg["hydraulic_model"] = next(iter(allowed_hydraulic))
                notices.append(f"{ch} hydraulic model adjusted for structure {ch_cfg['structure']}.")

        try:
            packed_cfg["htc_model"] = normalize_packed_heat_transfer_model(
                packed_cfg.get("htc_model", defaults["channels"][ch]["packed"]["htc_model"])
            )
        except ValueError:
            packed_cfg["htc_model"] = defaults["channels"][ch]["packed"]["htc_model"]
            notices.append(f"{ch} HTC model reset to default.")
        if packed_cfg["htc_model"] not in supported_htc or (
            allowed_htc_models and packed_cfg["htc_model"] not in allowed_htc_models
        ):
            packed_cfg["htc_model"] = defaults["channels"][ch]["packed"]["htc_model"]
            if allowed_htc_models and packed_cfg["htc_model"] not in allowed_htc_models:
                packed_cfg["htc_model"] = next(iter(allowed_htc_models))
                notices.append(f"{ch} HTC model adjusted for structure {ch_cfg['structure']}.")

        try:
            packed_cfg["ht_enhancement_model"] = normalize_wall_enhancement_model(
                packed_cfg.get(
                    "ht_enhancement_model",
                    defaults["channels"][ch]["packed"]["ht_enhancement_model"],
                )
            )
        except ValueError:
            packed_cfg["ht_enhancement_model"] = defaults["channels"][ch]["packed"]["ht_enhancement_model"]
            notices.append(f"{ch} wall enhancement model reset to default.")
        if packed_cfg["ht_enhancement_model"] not in supported_enhancement or (
            allowed_enhancement and packed_cfg["ht_enhancement_model"] not in allowed_enhancement
        ):
            packed_cfg["ht_enhancement_model"] = defaults["channels"][ch]["packed"]["ht_enhancement_model"]
            if allowed_enhancement and packed_cfg["ht_enhancement_model"] not in allowed_enhancement:
                packed_cfg["ht_enhancement_model"] = next(iter(allowed_enhancement))
                notices.append(f"{ch} wall enhancement adjusted for structure {ch_cfg['structure']}.")

        try:
            packed_cfg["kinetic_model"] = normalize_kinetic_model(
                packed_cfg.get("kinetic_model", defaults["channels"][ch]["packed"]["kinetic_model"])
            )
        except ValueError:
            packed_cfg["kinetic_model"] = defaults["channels"][ch]["packed"]["kinetic_model"]
            notices.append(f"{ch} kinetic model reset to default.")
        if packed_cfg["kinetic_model"] not in supported_kinetics or (
            allowed_kinetics and packed_cfg["kinetic_model"] not in allowed_kinetics
        ):
            packed_cfg["kinetic_model"] = defaults["channels"][ch]["packed"]["kinetic_model"]
            if allowed_kinetics and packed_cfg["kinetic_model"] not in allowed_kinetics:
                packed_cfg["kinetic_model"] = next(iter(allowed_kinetics))
                notices.append(f"{ch} kinetic model adjusted for structure {ch_cfg['structure']}.")

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


def load_cpfhx_preset(back_pressure_mpa, flowrate_ratio):
    """Load the CPFHX test configuration (Wang et al. 2024) into UI state."""
    cfg = create_cpfhx_config(back_pressure_mpa, flowrate_ratio)
    defaults = _extract_ui_state(create_default_config())
    preset_state = _extract_ui_state(cfg)
    merged, _ = sanitize_ui_state(preset_state, defaults)
    st.session_state.ui_state = merged
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


