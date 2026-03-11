"""ui.state -- UI state management: init, load, save, sanitize, presets."""

import copy
import json
import os
from datetime import datetime
from pathlib import Path

import streamlit as st

from solver.config import create_default_config, create_cpfhx_config
from correlations.thermohydraulic_correlations import ThermoHydraulicCorrelations
from models.packed_bed import SUPPORTED_PACKED_MODES

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


def sanitize_ui_state(state, defaults):
    merged = _deep_merge(defaults, state)
    notices = []
    supported_tpms = set(ThermoHydraulicCorrelations.get_supported_tpms_types())
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


