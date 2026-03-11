"""
Tests for solver.config — default config and CPFHX preset builder.

Run from TPMS_HE_main/:
    pytest tests/test_config.py -v
"""
import math
import pytest

from solver.config import (
    create_default_config,
    create_cpfhx_config,
    normalize_config,
    _CPFHX_TABLE6,
)


# ── create_default_config() structure ─────────────────────────────────────────

REQUIRED_TOP_LEVEL_KEYS = {
    "geometry", "tpms", "channels", "material",
    "operating", "solver", "output",
}

def test_default_config_has_required_keys():
    cfg = create_default_config()
    missing = REQUIRED_TOP_LEVEL_KEYS - cfg.keys()
    assert not missing, f"Missing top-level keys: {missing}"


def test_default_config_geometry_keys():
    geo = create_default_config()["geometry"]
    for key in ("length", "width", "height", "unit_cell_size",
                "porosity_hot", "porosity_cold", "wall_thickness"):
        assert key in geo, f"geometry missing '{key}'"


def test_default_config_has_both_channels():
    channels = create_default_config()["channels"]
    assert "hot" in channels and "cold" in channels


# ── Dh formula check (bare TPMS: Dh = 4·ε·a / 2π) ───────────────────────────

def test_dh_hot_formula():
    geo = create_default_config()["geometry"]
    porosity = geo["porosity_hot"]      # 0.65
    a        = geo["unit_cell_size"]    # 5e-3 m
    Dh_hot   = 4.0 * porosity * a / (2.0 * math.pi)
    assert abs(Dh_hot - 2.069e-3) < 5e-7, (
        f"Dh_hot = {Dh_hot:.5e} m, expected ≈ 2.069e-3 m"
    )


# ── normalize_config fills in all operating defaults ──────────────────────────

def test_normalize_fills_in_missing_Th_in():
    """normalize_config() uses setdefault to fill in all operating params, including Th_in."""
    cfg = create_default_config()
    del cfg["operating"]["Th_in"]
    result = normalize_config(cfg)
    assert "Th_in" in result["operating"]
    assert result["operating"]["Th_in"] == 78.0


# ── create_cpfhx_config(1.04, 2.7) ────────────────────────────────────────────

def test_cpfhx_config_temperatures():
    cfg = create_cpfhx_config(1.04, 2.7)
    assert abs(cfg["operating"]["Th_in"] - 63.1) < 0.5, \
        f"Th_in = {cfg['operating']['Th_in']}"
    assert abs(cfg["operating"]["Tc_in"] - 42.8) < 0.5, \
        f"Tc_in = {cfg['operating']['Tc_in']}"


def test_cpfhx_table6_entry():
    entry = _CPFHX_TABLE6[(1.04, 2.7)]
    assert entry["Th_in"] == 63.1
    assert entry["Tc_in"] == 42.8


# ── normalize_config is idempotent ────────────────────────────────────────────

def test_normalize_config_idempotent():
    cfg  = create_default_config()
    cfg1 = normalize_config(cfg)
    cfg2 = normalize_config(cfg1)
    # Key structure preserved on second call
    assert set(cfg1.keys()) == set(cfg2.keys())
