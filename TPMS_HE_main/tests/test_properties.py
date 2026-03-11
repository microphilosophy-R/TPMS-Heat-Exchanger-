"""
Tests for properties.ThermalProperties.

Run from TPMS_HE_main/:
    pytest tests/test_properties.py -v
"""
import numpy as np
import pytest

from properties.hydrogen_properties import ThermalProperties


@pytest.fixture(scope="module")
def props():
    return ThermalProperties()


# ── Equilibrium fraction in [0, 1] across cryogenic range ─────────────────────

@pytest.mark.parametrize("T_K", [20, 40, 80, 150, 200, 300])
def test_x_eq_in_unit_interval(T_K):
    x = ThermalProperties.get_equilibrium_fraction(T_K)
    assert 0.0 <= x <= 1.0, f"x_eq({T_K} K) = {x:.4f} out of [0, 1]"


def test_x_eq_300K_near_025():
    x = ThermalProperties.get_equilibrium_fraction(300.0)
    assert abs(x - 0.25) < 0.05, f"x_eq(300 K) = {x:.4f}, expected ≈ 0.25"


def test_x_eq_20K_near_1():
    x = ThermalProperties.get_equilibrium_fraction(20.0)
    assert x > 0.90, f"x_eq(20 K) = {x:.4f}, expected > 0.9 (mostly para)"


# ── Helium properties at 50 K, 2 MPa ──────────────────────────────────────────

def test_helium_density_physical(props):
    result = props.get_properties(50.0, 2e6, species='helium')
    rho = result["rho"]
    assert 5.0 <= rho <= 100.0, f"He rho = {rho:.2f} kg/m3 out of expected [5, 100]"
    assert np.isfinite(rho), "He density must be finite"


def test_helium_properties_finite(props):
    result = props.get_properties(50.0, 2e6, species='helium')
    for key in ("cp", "mu", "lambda"):
        val = result[key]
        assert np.isfinite(val) and val > 0, f"He {key}={val} must be finite and positive"


# ── Hydrogen mixture properties are finite ────────────────────────────────────

def test_hydrogen_properties_finite(props):
    result = props.get_properties(80.0, 2e6, species='hydrogen mixture', x_para=0.45)
    assert np.isfinite(result["rho"]) and result["rho"] > 0, \
        f"H2 rho={result['rho']}"
    assert np.isfinite(result["cp"]) and result["cp"] > 0, \
        f"H2 cp={result['cp']}"


def test_hydrogen_requires_x_para(props):
    with pytest.raises(ValueError):
        props.get_properties(80.0, 2e6, species='hydrogen mixture')
