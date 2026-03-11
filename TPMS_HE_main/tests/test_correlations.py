"""
Tests for correlations.ThermoHydraulicCorrelations.

Run from TPMS_HE_main/:
    pytest tests/test_correlations.py -v
"""
import numpy as np
import pytest

from correlations.thermohydraulic_correlations import ThermoHydraulicCorrelations


# ── Smoke: all supported types present ────────────────────────────────────────

def test_supported_types_complete():
    expected = {'PlateFin', 'SmoothPlateFin', 'Gyroid', 'Diamond',
                'Primitive', 'Neovius', 'FRD', 'FKS'}
    assert expected == set(ThermoHydraulicCorrelations.SUPPORTED_TPMS_TYPES)


# ── PlateFin at Re=500, Pr=0.72 ───────────────────────────────────────────────

def test_plate_fin_nu_re500():
    Nu, _ = ThermoHydraulicCorrelations.get_correlations('PlateFin', 500.0, 0.72, fluid_type='Gas')
    assert abs(Nu - 7.51) < 0.5, f"Expected Nu ≈ 7.51, got {Nu:.4f}"


def test_plate_fin_f_re500():
    _, f = ThermoHydraulicCorrelations.get_correlations('PlateFin', 500.0, 0.72, fluid_type='Gas')
    assert abs(f - 0.042) < 0.005, f"Expected f ≈ 0.042, got {f:.5f}"


# ── Diamond Gas at Re=1500 ─────────────────────────────────────────────────────

def test_diamond_gas_nu_range():
    Nu, f = ThermoHydraulicCorrelations.get_correlations('Diamond', 1500.0, 0.72, fluid_type='Gas')
    assert 20 <= Nu <= 80, f"Nu={Nu:.2f} out of expected range [20, 80]"
    assert f > 0, f"Friction factor must be positive, got {f}"


# ── Array input / output ───────────────────────────────────────────────────────

def test_array_input_same_shape():
    Re_arr = np.array([500.0, 1000.0, 2000.0])
    Nu, f = ThermoHydraulicCorrelations.get_correlations('Gyroid', Re_arr, 0.72, fluid_type='Gas')
    assert Nu.shape == Re_arr.shape, "Nu shape mismatch"
    assert f.shape == Re_arr.shape, "f shape mismatch"


def test_scalar_returns_scalar():
    Nu, f = ThermoHydraulicCorrelations.get_correlations('Diamond', 1000.0, 0.72, fluid_type='Gas')
    assert np.ndim(Nu) == 0, "Scalar Re should return scalar Nu"
    assert np.ndim(f) == 0, "Scalar Re should return scalar f"


# ── Physical limits ────────────────────────────────────────────────────────────

def test_nu_positive_all_types():
    Re_vals = np.array([100.0, 500.0, 1000.0, 3000.0])
    for tpms_type in ThermoHydraulicCorrelations.SUPPORTED_TPMS_TYPES:
        Nu, f = ThermoHydraulicCorrelations.get_correlations(tpms_type, Re_vals, 0.72, fluid_type='Gas')
        assert np.all(Nu > 0), f"{tpms_type}: Nu must be > 0"
        assert np.all(f >= 0), f"{tpms_type}: f must be >= 0"


# ── Nu monotonically increasing with Re (Gas correlations) ────────────────────

@pytest.mark.parametrize("tpms_type", [
    'Diamond', 'Gyroid', 'PlateFin', 'FKS',
])
def test_nu_monotone_increasing_with_re(tpms_type):
    Re_vals = np.array([300.0, 800.0, 2000.0, 5000.0])
    Nu, _ = ThermoHydraulicCorrelations.get_correlations(tpms_type, Re_vals, 0.72, fluid_type='Gas')
    diffs = np.diff(Nu)
    assert np.all(diffs > 0), (
        f"{tpms_type}: Nu should increase monotonically with Re, got {Nu}"
    )
