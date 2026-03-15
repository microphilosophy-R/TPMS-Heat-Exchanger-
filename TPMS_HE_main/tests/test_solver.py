"""
Integration tests for solver.TPMSHeatExchanger.

Each test constructs a configuration, runs the solver with a reduced element
count (n_elements=10) for speed, and checks physical consistency of the results.

Run from TPMS_HE_main/:
    pytest tests/test_solver.py -v
"""
import numpy as np
import pytest

from solver.calculator import TPMSHeatExchanger
from solver.config import create_default_config, create_cpfhx_config


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fast_config(base_cfg, n_elements=10):
    """Override solver settings for fast (smoke-test) runs."""
    cfg = base_cfg.copy()
    cfg["solver"] = dict(base_cfg.get("solver", {}))
    cfg["solver"]["n_elements"] = n_elements
    cfg["solver"]["max_iter"]   = 300
    cfg["solver"]["tolerance"]  = 1e-3
    return cfg


# ── Default config smoke test ─────────────────────────────────────────────────

@pytest.fixture(scope="module")
def default_result():
    """Run the solver once with default config; reused across tests in this module."""
    cfg = _fast_config(create_default_config())
    hx  = TPMSHeatExchanger(cfg)
    converged = hx.solve()
    return hx, converged


def test_default_converges(default_result):
    _, converged = default_result
    assert converged, "Default config must converge"


def test_hot_stream_cools(default_result):
    hx, _ = default_result
    Th_in = hx.config["operating"]["Th_in"]
    assert hx.Th[-1] < Th_in, (
        f"Hot stream must cool: Th_out={hx.Th[-1]:.2f} K, Th_in={Th_in:.2f} K"
    )


def test_cold_stream_heats(default_result):
    hx, _ = default_result
    Tc_in = hx.config["operating"]["Tc_in"]
    assert hx.Tc[0] > Tc_in, (
        f"Cold stream must heat: Tc_out={hx.Tc[0]:.2f} K, Tc_in={Tc_in:.2f} K"
    )


def test_no_temperature_crossover(default_result):
    hx, _ = default_result
    assert np.all(hx.Th >= hx.Tc), "Temperature crossover detected (Th < Tc)"


def test_positive_heat_transfer(default_result):
    hx, _ = default_result
    assert np.sum(hx.Q) > 0, "Total heat transferred must be positive"


def test_positive_pressure_drop(default_result):
    hx, _ = default_result
    dP_hot = hx.Ph[0] - hx.Ph[-1]
    assert dP_hot > 0, f"Hot-side pressure drop must be positive, got {dP_hot:.1f} Pa"


def test_para_fraction_increases(default_result):
    """For default hydrogen hot stream (bare), xh should be flat (no conversion)."""
    hx, _ = default_result
    # Bare channel: kinetics off → xh is constant
    assert hx.xh[-1] >= hx.xh[0] - 1e-6, (
        f"Para-fraction must not decrease: xh[0]={hx.xh[0]:.4f}, xh[-1]={hx.xh[-1]:.4f}"
    )


# ── CPFHX regression: (1.04 MPa, r=2.7) ──────────────────────────────────────

@pytest.fixture(scope="module")
def cpfhx_result():
    cfg = _fast_config(create_cpfhx_config(1.04, 2.7))
    hx  = TPMSHeatExchanger(cfg)
    converged = hx.solve()
    return hx, converged


def test_cpfhx_converges(cpfhx_result):
    _, converged = cpfhx_result
    assert converged, "CPFHX (1.04 MPa, r=2.7) must converge"


def test_cpfhx_outlet_temperature_in_range(cpfhx_result):
    """Wang et al. (2024) Table 6 reports Th_out_exp ≈ 55.1 K for this condition."""
    hx, _ = cpfhx_result
    Th_out = hx.Th[-1]
    assert abs(Th_out - 55.1) < 12.0, (
        f"CPFHX Th_out = {Th_out:.2f} K, expected near 55.1 ± 12 K"
    )

def test_arrhenius_kinetics_mode_smoke():
    cfg = _fast_config(create_default_config(), n_elements=6)
    cfg["channels"]["hot"]["mode"] = "packed"
    cfg["channels"]["hot"]["packed"]["kinetic_model"] = "arrhenius_first_order"
    cfg["channels"]["hot"]["packed"]["hydraulic_model"] = "phi_re_fit"
    cfg["channels"]["hot"]["packed"]["ht_enhancement_model"] = "from_phi"

    hx = TPMSHeatExchanger(cfg)
    hx._update_stream_physics("hot")
    hx._update_stream_physics("cold")
    hx._ortho_para_conversion()

    assert np.all(np.isfinite(hx.dx_dt))
    assert np.all(hx.xh >= 0.0) and np.all(hx.xh <= 1.0)
