import pytest

from models.packed_bed import PackedBedTPMSModel


def _build_model(hydraulic_model="psi_legacy", ht_enhancement_model="off"):
    catalyst = {
        "particle_diameter": 1.0e-3,
        "bed_porosity": 0.40,
        "k_solid": 10.0,
        "shape_factor": 1.0,
        "hydraulic_model": hydraulic_model,
        "phi_source": "ch3_f_re_fit",
        "ht_enhancement_model": ht_enhancement_model,
        "ht_nominal_rule": "geometric",
    }
    geometry = {
        "D_h": 3.3e-3,
        "wall_thickness": 0.5e-3,
        "k_wall": 237.0,
        "Afin_Abase_ratio": 1.2,
    }
    return PackedBedTPMSModel(catalyst, geometry)


def test_phi_fit_baseline_and_tpms_gain():
    model = _build_model(hydraulic_model="phi_re_fit")
    assert model.hydraulic_enhancement_phi("Plate", 1000.0) == pytest.approx(1.0)
    assert model.hydraulic_enhancement_phi("SmoothPlateFin", 1000.0) == pytest.approx(1.0)
    assert model.hydraulic_enhancement_phi("Diamond", 1000.0) > 1.0


def test_hydraulic_mode_switch_affects_friction():
    re_ch = 1200.0
    pr = 0.8
    k_f = 0.1

    model_psi = _build_model(hydraulic_model="psi_legacy")
    model_phi = _build_model(hydraulic_model="phi_re_fit")

    _, f_psi, d_psi = model_psi.get_htc_and_friction(re_ch, pr, k_f, tpms_type="Diamond")
    _, f_phi, d_phi = model_phi.get_htc_and_friction(re_ch, pr, k_f, tpms_type="Diamond")

    assert d_psi["hydraulic_model"] == "psi_legacy"
    assert d_phi["hydraulic_model"] == "phi_re_fit"
    assert f_psi == pytest.approx(d_psi["f_ergun_base"] * d_psi["correction_factor"])
    assert f_phi == pytest.approx(d_phi["f_ergun_base"] * d_phi["correction_factor"])
    assert f_phi > f_psi


def test_ht_enhancement_from_phi_interval():
    re_ch = 1200.0
    pr = 0.8
    k_f = 0.1

    model = _build_model(hydraulic_model="phi_re_fit", ht_enhancement_model="from_phi")

    h_l, _, d_l = model.get_htc_and_friction(re_ch, pr, k_f, tpms_type="Diamond", mode="lower")
    h_n, _, d_n = model.get_htc_and_friction(re_ch, pr, k_f, tpms_type="Diamond", mode="nominal")
    h_u, _, d_u = model.get_htc_and_friction(re_ch, pr, k_f, tpms_type="Diamond", mode="upper")

    phi = d_u["phi"]
    assert d_l["enh_used"] == pytest.approx(1.0)
    assert d_n["enh_used"] == pytest.approx(phi ** 0.5)
    assert d_u["enh_used"] == pytest.approx(phi)
    assert d_l["h_w"] < d_n["h_w"] < d_u["h_w"]
