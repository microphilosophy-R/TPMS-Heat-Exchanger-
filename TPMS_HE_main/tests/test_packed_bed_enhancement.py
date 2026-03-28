import pytest

from models.packed_bed import PackedBedTPMSModel


def _build_model(
    hydraulic_model="ergun_psi_tpms",
    ht_enhancement_model="off",
    ht_nominal_rule="geometric",
):
    catalyst = {
        "particle_diameter": 1.0e-3,
        "bed_porosity": 0.40,
        "k_solid": 10.0,
        "shape_factor": 1.0,
        "hydraulic_model": hydraulic_model,
        "phi_source": "ch3_f_re_fit",
        "ht_enhancement_model": ht_enhancement_model,
        "ht_nominal_rule": ht_nominal_rule,
    }
    geometry = {
        "D_h": 3.3e-3,
        "wall_thickness": 0.5e-3,
        "k_wall": 237.0,
        "Afin_Abase_ratio": 1.2,
    }
    return PackedBedTPMSModel(catalyst, geometry)


def test_phi_fit_baseline_and_tpms_gain():
    model = _build_model(hydraulic_model="ergun_phi_fit")
    assert model.hydraulic_enhancement_phi("Plate", 1000.0) == pytest.approx(1.0)
    assert model.hydraulic_enhancement_phi("SmoothPlateFin", 1000.0) == pytest.approx(1.0)
    assert model.hydraulic_enhancement_phi("Diamond", 1000.0) > 1.0


def test_hydraulic_mode_switch_affects_friction():
    re_ch = 1200.0
    pr = 0.8
    k_f = 0.1

    model_psi = _build_model(hydraulic_model="ergun_psi_tpms")
    model_phi = _build_model(hydraulic_model="ergun_phi_fit")

    _, f_psi, d_psi = model_psi.get_htc_and_friction(re_ch, pr, k_f, tpms_type="Diamond")
    _, f_phi, d_phi = model_phi.get_htc_and_friction(re_ch, pr, k_f, tpms_type="Diamond")

    assert d_psi["hydraulic_model"] == "ergun_psi_tpms"
    assert d_phi["hydraulic_model"] == "ergun_phi_fit"
    assert f_psi == pytest.approx(d_psi["f_ergun_base"] * d_psi["correction_factor"])
    assert f_phi == pytest.approx(d_phi["f_ergun_base"] * d_phi["correction_factor"])
    assert f_phi > f_psi


def test_ht_enhancement_from_phi_interval():
    re_ch = 1200.0
    pr = 0.8
    k_f = 0.1

    model = _build_model(hydraulic_model="ergun_phi_fit", ht_enhancement_model="wall_from_phi")

    h_l, _, d_l = model.get_htc_and_friction(re_ch, pr, k_f, tpms_type="Diamond", mode="lower")
    h_n, _, d_n = model.get_htc_and_friction(re_ch, pr, k_f, tpms_type="Diamond", mode="nominal")
    h_u, _, d_u = model.get_htc_and_friction(re_ch, pr, k_f, tpms_type="Diamond", mode="upper")

    phi = d_u["phi"]
    assert d_l["enh_used"] == pytest.approx(1.0)
    assert d_n["enh_used"] == pytest.approx(phi ** 0.5)
    assert d_u["enh_used"] == pytest.approx(phi)
    assert d_l["h_w"] < d_n["h_w"] < d_u["h_w"]


def test_ht_enhancement_from_phi_similarity_rule():
    re_ch = 1200.0
    pr = 0.8
    k_f = 0.1

    model = _build_model(
        hydraulic_model="ergun_phi_fit",
        ht_enhancement_model="wall_from_phi",
        ht_nominal_rule="similarity",
    )

    _, _, d_n = model.get_htc_and_friction(re_ch, pr, k_f, tpms_type="Diamond", mode="nominal")
    _, _, d_u = model.get_htc_and_friction(re_ch, pr, k_f, tpms_type="Diamond", mode="upper")

    phi = d_u["phi"]
    assert d_n["enh_nominal"] == pytest.approx(phi)
    assert d_n["enh_used"] == pytest.approx(phi)
    assert d_u["enh_used"] == pytest.approx(phi)


def test_wall_from_phi_changes_wall_terms_but_not_martin_bed_conductivity():
    re_ch = 1200.0
    pr = 0.8
    k_f = 0.1

    model_off = _build_model(hydraulic_model="ergun_phi_fit", ht_enhancement_model="off")
    model_on = _build_model(hydraulic_model="ergun_phi_fit", ht_enhancement_model="wall_from_phi")

    _, _, d_off = model_off.get_htc_and_friction(re_ch, pr, k_f, tpms_type="Diamond", htc_model="martin_nilles")
    _, _, d_on = model_on.get_htc_and_friction(re_ch, pr, k_f, tpms_type="Diamond", htc_model="martin_nilles")

    assert d_on["h_w"] > d_off["h_w"]
    assert d_on["Nu_w"] > d_off["Nu_w"]
    assert d_on["k_r_eff"] == pytest.approx(d_off["k_r_eff"])
    assert d_on["R_bed_conduction"] == pytest.approx(d_off["R_bed_conduction"])
    assert d_on["eta_fin"] == pytest.approx(d_off["eta_fin"])
    assert d_on["area_factor"] == pytest.approx(d_off["area_factor"])


def test_wall_from_phi_keeps_dixon_k_r_fixed_but_changes_bi():
    re_ch = 1200.0
    pr = 0.8
    k_f = 0.1

    model_off = _build_model(hydraulic_model="ergun_phi_fit", ht_enhancement_model="off")
    model_on = _build_model(hydraulic_model="ergun_phi_fit", ht_enhancement_model="wall_from_phi")

    _, _, d_off = model_off.get_htc_and_friction(re_ch, pr, k_f, tpms_type="Diamond", htc_model="dixon")
    _, _, d_on = model_on.get_htc_and_friction(re_ch, pr, k_f, tpms_type="Diamond", htc_model="dixon")

    assert d_on["k_r"] == pytest.approx(d_off["k_r"])
    assert d_on["Bi"] != pytest.approx(d_off["Bi"])
    assert d_on["R_total"] != pytest.approx(d_off["R_total"])
    assert d_on["area_factor"] == pytest.approx(d_off["area_factor"])


def test_area_factor_is_static_across_uncertainty_modes():
    re_ch = 1200.0
    pr = 0.8
    k_f = 0.1

    model = _build_model(hydraulic_model="ergun_phi_fit", ht_enhancement_model="wall_from_phi")

    _, _, d_l = model.get_htc_and_friction(re_ch, pr, k_f, tpms_type="Diamond", mode="lower")
    _, _, d_n = model.get_htc_and_friction(re_ch, pr, k_f, tpms_type="Diamond", mode="nominal")
    _, _, d_u = model.get_htc_and_friction(re_ch, pr, k_f, tpms_type="Diamond", mode="upper")

    assert d_l["area_factor"] == pytest.approx(d_n["area_factor"])
    assert d_n["area_factor"] == pytest.approx(d_u["area_factor"])


def test_wang_wall_htc_uses_wang_wall_but_martin_bed_framework():
    re_ch = 1200.0
    pr = 0.8
    k_f = 0.1

    model = _build_model()
    _, _, details = model.get_htc_and_friction(
        re_ch,
        pr,
        k_f,
        tpms_type="Diamond",
        htc_model="wang_wall_htc",
    )

    assert details["htc_model"] == "wang_wall_htc"
    assert details["wall_htc_source"] == "wang_wall_htc"
    assert details["bed_conduction_source"] == "martin_nilles"
    assert "k_r_eff" in details
