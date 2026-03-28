import sys
from pathlib import Path

import numpy as np

from solver.calculator import TPMSHeatExchanger


ROOT = Path(__file__).resolve().parents[4]
ANALYSIS_DIR = ROOT / "code" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.append(str(ANALYSIS_DIR))

import module8_packed_catalyst_matrix as module8


def test_build_run_plan_matches_expected_row_count():
    run_plan = module8.build_run_plan()

    assert len(run_plan) == 7 * 5 * 6
    assert "SmoothPlateFin" not in {row["Structure"] for row in run_plan}


def test_platefin_and_smooth_geometry_branches_use_dedicated_formulas():
    plate_geometry = dict(module8.BASE_CHANNEL_GEOMETRY)
    plate_geometry.update(module8.PLATE_FIN_GEOMETRY)
    plate_geometry["porosity"] = module8.plate_fin_porosity(plate_geometry)

    smooth_geometry = dict(module8.BASE_CHANNEL_GEOMETRY)
    smooth_geometry["porosity"] = module8.plate_fin_porosity(module8.PLATE_FIN_GEOMETRY)

    plate = module8.structure_geometry_properties("PlateFin", plate_geometry)
    smooth = module8.structure_geometry_properties("SmoothPlateFin", smooth_geometry)

    expected_plate_dh = 2.0 * (
        (plate_geometry["fin_height"] - plate_geometry["fin_thickness"])
        * (plate_geometry["fin_spacing"] - plate_geometry["fin_thickness"])
    ) / (
        plate_geometry["fin_height"]
        + plate_geometry["fin_spacing"]
        - 2.0 * plate_geometry["fin_thickness"]
    )
    expected_smooth_dh = module8.smooth_plate_fin_hydraulic_diameter(
        width=smooth_geometry["width"],
        height=smooth_geometry["height"],
        porosity=smooth_geometry["porosity"],
    )

    assert plate["branch"] == "plate_fin"
    assert smooth["branch"] == "smooth_plate_fin"
    assert plate["Dh"] == expected_plate_dh
    assert smooth["Dh"] == expected_smooth_dh


def test_small_subset_result_schema_and_count():
    df_results = module8.generate_results(
        structures=["Gyroid", "PlateFin"],
        catalyst_case_names=["baseline", "high_k"],
        re_targets=[500, 1000],
        n_elements=6,
        max_iter=120,
        tolerance=1e-3,
        progress=False,
    )

    required_columns = {
        "Structure",
        "CatalystCase",
        "Re_target",
        "Re_hot_mean",
        "Re_cold_mean",
        "mh_kg_s",
        "mc_kg_s",
        "T_out_K",
        "x_in",
        "x_out",
        "x_eq_out",
        "Conversion_Efficiency_pct",
        "Pressure_Drop_hot_kPa",
        "U_avg_W_m2K",
        "FOM",
        "Iterations",
        "Converged",
    }

    assert len(df_results) == 8
    assert required_columns.issubset(df_results.columns)


def test_all_packed_structures_single_point_smoke():
    for structure in module8.PACKED_STRUCTURES:
        result = module8.run_case(
            structure=structure,
            re_target=500,
            catalyst_case_name="baseline",
            n_elements=6,
            max_iter=120,
            tolerance=1e-3,
        )

        assert result["Converged"], structure
        assert np.isfinite(result["U_avg_W_m2K"]), structure
        assert np.isfinite(result["Pressure_Drop_hot_kPa"]), structure
        assert np.isfinite(result["Conversion_Efficiency_pct"]), structure


def test_build_case_config_keeps_hot_channel_packed_for_conversion():
    cfg = module8.build_case_config(
        structure="Gyroid",
        re_target=500,
        catalyst_case_name="baseline",
        n_elements=6,
        max_iter=120,
        tolerance=1e-3,
    )

    assert cfg["channels"]["hot"]["mode"] == "packed"
    assert cfg["channels"]["cold"]["mode"] == "bare"

    hx = TPMSHeatExchanger(cfg)
    hx._update_stream_physics("hot")
    hx._update_stream_physics("cold")
    xh_before = hx.xh.copy()
    hx._ortho_para_conversion()

    assert np.any(np.abs(hx.xh - xh_before) > 1e-9)


def test_alt_kinetics_case_stays_below_equilibrium_limit():
    result = module8.run_case(
        structure="Gyroid",
        re_target=1000,
        catalyst_case_name="alt_kinetics",
        n_elements=6,
        max_iter=120,
        tolerance=1e-3,
    )

    assert result["Converged"]
    assert result["x_out"] <= result["x_eq_out"] + 1e-6
    assert result["Conversion_Efficiency_pct"] <= 100.0 + 1e-6


def test_auxiliary_bare_reference_includes_mass_flow_view():
    df_bare = module8.compute_auxiliary_bare_reference(re_targets=[500, 1000])

    assert "mh_kg_s" in df_bare.columns
    assert np.all(df_bare["mh_kg_s"] > 0.0)
    assert len(df_bare) == 4
