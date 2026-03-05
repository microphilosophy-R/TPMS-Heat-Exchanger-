import os
from pathlib import Path

import pandas as pd
import streamlit as st

from tpms_correlations import TPMSCorrelations
from tpms_thermo_hydraulic_calculator import TPMSHeatExchanger, create_default_config


def _channel_form(channel_name, defaults):
    st.subheader(f"{channel_name.capitalize()} Channel")
    mode = st.selectbox(
        f"{channel_name} mode",
        options=["bare", "packed"],
        index=["bare", "packed"].index(defaults["mode"]),
        key=f"{channel_name}_mode",
    )
    structure = st.selectbox(
        f"{channel_name} structure",
        options=list(TPMSCorrelations.get_supported_tpms_types()),
        index=list(TPMSCorrelations.get_supported_tpms_types()).index(defaults["structure"]),
        key=f"{channel_name}_structure",
    )

    packed = defaults["packed"].copy()
    packed_mode = packed.get("mode", "nominal")
    if mode == "packed":
        packed_mode = st.selectbox(
            f"{channel_name} packed model mode",
            options=["lower", "nominal", "upper"],
            index=["lower", "nominal", "upper"].index(packed_mode),
            key=f"{channel_name}_packed_mode",
        )
        packed["particle_diameter"] = st.number_input(
            f"{channel_name} particle diameter [m]",
            min_value=1e-5,
            max_value=1e-1,
            value=float(packed["particle_diameter"]),
            format="%.6f",
            key=f"{channel_name}_particle_diameter",
        )
        packed["bed_porosity"] = st.number_input(
            f"{channel_name} bed porosity [-]",
            min_value=0.05,
            max_value=0.95,
            value=float(packed["bed_porosity"]),
            format="%.3f",
            key=f"{channel_name}_bed_porosity",
        )
        packed["k_solid"] = st.number_input(
            f"{channel_name} solid conductivity [W/m-K]",
            min_value=0.01,
            max_value=5000.0,
            value=float(packed["k_solid"]),
            format="%.3f",
            key=f"{channel_name}_k_solid",
        )
        packed["shape_factor"] = st.number_input(
            f"{channel_name} shape factor [-]",
            min_value=0.01,
            max_value=5.0,
            value=float(packed["shape_factor"]),
            format="%.3f",
            key=f"{channel_name}_shape_factor",
        )

    packed["mode"] = packed_mode
    return {"mode": mode, "structure": structure, "packed": packed}


def main():
    st.set_page_config(page_title="TPMS HE Controller", layout="wide")
    st.title("TPMS Heat Exchanger Controller (MVP)")
    st.caption("Two-level model: exchanger orchestration + channel closure dispatch")

    default_cfg = create_default_config()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Geometry")
        length = st.number_input("Length [m]", value=float(default_cfg["geometry"]["length"]))
        width = st.number_input("Width [m]", value=float(default_cfg["geometry"]["width"]))
        height = st.number_input("Height [m]", value=float(default_cfg["geometry"]["height"]))
        unit_cell_size = st.number_input(
            "Unit cell size [m]", min_value=1e-4, value=float(default_cfg["geometry"]["unit_cell_size"]), format="%.6f"
        )
        wall_thickness = st.number_input(
            "Wall thickness [m]", min_value=1e-5, value=float(default_cfg["geometry"]["wall_thickness"]), format="%.6f"
        )
        surface_area_density = st.number_input(
            "Surface area density [1/m]", min_value=1.0, value=float(default_cfg["geometry"]["surface_area_density"])
        )
        porosity_hot = st.number_input("Hot porosity [-]", min_value=0.05, max_value=0.95, value=float(default_cfg["geometry"]["porosity_hot"]))
        porosity_cold = st.number_input("Cold porosity [-]", min_value=0.05, max_value=0.95, value=float(default_cfg["geometry"]["porosity_cold"]))

    with c2:
        st.subheader("Operating")
        fluid_hot = st.selectbox("Hot fluid", options=["hydrogen mixture", "normal hydrogen", "helium", "argon"], index=0)
        fluid_cold = st.selectbox("Cold fluid", options=["helium", "argon", "hydrogen mixture"], index=0)
        Th_in = st.number_input("Th_in [K]", value=float(default_cfg["operating"]["Th_in"]))
        Tc_in = st.number_input("Tc_in [K]", value=float(default_cfg["operating"]["Tc_in"]))
        Ph_in = st.number_input("Ph_in [Pa]", value=float(default_cfg["operating"]["Ph_in"]), format="%.1f")
        Pc_in = st.number_input("Pc_in [Pa]", value=float(default_cfg["operating"]["Pc_in"]), format="%.1f")
        mh = st.number_input("m_hot [kg/s]", min_value=1e-5, value=float(default_cfg["operating"]["mh"]), format="%.5f")
        mc = st.number_input("m_cold [kg/s]", min_value=1e-5, value=float(default_cfg["operating"]["mc"]), format="%.5f")
        xh_in = st.number_input("Hot para fraction xh_in [-]", min_value=0.0, max_value=1.0, value=float(default_cfg["operating"]["xh_in"]))

    st.divider()
    c3, c4 = st.columns(2)
    with c3:
        hot_channel = _channel_form("hot", default_cfg["channels"]["hot"])
    with c4:
        cold_channel = _channel_form("cold", default_cfg["channels"]["cold"])

    st.divider()
    c5, c6 = st.columns(2)
    with c5:
        st.subheader("Solver")
        n_elements = st.number_input("n_elements", min_value=5, max_value=2000, value=int(default_cfg["solver"]["n_elements"]))
        max_iter = st.number_input("max_iter", min_value=5, max_value=5000, value=int(default_cfg["solver"]["max_iter"]))
        tolerance = st.number_input("tolerance", min_value=1e-8, max_value=1.0, value=float(default_cfg["solver"]["tolerance"]), format="%.6f")
        relax_thermal = st.number_input("relax_thermal", min_value=0.01, max_value=1.0, value=float(default_cfg["solver"]["relax_thermal"]), format="%.3f")
        relax_hydraulic = st.number_input("relax_hydraulic", min_value=0.01, max_value=1.0, value=float(default_cfg["solver"]["relax_hydraulic"]), format="%.3f")
        relax_kinetics = st.number_input("relax_kinetics", min_value=0.01, max_value=1.0, value=float(default_cfg["solver"]["relax_kinetics"]), format="%.3f")
        q_damping = st.number_input("Q_damping", min_value=0.01, max_value=1.0, value=float(default_cfg["solver"]["Q_damping"]), format="%.3f")

    with c6:
        st.subheader("Output")
        results_csv = st.text_input("results_csv", value=default_cfg["output"]["results_csv"])
        convergence_csv = st.text_input("convergence_csv", value=default_cfg["output"]["convergence_csv"])
        performance_plot = st.text_input("performance_plot", value=default_cfg["output"]["performance_plot"])
        convergence_plot = st.text_input("convergence_plot", value=default_cfg["output"]["convergence_plot"])

    run = st.button("Run Simulation", type="primary")
    if not run:
        return

    cfg = create_default_config()
    cfg["geometry"].update(
        {
            "length": length,
            "width": width,
            "height": height,
            "unit_cell_size": unit_cell_size,
            "wall_thickness": wall_thickness,
            "surface_area_density": surface_area_density,
            "porosity_hot": porosity_hot,
            "porosity_cold": porosity_cold,
        }
    )
    cfg["operating"].update(
        {
            "fluid_hot": fluid_hot,
            "fluid_cold": fluid_cold,
            "Th_in": Th_in,
            "Tc_in": Tc_in,
            "Ph_in": Ph_in,
            "Pc_in": Pc_in,
            "mh": mh,
            "mc": mc,
            "xh_in": xh_in,
        }
    )
    cfg["channels"]["hot"] = hot_channel
    cfg["channels"]["cold"] = cold_channel

    # Keep legacy catalyst defaults synced for compatibility callers.
    cfg["catalyst"].update(hot_channel["packed"])

    cfg["solver"].update(
        {
            "n_elements": int(n_elements),
            "max_iter": int(max_iter),
            "tolerance": float(tolerance),
            "relax": float(relax_thermal),
            "relax_thermal": float(relax_thermal),
            "relax_hydraulic": float(relax_hydraulic),
            "relax_kinetics": float(relax_kinetics),
            "Q_damping": float(q_damping),
        }
    )
    cfg["output"].update(
        {
            "results_csv": results_csv,
            "convergence_csv": convergence_csv,
            "performance_plot": performance_plot,
            "convergence_plot": convergence_plot,
        }
    )

    with st.spinner("Running solver..."):
        he = TPMSHeatExchanger(cfg)
        converged = he.solve(
            max_iter=cfg["solver"]["max_iter"],
            tolerance=cfg["solver"]["tolerance"],
        )
        he.finalize_simulation()

    st.success("Simulation finished")
    st.write(f"Converged: `{converged}`")

    c7, c8, c9 = st.columns(3)
    c7.metric("Q_total [W]", f"{he.Q.sum():.2f}")
    c8.metric("Hot outlet [K]", f"{he.Th[-1]:.2f}")
    c9.metric("Cold outlet [K]", f"{he.Tc[0]:.2f}")

    st.subheader("Output Files")
    output_paths = [
        cfg["output"]["results_csv"],
        cfg["output"]["convergence_csv"],
        cfg["output"]["performance_plot"],
        cfg["output"]["convergence_plot"],
    ]
    for path in output_paths:
        full_path = Path(path)
        st.write(f"- `{full_path}` {'(found)' if full_path.exists() else '(missing)'}")

    if os.path.exists(cfg["output"]["performance_plot"]):
        st.image(cfg["output"]["performance_plot"], caption="Performance Profile")

    if os.path.exists(cfg["output"]["convergence_plot"]):
        st.image(cfg["output"]["convergence_plot"], caption="Convergence Diagnostics")

    if os.path.exists(cfg["output"]["results_csv"]):
        st.subheader("Results Preview")
        st.dataframe(pd.read_csv(cfg["output"]["results_csv"]).head(20))


if __name__ == "__main__":
    main()
