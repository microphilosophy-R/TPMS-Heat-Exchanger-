"""ui.step_operating -- Step 2: Operating Conditions rendering."""

import streamlit as st

from ui.components import _k

def render_step_operating(state):
    st.subheader("Operating Conditions")
    c1, c2 = st.columns(2)
    with c1:
        hot_options = ["hydrogen mixture", "normal hydrogen", "helium", "argon"]
        idx_hot = hot_options.index(state["operating"]["fluid_hot"]) if state["operating"]["fluid_hot"] in hot_options else 0
        state["operating"]["fluid_hot"] = st.selectbox(
            "Hot fluid", options=hot_options, index=idx_hot, key=_k("op_hot_fluid")
        )
        state["operating"]["Th_in"] = st.number_input(
            "Th_in [K]",
            min_value=1.0,
            value=float(state["operating"]["Th_in"]),
            key=_k("op_Th"),
        )
        state["operating"]["Ph_in"] = st.number_input(
            "Ph_in [Pa]",
            min_value=1.0,
            value=float(state["operating"]["Ph_in"]),
            format="%.1f",
            key=_k("op_Ph"),
        )
        state["operating"]["mh"] = st.number_input(
            "m_hot [kg/s]",
            min_value=1e-6,
            value=float(state["operating"]["mh"]),
            format="%.6f",
            key=_k("op_mh"),
        )

    with c2:
        cold_options = ["helium", "argon", "hydrogen mixture", "normal hydrogen"]
        idx_cold = cold_options.index(state["operating"]["fluid_cold"]) if state["operating"]["fluid_cold"] in cold_options else 0
        state["operating"]["fluid_cold"] = st.selectbox(
            "Cold fluid",
            options=cold_options,
            index=idx_cold,
            key=_k("op_cold_fluid"),
        )
        state["operating"]["Tc_in"] = st.number_input(
            "Tc_in [K]",
            min_value=1.0,
            value=float(state["operating"]["Tc_in"]),
            key=_k("op_Tc"),
        )
        state["operating"]["Pc_in"] = st.number_input(
            "Pc_in [Pa]",
            min_value=1.0,
            value=float(state["operating"]["Pc_in"]),
            format="%.1f",
            key=_k("op_Pc"),
        )
        state["operating"]["mc"] = st.number_input(
            "m_cold [kg/s]",
            min_value=1e-6,
            value=float(state["operating"]["mc"]),
            format="%.6f",
            key=_k("op_mc"),
        )

    state["operating"]["xh_in"] = st.slider(
        "Hot inlet para fraction xh_in [-]",
        min_value=0.0,
        max_value=1.0,
        value=float(state["operating"]["xh_in"]),
        step=0.001,
        key=_k("op_xh"),
    )
    state["operating"]["T_ambient"] = st.number_input(
        "Ambient (dead-state) temperature T₀ [K]",
        min_value=200.0,
        max_value=400.0,
        value=float(state["operating"].get("T_ambient", 298.0)),
        format="%.1f",
        key=_k("op_T0"),
        help="Reference temperature for exergy analysis. Use ambient (~298 K) for "
             "liquefaction applications. Both streams are sub-ambient, so cold exergy "
             "is measured relative to this value.",
    )


