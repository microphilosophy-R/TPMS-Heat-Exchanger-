"""ui.step_solver -- Step 3: Solver Controls rendering."""

import streamlit as st

from ui.components import _k

def render_step_solver(state):
    st.subheader("Solver Controls")
    c1, c2 = st.columns(2)
    with c1:
        state["solver"]["n_elements"] = int(
            st.number_input(
                "n_elements",
                min_value=5,
                max_value=5000,
                value=int(state["solver"]["n_elements"]),
                key=_k("solver_n"),
            )
        )
        state["solver"]["max_iter"] = int(
            st.number_input(
                "max_iter",
                min_value=1,
                max_value=20000,
                value=int(state["solver"]["max_iter"]),
                key=_k("solver_max_iter"),
            )
        )
        state["solver"]["tolerance"] = st.number_input(
            "tolerance",
            min_value=1e-12,
            max_value=1.0,
            value=float(state["solver"]["tolerance"]),
            format="%.8f",
            key=_k("solver_tol"),
        )
    with c2:
        state["solver"]["relax_thermal"] = st.slider(
            "relax_thermal",
            min_value=0.01,
            max_value=1.0,
            value=float(state["solver"]["relax_thermal"]),
            step=0.01,
            key=_k("solver_rt"),
        )
        state["solver"]["relax_hydraulic"] = st.slider(
            "relax_hydraulic",
            min_value=0.01,
            max_value=1.0,
            value=float(state["solver"]["relax_hydraulic"]),
            step=0.01,
            key=_k("solver_rh"),
        )
        state["solver"]["relax_kinetics"] = st.slider(
            "relax_kinetics",
            min_value=0.01,
            max_value=1.0,
            value=float(state["solver"]["relax_kinetics"]),
            step=0.01,
            key=_k("solver_rk"),
        )
        state["solver"]["Q_damping"] = st.slider(
            "Q_damping",
            min_value=0.01,
            max_value=1.0,
            value=float(state["solver"]["Q_damping"]),
            step=0.01,
            key=_k("solver_qd"),
        )
    state["solver"]["relax"] = state["solver"]["relax_thermal"]


