"""ui.step_output -- Step 4: Output Paths rendering."""

import streamlit as st

from ui.components import _k

def render_step_output(state):
    st.subheader("Output Paths")
    state["output"]["results_csv"] = st.text_input(
        "results_csv", value=state["output"]["results_csv"], key=_k("out_results")
    )
    state["output"]["convergence_csv"] = st.text_input(
        "convergence_csv",
        value=state["output"]["convergence_csv"],
        key=_k("out_conv_csv"),
    )
    state["output"]["performance_plot"] = st.text_input(
        "performance_plot",
        value=state["output"]["performance_plot"],
        key=_k("out_perf"),
    )
    state["output"]["convergence_plot"] = st.text_input(
        "convergence_plot",
        value=state["output"]["convergence_plot"],
        key=_k("out_conv_plot"),
    )

