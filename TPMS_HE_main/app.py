"""TPMS Heat Exchanger — Streamlit entry point.  Run: streamlit run app.py"""
from datetime import datetime

import streamlit as st

from ui.state import (
    AUTOSAVE_PATH,
    init_ui_state,
    maybe_autosave,
    reload_autosave,
    reset_to_defaults,
)
from ui.components import (
    STEP_DEFS,
    has_blocking_issues,
    render_channel_summary_strip,
    render_nav_bar,
    render_summary_table,
    render_validation_panel,
)
from ui.step_geometry import render_step_geometry
from ui.step_operating import render_step_operating
from ui.step_channels import render_step_channels
from ui.step_solver import render_step_solver
from ui.step_output import render_step_output
from ui.step_results import render_run_result, run_simulation
from ui.validation import validate_ui_state


def main():
    st.set_page_config(page_title="TPMS HE Controller", layout="wide")
    init_ui_state()

    st.title("TPMS Heat Exchanger Wizard")
    st.caption("Step-by-step setup with autosave, validation, and channel-model visibility")

    for msg in st.session_state.get("init_messages", []):
        st.warning(msg)
    st.session_state.init_messages = []

    c_top1, c_top2, c_top3 = st.columns([2, 1, 1])
    c_top1.caption(f"Autosave file: `{AUTOSAVE_PATH}`")
    c_top1.caption(f"Last autosave: {st.session_state.last_saved_at}")
    if c_top2.button("Reload Autosave"):
        reload_autosave()
    if c_top3.button("Reset Defaults"):
        reset_to_defaults()

    state = st.session_state.ui_state
    render_channel_summary_strip(state)

    current  = st.session_state.current_step
    step_def = STEP_DEFS[current]
    render_nav_bar(current, position="top")
    st.subheader(step_def["title"])

    if step_def["key"] == "geometry":
        render_step_geometry(state)
    elif step_def["key"] == "operating":
        render_step_operating(state)
    elif step_def["key"] == "solver":
        render_step_solver(state)
    elif step_def["key"] == "output":
        render_step_output(state)
    elif step_def["key"] == "results":
        render_run_result()

    maybe_autosave()
    issues = validate_ui_state(state)

    if step_def["key"] == "confirm":
        render_summary_table(state, issues)

    render_validation_panel(issues, step_def["sections"])

    has_error_current = has_blocking_issues(issues, step_def["sections"])
    has_error_global  = has_blocking_issues(
        issues, ["geometry", "operating", "channels", "solver", "output"]
    )

    render_nav_bar(current, position="bottom")
    st.divider()
    _, c_nav1, c_nav2, _ = st.columns([2, 1, 1, 2])
    if current > 0 and c_nav1.button("← Back"):
        st.session_state.current_step -= 1
        st.rerun()
    if step_def["key"] == "confirm":
        if c_nav2.button("Run Simulation", type="primary", disabled=has_error_global):
            with st.spinner("Running solver..."):
                st.session_state.run_result = run_simulation(state)
            st.session_state.last_run_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            maybe_autosave(force=True)
            st.session_state.current_step = len(STEP_DEFS) - 1   # jump to Results page
            st.rerun()
    elif step_def["key"] != "results":
        if c_nav2.button("Next →", disabled=has_error_current):
            st.session_state.current_step += 1
            st.rerun()


if __name__ == "__main__":
    main()
