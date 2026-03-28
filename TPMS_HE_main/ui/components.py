"""ui.components -- Shared UI components: nav bar, summary table, validation panel."""

import streamlit as st
import pandas as pd

from ui.validation import build_solver_config

STEP_DEFS = [
    {"key": "geometry", "title": "1. Geometry & Channels", "sections": ["geometry", "channels"]},
    {"key": "operating", "title": "2. Operating", "sections": ["operating"]},
    {"key": "solver", "title": "3. Solver", "sections": ["solver"]},
    {"key": "output", "title": "4. Output", "sections": ["output"]},
    {
        "key": "confirm",
        "title": "5. Confirm & Run",
        "sections": ["geometry", "operating", "channels", "solver", "output"],
    },
    {"key": "results", "title": "6. Results", "sections": []},
]

def render_channel_summary_strip(state):
    hot = state["channels"]["hot"]
    cold = state["channels"]["cold"]
    c1, c2 = st.columns(2)
    c1.info(
        f"Hot: {hot['mode']} / {hot['structure']} / packed uncertainty {hot['packed'].get('uncertainty_mode', 'nominal')}"
    )
    c2.info(
        f"Cold: {cold['mode']} / {cold['structure']} / packed uncertainty {cold['packed'].get('uncertainty_mode', 'nominal')}"
    )


def _k(name):
    return f"{name}_v{st.session_state.ui_version}"


def _summary_df(rows):
    return pd.DataFrame([(p, str(v)) for p, v in rows], columns=["Parameter", "Value"])


def render_summary_table(state, issues):
    st.subheader("Configuration Summary")
    st.markdown("Review all settings before execution.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Geometry**")
        st.table(
            _summary_df(
                [
                    ("length [m]", state["geometry"]["length"]),
                    ("width [m]", state["geometry"]["width"]),
                    ("plate_thickness [m]", state["geometry"].get("plate_thickness", "—")),
                    ("ε_hot [-]", state["channels"]["hot"].get("geometry", {}).get("porosity", "—")),
                    ("ε_cold [-]", state["channels"]["cold"].get("geometry", {}).get("porosity", "—")),
                    ("α_hot [1/m]", state["channels"]["hot"].get("surface_area_density", "—")),
                    ("α_cold [1/m]", state["channels"]["cold"].get("surface_area_density", "—")),
                ]
            )
        )
        st.markdown("**Operating**")
        st.table(
            _summary_df(
                [
                    ("fluid_hot", state["operating"]["fluid_hot"]),
                    ("fluid_cold", state["operating"]["fluid_cold"]),
                    ("Th_in [K]", state["operating"]["Th_in"]),
                    ("Tc_in [K]", state["operating"]["Tc_in"]),
                    ("Ph_in [Pa]", state["operating"]["Ph_in"]),
                    ("Pc_in [Pa]", state["operating"]["Pc_in"]),
                    ("mh [kg/s]", state["operating"]["mh"]),
                    ("mc [kg/s]", state["operating"]["mc"]),
                    ("xh_in", state["operating"]["xh_in"]),
                ]
            )
        )

    with c2:
        for ch in ("hot", "cold"):
            ch_cfg = state["channels"][ch]
            st.markdown(f"**{ch.capitalize()} Channel**")
            rows = [
                ("mode", ch_cfg["mode"]),
                ("structure", ch_cfg["structure"]),
                ("packed.uncertainty_mode", ch_cfg["packed"].get("uncertainty_mode", "nominal")),
            ]
            if ch_cfg["mode"] == "packed":
                rows.extend(
                    [
                        ("packed.htc_model", ch_cfg["packed"]["htc_model"]),
                        ("packed.hydraulic_model", ch_cfg["packed"]["hydraulic_model"]),
                        ("packed.ht_enhancement_model", ch_cfg["packed"]["ht_enhancement_model"]),
                        ("packed.kinetic_model", ch_cfg["packed"]["kinetic_model"]),
                        ("packed.particle_diameter [m]", ch_cfg["packed"]["particle_diameter"]),
                        ("packed.bed_porosity", ch_cfg["packed"]["bed_porosity"]),
                        ("packed.k_solid [W/m-K]", ch_cfg["packed"]["k_solid"]),
                        ("packed.shape_factor", ch_cfg["packed"]["shape_factor"]),
                    ]
                )
            st.table(_summary_df(rows))

        st.markdown("**Solver**")
        st.table(
            _summary_df(
                [
                    ("n_elements", state["solver"]["n_elements"]),
                    ("max_iter", state["solver"]["max_iter"]),
                    ("tolerance", state["solver"]["tolerance"]),
                    ("relax_thermal", state["solver"]["relax_thermal"]),
                    ("relax_hydraulic", state["solver"]["relax_hydraulic"]),
                    ("relax_kinetics", state["solver"]["relax_kinetics"]),
                    ("Q_damping", state["solver"]["Q_damping"]),
                ]
            )
        )
        st.markdown("**Output**")
        st.table(
            _summary_df(
                [
                    ("results_csv", state["output"]["results_csv"]),
                    ("convergence_csv", state["output"]["convergence_csv"]),
                    ("performance_plot", state["output"]["performance_plot"]),
                    ("convergence_plot", state["output"]["convergence_plot"]),
                ]
            )
        )

    with st.expander("Effective Config JSON"):
        st.json(build_solver_config(state))

    error_count = sum(1 for x in issues if x["level"] == "error")
    warning_count = sum(1 for x in issues if x["level"] == "warning")
    info_count = sum(1 for x in issues if x["level"] == "info")
    st.caption(
        f"Validation snapshot: {error_count} errors, {warning_count} warnings, {info_count} info"
    )


def render_validation_panel(issues, visible_sections):
    relevant = [x for x in issues if x["section"] in visible_sections]
    errors = [x for x in relevant if x["level"] == "error"]
    warnings_ = [x for x in relevant if x["level"] == "warning"]
    infos = [x for x in relevant if x["level"] == "info"]

    st.markdown("### Validation")
    if not errors and not warnings_ and not infos:
        st.success("No validation issues in this step.")
        return

    if errors:
        st.error(f"{len(errors)} blocking error(s) detected.")
        for item in errors:
            st.write(f"- `{item['section']}.{item['field']}`: {item['message']}")
    if warnings_:
        st.warning(f"{len(warnings_)} warning(s).")
        for item in warnings_:
            st.write(f"- `{item['section']}.{item['field']}`: {item['message']}")
    if infos:
        st.info(f"{len(infos)} info message(s).")
        for item in infos:
            st.write(f"- `{item['section']}.{item['field']}`: {item['message']}")


def has_blocking_issues(issues, sections):
    return any(x["level"] == "error" and x["section"] in sections for x in issues)


def render_nav_bar(current: int, position: str = "top"):
    """Render a row of step-pill buttons. Active step is highlighted; Results pill
    is disabled until a simulation result exists.

    Args:
        current: Index of the active step.
        position: 'top' or 'bottom' — used to generate unique widget keys.
    """
    cols = st.columns(len(STEP_DEFS))
    for i, (col, sdef) in enumerate(zip(cols, STEP_DEFS)):
        is_active = (i == current)
        is_disabled = is_active or (
            sdef["key"] == "results" and not st.session_state.get("run_result")
        )
        if col.button(
            sdef["title"],
            key=f"navpill_{position}_{i}",
            type="primary" if is_active else "secondary",
            disabled=is_disabled,
            use_container_width=True,
        ):
            st.session_state.current_step = i
            st.rerun()


