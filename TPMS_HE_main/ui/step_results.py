"""ui.step_results -- Step 6: run_simulation and render_run_result."""

import copy
import os
from pathlib import Path

import streamlit as st
import pandas as pd

from solver.calculator import TPMSHeatExchanger
from ui.validation import build_solver_config

def run_simulation(state):
    cfg = build_solver_config(state)
    try:
        he = TPMSHeatExchanger(cfg)
        converged = he.solve(
            max_iter=cfg["solver"]["max_iter"],
            tolerance=cfg["solver"]["tolerance"],
        )
    except Exception as exc:
        return {"error": str(exc)}

    result = {
        "error": None,
        "converged": converged,
        "q_total": float(he.Q.sum()),
        "hot_out": float(he.Th[-1]),
        "cold_out": float(he.Tc[0]),
        "eta_ex":     he.perf.get('eta_ex',      None),
        "S_gen":      he.perf.get('S_gen_total',  None),
        "PEC_mean_h": he.perf.get('PEC_mean_h',  None),
        "PEC_mean_c": he.perf.get('PEC_mean_c',  None),
        "output": copy.deepcopy(cfg["output"]),
    }
    try:
        he.finalize_simulation()
        result["output"] = copy.deepcopy(he.config["output"])
    except Exception as exc:
        result["warning"] = f"Outputs partially failed: {exc}"
    return result


def render_run_result():
    result = st.session_state.get("run_result")
    if not result:
        st.info("No results yet. Complete the setup on pages 1–5 and press **Run Simulation**.")
        return

    last_run = st.session_state.get("last_run_at")
    if last_run:
        st.caption(f"Last run: {last_run}")

    st.divider()
    st.subheader("Latest Run Result")
    if result.get("error"):
        st.error(f"Run failed: {result['error']}")
        return

    st.success("Simulation finished")
    if result.get("warning"):
        st.warning(result["warning"])
    st.write(f"Converged: `{result['converged']}`")
    c1, c2, c3 = st.columns(3)
    c1.metric("Q_total [W]", f"{result['q_total']:.2f}")
    c2.metric("Hot outlet [K]", f"{result['hot_out']:.2f}")
    c3.metric("Cold outlet [K]", f"{result['cold_out']:.2f}")

    # Performance evaluation indicators row
    eta_ex = result.get('eta_ex')
    s_gen  = result.get('S_gen')
    pec_h  = result.get('PEC_mean_h')
    pec_c  = result.get('PEC_mean_c')
    if any(v is not None for v in (eta_ex, s_gen, pec_h, pec_c)):
        c4, c5, c6, c7 = st.columns(4)
        c4.metric("η_ex [%]",    f"{eta_ex*100:.1f}"  if eta_ex is not None else "—")
        c5.metric("S_gen [mW/K]", f"{s_gen*1e3:.3f}"  if s_gen  is not None else "—")
        c6.metric("PEC hot",     f"{pec_h:.4f}"        if pec_h  is not None else "—")
        c7.metric("PEC cold",    f"{pec_c:.4f}"        if pec_c  is not None else "—")

    out = result["output"]
    st.markdown("**Output Files**")
    for key in ("results_csv", "convergence_csv", "performance_plot", "convergence_plot",
                "resistance_pie", "performance_eval_plot"):
        p = Path(out.get(key, ""))
        if str(p):
            st.write(f"- `{p}` {'(found)' if p.exists() else '(missing)'}")

    for _img_key, _caption in [
        ("performance_plot",     "Performance Profile"),
        ("resistance_pie",       "Thermal Resistance Breakdown"),
        ("performance_eval_plot","Performance Evaluation — Exergy & PEC"),
        ("convergence_plot",     "Convergence Diagnostics"),
    ]:
        _p = out.get(_img_key, "")
        if _p and os.path.exists(_p):
            with open(_p, "rb") as _f:
                st.image(_f.read(), caption=_caption)
    if os.path.exists(out["results_csv"]):
        st.subheader("Results Preview")
        try:
            st.dataframe(pd.read_csv(out["results_csv"]).head(20))
        except Exception as exc:
            st.warning(f"Could not load CSV preview: {exc}")


