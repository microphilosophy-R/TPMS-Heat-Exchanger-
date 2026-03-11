"""ui.step_channels -- Step 1 (channels sub-section): channel card rendering."""

import streamlit as st
import numpy as np

from correlations.thermohydraulic_correlations import ThermoHydraulicCorrelations
from models.packed_bed import SUPPORTED_PACKED_MODES
from ui.components import _k
from ui.step_geometry import _render_channel_estimator


def _render_derived_channel_metrics(state, channel_name, channel_state):
    """Display read-only derived geometry metrics for one channel."""
    sk = channel_name.lower()
    geo = state["geometry"]
    ch_geo = channel_state.get("geometry", {}) or {}

    ch_L = float(ch_geo.get("length")  or geo["length"])
    ch_W = float(ch_geo.get("width")   or geo["width"])
    ch_H = float(ch_geo.get("height")  or geo["height"])
    L_cell = float(ch_geo.get("unit_cell_size") or geo["unit_cell_size"])
    struct = channel_state["structure"]

    if struct == "PlateFin":
        Hf = float(ch_geo.get("fin_height")    or geo.get("fin_height",    9.5e-3))
        sf = float(ch_geo.get("fin_spacing")   or geo.get("fin_spacing",   3.2e-3))
        tf = float(ch_geo.get("fin_thickness") or geo.get("fin_thickness", 0.6e-3))
        eps = (sf - tf) / max(sf, 1e-12)
        Dh  = 2.0 * (Hf - tf) * (sf - tf) / max(Hf + sf - 2.0 * tf, 1e-12)
    elif struct == "SmoothPlateFin":
        eps = float(geo.get(f"porosity_{sk}", 0.5))
        Dh  = 4.0 * ch_W * ch_H * eps / max(2.0 * (ch_W + ch_H * eps), 1e-12)
    else:
        eps = float(geo.get(f"porosity_{sk}", 0.5))
        a   = L_cell / (2.0 * np.pi)
        Dh  = 4.0 * eps * a

    V_total = ch_L * ch_W * ch_H
    V_f     = eps * V_total
    V_s     = (1.0 - eps) * V_total
    SAD     = float(channel_state.get("surface_area_density", 0.0))
    A_HX    = SAD * V_total

    is_packed = channel_state.get("mode") == "packed"
    n_cols = 7 if is_packed else 5
    st.markdown("**Derived Channel Geometry**")
    cols = st.columns(n_cols)
    cols[0].metric("Dh [mm]",      f"{Dh * 1e3:.3f}")
    cols[1].metric("ε [-]",        f"{eps:.3f}")
    cols[2].metric("A_HX [m²]",    f"{A_HX:.4f}")
    cols[3].metric("V_fluid [L]",  f"{V_f * 1e3:.4f}")
    cols[4].metric("V_solid [L]",  f"{V_s * 1e3:.4f}")
    if is_packed:
        packed = channel_state.get("packed", {})
        dp_mm  = float(packed.get("particle_diameter", 0.0)) * 1e3
        eps_bed = float(packed.get("bed_porosity", 0.0))
        cols[5].metric("d_p [mm]",   f"{dp_mm:.3f}")
        cols[6].metric("ε_bed [-]",  f"{eps_bed:.3f}")


def _render_htc_model_equations(htc_model: str):
    """Show governing equations for the selected packed-bed HTC model."""
    if htc_model == 'dixon':
        with st.expander("Dixon model equations (Eqs 0.1 \u2013 0.6)", expanded=False):
            st.markdown("**Overall tube-side heat transfer coefficient [109]:**")
            st.latex(r"\frac{1}{h_i} = \frac{1}{h_w} + \frac{d_i}{6k_r}\,\frac{\mathrm{Bi}+3}{\mathrm{Bi}+4}")
            st.markdown("**Biot number:**")
            st.latex(r"\mathrm{Bi} = \frac{h_w\,d_i}{2k_r}")
            st.markdown("**Effective radial conductivity [110, 112]:**")
            st.latex(
                r"k_r = \lambda_h\!\left(\frac{\lambda_s}{\lambda_h}\right)^{\!0.28"
                r" - 0.757\log\varepsilon - 0.057\log\!\frac{\lambda_s}{\lambda_h}}"
                r" + \frac{\lambda_h}{\mathrm{Pe}_r}\,\mathrm{Re}\,\mathrm{Pr}"
            )
            st.markdown("**Wall heat transfer coefficient [111]:**")
            st.latex(r"h_w = \frac{\mathrm{Nu}_w\,\lambda_h}{d_p}")
            st.latex(
                r"\mathrm{Nu}_w = \mathrm{Nu}_{w,0} + 110.3\,\mathrm{Pr}^{1/3}\,\mathrm{Re}^{0.75}"
                r" + 10.054\,\mathrm{Re}\,\mathrm{Pr} \quad (\mathrm{Nu}_{w,0}=20)"
            )
            st.markdown("**Radial Peclet number [112]:**")
            st.latex(r"\mathrm{Pe}_r = \frac{1}{0.11 + \dfrac{20.64}{\mathrm{Re}}}")
            st.caption(
                r"$d_i$ = TPMS hydraulic diameter $D_h$; "
                r"$d_p$ = particle diameter; "
                r"$\lambda_h$ = fluid conductivity; "
                r"$\lambda_s$ = solid conductivity; "
                r"$\varepsilon$ = bed porosity; "
                r"$\mathrm{Re} = \rho u d_p / \mu$."
            )
    else:  # martin_nilles
        with st.expander("Martin-Nilles / ZBS model equations", expanded=False):
            st.markdown("**Overall heat transfer coefficient (two-resistance model):**")
            st.latex(
                r"\frac{1}{h_\mathrm{eff}} = \frac{1}{h_w}"
                r" + \frac{D_h}{C_\mathrm{shape}\,k_{r,\mathrm{eff}}}"
            )
            st.markdown("**Wall HTC \u2014 Martin-Nilles correlation:**")
            st.latex(
                r"\mathrm{Nu}_w = \left(1.3 + \frac{5}{D_h/d_p}\right)\frac{k_{r,\mathrm{eff}}}{k_f}"
                r" + 0.19\,\mathrm{Re}^{0.75}\,\mathrm{Pr}^{1/3}"
            )
            st.markdown("**Stagnant effective conductivity \u2014 ZBS model:**")
            st.latex(
                r"k_{r,\mathrm{eff}}^{0} = \bigl(1-\sqrt{1-\varepsilon}\bigr)\,k_f"
                r" + \sqrt{1-\varepsilon}\,k_\mathrm{cell}"
            )
            st.markdown("**Dispersion contribution (Wen-Fan):**")
            st.latex(r"k_\mathrm{disp} = 0.1\,\mathrm{Re}\,\mathrm{Pr}\,k_f")
            st.caption(
                r"$C_\mathrm{shape}$: geometric factor (lower = 8, nominal = 6, upper = 4). "
                "TPMS fin area enhancement applied on top."
            )


def _render_tpms_bare_equations(tpms_type: str):
    """Show Nu and f correlations for the selected TPMS structure (bare channel)."""
    corr_map = {
        'Gyroid': {
            'gas': (r"\mathrm{Nu} = 0.3250\,\mathrm{Re}^{0.7002}\,\mathrm{Pr}^{0.36}", r"f = 2.5\,\mathrm{Re}^{-0.2}", "Re: 2000–8170 (Gas/Air)"),
            'water': (r"\mathrm{Nu} = 0.471\,\mathrm{Re}^{0.627}\,\mathrm{Pr}^{1/3}", r"f = 2.577\,\mathrm{Re}^{-0.095}", "Re: 150–3000 (Water)"),
        },
        'Diamond': {
            'gas': (r"\mathrm{Nu} = 0.409\,\mathrm{Re}^{0.625}\,\mathrm{Pr}^{0.4}", r"f = 2.5892\,\mathrm{Re}^{-0.1940}", "Re: 800–9590 (Gas)"),
            'water': (r"\mathrm{Nu} = 0.12504\,\mathrm{Re}^{0.73143}\,\mathrm{Pr}^{1/3}", r"f = 2.74632\,\mathrm{Re}^{-0.36099}", "Re: 80–1500 (Water)"),
        },
        'Primitive': {
            'gas': (r"\mathrm{Nu} = 0.1\,\mathrm{Re}^{0.75}\,\mathrm{Pr}^{0.36}", r"f = 4.0\,\mathrm{Re}^{-0.25}", "Gas (estimated)"),
            'water': (r"\mathrm{Nu} = 0.05513\,\mathrm{Re}^{0.81370}\,\mathrm{Pr}^{1/3}", r"f = 3.96709\,\mathrm{Re}^{-0.23326}", "Re: 80–1500 (Water)"),
        },
        'Neovius': {
            'gas': (r"\mathrm{Nu} = 0.15\,\mathrm{Re}^{0.7}\,\mathrm{Pr}^{0.36}", r"f = 5.0\,\mathrm{Re}^{-0.3}", "Gas (estimated)"),
            'water': (r"\mathrm{Nu} = 2.48\,\mathrm{Re}^{0.45}", r"f = 59.2\,\mathrm{Re}^{-0.63}", "Re: 10–75 (Water)"),
        },
        'FRD': {
            'gas': (r"\mathrm{Nu} = 0.3\,\mathrm{Re}^{0.65}\,\mathrm{Pr}^{0.36}", r"f = 3.0\,\mathrm{Re}^{-0.2}", "Gas (estimated)"),
            'water': (r"\mathrm{Nu} = 1.74\,\mathrm{Re}^{0.54}", r"f = 11.5\,\mathrm{Re}^{-0.41}", "Re: 35–290 (Water)"),
        },
        'FKS': {
            'gas': (r"\mathrm{Nu} = 0.52\,\mathrm{Re}^{0.61}\,\mathrm{Pr}^{0.4}", r"f = 2.1335\,\mathrm{Re}^{-0.1334}", "Re: 730–10230 (Gas)"),
            'water': (r"\mathrm{Nu} = 3.02\,\mathrm{Re}^{0.40}", r"f = 25.0\,\mathrm{Re}^{-0.73}", "Re: 10–140 (Water)"),
        },
        'SmoothPlateFin': {
            'gas': (r"\mathrm{Nu} = 0.023\,\mathrm{Re}^{0.8}\,\mathrm{Pr}^{0.4}", r"f = \frac{(0.790\ln\mathrm{Re}-1.64)^{-2}}{4}", "Re > 2300 (Dittus-Boelter / Petukhov-Filonenko); laminar: Nu=3.66, f=16/Re"),
            'water': (r"\mathrm{Nu} = 0.023\,\mathrm{Re}^{0.8}\,\mathrm{Pr}^{0.4}", r"f = \frac{(0.790\ln\mathrm{Re}-1.64)^{-2}}{4}", "Re > 2300 (same correlation)"),
        },
        'PlateFin': {
            'gas': (
                r"\ln j = -2.641\!\times\!10^{-2}(\ln\mathrm{Re})^3+0.556(\ln\mathrm{Re})^2"
                r"-4.092\ln\mathrm{Re}+6.217,\quad \mathrm{Nu}=j\,\mathrm{Re}\,\mathrm{Pr}^{1/3}",
                r"f \approx 2.5\,j \quad \text{(perforated-fin approx., no }f\text{ in paper)}",
                "Perforated fins — Wang et al. (2024) / Li (2018)",
            ),
            'water': (
                r"\ln j = -2.641\!\times\!10^{-2}(\ln\mathrm{Re})^3+0.556(\ln\mathrm{Re})^2"
                r"-4.092\ln\mathrm{Re}+6.217,\quad \mathrm{Nu}=j\,\mathrm{Re}\,\mathrm{Pr}^{1/3}",
                r"f \approx 2.5\,j",
                "Same j-factor correlation applied",
            ),
        },
    }
    corr = corr_map.get(tpms_type)
    if not corr:
        return
    if tpms_type == "SmoothPlateFin":
        title = "SmoothPlateFin baseline correlations (Nu & f)"
    elif tpms_type == "PlateFin":
        title = "PlateFin perforated-fin correlations (Nu, f & fin efficiency)"
    else:
        title = f"{tpms_type} bare-channel correlations (Nu & f)"
    with st.expander(title, expanded=False):
        st.markdown(f"**Gas / Cryogenic** ({corr['gas'][2]}):")
        st.latex(corr['gas'][0])
        st.latex(corr['gas'][1])
        if tpms_type != "PlateFin":
            st.markdown(f"**Water** ({corr['water'][2]}):")
            st.latex(corr['water'][0])
            st.latex(corr['water'][1])
        st.markdown("**HTC from Nu:**")
        st.latex(r"h = \mathrm{Nu}\,k_f / D_h")
        if tpms_type == "PlateFin":
            st.markdown("**Fin efficiency (Wang et al. 2024, Eqs. 10–12):**")
            st.latex(r"m = \sqrt{\dfrac{2h}{k_w\,t_f}},\quad \eta_f = \dfrac{\tanh(m H_f)}{m H_f},\quad \eta_h = 1 - \dfrac{A_f}{A_h}(1-\eta_f)")
            st.markdown("**Hydraulic diameter (Eq. 2):**")
            st.latex(r"D_h = \dfrac{2(H_f - t_f)(s_f - t_f)}{H_f + s_f - 2t_f}")
            st.caption(
                r"$H_f$ = fin height; $s_f$ = fin spacing; $t_f$ = fin thickness; "
                r"$k_w$ = wall conductivity; $f$ is an empirical approximation."
            )
        elif tpms_type == "SmoothPlateFin":
            st.caption(
                r"$f$ = Fanning friction factor; "
                r"$D_h = 4 A_c / P$ (rectangular duct); "
                r"SAD = $2/H$ (two flat walls); "
                r"$\mathrm{Re} = \rho u D_h / \mu$."
            )
        else:
            st.caption(
                r"$f$ = Fanning friction factor; "
                r"$D_h = 4\varepsilon L_\mathrm{cell}/(2\pi)$; "
                r"$\mathrm{Re} = \rho u D_h / \mu$."
            )


def _render_ergun_equations():
    """Show Ergun pressure drop and equivalent friction factor for packed bed."""
    with st.expander("Ergun pressure drop equations", expanded=False):
        st.markdown("**Ergun equation:**")
        st.latex(
            r"\frac{\Delta P}{L} = \frac{150\,\mu\,u_s\,(1-\varepsilon_b)^2}{\varepsilon_b^3\,d_p^2}"
            r" + \frac{1.75\,\rho\,u_s^2\,(1-\varepsilon_b)}{\varepsilon_b^3\,d_p}"
        )
        st.markdown("**Equivalent Fanning friction factor:**")
        st.latex(
            r"f_\mathrm{equiv} = \frac{D_h}{d_p}\,\frac{1-\varepsilon_b}{\varepsilon_b^3}"
            r"\!\left(\frac{300\,(1-\varepsilon_b)}{\mathrm{Re}_p} + 3.5\right)"
            r"\cdot \psi_\mathrm{TPMS}"
        )
        st.markdown("**TPMS pressure correction factors** $\\psi$:")
        st.markdown(
            "| Structure | ψ |\n"
            "|-----------|-----|\n"
            "| Gyroid | 1.15 |\n"
            "| Diamond | 1.20 |\n"
            "| Primitive | 1.10 |\n"
            "| Neovius | 1.30 |\n"
            "| FRD | 1.18 |\n"
            "| FKS | 1.12 |"
        )
        st.caption(
            r"$u_s$ = superficial velocity; $\varepsilon_b$ = bed porosity; "
            r"$d_p$ = particle diameter; $\mathrm{Re}_p = \rho u_s d_p / \mu$."
        )


def render_channel_card(channel_name, channel_state, state):
    st.markdown(f"#### {channel_name.capitalize()} Channel")
    mode_options = ["bare", "packed"]
    mode_idx = mode_options.index(channel_state["mode"]) if channel_state["mode"] in mode_options else 0
    channel_state["mode"] = st.selectbox(
        f"{channel_name} mode",
        options=mode_options,
        index=mode_idx,
        key=_k(f"{channel_name}_mode"),
    )
    structures = list(ThermoHydraulicCorrelations.get_supported_tpms_types())
    structure_idx = structures.index(channel_state["structure"]) if channel_state["structure"] in structures else 0
    channel_state["structure"] = st.selectbox(
        f"{channel_name} TPMS structure",
        options=structures,
        index=structure_idx,
        key=_k(f"{channel_name}_structure"),
    )

    # --- PlateFin fin geometry (per-channel) ---
    if channel_state["structure"] == "PlateFin":
        ch_geo = channel_state.setdefault("geometry", {})
        g = state["geometry"]   # global fallbacks
        pf1, pf2, pf3 = st.columns(3)
        ch_geo["fin_height"] = pf1.number_input(
            f"{channel_name} fin height Hf [m]", min_value=1e-4,
            value=float(ch_geo.get("fin_height") or g.get("fin_height", 9.5e-3)),
            format="%.5f", key=_k(f"{channel_name}_fin_H"),
            help="Distance between plates (fin height). Determines Dh and fin efficiency.",
        )
        ch_geo["fin_spacing"] = pf2.number_input(
            f"{channel_name} fin spacing sf [m]", min_value=1e-5,
            value=float(ch_geo.get("fin_spacing") or g.get("fin_spacing", 3.2e-3)),
            format="%.5f", key=_k(f"{channel_name}_fin_s"),
            help="Centre-to-centre fin pitch. Porosity ε = (sf−tf)/sf.",
        )
        ch_geo["fin_thickness"] = pf3.number_input(
            f"{channel_name} fin thickness tf [m]", min_value=1e-6,
            value=float(ch_geo.get("fin_thickness") or g.get("fin_thickness", 0.6e-3)),
            format="%.6f", key=_k(f"{channel_name}_fin_t"),
            help="Fin wall thickness. Used in Dh (Eq. 2) and fin efficiency (Eqs. 11–12).",
        )

    channel_state["surface_area_density"] = st.number_input(
        f"{channel_name} surface area density α [1/m]",
        min_value=1.0,
        value=float(channel_state.get("surface_area_density", 60.0)),
        help="Specific wetted area of this channel's TPMS structure. "
             "Use the Geometry Estimator in Step 1 to estimate this value.",
        key=_k(f"{channel_name}_sad"),
    )
    if channel_state["surface_area_density"] > 200:
        st.warning(
            f"**High SAD detected ({channel_state['surface_area_density']:.0f} 1/m).** "
            "A very large surface area density produces a stiff solver (high NTU), which "
            "can cause the heat-flux residual (dQ) to oscillate and prevent convergence. "
            "The solver will auto-reduce Q_damping, but if convergence fails you can also "
            "manually lower **relax_thermal** (< 0.10) and **Q_damping** (< 0.1) in the "
            "Solver Settings panel."
        )

    if channel_state["mode"] == "bare":
        _render_tpms_bare_equations(channel_state["structure"])

    if channel_state["mode"] == "packed":
        packed = channel_state["packed"]
        mode_idx = list(SUPPORTED_PACKED_MODES).index(packed["mode"]) if packed["mode"] in SUPPORTED_PACKED_MODES else 1
        packed["mode"] = st.selectbox(
            f"{channel_name} packed mode",
            options=list(SUPPORTED_PACKED_MODES),
            index=mode_idx,
            key=_k(f"{channel_name}_packed_mode"),
        )
        _HTC_MODELS = ('martin_nilles', 'dixon')
        _htc_default = packed.get("htc_model", "martin_nilles")
        _htc_idx = _HTC_MODELS.index(_htc_default) if _htc_default in _HTC_MODELS else 0
        packed["htc_model"] = st.selectbox(
            f"{channel_name} HTC model",
            options=list(_HTC_MODELS),
            index=_htc_idx,
            key=_k(f"{channel_name}_htc_model"),
        )
        _render_htc_model_equations(packed["htc_model"])
        _render_ergun_equations()
        packed["particle_diameter"] = st.number_input(
            f"{channel_name} particle diameter [m]",
            min_value=1e-6,
            value=float(packed["particle_diameter"]),
            format="%.6f",
            key=_k(f"{channel_name}_dp"),
        )
        packed["bed_porosity"] = st.slider(
            f"{channel_name} bed porosity [-]",
            min_value=0.05,
            max_value=0.95,
            value=float(packed["bed_porosity"]),
            step=0.01,
            key=_k(f"{channel_name}_bed_por"),
        )
        packed["k_solid"] = st.number_input(
            f"{channel_name} solid conductivity [W/m-K]",
            min_value=0.01,
            value=float(packed["k_solid"]),
            format="%.3f",
            key=_k(f"{channel_name}_ks"),
        )
        packed["shape_factor"] = st.slider(
            f"{channel_name} shape factor [-]",
            min_value=0.01,
            max_value=5.0,
            value=float(packed["shape_factor"]),
            step=0.01,
            key=_k(f"{channel_name}_shape"),
        )

    st.divider()
    _render_derived_channel_metrics(state, channel_name, channel_state)
    _render_channel_estimator(state, channel_name)


def render_step_channels(state):
    st.subheader("Channel Modeling")
    c1, c2 = st.columns(2)
    with c1:
        render_channel_card("hot", state["channels"]["hot"], state)
    with c2:
        render_channel_card("cold", state["channels"]["cold"], state)


