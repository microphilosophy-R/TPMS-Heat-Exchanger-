"""
TPMS Heat Exchanger Solver — main calculator class.

The TPMSHeatExchanger class contains the iterative nodal solver
for counter-flow TPMS and plate-fin heat exchangers.
"""

import copy
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fsolve
import warnings
import os

# Ensure UTF-8 output on Windows terminals that default to GBK/CP936
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Package imports (replaces flat imports from old tpms_thermo_hydraulic_calculator.py)
from correlations.thermohydraulic_correlations import ThermoHydraulicCorrelations
from visualization.plotter import TPMSVisualizer
from properties.hydrogen_properties import ThermalProperties
from visualization.convergence import ConvergenceTracker
from models.packed_bed import SUPPORTED_PACKED_MODES, create_packed_bed_model
from models.plate_fin import plate_fin_fin_efficiency

# Config helpers (extracted to solver/config.py to avoid circular imports)
from solver.config import (
    normalize_config, _default_output_paths,
    _infer_fluid_type, _validate_tpms_structure, _normalize_single_channel,
    SUPPORTED_CHANNEL_MODES, SUPPORTED_HTC_MODELS,
)

SUPPORTED_CHANNEL_MODES = SUPPORTED_CHANNEL_MODES  # re-export
SUPPORTED_HTC_MODELS    = SUPPORTED_HTC_MODELS      # re-export

warnings.filterwarnings("ignore")

class TPMSHeatExchanger:
    """
    Unified TPMS Heat Exchanger Solver.
    Data is standardized into three categories:
    1. Nodal Data (T, P, x, Props) - Size N+1
    2. Elemental Data (Q, U, Flow Params) - Size N
    3. Global Data (Geometry, Streams)
    """

    def __init__(self, config):
        self.config = normalize_config(config)

        # 1. Initialize Properties Engine
        try:
            self.h2_props = ThermalProperties()
        except ImportError:
            raise ImportError("Critical: 'hydrogen_properties.py' not found.")

        # 2. Extract Geometry — per-channel (falls back to global)
        self.N = self.config['solver']['n_elements']

        def _ch_geo(sk):
            """Return (L, W, H, Lc, tw, Hf, sf, tf, eps) for channel sk."""
            ch_geo = self.config['channels'][sk].get('geometry', {}) or {}
            g = self.config['geometry']
            # length/width are always in per-channel geo (mirrored from shared global)
            L  = ch_geo.get('length')  or g['length']
            W  = ch_geo.get('width')   or g['width']
            H  = ch_geo.get('height',  0.25)
            Lc = ch_geo.get('unit_cell_size', 5e-3)
            tw = ch_geo.get('wall_thickness', 5e-4)
            Hf = ch_geo.get('fin_height',    9.5e-3)
            sf = ch_geo.get('fin_spacing',   3.2e-3)
            tf = ch_geo.get('fin_thickness', 0.6e-3)
            eps = ch_geo.get('porosity', 0.65 if sk == 'hot' else 0.70)
            return (float(L), float(W), float(H), float(Lc), float(tw),
                    float(Hf), float(sf), float(tf), float(eps))

        L_h, W_h, H_h, Lc_h, tw_h, Hf_h, sf_h, tf_h, eps_h = _ch_geo('hot')
        L_c, W_c, H_c, Lc_c, tw_c, Hf_c, sf_c, tf_c, eps_c = _ch_geo('cold')

        # Per-channel surface area densities [1/m]
        alpha_h = self.config['channels']['hot']['surface_area_density']
        alpha_c = self.config['channels']['cold']['surface_area_density']

        # Per-channel elemental heat transfer areas [m²]
        self.A_elem_h = L_h * W_h * H_h * alpha_h / self.N
        self.A_elem_c = L_c * W_c * H_c * alpha_c / self.N
        # Legacy attributes (hot-side reference)
        self.L_HE         = L_h
        self.L_HE_c       = L_c
        self.A_heat_total  = L_h * W_h * H_h * alpha_h
        self.A_elem        = self.A_elem_h
        self.L_elem        = L_h / self.N    # hot-channel element length
        self.L_elem_c      = L_c / self.N    # cold-channel element length
        self.W             = W_h             # hot-channel cross-section width  [m]
        self.H             = H_h             # hot-channel cross-section height [m]
        self.wall_thickness   = tw_h     # hot-side TPMS skeleton thickness (legacy name)
        self.wall_thickness_c = tw_c     # cold-side TPMS skeleton thickness
        self.plate_thickness  = float(self.config['geometry'].get('plate_thickness', 1e-3))
        self.k_wall         = self.config['material']['k_wall']
        # Dividing-plate elemental areas [m²] — Abase = width × length per element
        # Used for packed-channel conductance and plate-conduction resistance
        self.Abase_elem_h = W_h * L_h / self.N
        self.Abase_elem_c = W_c * L_c / self.N

        # 3. Initialize Stream Constants (per-channel geometry applied)
        por_h = eps_h
        por_c = eps_c
        self.streams = {
            'hot': {
                'species': self.config['operating'].get('fluid_hot', 'hydrogen mixture'),
                'm': self.config['operating']['mh'],
                'tpms': self.config['channels']['hot']['structure'],
                'mode': self.config['channels']['hot']['mode'],
                'packed_mode': self.config['channels']['hot']['packed']['mode'],
                'htc_model': self.config['channels']['hot']['packed']['htc_model'],
                'porosity': por_h,
                'Ac': W_h * H_h * por_h,
                'Dh': 4 * por_h * Lc_h / (2 * np.pi),
                'fluid_type': _infer_fluid_type(self.config['operating'].get('fluid_hot', 'hydrogen mixture')),
            },
            'cold': {
                'species': self.config['operating'].get('fluid_cold', 'helium'),
                'm': self.config['operating']['mc'],
                'tpms': self.config['channels']['cold']['structure'],
                'mode': self.config['channels']['cold']['mode'],
                'packed_mode': self.config['channels']['cold']['packed']['mode'],
                'htc_model': self.config['channels']['cold']['packed']['htc_model'],
                'porosity': por_c,
                'Ac': W_c * H_c * por_c,
                'Dh': 4 * por_c * Lc_c / (2 * np.pi),
                'fluid_type': _infer_fluid_type(self.config['operating'].get('fluid_cold', 'helium')),
            }
        }

        # 3b. SmoothPlateFin override: replace TPMS Dh with rectangular-duct Dh
        _ch_WH = {'hot': (W_h, H_h), 'cold': (W_c, H_c)}
        for sk in ('hot', 'cold'):
            if self.streams[sk]['tpms'] == 'SmoothPlateFin':
                eps = self.streams[sk]['porosity']
                Wsk, Hsk = _ch_WH[sk]
                Ac_rect = Wsk * Hsk * eps
                Dh_rect = 4 * Ac_rect / (2 * (Wsk + Hsk * eps))
                self.streams[sk]['Ac'] = Ac_rect
                self.streams[sk]['Dh'] = Dh_rect

        # 3c. PlateFin override: hydraulic diameter from fin geometry (Wang et al. 2024, Eq. 2)
        #     Dh = 2(Hf - tf)(sf - tf) / (Hf + sf - 2tf)
        _ch_finparams = {
            'hot':  (Hf_h, sf_h, tf_h),
            'cold': (Hf_c, sf_c, tf_c),
        }
        for sk in ('hot', 'cold'):
            if self.streams[sk]['tpms'] == 'PlateFin':
                Hf, sf, tf = _ch_finparams[sk]
                eps_pf = (sf - tf) / sf
                Wsk, Hsk = _ch_WH[sk]
                Dh_pf = 2.0 * (Hf - tf) * (sf - tf) / max(Hf + sf - 2.0 * tf, 1e-12)
                self.streams[sk]['Ac']  = Wsk * Hsk * eps_pf
                self.streams[sk]['Dh']  = Dh_pf
                self.streams[sk]['fin_height']    = Hf
                self.streams[sk]['fin_spacing']   = sf
                self.streams[sk]['fin_thickness'] = tf
                # Pre-compute Af/Ah ratio for fin efficiency (Eqs. 8–10, Wang et al. 2024)
                # Without perforations (n=0): Af = (2Hf - tf + sf - tf), Ah = (2Hf - tf + 2*(sf - tf))
                n_d = self.config['channels'][sk]['geometry'].get('perf_density', 0.0)
                r_p = self.config['channels'][sk]['geometry'].get('perf_radius',  0.0)
                if n_d > 0 and r_p > 0:
                    # Per unit HE length, Af and Ah from Eqs. 8–9 (W/sf factor cancels in ratio)
                    Af_base = (2 * Hf - tf) + (sf - tf)
                    Ah_base = (2 * Hf - tf) + 2 * (sf - tf)
                    # Perforation corrections per unit area of fin face (n_d per m², L=1 m)
                    perf_af = 3 * n_d * np.pi * r_p**2 - 2 * n_d * np.pi * (2 * r_p) * tf
                    perf_ah = 2 * n_d * np.pi * r_p**2 - 2 * n_d * np.pi * (2 * r_p) * tf
                    Af_val = max(Af_base - perf_af, 1e-12)
                    Ah_val = max(Ah_base - perf_ah, 1e-12)
                else:
                    Af_val = (2 * Hf - tf) + (sf - tf)
                    Ah_val = (2 * Hf - tf) + 2 * (sf - tf)
                self.streams[sk]['Af_Ah_ratio'] = Af_val / max(Ah_val, 1e-12)

        # Store per-channel SAD and elemental area in stream dicts for reference
        self.streams['hot']['surface_area_density'] = alpha_h
        self.streams['cold']['surface_area_density'] = alpha_c
        self.streams['hot']['A_elem'] = self.A_elem_h
        self.streams['cold']['A_elem'] = self.A_elem_c

        # 4. Relaxation Factors (Global Attribution)
        self.relax_thermal = self.config['solver'].get('relax_thermal', self.config['solver'].get('relax', 0.15))
        self.relax_hydraulic = self.config['solver'].get('relax_hydraulic', 0.5)
        self.relax_kinetics = self.config['solver'].get('relax_kinetics', 1.0)
        self.relax_Q = self.config['solver'].get('Q_damping', 0.5)  # Heat Flux Damping

        # 4.5 Initialize channel-level closure registry
        self._initialize_channel_closures()

        # 5. Initialize State Arrays
        self._initialize_state()

        # 6. Initialize Tracker
        self.tracker = ConvergenceTracker()

    def _initialize_channel_closures(self):
        """Build per-channel closure registry (bare or packed)."""
        self.channel_closure_registry = {}
        for stream_key in ('hot', 'cold'):
            ch_cfg = self.config['channels'][stream_key]
            mode = ch_cfg['mode']
            if mode == 'packed':
                ch_geo_ov = ch_cfg.get('geometry', {}) or {}
                cell_size_ov = ch_geo_ov.get('unit_cell_size') or None
                t_wall_ov = ch_geo_ov.get('wall_thickness') or None
                packed_model = create_packed_bed_model(
                    self.config, stream_key=stream_key,
                    cell_size_override=cell_size_ov,
                    t_wall_override=t_wall_ov,
                )
            else:
                packed_model = None
            self.channel_closure_registry[stream_key] = {
                'mode': mode,
                'structure': ch_cfg['structure'],
                'packed_mode': ch_cfg['packed']['mode'],
                'htc_model': ch_cfg['packed']['htc_model'],
                'packed_model': packed_model,
            }

    def get_channel_closure(self, stream_key, Re_channel, Pr, k_f, context):
        """
        Unified level-2 closure interface.

        Returns
        -------
        Nu : float
        f : float
        htc : float
        details : dict
        """
        reg = self.channel_closure_registry[stream_key]
        structure = reg['structure']
        dh = context['Dh']

        if reg['mode'] == 'bare':
            Nu, f = ThermoHydraulicCorrelations.get_correlations(
                structure, Re_channel, Pr, context.get('fluid_type', 'Gas')
            )
            htc = Nu * k_f / max(dh, 1e-12)
            details = {'mode': 'bare', 'structure': structure}
            # PlateFin: apply weighted fin efficiency (Eqs. 10–12, Wang et al. 2024)
            if structure == 'PlateFin':
                Hf      = context.get('fin_height',    9.5e-3)
                tf_fin  = context.get('fin_thickness', 0.6e-3)
                Af_Ah   = context.get('Af_Ah_ratio',   0.5)
                eta_h   = plate_fin_fin_efficiency(htc, Hf, tf_fin, self.k_wall, Af_Ah)
                htc     = htc * eta_h
                Nu      = htc * max(dh, 1e-12) / max(k_f, 1e-12)
                details['eta_h'] = float(eta_h)
            return Nu, f, htc, details

        packed_model = reg['packed_model']
        h_eff, f_equiv, details = packed_model.get_htc_and_friction(
            Re_channel=Re_channel,
            Pr=Pr,
            k_f=k_f,
            tpms_type=structure,
            mode=reg['packed_mode'],
            htc_model=reg['htc_model'],
        )
        Nu_equiv = h_eff * dh / max(k_f, 1e-12)
        details = {
            **details,
            'mode': 'packed',
            'structure': structure,
            'packed_mode': reg['packed_mode'],
            'htc_model': reg['htc_model'],
            'Nu_equivalent': Nu_equiv,
        }
        return Nu_equiv, f_equiv, h_eff, details

    def _initialize_state(self):
        """Initializes Nodal, Elemental, and Global data structures with smart initial guess."""
        N_nodes, N_elems = self.N + 1, self.N
        ops = self.config['operating']

        # --- 1. SMART INITIAL GUESS (Epsilon-NTU based) ---
        # Get inlet properties to estimate Heat Capacity Rates
        try:
            # Inlet specific heats
            prop_h_in = self.h2_props.get_properties(ops['Th_in'], ops['Ph_in'], self.streams['hot']['species'], ops['xh_in'])
            prop_c_in = self.h2_props.get_properties(ops['Tc_in'], ops['Pc_in'], self.streams['cold']['species'])

            Cp_h = prop_h_in['cp']
            Cp_c = prop_c_in['cp']

            # Heat capacity rates
            Ch = self.streams['hot']['m'] * Cp_h
            Cc = self.streams['cold']['m'] * Cp_c

            Cmin = min(Ch, Cc)

            # Assume high effectiveness for counter-flow TPMS (approx 0.9)
            epsilon_guess = 0.9

            # Estimated Heat Load: Q = epsilon * Cmin * (Th_in - Tc_in)
            Q_guess = epsilon_guess * Cmin * (ops['Th_in'] - ops['Tc_in'])

            # Calculate estimated outlet temperatures
            # Hot (cooling): Th_out = Th_in - Q / Ch
            Th_out_est = ops['Th_in'] - Q_guess / Ch

            # Cold (heating): Tc_out = Tc_in + Q / Cc
            Tc_out_est = ops['Tc_in'] + Q_guess / Cc

        except Exception as e:
            print(f"Warning: Smart initialization failed ({e}), reverting to fallback.")
            Th_out_est = ops['Th_in'] - 20
            Tc_out_est = ops['Tc_in'] + 20

        # --- 2. NODAL DATA (Size N+1) ---
        # Primary Variables - Linear profiles based on smart endpoints
        self.Th = np.linspace(ops['Th_in'], Th_out_est, N_nodes)

        # Cold flows N->0. Index 0 is outlet, Index N is inlet.
        # Profile needs to go from Tc_out (at 0) to Tc_in (at N)
        self.Tc = np.linspace(Tc_out_est, ops['Tc_in'], N_nodes)

        self.Ph = np.linspace(ops['Ph_in'], ops['Ph_in'] * 0.99, N_nodes)
        self.Pc = np.linspace(ops['Pc_in'] * 0.99, ops['Pc_in'], N_nodes)

        if 'hydrogen' in self.streams['hot']['species']:
            # Keep initial para-fraction flat; avoids fake conversion profile when kinetics are disabled.
            self.xh = np.full(N_nodes, ops['xh_in'])
        else:
            self.xh = np.zeros(N_nodes)

        # Derived Properties (Dict of Arrays)
        # keys: rho, mu, cp, k (conductivity), h (enthalpy), s (entropy)
        prop_keys = ['rho', 'mu', 'cp', 'k', 'h', 's']
        self.props_h = {k: np.zeros(N_nodes) for k in prop_keys}
        self.props_c = {k: np.zeros(N_nodes) for k in prop_keys}

        # --- 3. ELEMENTAL DATA (Size N) ---
        # Flow Physics & Heat Transfer
        elem_keys = ['Re', 'Pr', 'Nu', 'f', 'htc']
        self.elem_h = {k: np.zeros(N_elems) for k in elem_keys}
        self.elem_c = {k: np.zeros(N_elems) for k in elem_keys}
        self.elem_details = {'hot': [None] * N_elems, 'cold': [None] * N_elems}

        self.Q = np.zeros(N_elems)
        self.U = np.zeros(N_elems)
        self.dx_dt = np.zeros(N_elems)  # kinetic para-fraction rate [1/s], signed

        # Thermal resistance arrays per element [K/W] — filled by _compute_energy_balance()
        self.R_hot  = np.zeros(N_elems)   # total hot-side resistance
        self.R_cold = np.zeros(N_elems)   # total cold-side resistance
        self.R_wall = np.zeros(N_elems)   # wall conduction resistance
        # Sub-resistances for packed channels (equals R_hot/R_cold when bare)
        self.R_hot_wall_film  = np.zeros(N_elems)
        self.R_hot_bed_cond   = np.zeros(N_elems)
        self.R_cold_wall_film = np.zeros(N_elems)
        self.R_cold_bed_cond  = np.zeros(N_elems)

        # Performance evaluation metrics (populated by _compute_performance_metrics)
        self.perf = {}

        # --- 4. CONVERGENCE MEMORY ---
        self.Th_old = np.zeros_like(self.Th)
        self.Tc_old = np.zeros_like(self.Tc)
        self.Ph_old = np.zeros_like(self.Ph)
        self.Pc_old = np.zeros_like(self.Pc)
        self.Q_old = np.zeros_like(self.Q)

        # --- 5. GLOBAL THERMODYNAMIC CAPACITY (Qmax) ---
        # Calculate the theoretical maximum heat transfer based on inlet conditions
        try:
            # Hot inlet properties
            h_h_in = self.h2_props.get_properties(ops['Th_in'], ops['Ph_in'], self.streams['hot']['species'], ops['xh_in'])['h']
            # Hot fluid at cold inlet temp at equilibrium composition (max cooling + max conversion)
            # Using x_eq(Tc_in) ensures Q_max_hot ≥ Q_actual → ε ≤ 100% even with ortho-para conversion heat
            xh_eq_at_Tc = float(self.h2_props.get_equilibrium_fraction(ops['Tc_in']))
            h_h_min = self.h2_props.get_properties(ops['Tc_in'], ops['Ph_in'], self.streams['hot']['species'], xh_eq_at_Tc)['h']

            # Cold inlet properties
            h_c_in = self.h2_props.get_properties(ops['Tc_in'], ops['Pc_in'], self.streams['cold']['species'])['h']
            # Cold fluid at hot inlet temp (max heating)
            h_c_max = self.h2_props.get_properties(ops['Th_in'], ops['Pc_in'], self.streams['cold']['species'])['h']

            Q_max_hot = self.streams['hot']['m'] * (h_h_in - h_h_min)
            Q_max_cold = self.streams['cold']['m'] * (h_c_max - h_c_in)

            # When ortho-para conversion is active, the conversion is an internal heat
            # source in the hot stream — the cold outlet can exceed Th_in, so Q_max_cold
            # is not the binding constraint.  Use Q_max_hot (full enthalpy including
            # conversion to equilibrium at Tc_in) as the sole capacity bound.
            # For bare channels (no conversion kinetics) use the standard min() rule.
            conversion_active = (
                'hydrogen' in self.streams['hot']['species'] and
                self.streams['hot']['mode'] == 'packed'
            )
            self.Q_max_capacity = Q_max_hot if conversion_active else min(Q_max_hot, Q_max_cold)
        except:
            self.Q_max_capacity = 1e6 # Fallback

    def _update_stream_physics(self, stream_key):
        """
        Unified Physics Loop: Populates Nodal Properties and Heat Transfer Coefficients.
        NOTE: Pressure drop is handled separately in _update_hydraulics.
        """
        is_hot = (stream_key == 'hot')

        # Select Pointers
        if is_hot:
            T, P, x = self.Th, self.Ph, self.xh
            props_dict, elem_dict = self.props_h, self.elem_h
        else:
            T, P, x = self.Tc, self.Pc, None
            props_dict, elem_dict = self.props_c, self.elem_c

        # Unpack Constants
        s = self.streams[stream_key]
        species, m_dot = s['species'], s['m']
        Ac, Dh = s['Ac'], s['Dh']

        # Iteration direction: Hot (0 -> N), Cold (N -> 0)
        indices = range(self.N + 1) if is_hot else range(self.N, -1, -1)

        for i in indices:
            # --- A. Update Nodal Properties (at Node i) ---
            try:
                p_val = self.h2_props.get_properties(T[i], P[i], species, x[i] if x is not None else None)
            except ValueError:
                p_val = self.h2_props.get_properties(max(10, T[i]), P[i], species, x[i] if x is not None else None)

            # Map to Class Dict
            props_dict['rho'][i] = p_val['rho']
            props_dict['mu'][i]  = max(p_val['mu'], 1e-7)
            props_dict['cp'][i]  = p_val['cp']
            props_dict['k'][i]   = p_val['lambda']
            props_dict['h'][i]   = p_val['h']
            props_dict['s'][i]   = p_val.get('s', 0.0)   # specific entropy [J/kg·K]

            # --- B. Update Elemental Physics (HTC only) ---
            # Hot: Element i corresponds to Node i (Upwind)
            # Cold: Element i-1 corresponds to Node i (Upwind)
            if is_hot and i < self.N:
                elem_idx = i
                calc_elem = True
            elif not is_hot and i > 0:
                elem_idx = i - 1
                calc_elem = True
            else:
                calc_elem = False

            if calc_elem:
                # Use Upwind Properties (Node i)
                rho, mu = props_dict['rho'][i], props_dict['mu'][i]
                k_therm, cp = props_dict['k'][i], props_dict['cp'][i]

                # Velocity & Reynolds
                u = m_dot / (rho * Ac)
                Re = rho * u * Dh / mu
                Pr = mu * cp / k_therm

                # Heat-transfer and friction closure from level-2 registry.
                _ctx = {
                    'Dh': Dh,
                    'fluid_type': s.get('fluid_type', 'Gas'),
                }
                # Pass PlateFin fin geometry into context for fin-efficiency calculation
                if s.get('tpms') == 'PlateFin':
                    _ctx['fin_height']    = s.get('fin_height',    9.5e-3)
                    _ctx['fin_thickness'] = s.get('fin_thickness', 0.6e-3)
                    _ctx['Af_Ah_ratio']   = s.get('Af_Ah_ratio',  0.5)
                Nu, f, htc, details = self.get_channel_closure(
                    stream_key=stream_key,
                    Re_channel=Re,
                    Pr=Pr,
                    k_f=k_therm,
                    context=_ctx,
                )

                # Store Data
                elem_dict['Re'][elem_idx] = Re
                elem_dict['Pr'][elem_idx] = Pr
                elem_dict['Nu'][elem_idx] = Nu
                elem_dict['f'][elem_idx] = f  # Store friction factor
                elem_dict['htc'][elem_idx] = htc
                self.elem_details[stream_key][elem_idx] = details

    def _compute_hydraulic_balance(self):
        """
        Updates the Hydraulic Balance (Pressure Drop) for both streams.
        Includes relaxation for stability.
        """
        for stream_key in ['hot', 'cold']:
            is_hot = (stream_key == 'hot')

            # Select Pointers
            if is_hot:
                P_current, props_dict, elem_dict = self.Ph, self.props_h, self.elem_h
                P_old_state = self.Ph_old
            else:
                P_current, props_dict, elem_dict = self.Pc, self.props_c, self.elem_c
                P_old_state = self.Pc_old

            # Temporary array for the calculated pressure profile (Strict Continuity)
            P_calc = P_current.copy()

            s = self.streams[stream_key]
            m_dot, Ac, Dh = s['m'], s['Ac'], s['Dh']

            # Loop elements
            # Hot: 0 -> N (calc element i, update P[i+1])
            # Cold: N -> 0 (calc element i-1 using node i props, update P[i-1])
            indices = range(self.N) if is_hot else range(self.N, 0, -1)

            for i in indices:
                if is_hot:
                    node_idx = i
                    elem_idx = i
                    next_node_idx = i + 1
                    sign = -1.0 # P drops downstream
                else:
                    node_idx = i
                    elem_idx = i - 1
                    next_node_idx = i - 1
                    sign = -1.0 # P drops downstream

                # Properties from Upwind Node
                rho = props_dict['rho'][node_idx]
                f = elem_dict['f'][elem_idx]
                u = m_dot / (rho * Ac)

                # Explicit Pressure Drop
                L_elem_sk = self.L_elem if is_hot else self.L_elem_c
                dP = f * (L_elem_sk / Dh) * (rho * u**2 / 2)

                # Update P_calc strictly
                P_calc[next_node_idx] = P_calc[node_idx] + (sign * dP)

            # Apply Relaxation: P_new = P_old + relax * (P_calc - P_old)
            # using P_old_state (from start of iteration) for robust damping
            if is_hot:
                self.Ph = P_old_state + self.relax_hydraulic * (P_calc - P_old_state)
            else:
                self.Pc = P_old_state + self.relax_hydraulic * (P_calc - P_old_state)

    def _compute_energy_balance(self, relax):
        """Solves Energy Balance using Enthalpy Relaxation with Thermodynamic Limits."""
        mh = self.streams['hot']['m']
        mc = self.streams['cold']['m']

        # 1. Calculate Heat Load & UA (Elemental)
        # For packed channels h_eff is referenced to Abase (= width × length per element)
        # so that G = h_eff * Abase_elem correctly gives W/K = q/dt.
        # For bare channels h_eff is referenced to total TPMS surface area A_elem.
        _hot_packed  = (self.channel_closure_registry['hot']['mode']  == 'packed')
        _cold_packed = (self.channel_closure_registry['cold']['mode'] == 'packed')
        A_ref_h = self.Abase_elem_h if _hot_packed  else self.A_elem_h
        A_ref_c = self.Abase_elem_c if _cold_packed else self.A_elem_c

        for i in range(self.N):
            # Per-element thermal conductances [W/K]
            h_hot  = max(self.elem_h['htc'][i], 1e-5)
            h_cold = max(self.elem_c['htc'][i], 1e-5)

            G_hot  = h_hot  * A_ref_h   # hot-side conductance [W/K]
            G_cold = h_cold * A_ref_c   # cold-side conductance [W/K]
            # Dividing-plate conduction uses actual plate area (Abase_elem_h) [W/K]
            G_wall = self.k_wall * self.Abase_elem_h / max(self.plate_thickness, 1e-9)

            # Total conductance UA [W/K] — reference-area-independent
            UA_elem = 1.0 / (1.0/G_hot + 1.0/G_wall + 1.0/G_cold)
            # Store U referenced to hot-side reference area for output compatibility
            self.U[i] = UA_elem / A_ref_h

            # --- Thermal resistance breakdown [K/W] ---
            R_hot_total  = 1.0 / G_hot
            R_cold_total = 1.0 / G_cold
            R_wall_total = 1.0 / G_wall
            self.R_hot[i]  = R_hot_total
            self.R_cold[i] = R_cold_total
            self.R_wall[i] = R_wall_total

            # Sub-split for packed hot channel
            d_h = self.elem_details['hot'][i]
            if d_h and d_h.get('mode') == 'packed' and 'R_wall_film' in d_h:
                r_pack = max(d_h['R_total'], 1e-30)
                self.R_hot_wall_film[i] = (d_h['R_wall_film']      / r_pack) * R_hot_total
                self.R_hot_bed_cond[i]  = (d_h['R_bed_conduction'] / r_pack) * R_hot_total
            else:
                self.R_hot_wall_film[i] = R_hot_total   # bare: single bucket
                self.R_hot_bed_cond[i]  = 0.0

            # Sub-split for packed cold channel
            d_c = self.elem_details['cold'][i]
            if d_c and d_c.get('mode') == 'packed' and 'R_wall_film' in d_c:
                r_pack = max(d_c['R_total'], 1e-30)
                self.R_cold_wall_film[i] = (d_c['R_wall_film']      / r_pack) * R_cold_total
                self.R_cold_bed_cond[i]  = (d_c['R_bed_conduction'] / r_pack) * R_cold_total
            else:
                self.R_cold_wall_film[i] = R_cold_total  # bare: single bucket
                self.R_cold_bed_cond[i]  = 0.0

            # Average Temps for Element
            Th_avg = 0.5 * (self.Th[i] + self.Th[i + 1])
            Tc_avg = 0.5 * (self.Tc[i] + self.Tc[i + 1])

            # Raw Heat Transfer [W] — UA·ΔT, no explicit area needed
            Q_raw = UA_elem * (Th_avg - Tc_avg)

            # --- THERMODYNAMIC LIMIT CHECK (Local Enthalpy Potential) ---
            # Q cannot exceed the capacity of the hot stream to cool to Tc_avg
            # nor the capacity of the cold stream to heat to Th_avg
            try:
                # Hot stream potential: mh * (h(Th_avg) - h(Tc_avg))
                # Note: We use average temps to estimate the element limit
                props_h_at_Th = self.h2_props.get_properties(Th_avg, self.Ph[i], self.streams['hot']['species'], self.xh[i])
                props_h_at_Tc = self.h2_props.get_properties(Tc_avg, self.Ph[i], self.streams['hot']['species'], self.xh[i])
                dQ_max_hot = mh * (props_h_at_Th['h'] - props_h_at_Tc['h'])

                # Cold stream potential: mc * (h(Th_avg) - h(Tc_avg))
                props_c_at_Tc = self.h2_props.get_properties(Tc_avg, self.Pc[i], self.streams['cold']['species'])
                props_c_at_Th = self.h2_props.get_properties(Th_avg, self.Pc[i], self.streams['cold']['species'])
                dQ_max_cold = mc * (props_c_at_Th['h'] - props_c_at_Tc['h'])

                # The physical limit for this element
                Q_limit = min(max(0, dQ_max_hot), max(0, dQ_max_cold))

                # Calculate Target Q (Instantaneous calculation)
                if Q_raw >= 0:
                    Q_target = min(Q_raw, Q_limit)
                else:
                    Q_target = max(Q_raw, -Q_limit)

                # --- APPLY EXPLICIT DAMPING TO HEAT FLUX ---
                # This breaks the resonance loop between Temperature and HTC
                # Q_new = Q_old + relax_Q * (Q_target - Q_old)
                self.Q[i] = self.Q[i] + self.relax_Q * (Q_target - self.Q[i])

            except:
                # Fallback if properties fail
                self.Q[i] = Q_raw

        # 2. Enthalpy Targeting & Relaxation (Nodal)

        # Hot Stream (Forward)
        H_h_targ = np.zeros(self.N + 1)
        H_h_targ[0] = self.props_h['h'][0] # Inlet fixed
        for i in range(self.N):
            H_h_targ[i + 1] = H_h_targ[i] - self.Q[i] / mh

        # Cold Stream (Backward)
        H_c_targ = np.zeros(self.N + 1)
        H_c_targ[-1] = self.props_c['h'][-1] # Inlet fixed (at end)
        for i in range(self.N - 1, -1, -1):
            H_c_targ[i] = H_c_targ[i + 1] + self.Q[i] / mc

        # Relax: New Enthalpy = Current Thermo H + relax * (Target - Current Thermo H)
        H_h_new = self.props_h['h'] + relax * (H_h_targ - self.props_h['h'])
        H_c_new = self.props_c['h'] + relax * (H_c_targ - self.props_c['h'])

        # 3. Temperature Inversion
        self.Th = self._invert_enthalpy(H_h_new, 'hot', self.Th)
        self.Tc = self._invert_enthalpy(H_c_new, 'cold', self.Tc)

    def _invert_enthalpy(self, H_arr, stream_key, T_guess_arr):
        """Helper to invert Enthalpy to Temperature."""
        T_out = np.zeros_like(T_guess_arr)
        species = self.streams[stream_key]['species']
        P_arr = self.Ph if stream_key == 'hot' else self.Pc

        for i in range(len(H_arr)):
            x_val = self.xh[i] if stream_key == 'hot' else None

            def res(T):
                try:
                    p = self.h2_props.get_properties(T, P_arr[i], species, x_val)
                    return p['h'] - H_arr[i]
                except: return 1e6

            try:
                sol = fsolve(res, T_guess_arr[i], xtol=1e-3)
                T_val = np.clip(float(sol[0]), T_guess_arr[i] - 10, T_guess_arr[i] + 10) # Clamp
                T_out[i] = max(10.0, min(500.0, T_val))
            except:
                T_out[i] = T_guess_arr[i]
        return T_out

    def _ortho_para_conversion(self):
        """Calculates kinetic conversion of Ortho->Para Hydrogen with Relaxation."""
        if 'hydrogen' not in self.streams['hot']['species']:
            return

        if self.streams['hot'].get('mode', 'bare') != 'packed':
            # Packed catalyst absent: force zero reaction-rate behavior.
            self.xh[:] = self.xh[0]
            return

        xh_calc = self.xh.copy() # Temporary array for calculated profile
        Ac = self.streams['hot']['Ac']
        mh = self.streams['hot']['m']
        Tc_H2, Pc_H2 = 32.938, 1.284e6

        for i in range(self.N):
            T = 0.5 * (self.Th[i] + self.Th[i + 1])
            P = 0.5 * (self.Ph[i] + self.Ph[i + 1])
            x = 0.5 * (self.xh[i] + self.xh[i + 1])

            if T > 100:
                xh_calc[i + 1] = self.xh[i]
                continue

            try:
                x_eq = self.h2_props.get_equilibrium_fraction(T)
                props = self.h2_props.get_properties(T, P, "hydrogen mixture", x)
                rho = props['rho']

                C_H2 = rho / 0.002016
                kw = 34.76 - 220.9 * (T / Tc_H2) - 20.65 * (P / Pc_H2)

                # Guard against blow-up at pure limits (x→0 or x→1)
                x_s    = np.clip(x,    1e-9, 1.0 - 1e-9)
                x_eq_s = np.clip(x_eq, 1e-9, 1.0 - 1e-9)
                term   = (1.0 - x_eq_s) / (1.0 - x_s)
                # Reversible: rate > 0 (forward, ortho→para) when x < x_eq
                #             rate < 0 (backward, para→ortho) when x > x_eq
                # kw < 0 at cryogenic T → sign convention consistent with thermodynamics
                rate = (kw / C_H2) * np.log(term)   # [1/s], signed
                rate = np.clip(rate, -10.0, 10.0)    # symmetric magnitude cap only

                u = mh / (rho * Ac)
                # Calculate downstream x based on upstream x (xh[i])
                # Note: using self.xh[i] (current best guess) as base is standard for spatial marching
                xh_calc[i + 1] = np.clip(self.xh[i] + rate * (self.L_elem / u), 0.0, 1.0)
                self.dx_dt[i]  = rate   # kinetic para-fraction rate [1/s], signed
            except:
                xh_calc[i + 1] = self.xh[i]

        # Apply Relaxation: x_new = x_old + relax * (x_calc - x_old)
        # Using implicit memory of previous iteration (self.xh before update is functionally x_old_iter)
        # But strictly, self.xh changes during the loop if we were doing gauss-seidel.
        # Here xh_calc is a fresh profile based on current T/P.
        # We relax it against the previous iteration's profile (which is self.xh currently)

        self.xh = self.xh + self.relax_kinetics * (xh_calc - self.xh)

    def _calculate_unified_error(self):
        """Calculates dimensionless unified error metric."""
        # 1. Temperature Error
        err_T = max(
            np.max(np.abs(self.Th - self.Th_old) / (np.abs(self.Th) + 1e-5)),
            np.max(np.abs(self.Tc - self.Tc_old) / (np.abs(self.Tc) + 1e-5))
        )

        # 2. Pressure Error
        err_P = max(
            np.max(np.abs(self.Ph - self.Ph_old) / (np.abs(self.Ph) + 1e-5)),
            np.max(np.abs(self.Pc - self.Pc_old) / (np.abs(self.Pc) + 1e-5))
        )

        # 3. Heat Flux Error
        Q_scale = np.mean(np.abs(self.Q)) if np.mean(np.abs(self.Q)) > 1e-6 else 1.0
        err_Q = np.max(np.abs(self.Q - self.Q_old)) / Q_scale

        # 4. Weighted Sum
        # Weights: T=1.0, Q=1.0, P=0.5
        w_T, w_Q, w_P = 1.0, 1.0, 0.5
        total_error = (w_T * err_T + w_Q * err_Q + w_P * err_P) / (w_T + w_Q + w_P)

        return total_error, {'total': total_error, 'dT': err_T, 'dP': err_P, 'dQ': err_Q}

    def solve(self, max_iter=500, tolerance=1e-4):
        print("=" * 70)
        print(
            "TPMS Solver Started | "
            f"Hot: {self.streams['hot']['species']} ({self.streams['hot']['mode']}) | "
            f"Cold: {self.streams['cold']['species']} ({self.streams['cold']['mode']})"
        )
        print(f"Relaxation: Therm={self.relax_thermal}, Hydro={self.relax_hydraulic}, Kin={self.relax_kinetics}")
        if 'hydrogen' in self.streams['hot']['species'] and self.streams['hot']['mode'] == 'packed':
            print("Conversion model: ON (hot channel packed)")
        elif 'hydrogen' in self.streams['hot']['species']:
            print("Conversion model: OFF (hot channel bare)")
        else:
            print("Conversion model: OFF (hot fluid non-hydrogen)")
        print("=" * 70)

        # Initial Physics Pass to populate properties
        self._update_stream_physics('hot')
        self._update_stream_physics('cold')

        # Auto-tune Q_damping (relax_Q) based on estimated NTU.
        # High surface-to-volume ratio (high SAD) produces large UA_elem, making the
        # solver stiff. A large relax_Q (0.5) causes Q to overshoot the thermodynamic
        # limit each iteration → oscillatory non-convergence. Reducing relax_Q for
        # high-NTU cases stabilises the Q update and allows dQ to decay.
        try:
            h_hot_mean  = max(np.mean(self.elem_h['htc']), 1e-9)
            h_cold_mean = max(np.mean(self.elem_c['htc']), 1e-9)
            # Use the same reference area as _compute_energy_balance:
            # packed channels → h is referenced to Abase (W×L/N), not TPMS surface area
            _hot_packed_est  = (self.channel_closure_registry['hot']['mode']  == 'packed')
            _cold_packed_est = (self.channel_closure_registry['cold']['mode'] == 'packed')
            _A_ref_h_est = self.Abase_elem_h if _hot_packed_est  else self.A_elem_h
            _A_ref_c_est = self.Abase_elem_c if _cold_packed_est else self.A_elem_c
            G_hot_elem  = h_hot_mean  * _A_ref_h_est
            G_cold_elem = h_cold_mean * _A_ref_c_est
            G_wall_elem = self.k_wall * self.Abase_elem_h / max(self.plate_thickness, 1e-9)
            UA_elem_est = 1.0 / (1.0 / max(G_hot_elem, 1e-30) +
                                 1.0 / max(G_wall_elem, 1e-30) +
                                 1.0 / max(G_cold_elem, 1e-30))
            UA_total_est = UA_elem_est * self.N
            C_min_est = min(
                self.streams['hot']['m']  * max(np.mean(self.props_h['cp']), 1.0),
                self.streams['cold']['m'] * max(np.mean(self.props_c['cp']), 1.0)
            )
            NTU_est = UA_total_est / max(C_min_est, 1e-6)
            # Scale relax_Q DOWN for stiff (high-NTU) systems
            if NTU_est > 20:
                auto_relax_Q = 0.05
            elif NTU_est > 5:
                auto_relax_Q = 0.10
            elif NTU_est > 1:
                auto_relax_Q = 0.25
            else:
                auto_relax_Q = self.relax_Q  # no change for NTU < 1
            self.relax_Q = min(self.relax_Q, auto_relax_Q)
            print(f"NTU estimate = {NTU_est:.2f} → Q_damping auto-set to {self.relax_Q:.3f}")
        except Exception:
            pass  # Keep user-configured relax_Q on any error

        # Error-tracking adaptive relaxation state
        # Initialized here so error_val is always defined before the update step.
        adaptive_relax = self.relax_thermal
        _prev_error = float('inf')

        for iteration in range(max_iter):
            # Snapshot Old State
            self.Th_old, self.Tc_old = self.Th.copy(), self.Tc.copy()
            self.Ph_old, self.Pc_old = self.Ph.copy(), self.Pc.copy()
            self.Q_old = self.Q.copy()

            # 1. Physics (Properties & HTC)
            self._update_stream_physics('hot')
            self._update_stream_physics('cold')

            # 2. Kinetics
            self._ortho_para_conversion()

            # 3. Hydraulics (Pressure Drop)
            self._compute_hydraulic_balance()

            # 4. Energy Balance (Q & T) — uses adaptive_relax from previous iteration's error check
            self._compute_energy_balance(adaptive_relax)

            # 5. Error Check
            error_val, error_dict = self._calculate_unified_error()

            # Update adaptive_relax for NEXT iteration using error-tracking strategy.
            # High-SAD (stiff) systems need smaller relaxation when error grows, not larger.
            # The old monotonic ramp (relax_thermal + 0.01*iter) increased into the unstable
            # regime for stiff problems. This version backs off when the error increases.
            if error_val > _prev_error * 1.05:
                # Error grew → reduce relaxation to damp oscillations
                adaptive_relax = max(0.05, adaptive_relax * 0.85)
            else:
                # Error shrinking or flat → cautiously increase up to 0.5
                adaptive_relax = min(0.5, adaptive_relax * 1.03)
            _prev_error = error_val

            # 6. Tracking
            self.tracker.update(iteration, error_dict, self, adaptive_relax)
            if (iteration + 1) % 10 == 0 or iteration < 5:
                print(f"Iter {iteration + 1:3d} | Err: {error_val:.2e} (dT:{error_dict['dT']:.1e}, dQ:{error_dict['dQ']:.1e}, dP:{error_dict['dP']:.1e}) | Q_tot: {np.sum(self.Q):.2f}W | relax:{adaptive_relax:.3f}")
                if np.isnan(error_val) or error_val > 1e4:
                    print("!!! Divergence Detected !!!"); break

            if error_val < tolerance:
                print(f"\n*** CONVERGED in {iteration + 1} iterations ***")
                self._print_results()
                return True

        print("!!! Max Iterations Reached !!!")
        self._print_results()
        return False

    def _compute_performance_metrics(self):
        """Compute post-processing performance indicators.

        Exergy analysis — Gouy-Stodola decomposition for a cryogenic liquefaction HX
        -------------------------------------------------------------------------------
        Dead state: T0 = T_ambient (default 298 K), NOT Tc_in.  Both streams operate
        below ambient, so their cold exergy is positive and large relative to 298 K.

        Physical roles in a sub-ambient counter-flow cooler (element i: hot i→i+1, cold i+1→i):
          • Refrigerant (cold stream, warmer of the two) SUPPLIES cold exergy as it warms.
          • Product    (hot  stream, cooled to lower T) RECEIVES cold exergy as it cools.

        Three irreversibility sources (Gouy-Stodola, T0 × Sgen):
          1. Heat transfer across finite ΔT  → Ex_dest_HT
          2. Pressure drop (negligible)      → Ex_dest_dP  (= 0)
          3. Ortho-para chemical conversion  → Ex_dest_chem  (packed-bed hot channel)
        """
        ops = self.config['operating']
        # --- Dead state: ambient temperature (NOT Tc_in) ---
        T0 = float(ops.get('T_ambient', 298.0))
        mh = self.streams['hot']['m']
        mc = self.streams['cold']['m']
        N = self.N

        # --- 1. j-factor and PEC (per element) — unchanged ---
        j_h   = np.zeros(N)
        j_c   = np.zeros(N)
        PEC_h = np.zeros(N)
        PEC_c = np.zeros(N)
        for i in range(N):
            Re_h = max(self.elem_h['Re'][i], 1e-6)
            Re_c = max(self.elem_c['Re'][i], 1e-6)
            Pr_h = max(self.elem_h['Pr'][i], 1e-6)
            Pr_c = max(self.elem_c['Pr'][i], 1e-6)
            f_h  = max(self.elem_h['f'][i],  1e-12)
            f_c  = max(self.elem_c['f'][i],  1e-12)
            j_h[i]   = self.elem_h['Nu'][i] * Pr_h**(-1.0/3.0) / Re_h
            j_c[i]   = self.elem_c['Nu'][i] * Pr_c**(-1.0/3.0) / Re_c
            PEC_h[i] = j_h[i] / f_h**(1.0/3.0)
            PEC_c[i] = j_c[i] / f_c**(1.0/3.0)

        # --- 2. Dead-state reference properties ---
        # Hot dead-state: T0 at inlet pressure, equilibrium para-fraction at T0
        try:
            x0_h = float(self.h2_props.get_equilibrium_fraction(T0))  # ≈0.25 at 298 K
            ref_h = self.h2_props.get_properties(
                T0, ops['Ph_in'], self.streams['hot']['species'], x0_h)
            h0_h, s0_h = ref_h['h'], ref_h.get('s', 0.0)
        except Exception:
            h0_h, s0_h = 0.0, 0.0

        try:
            ref_c = self.h2_props.get_properties(
                T0, ops['Pc_in'], self.streams['cold']['species'])
            h0_c, s0_c = ref_c['h'], ref_c.get('s', 0.0)
        except Exception:
            h0_c, s0_c = 0.0, 0.0

        # --- 3. Nodal flow exergy  ex = (h - h0) - T0*(s - s0)  [J/kg] ---
        # With T0 = 298 K and streams at 66-96 K: s < s0, h < h0,
        # but −T0*(s−s0) dominates → ex_i > 0  (correct cold exergy sign)
        ex_h = (self.props_h['h'] - h0_h) - T0 * (self.props_h['s'] - s0_h)
        ex_c = (self.props_c['h'] - h0_c) - T0 * (self.props_c['s'] - s0_c)

        # --- 4. Per-element exergy balance + Gouy-Stodola decomposition ---
        # Counter-flow: hot 0→N, cold N→0
        # Refrigerant (cold stream) supplies cold exergy  [> 0 when it warms up]
        # Product     (hot  stream) receives cold exergy  [> 0 when it cools down]
        Ex_cold_supplied = np.zeros(N)   # [W]  legacy per-element tracker
        Ex_hot_received  = np.zeros(N)   # [W]  legacy per-element tracker
        Ex_dest_HT       = np.zeros(N)   # Source 1: finite-ΔT heat transfer [W]
        Ex_dest_chem     = np.zeros(N)   # Source 3: ortho-para conversion   [W]
        Ex_dest_dP       = np.zeros(N)   # Source 2: pressure drop           [W]
        S_gen_HT         = np.zeros(N)   # [W/K]
        S_gen_dP         = np.zeros(N)   # [W/K]
        S_gen_chem       = np.zeros(N)   # [W/K]

        for i in range(N):
            # Legacy per-element exergy trackers (for backward-compat output)
            Ex_cold_supplied[i] = mc * (ex_c[i + 1] - ex_c[i])
            Ex_hot_received[i]  = mh * (ex_h[i + 1] - ex_h[i])

            Th_avg = 0.5 * (self.Th[i] + self.Th[i + 1])
            Tc_avg = 0.5 * (self.Tc[i] + self.Tc[i + 1])

            # Source 1 — Heat-transfer irreversibility (Gouy-Stodola)
            if Th_avg > Tc_avg > 0.0:
                S_gen_HT[i] = self.Q[i] * (1.0 / Tc_avg - 1.0 / Th_avg)
            Ex_dest_HT[i] = T0 * max(S_gen_HT[i], 0.0)

            # Source 2 — Pressure-drop irreversibility: ṁ/(ρ·T) × |ΔP|
            # Hot flows 0→N: pressure drops, dP_h = Ph[i] - Ph[i+1] ≥ 0
            # Cold flows N→0: pressure drops, dP_c = Pc[i+1] - Pc[i] ≥ 0
            rho_h_avg = 0.5 * (self.props_h['rho'][i] + self.props_h['rho'][i + 1])
            rho_c_avg = 0.5 * (self.props_c['rho'][i] + self.props_c['rho'][i + 1])
            dP_h = max(self.Ph[i] - self.Ph[i + 1], 0.0)
            dP_c = max(self.Pc[i + 1] - self.Pc[i], 0.0)
            S_gen_dP_h_i = mh * dP_h / (max(rho_h_avg, 1e-12) * max(Th_avg, 1.0))
            S_gen_dP_c_i = mc * dP_c / (max(rho_c_avg, 1e-12) * max(Tc_avg, 1.0))
            S_gen_dP[i]  = S_gen_dP_h_i + S_gen_dP_c_i
            Ex_dest_dP[i] = T0 * S_gen_dP[i]

            # Source 3 — Ortho-para chemical conversion irreversibility
            # Derived from the element exergy balance (Gouy-Stodola residual method):
            #   Ex_dest_chem[i] = dEx_c[i] - dEx_h[i] - Ex_dest_HT[i] - Ex_dest_dP[i]
            # This is thermodynamically exact: it uses the same ex_h/ex_c nodal arrays
            # that define Ex_H2_total_gain and Ex_He_consumed, so the global balance
            # closes to zero by construction (modulo the small |Δμ/RT| residual from
            # CoolProp's independent para/ortho reference states).
            # Negative values are clamped to zero (unphysical rounding in near-equilibrium
            # elements due to the finite Gibbs-anchor error at temperatures ≠ 300 K).
            if 'hydrogen' in self.streams['hot']['species']:
                dEx_c_i = mc * (ex_c[i + 1] - ex_c[i])
                dEx_h_i = mh * (ex_h[i + 1] - ex_h[i])
                ex_dest_chem_i = dEx_c_i - dEx_h_i - Ex_dest_HT[i] - Ex_dest_dP[i]
                Ex_dest_chem[i] = max(ex_dest_chem_i, 0.0)
                S_gen_chem[i]   = Ex_dest_chem[i] / T0

        Ex_dest_total = Ex_dest_HT + Ex_dest_chem + Ex_dest_dP

        # --- 5. Global totals ---
        Ex_dest_HT_tot   = float(np.sum(Ex_dest_HT))
        Ex_dest_chem_tot = float(np.sum(Ex_dest_chem))
        Ex_dest_dP_tot   = float(np.sum(Ex_dest_dP))
        Ex_dest_grand    = Ex_dest_HT_tot + Ex_dest_chem_tot + Ex_dest_dP_tot

        # Global entropy totals (all irreversibility sources)
        S_gen_HT_tot   = float(np.sum(np.maximum(S_gen_HT,   0.0)))
        S_gen_dP_tot   = float(np.sum(np.maximum(S_gen_dP,   0.0)))
        S_gen_chem_tot = float(np.sum(np.maximum(S_gen_chem, 0.0)))
        S_gen_total    = S_gen_HT_tot + S_gen_dP_tot + S_gen_chem_tot

        # --- 5b. Corrected exergetic efficiency (always ≤ 1) ---
        # He (cold stream) is the sole external exergy supplier.
        # Its cold exergy consumed = mc*(ex_c[inlet] − ex_c[outlet]) = mc*(ex_c[-1] − ex_c[0]).
        # The HX efficiency is evaluated for the *heat-transfer* function only:
        #   η_ex = 1 − T0·(Ṡ_gen_HT + Ṡ_gen_dP) / Ex_He_consumed  ∈ [0, 1]
        # Chemical exergy from ortho-para conversion is a separate energy source
        # and is reported independently as Ex_chem_net.
        Ex_He_consumed   = mc * float(ex_c[-1] - ex_c[0])   # always > 0
        Ex_H2_total_gain = mh * float(ex_h[-1] - ex_h[0])   # thermal + chemical

        S_gen_HT_dP_tot  = S_gen_HT_tot + S_gen_dP_tot
        Ex_dest_thermal  = T0 * S_gen_HT_dP_tot

        eta_ex = float(np.clip(
            1.0 - Ex_dest_thermal / max(Ex_He_consumed, 1e-12),
            0.0, 1.0))

        # Chemical exergy net contribution: how much of H2's exergy gain came from
        # ortho-para conversion (internal source) rather than the He refrigerant.
        # Ex_chem_net > 0 when conversion adds exergy beyond what He supplied (thermally).
        Ex_chem_net = Ex_H2_total_gain - (Ex_He_consumed - Ex_dest_thermal)

        # Global balance residual (Gouy-Stodola): Ex_supplied - Ex_gained - Ex_destroyed = 0
        # Correct form: He exergy in - H2 exergy gain - all destruction sources.
        # (Ex_chem_net must NOT appear here; it is already embedded in Ex_H2_total_gain
        #  and Ex_dest_chem_tot via the property model's h and s values.)
        # This residual ≈ 0 iff enthalpy/entropy accounting is thermodynamically consistent.
        Ex_balance_residual = Ex_He_consumed - Ex_H2_total_gain - Ex_dest_grand

        # --- Legacy aliases (backward compatibility) ---
        Ex_cold_net  = mc * float(ex_c[-1] - ex_c[0])   # = Ex_He_consumed
        Ex_hot_net   = mh * float(ex_h[-1] - ex_h[0])   # = Ex_H2_total_gain
        Ex_balance   = Ex_He_consumed - Ex_H2_total_gain - Ex_dest_grand
        Ex_hot_lost  = -Ex_hot_received
        Ex_cold_gain = Ex_cold_supplied
        Ex_dest      = Ex_dest_total
        S_gen        = S_gen_HT + S_gen_dP + S_gen_chem  # total per-element

        self.perf = {
            'T0':                    T0,
            'ex_h':                  ex_h,
            'ex_c':                  ex_c,
            # --- Corrected exergy efficiency (always ≤ 1) ---
            'eta_ex':                eta_ex,
            'Ex_He_consumed':        Ex_He_consumed,
            'Ex_H2_total_gain':      Ex_H2_total_gain,
            'Ex_chem_net':           Ex_chem_net,
            'Ex_dest_thermal':       Ex_dest_thermal,
            'Ex_balance_residual':   Ex_balance_residual,
            # --- Per-element arrays ---
            'Ex_cold_supplied':      Ex_cold_supplied,
            'Ex_hot_received':       Ex_hot_received,
            'Ex_dest_HT':            Ex_dest_HT,
            'Ex_dest_dP':            Ex_dest_dP,
            'Ex_dest_chem':          Ex_dest_chem,
            'Ex_dest_total':         Ex_dest_total,
            # --- Global totals (all three sources) ---
            'Ex_dest_HT_tot':        Ex_dest_HT_tot,
            'Ex_dest_dP_tot':        Ex_dest_dP_tot,
            'Ex_dest_chem_tot':      Ex_dest_chem_tot,
            'Ex_dest_grand':         Ex_dest_grand,
            # --- Entropy generation ---
            'S_gen_HT':              S_gen_HT,
            'S_gen_dP':              S_gen_dP,
            'S_gen_chem':            S_gen_chem,
            'S_gen':                 S_gen,
            'S_gen_HT_tot':          S_gen_HT_tot,
            'S_gen_dP_tot':          S_gen_dP_tot,
            'S_gen_chem_tot':        S_gen_chem_tot,
            'S_gen_total':           S_gen_total,
            # --- Legacy aliases (backward compatibility) ---
            'Ex_cold_net':           Ex_cold_net,
            'Ex_hot_net':            Ex_hot_net,
            'Ex_cold_total':         Ex_cold_net,
            'Ex_hot_total':          Ex_hot_net,
            'Ex_balance':            Ex_balance,
            'Ex_hot_lost':           Ex_hot_lost,
            'Ex_cold_gain':          Ex_cold_gain,
            'Ex_dest':               Ex_dest,
            # --- j-factor / PEC ---
            'j_h':                   j_h,
            'j_c':                   j_c,
            'PEC_h':                 PEC_h,
            'PEC_c':                 PEC_c,
            'j_mean_h':              float(np.mean(j_h)),
            'j_mean_c':              float(np.mean(j_c)),
            'PEC_mean_h':            float(np.mean(PEC_h)),
            'PEC_mean_c':            float(np.mean(PEC_c)),
        }

    def _print_results(self):
        """Print comprehensive results"""
        # Compute performance metrics before printing
        try:
            self._compute_performance_metrics()
        except Exception as _e:
            print(f"Warning: Performance metrics computation failed: {_e}")

        print("=" * 70)
        print("RESULTS - Thermo-Hydraulic Performance")
        print("=" * 70)

        # Temperatures
        print("\nTemperatures:")
        print(f"  Hot:  {self.Th[0]:.2f} K → {self.Th[-1]:.2f} K (ΔT = {self.Th[0] - self.Th[-1]:.2f} K)")
        print(f"  Cold: {self.Tc[-1]:.2f} K → {self.Tc[0]:.2f} K (ΔT = {self.Tc[0] - self.Tc[-1]:.2f} K)")

        # Pressures
        dP_hot = self.Ph[0] - self.Ph[-1]
        dP_cold = self.Pc[-1] - self.Pc[0]
        print("\nPressures:")
        print(f"  Hot:  {self.Ph[0]/1e6:.3f} MPa → {self.Ph[-1]/1e6:.3f} MPa (ΔP = {dP_hot/1e3:.2f} kPa)")
        print(f"  Cold: {self.Pc[-1]/1e6:.3f} MPa → {self.Pc[0]/1e6:.3f} MPa (ΔP = {dP_cold/1e3:.2f} kPa)")

        # Heat transfer — use inlet/outlet enthalpy states (includes conversion heat)
        mh = self.streams['hot']['m']
        Q_total = mh * (self.props_h['h'][0] - self.props_h['h'][-1])
        print("\nHeat Transfer:")
        print(f"  Total heat load: {Q_total:.2f} W")
        print(f"  Theoretical Max Capacity (Global): {self.Q_max_capacity:.2f} W")
        print(f"  Effectiveness: {Q_total/max(self.Q_max_capacity,1e-12)*100:.1f} %")
        print(f"  Avg U: {np.mean(self.U):.2f} W/m2K")

        # Conversion
        print("\nConversion:")
        print(f"  Para-H2: {self.xh[0]:.4f} -> {self.xh[-1]:.4f}")
        if 'hydrogen' in self.streams['hot']['species'] and self.streams['hot']['mode'] == 'packed':
            print("  Conversion model: ON (hot channel packed)")
        elif 'hydrogen' in self.streams['hot']['species']:
            print("  Conversion model: OFF (hot channel bare)")
        else:
            print("  Conversion model: OFF (hot fluid non-hydrogen)")

        # Energy balance check
        try:
            h_h_in, h_h_out = self.props_h['h'][0], self.props_h['h'][-1]
            h_c_in, h_c_out = self.props_c['h'][-1], self.props_c['h'][0]

            Q_hot = self.config['operating']['mh'] * (h_h_in - h_h_out)
            Q_cold = self.config['operating']['mc'] * (h_c_out - h_c_in)
            imbalance = abs(Q_hot - Q_cold) / max(abs(Q_hot), 1e-5) * 100

            print("\nEnergy Balance:")
            print(f"  Hot stream loss:  {Q_hot:.2f} W")
            print(f"  Cold stream gain: {Q_cold:.2f} W")
            print(f"  Imbalance: {imbalance:.2f}%")
        except Exception as e:
            print(f"\nEnergy Balance: Could not calculate - {e}")

        # Performance Evaluation Indicators
        if self.perf:
            p = self.perf
            print("\nExergy Analysis (Gouy-Stodola, 3-source decomposition):")
            print(f"  Dead-state T0:             {p['T0']:.2f} K")
            print(f"  η_ex (HX, thermal):        {p['eta_ex']*100:.1f}%  (always ≤ 100%)")
            print(f"  He cold exergy consumed:   {p['Ex_He_consumed']:.4f} W")
            print(f"  H2 total exergy gained:    {p['Ex_H2_total_gain']:.4f} W")
            print(f"  Ex destroyed (HT+dP):      {p['Ex_dest_thermal']:.4f} W")
            print(f"  Ex destroyed (chem rxn):   {p['Ex_dest_chem_tot']:.4f} W")
            print(f"  Ex destroyed (grand total):{p['Ex_dest_grand']:.4f} W")
            print(f"  Balance residual:          {p['Ex_balance_residual']:.4f} W  (≈ 0 if consistent)")
            print(f"  S_gen total:               {p['S_gen_total']*1e3:.3f} mW/K")
            print(f"    ↳ Heat transfer ΔT:      {p['S_gen_HT_tot']*1e3:.3f} mW/K")
            print(f"    ↳ Pressure drop ΔP:      {p['S_gen_dP_tot']*1e3:.3f} mW/K")
            print(f"    ↳ Ortho-para reaction:   {p['S_gen_chem_tot']*1e3:.3f} mW/K")
            print(f"  j-factor mean (hot/cold):  {p['j_mean_h']:.4f} / {p['j_mean_c']:.4f}")
            print(f"  PEC mean (hot/cold):       {p['PEC_mean_h']:.4f} / {p['PEC_mean_c']:.4f}")
        print("=" * 70)

    def finalize_simulation(self):
        """
        Outputs all final data:
        1. Comprehensive Performance Plots
        2. Convergence History Plot
        3. Detailed Results CSV
        4. Convergence Data CSV
        Uses paths defined in self.config['output']
        """
        # Ensure output config exists
        out_cfg = self.config.get('output', {})

        # Define paths with defaults
        path_results_csv = out_cfg.get('results_csv', 'output/tpms_final_results.csv')
        path_conv_csv = out_cfg.get('convergence_csv', 'output/tpms_convergence.csv')
        path_perf_plot = out_cfg.get('performance_plot', 'output/tpms_performance.png')
        path_conv_plot = out_cfg.get('convergence_plot', 'output/tpms_convergence.png')
        path_res_pie = out_cfg.get('resistance_pie', 'output/tpms_resistance_pie.png')

        # Create directories if they don't exist
        for path in [path_results_csv, path_conv_csv, path_perf_plot, path_conv_plot, path_res_pie]:
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                try:
                    os.makedirs(directory)
                except OSError as e:
                    print(f"Warning: Could not create directory {directory}: {e}")

        print("\n" + "="*70)
        print("FINALIZING SIMULATION OUTPUTS")
        print("="*70)

        # 1. Visualization & Results CSV
        vis = TPMSVisualizer(self)

        print(f"Generating performance plot: {path_perf_plot}...")
        vis.plot_comprehensive(save_path=path_perf_plot)

        print(f"Exporting detailed results: {path_results_csv}...")
        vis.export_results_to_csv(filename=path_results_csv)

        print(f"Generating resistance pie chart: {path_res_pie}...")
        vis.plot_resistance_pie(save_path=path_res_pie)
        # Write path back into output config so run_simulation() can return it
        self.config['output']['resistance_pie'] = path_res_pie

        # Performance evaluation plot (exergy + j-factor + PEC)
        path_perf_eval = out_cfg.get('performance_eval_plot', 'results/performance_evaluation.png')
        print(f"Generating performance evaluation plot: {path_perf_eval}...")
        vis.plot_performance_evaluation(save_path=path_perf_eval)
        self.config['output']['performance_eval_plot'] = path_perf_eval

        # 2. Convergence Tracking
        print(f"Generating convergence plot: {path_conv_plot}...")
        self.tracker.plot(save_path=path_conv_plot)

        print(f"Exporting convergence history: {path_conv_csv}...")
        self.tracker.export_csv(filepath=path_conv_csv)

        print("="*70 + "\n")



if __name__ == "__main__":
    from solver.config import create_default_config
    config = create_default_config()
    he = TPMSHeatExchanger(config)
    he.solve(
        max_iter=config['solver'].get('max_iter', 500),
        tolerance=config['solver'].get('tolerance', 1e-4),
    )
    he.finalize_simulation()
