"""
TPMS Heat Exchanger Simulation - Unified & Optimized
Content:
1. ConvergenceTracker: Tracks solution stability and exports data.
2. TPMSHeatExchanger: Unified solver with optimized physics and enthalpy relaxation.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fsolve
import warnings
import os

# Import local modules
from tpms_correlations import TPMSCorrelations
from tpms_visualization import TPMSVisualizer
from hydrogen_properties import ThermalProperties
from convergence_tracker import ConvergenceTracker

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
        self.config = config

        # 1. Initialize Properties Engine
        try:
            self.h2_props = ThermalProperties()
        except ImportError:
            raise ImportError("Critical: 'hydrogen_properties.py' not found.")

        # 2. Extract Geometry (Global Data)
        self.N = config['solver']['n_elements']
        self.L_HE = config['geometry']['length']
        self.A_heat_total = self.L_HE * config['geometry']['width'] * config['geometry']['height'] * config['geometry'][
            'surface_area_density']
        self.A_elem = self.A_heat_total / self.N
        self.L_elem = self.L_HE / self.N
        self.wall_thickness = config['geometry']['wall_thickness']
        self.k_wall = config['material']['k_wall']

        # 3. Initialize Stream Constants (Global Data)
        self.streams = {
            'hot': {
                'species': config['operating'].get('fluid_hot', 'hydrogen mixture'),
                'm': config['operating']['mh'],
                'tpms': config['tpms']['type_hot'],
                'porosity': config['geometry']['porosity_hot'],
                'Ac': config['geometry']['width'] * config['geometry']['height'] * config['geometry']['porosity_hot'],
                'Dh': 4 * config['geometry']['porosity_hot'] * config['geometry']['unit_cell_size'] / (2 * np.pi)
            },
            'cold': {
                'species': config['operating'].get('fluid_cold', 'helium'),
                'm': config['operating']['mc'],
                'tpms': config['tpms']['type_cold'],
                'porosity': config['geometry']['porosity_cold'],
                'Ac': config['geometry']['width'] * config['geometry']['height'] * config['geometry']['porosity_cold'],
                'Dh': 4 * config['geometry']['porosity_cold'] * config['geometry']['unit_cell_size'] / (2 * np.pi)
            }
        }

        # 4. Relaxation Factors (Global Attribution)
        self.relax_thermal = config['solver'].get('relax_thermal', 0.15)
        self.relax_hydraulic = config['solver'].get('relax_hydraulic', 0.5)
        self.relax_kinetics = config['solver'].get('relax_kinetics', 1.0)
        self.relax_Q = config['solver'].get('Q_damping', 0.5)  # Heat Flux Damping

        # 5. Initialize State Arrays
        self._initialize_state()

        # 6. Initialize Tracker
        self.tracker = ConvergenceTracker()

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

        self.xh = np.linspace(ops['xh_in'], 0.9, N_nodes) if 'hydrogen' in self.streams['hot']['species'] else np.zeros(N_nodes)

        # Derived Properties (Dict of Arrays)
        # keys: rho, mu, cp, k (conductivity), h (enthalpy)
        prop_keys = ['rho', 'mu', 'cp', 'k', 'h']
        self.props_h = {k: np.zeros(N_nodes) for k in prop_keys}
        self.props_c = {k: np.zeros(N_nodes) for k in prop_keys}

        # --- 3. ELEMENTAL DATA (Size N) ---
        # Flow Physics & Heat Transfer
        elem_keys = ['Re', 'Pr', 'Nu', 'f', 'htc']
        self.elem_h = {k: np.zeros(N_elems) for k in elem_keys}
        self.elem_c = {k: np.zeros(N_elems) for k in elem_keys}

        self.Q = np.zeros(N_elems)
        self.U = np.zeros(N_elems)

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
            # Hot fluid at cold inlet temp (max cooling)
            h_h_min = self.h2_props.get_properties(ops['Tc_in'], ops['Ph_in'], self.streams['hot']['species'], ops['xh_in'])['h']

            # Cold inlet properties
            h_c_in = self.h2_props.get_properties(ops['Tc_in'], ops['Pc_in'], self.streams['cold']['species'])['h']
            # Cold fluid at hot inlet temp (max heating)
            h_c_max = self.h2_props.get_properties(ops['Th_in'], ops['Pc_in'], self.streams['cold']['species'])['h']

            Q_max_hot = self.streams['hot']['m'] * (h_h_in - h_h_min)
            Q_max_cold = self.streams['cold']['m'] * (h_c_max - h_c_in)

            self.Q_max_capacity = min(Q_max_hot, Q_max_cold)
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
        Ac, Dh, tpms = s['Ac'], s['Dh'], s['tpms']

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

                # Heat Transfer Coefficient
                # Update: Capture Friction Factor 'f' here
                Nu, f = TPMSCorrelations.get_correlations(tpms, Re, Pr, 'Gas')

                # Store Data
                elem_dict['Re'][elem_idx] = Re
                elem_dict['Pr'][elem_idx] = Pr
                elem_dict['Nu'][elem_idx] = Nu
                elem_dict['f'][elem_idx] = f  # Store friction factor

                factor = 1.2 if is_hot else 1.0
                elem_dict['htc'][elem_idx] = factor * Nu * k_therm / Dh

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
                dP = f * (self.L_elem / Dh) * (rho * u**2 / 2)

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

        # 1. Calculate Heat Load & U (Elemental)
        for i in range(self.N):
            # Thermal Resistance
            h_hot = max(self.elem_h['htc'][i], 1e-5)
            h_cold = max(self.elem_c['htc'][i], 1e-5)

            R_total = (1 / h_hot) + (self.wall_thickness / self.k_wall) + (1 / h_cold)
            self.U[i] = 1 / R_total

            # Average Temps for Element
            Th_avg = 0.5 * (self.Th[i] + self.Th[i + 1])
            Tc_avg = 0.5 * (self.Tc[i] + self.Tc[i + 1])

            # Raw Heat Flux
            Q_raw = self.U[i] * self.A_elem * (Th_avg - Tc_avg)

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

                term = (1 - x_eq) / (1 - x + 1e-9)
                rate = (kw / C_H2) * np.log(term) if (x < x_eq and term > 0) else 0
                rate = max(0, min(rate, 10.0))

                u = mh / (rho * Ac)
                # Calculate downstream x based on upstream x (xh[i])
                # Note: using self.xh[i] (current best guess) as base is standard for spatial marching
                xh_calc[i + 1] = np.clip(self.xh[i] + rate * (self.L_elem / u), 0, 1.0)
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
        print(f"TPMS Solver Started | Hot: {self.streams['hot']['species']} | Cold: {self.streams['cold']['species']}")
        print(f"Relaxation: Therm={self.relax_thermal}, Hydro={self.relax_hydraulic}, Kin={self.relax_kinetics}")
        print("=" * 70)

        # Initial Physics Pass to populate properties
        self._update_stream_physics('hot')
        self._update_stream_physics('cold')

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

            # 4. Energy Balance (Q & T)
            # Use dynamic relax for thermal if needed, or stick to configured self.relax_thermal
            # Combining adaptive strategy with base configured value
            adaptive_relax = min(0.5, self.relax_thermal + 0.01 * iteration) if iteration > 10 else self.relax_thermal
            self._compute_energy_balance(adaptive_relax)

            # 5. Error Check
            error_val, error_dict = self._calculate_unified_error()

            # 6. Tracking
            self.tracker.update(iteration, error_dict, self, adaptive_relax)
            if (iteration + 1) % 10 == 0 or iteration < 5:
                print(f"Iter {iteration + 1:3d} | Err: {error_val:.2e} (dT:{error_dict['dT']:.1e}, dQ:{error_dict['dQ']:.1e}, dP:{error_dict['dP']:.1e}) | Q_tot: {np.sum(self.Q):.2f}W")
                if np.isnan(error_val) or error_val > 1e4:
                    print("!!! Divergence Detected !!!"); break

            if error_val < tolerance:
                print(f"\n*** CONVERGED in {iteration + 1} iterations ***")
                self._print_results()
                return True

        print("!!! Max Iterations Reached !!!")
        self._print_results()
        return False

    def _print_results(self):
        """Print comprehensive results"""
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

        # Heat transfer
        Q_total = np.sum(self.Q)
        print("\nHeat Transfer:")
        print(f"  Total heat load: {Q_total:.2f} W")
        print(f"  Theoretical Max Capacity (Global): {self.Q_max_capacity:.2f} W")
        print(f"  Effectiveness: {Q_total/self.Q_max_capacity*100:.1f} %")
        print(f"  Avg U: {np.mean(self.U):.2f} W/m2K")

        # Conversion
        print("\nConversion:")
        print(f"  Para-H₂: {self.xh[0]:.4f} → {self.xh[-1]:.4f}")

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

        # Create directories if they don't exist
        for path in [path_results_csv, path_conv_csv, path_perf_plot, path_conv_plot]:
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

        # 2. Convergence Tracking
        print(f"Generating convergence plot: {path_conv_plot}...")
        self.tracker.plot(save_path=path_conv_plot)

        print(f"Exporting convergence history: {path_conv_csv}...")
        self.tracker.export_csv(filepath=path_conv_csv)

        print("="*70 + "\n")


def create_default_config():
    """Create default configuration"""
    return {
        'geometry': {
            'length': 0.94, 'width': 0.25, 'height': 0.25,
            'porosity_hot': 0.65, 'porosity_cold': 0.70, 'unit_cell_size': 5e-3,
            'wall_thickness': 0.5e-3, 'surface_area_density': 60
        },
        'tpms': {'type_hot': 'Diamond', 'type_cold': 'Gyroid'},
        'material': {'k_wall': 237},
        'operating': {
            'Th_in': 78, 'Tc_in': 43, 'Ph_in': 2e6, 'Pc_in': 1.5e6,
            'mh': 2e-2, 'mc': 6e-2, 'xh_in': 0.452
        },
        'catalyst': {'enhancement': 1.2},
        'solver': {
            'n_elements': 100, 'max_iter': 500, 'tolerance': 1e-3,
            'relax': 0.15, 'relax_hydraulic': 0.5, 'relax_kinetics': 1.0,
            'Q_damping': 0.5, 'adaptive_damping': True
        },
        'output': {
            'results_csv': 'results/final_results.csv',
            'convergence_csv': 'results/convergence_history.csv',
            'performance_plot': 'results/performance_profile.png',
            'convergence_plot': 'results/convergence_diagnostics.png'
        }
    }


# Set plot style
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.dpi'] = 100

if __name__ == "__main__":
    # Example Config
    config = create_default_config()
    # Update with specific case values if needed
    config['operating']['fluid_hot'] = 'hydrogen mixture'
    config['operating']['fluid_cold'] = 'helium'

    # Solve
    he = TPMSHeatExchanger(config)
    he.solve()

    # Finalize & Output
    he.finalize_simulation()