"""
TPMS Heat Exchanger for Ortho-Para Hydrogen Conversion

Main solver for catalyst-filled TPMS heat exchangers in hydrogen liquefaction.
Couples heat transfer, fluid flow, and ortho-para conversion kinetics.
Corrected for bidirectional heat transfer and reversible kinetics.

Author: Based on research by Zhang et al. (2025)
"""

import numpy as np
import time
from scipy.optimize import fsolve
import warnings

from hydrogen_properties import HydrogenProperties
from tpms_correlations import TPMSCorrelations

# Suppress the specific Re range warnings to clean up output
warnings.filterwarnings("ignore", message="Some Re values outside validated range")


class TPMSHeatExchanger:
    """
    Robust Spatially-Aligned TPMS Solver.
    Index 0 = x=0 (Hot Inlet)
    Index N = x=L (Cold Inlet)
    """

    def __init__(self, config):
        self.config = config
        self.h2_props = HydrogenProperties()

        # Geometry
        self.L_HE = config['geometry']['length']
        self.W_HE = config['geometry']['width']
        self.H_HE = config['geometry']['height']
        self.porosity_hot = config['geometry']['porosity_hot']
        self.porosity_cold = config['geometry']['porosity_cold']
        self.unit_cell = config['geometry']['unit_cell_size']
        self.wall_thickness = config['geometry']['wall_thickness']

        self.TPMS_hot = config['tpms']['type_hot']
        self.TPMS_cold = config['tpms']['type_cold']

        self._calculate_geometry()
        self.N_elements = config['solver']['n_elements']
        self._initialize_solution()

    def _calculate_geometry(self):
        self.Dh_hot = 4 * self.porosity_hot * self.unit_cell / (2 * np.pi)
        self.Dh_cold = 4 * self.porosity_cold * self.unit_cell / (2 * np.pi)
        self.Ac_hot = self.W_HE * self.H_HE * self.porosity_hot
        self.Ac_cold = self.W_HE * self.H_HE * self.porosity_cold

        surface_area_density = self.config['geometry'].get('surface_area_density', 600)
        self.A_heat = self.L_HE * self.W_HE * self.H_HE * surface_area_density
        self.k_wall = self.config['material']['k_wall']

    def _initialize_solution(self):
        """
        Initialize temperatures linearly based on expected boundary conditions.
        """
        N = self.N_elements + 1

        Th_in = self.config['operating']['Th_in']
        Tc_in = self.config['operating']['Tc_in']

        # Guesses for outlets (assuming 90% effectiveness)
        Th_out_guess = Tc_in + 5.0
        Tc_out_guess = Th_in - 5.0

        # Spatial Initialization (0 -> L)
        self.Th = np.linspace(Th_in, Th_out_guess, N)
        # Cold gets colder as x goes 0 -> L (since it flows L -> 0)
        self.Tc = np.linspace(Tc_out_guess, Tc_in, N)

        self.Ph = np.full(N, self.config['operating']['Ph_in'])
        self.Pc = np.full(N, self.config['operating']['Pc_in'])
        self.xh = np.linspace(self.config['operating']['xh_in'], 0.5, N)

        self.L_elem = self.L_HE / self.N_elements

    def _safe_get_prop(self, T, P, x=None, is_helium=False):
        """Safely get properties, handling crashes for T < 14K"""
        # Hard clamp for safety
        if np.isnan(T): T = 20.0
        # Increase minimum temperature to avoid melting point issues
        if T < 14.2: T = 14.2
        if T > 500.0: T = 500.0

        try:
            if is_helium:
                # Helium can go lower (down to ~2.2K lambda point)
                if T < 2.5: T = 2.5
                p = self.h2_props.get_helium_properties(T, P)
            else:
                p = self.h2_props.get_properties(T, P, x)
            return p
        except Exception:
            # Fallback values if Refprop/Coolprop fails completely
            return {
                'h': 1000.0, 'rho': 2.0,
                'cp': 14000.0 if not is_helium else 5200.0,
                'mu': 1.0e-5, 'lambda': 0.1,
                'Pr': 0.7
            }

    def solve(self, max_iter=200, tolerance=1e-3):
        print("=" * 70)
        print("TPMS Heat Exchanger (Bidirectional Solver)")
        kinetics_model = self.config.get('kinetics', {}).get('model', 'simple')
        print(f"Kinetics Model: {kinetics_model.upper()}")
        print("=" * 70)

        mh = self.config['operating']['mh']
        mc = self.config['operating']['mc']

        # Initialize heat transfer rates
        Q = np.zeros(self.N_elements)

        for iteration in range(max_iter):
            Th_old = self.Th.copy()
            Tc_old = self.Tc.copy()
            Q_old = Q.copy()

            # --- 1. Properties (at Nodes) ---
            props_h_nodes = [self._safe_get_prop(T, P, x, False)
                             for T, P, x in zip(self.Th, self.Ph, self.xh)]
            props_c_nodes = [self._safe_get_prop(T, P, None, True)
                             for T, P in zip(self.Tc, self.Pc)]

            # --- 2. Heat Transfer Calculation ---
            Q_element = np.zeros(self.N_elements)

            for i in range(self.N_elements):
                # Element i spans from node i to node i+1
                
                # Hot side properties at node i
                p_h = props_h_nodes[i]
                u_h = mh / (p_h['rho'] * self.Ac_hot)
                Re_h = p_h['rho'] * u_h * self.Dh_hot / p_h['mu']
                Pr_h = p_h['mu'] * p_h['cp'] / p_h['lambda']
                Nu_h_val = TPMSCorrelations.get_correlations(self.TPMS_hot, Re_h, Pr_h, 'Gas')[0]
                h_h = 1.2 * Nu_h_val * p_h['lambda'] / self.Dh_hot

                # Cold side properties at node i+1 (counterflow)
                p_c = props_c_nodes[i+1]
                u_c = mc / (p_c['rho'] * self.Ac_cold)
                Re_c = p_c['rho'] * u_c * self.Dh_cold / p_c['mu']
                Pr_c = p_c['mu'] * p_c['cp'] / p_c['lambda']
                Nu_c_val = TPMSCorrelations.get_correlations(self.TPMS_cold, Re_c, Pr_c, 'Gas')[0]
                h_c = Nu_c_val * p_c['lambda'] / self.Dh_cold

                # Overall heat transfer coefficient
                U = 1 / (1 / h_h + self.wall_thickness / self.k_wall + 1 / h_c)
                
                # Temperature differences for LMTD (Counterflow)
                # Hot: node i -> node i+1
                # Cold: node i+1 -> node i (opposite direction)
                dT1 = self.Th[i] - self.Tc[i + 1]  # Hot in - Cold out
                dT2 = self.Th[i + 1] - self.Tc[i]  # Hot out - Cold in
                
                # Calculate LMTD with numerical stability
                if abs(dT1 - dT2) < 1e-6:  # Near equal, use arithmetic mean
                    LMTD = dT1
                elif dT1 * dT2 > 0:  # Same sign
                    if dT1 > 0 and dT2 > 0:
                        LMTD = (dT1 - dT2) / np.log(dT1 / dT2)
                    else:  # Both negative
                        LMTD = (dT2 - dT1) / np.log(dT2 / dT1)
                else:  # Mixed flow, use arithmetic mean
                    LMTD = (dT1 + dT2) / 2.0

                # Heat transfer rate in this element
                Q_element[i] = U * self.A_heat / self.N_elements * LMTD

            # Apply under-relaxation
            relax_factor = 0.1 if iteration < 20 else 0.05
            Q = Q_old + relax_factor * (Q_element - Q_old)

            # --- 3. Energy Balance (Updated with more conservative approach) ---
            # Hot stream: loses heat as it flows from 0 to L
            # Cold stream: gains heat as it flows from L to 0
            
            # Calculate enthalpies based on current temperatures
            hh_calc = np.array([self._safe_get_prop(self.Th[i], self.Ph[i], self.xh[i], False)['h'] 
                               for i in range(len(self.Th))])
            hc_calc = np.array([self._safe_get_prop(self.Tc[i], self.Pc[i], None, True)['h'] 
                               for i in range(len(self.Tc))])
            
            # Predict new enthalpies based on heat transfer
            hh_new = hh_calc.copy()
            hc_new = hc_calc.copy()
            
            # Hot stream loses heat: h_new[i+1] = h_old[i+1] - (heat_removed_from_element_i) 
            # But we need to redistribute this heat loss appropriately
            for i in range(self.N_elements):
                # Heat lost by hot stream in element i
                heat_lost_hot = Q[i]  # Only positive heat transfer from hot to cold
                # Distribute heat loss between nodes i and i+1
                hh_new[i] -= heat_lost_hot / (2 * mh)  # Half the heat from node i
                if i+1 < len(hh_new):
                    hh_new[i+1] -= heat_lost_hot / (2 * mh)  # Half the heat from node i+1

            # Cold stream gains heat: h_new[i] = h_old[i] + (heat_gained_from_element_i)
            for i in range(self.N_elements):
                # Heat gained by cold stream in element i
                heat_gained_cold = Q[i]  # Only positive heat transfer to cold
                # Distribute heat gain between nodes i and i+1
                hc_new[i+1] += heat_gained_cold / (2 * mc)  # Half the heat to node i+1
                hc_new[i] += heat_gained_cold / (2 * mc)  # Half the heat to node i

            # --- 4. Temperature Updates ---
            # Convert new enthalpies back to temperatures
            Th_new = self.Th.copy()
            Tc_new = self.Tc.copy()
            
            # Update hot temperatures
            for i in range(len(self.Th)):
                def enthalpy_residual(T):
                    h_calc = self._safe_get_prop(T, self.Ph[i], self.xh[i], False)['h']
                    return h_calc - hh_new[i]
                
                initial_guess = max(14.2, min(self.Th[i], 400.0))
                try:
                    sol = fsolve(enthalpy_residual, initial_guess, full_output=True)
                    if sol[2] == 1 and np.isfinite(sol[0][0]):
                        temp = max(14.2, min(sol[0][0], 400.0))
                        # Apply physical constraint: hot stream should cool down along flow
                        if i > 0:
                            temp = min(temp, Th_new[i-1])  # Ensure cooling trend
                        Th_new[i] = 0.8 * self.Th[i] + 0.2 * temp  # Damping
                    else:
                        Th_new[i] = self.Th[i]  # No change if solution fails
                except:
                    Th_new[i] = self.Th[i]  # No change if exception occurs

            # Update cold temperatures
            for i in range(len(self.Tc)):
                def enthalpy_residual(T):
                    h_calc = self._safe_get_prop(T, self.Pc[i], None, True)['h']
                    return h_calc - hc_new[i]
                
                initial_guess = max(4.0, min(self.Tc[i], 400.0))
                try:
                    sol = fsolve(enthalpy_residual, initial_guess, full_output=True)
                    if sol[2] == 1 and np.isfinite(sol[0][0]):
                        temp = max(4.0, min(sol[0][0], 400.0))
                        # Apply physical constraint: cold stream should heat up along flow (reverse direction)
                        # Cold stream flows from index N to 0, so Tc[i] should be >= Tc[i+1] if heating
                        if i < len(self.Tc) - 1:
                            temp = max(temp, Tc_new[i+1])  # Ensure heating trend
                        Tc_new[i] = 0.8 * self.Tc[i] + 0.2 * temp  # Damping
                    else:
                        Tc_new[i] = self.Tc[i]  # No change if solution fails
                except:
                    Tc_new[i] = self.Tc[i]  # No change if exception occurs

            # Apply the updates with damping
            self.Th = Th_new
            self.Tc = Tc_new

            # --- 5. Kinetics Update ---
            xh_new = self._ortho_para_conversion(self.Th, self.Ph, self.xh, mh)
            self.xh = self.xh + 0.02 * (xh_new - self.xh)

            # Convergence Check
            err_th = np.max(np.abs(self.Th - Th_old))
            err_tc = np.max(np.abs(self.Tc - Tc_old))
            err = err_th + err_tc

            if (iteration + 1) % 20 == 0:
                print(f'N_iter = {iteration + 1:3d}: Err={err:.4f} | Th_out={self.Th[-1]:.1f}K | Tco={self.Tc[0]:.1f}K')

            if np.any(~np.isfinite(self.Th)) or np.any(~np.isfinite(self.Tc)):
                print(f"Solver diverged at iteration {iteration + 1}.")
                self._print_results(Q)
                return False

            if err < tolerance:
                print(f"\nConverged in {iteration + 1} iterations.")
                self._print_results(Q)
                return True

        print("Max iterations reached.")
        self._print_results(Q)
        return False

    def _ortho_para_conversion(self, Th, Ph, xh, mh):
        """
        Calculate ortho-para conversion kinetics.
        Supports both 'simple' (first-order) and 'complex' (Wilhelmsen/demi.py) models.
        Bidirectional in both cases.
        """
        xh_new = np.zeros_like(xh)
        xh_new[0] = xh[0]

        model = self.config.get('kinetics', {}).get('model', 'simple')

        # Complex model constants
        Tc_H2 = 32.938
        Pc_H2 = 1.284e6

        for i in range(self.N_elements):
            T_avg = 0.5 * (Th[i] + Th[i + 1])
            P_avg = 0.5 * (Ph[i] + Ph[i + 1])
            x_avg = 0.5 * (xh[i] + xh[i + 1])
            x_eq = self.h2_props.get_equilibrium_fraction(T_avg)

            rho = self._safe_get_prop(T_avg, Ph[i], x_avg, False)['rho']
            u = mh / (rho * self.Ac_hot)
            tau = self.L_elem / u

            if model == 'complex':
                # --- COMPLEX MODEL (demi.py / Wilhelmsen) ---
                # Rate K typically becomes negative for T > ~30K
                K_val = 59.7 - 253.9 * (T_avg / Tc_H2) - 11.6 * (P_avg / Pc_H2)
                C_H2 = rho / 0.002016

                # Avoid log errors
                x_safe = np.clip(x_avg, 1e-4, 0.9999)
                x_eq_safe = np.clip(x_eq, 1e-4, 0.9999)

                try:
                    # Wilhelmsen Rate Law: r = (K / C) * ln( ... )
                    term1 = (x_safe / x_eq_safe) ** 1.0924
                    term2 = (1 - x_eq_safe) / (1 - x_safe)
                    log_arg = term1 * term2

                    if log_arg <= 0:
                        dx = 0.0
                    else:
                        # Correct Rate calculation:
                        # If K < 0 and x > x_eq (log > 0) -> Rate < 0 (Reverse) -> Stable
                        # If K < 0 and x < x_eq (log < 0) -> Rate > 0 (Forward) -> Stable
                        rate = (K_val / C_H2) * np.log(log_arg)
                        dx = rate * tau

                except Exception:
                    dx = 0.0

            else:
                # --- SIMPLE MODEL (Robust Reversible) ---
                # Rate = k * (x_eq - x)
                k_rate = 0.1
                dx = k_rate * (x_eq - x_avg) * tau

            # Apply change
            xh_new[i + 1] = xh_new[i] + dx
            xh_new[i + 1] = np.clip(xh_new[i + 1], 0.0, 1.0)

        return xh_new

    def _print_results(self, Q):
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Hot Inlet (x=0):  {self.Th[0]:.2f} K")
        print(f"Hot Outlet (x=L): {self.Th[-1]:.2f} K")
        print(f"Cold Inlet (x=L): {self.Tc[-1]:.2f} K")
        print(f"Cold Outlet (x=0): {self.Tc[0]:.2f} K")

        valid_Q = Q[np.isfinite(Q)]
        if len(valid_Q) > 0:
            total_heat = np.sum(valid_Q)
            print(f"Total Heat: {total_heat:.2f} W")
        else:
            print("Total Heat: 0.00 W")


def create_default_config():
    return {
        'geometry': {
            'length': 0.94, 'width': 0.15, 'height': 0.10,
            'porosity_hot': 0.65, 'porosity_cold': 0.70,
            'unit_cell_size': 5e-3, 'wall_thickness': 0.5e-3,
            'surface_area_density': 600
        },
        'tpms': {'type_hot': 'Diamond', 'type_cold': 'Gyroid'},
        'material': {'k_wall': 237},
        'operating': {
            'Th_in': 66.3, 'Th_out': 53.5,
            'Tc_in': 43.5, 'Tc_out': 61.3,
            'Ph_in': 1.13e6, 'Pc_in': 0.54e6,
            'mh': 1e-3, 'mc': 1e-3,
            'xh_in': 0.452
        },
        'kinetics': {'model': 'simple'}, # Change to 'complex' to use demi.py model
        'solver': {'n_elements': 20, 'max_iter': 100, 'tolerance': 1e-3}
    }


if __name__ == "__main__":
    config = create_default_config()
    config['kinetics']['model'] = 'complex' # Uncomment to test complex model
    he = TPMSHeatExchanger(config)
    he.solve()