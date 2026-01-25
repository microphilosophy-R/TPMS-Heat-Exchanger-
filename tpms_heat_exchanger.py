"""
TPMS Heat Exchanger for Ortho-Para Hydrogen Conversion Simulation

This module simulates heat transfer in a TPMS (Triply Periodic Minimal Surface) heat exchanger
specifically designed for ortho-para hydrogen conversion processes.

CORRECTION NOTE:
Replaced Epsilon-NTU method with a direct Enthalpy-Based Discretized solver.
This correctly handles the internal heat generation from ortho-para conversion by
decoupling wall heat transfer (driven by T) from energy balance (tracked by H).
"""

import numpy as np
import time
from scipy.optimize import fsolve
import warnings

from hydrogen_properties import HydrogenProperties
from tpms_correlations import TPMSCorrelations

# Suppress warnings about Reynolds number range validation
warnings.filterwarnings("ignore", message="Some Re values outside validated range")


class TPMSHeatExchanger:
    """
    TPMS Heat Exchanger Solver using Enthalpy-Based Segmental Method.

    Index Convention:
    - Nodes: 0 to N (Temperature, Pressure, Enthalpy defined here)
    - Elements: 0 to N-1 (Heat transfer Q, U defined here)
    - Hot Flow: 0 -> N
    - Cold Flow: N -> 0
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

        # Area per element
        self.A_elem = self.A_heat / self.config['solver']['n_elements']

    def _initialize_solution(self):
        """
        Initialize temperatures linearly.
        """
        N = self.N_elements + 1

        Th_in = self.config['operating']['Th_in']
        Tc_in = self.config['operating']['Tc_in']

        # Linear guesses
        Th_out_guess = Tc_in + 5.0
        Tc_out_guess = Th_in - 5.0

        # Spatial Initialization (0 -> L)
        self.Th = np.linspace(Th_in, Th_out_guess, N)
        self.Tc = np.linspace(Tc_out_guess, Tc_in, N)

        self.Ph = np.full(N, self.config['operating']['Ph_in'])
        self.Pc = np.full(N, self.config['operating']['Pc_in'])

        # Initial para fraction guess (linear)
        self.xh = np.linspace(self.config['operating']['xh_in'], 0.5, N)

        self.L_elem = self.L_HE / self.N_elements

    def _safe_get_prop(self, T, P, x=None, is_helium=False):
        """Safely get properties with bounds checking"""
        if np.isnan(T): T = 20.0

        # Clamp temperatures for property lookup stability
        if is_helium:
            T = np.clip(T, 2.5, 400.0)
            try:
                return self.h2_props.get_helium_properties(T, P)
            except:
                return {'h': 1000.0, 'rho': 2.0, 'cp': 5200.0, 'mu': 1e-5, 'lambda': 0.1, 'Pr': 0.7}
        else:
            T = np.clip(T, 20.0, 400.0)
            if x is not None: x = np.clip(x, 0.0, 1.0)
            try:
                return self.h2_props.get_properties(T, P, x)
            except:
                return {'h': 1000.0, 'rho': 2.0, 'cp': 14000.0, 'mu': 1e-5, 'lambda': 0.1, 'Pr': 0.7}

    def solve(self, max_iter=500, tolerance=1e-4, relaxation=0.2):
        print("=" * 70)
        print("TPMS Heat Exchanger (Enthalpy Balance Method)")
        print("=" * 70)

        mh = self.config['operating']['mh']
        mc = self.config['operating']['mc']

        # Q represents heat transferred FROM hot TO cold in each element
        Q = np.zeros(self.N_elements)

        start_time = time.time()

        for iteration in range(max_iter):
            Th_old = self.Th.copy()
            Tc_old = self.Tc.copy()
            Q_old = Q.copy()

            # --- 1. Kinetics Update ---
            # Update conversion based on current temperature profile
            # This is done FIRST so the correct x is used for enthalpy lookup
            xh_new = self._ortho_para_conversion(self.Th, self.Ph, self.xh, mh)
            self.xh = self.xh + 0.1 * (xh_new - self.xh) # Damped update

            # --- 2. Calculate Properties & Heat Transfer Coefficients ---
            U_vals = np.zeros(self.N_elements)

            # Element-wise calculation
            for i in range(self.N_elements):
                # Use Average Temperature in element for properties
                Th_avg = 0.5 * (self.Th[i] + self.Th[i+1])
                Tc_avg = 0.5 * (self.Tc[i] + self.Tc[i+1])
                xh_avg = 0.5 * (self.xh[i] + self.xh[i+1])

                # Hot Props
                ph = self._safe_get_prop(Th_avg, self.Ph[i], xh_avg, False)
                u_h = mh / (ph['rho'] * self.Ac_hot)
                Re_h = ph['rho'] * u_h * self.Dh_hot / ph['mu']
                Pr_h = ph['mu'] * ph['cp'] / ph['lambda']

                Nu_h = TPMSCorrelations.get_correlations(self.TPMS_hot, Re_h, Pr_h, 'Gas')[0]
                h_h = 1.2 * Nu_h * ph['lambda'] / self.Dh_hot # 1.2 enhancement

                # Cold Props
                pc = self._safe_get_prop(Tc_avg, self.Pc[i], None, True)
                u_c = mc / (pc['rho'] * self.Ac_cold)
                Re_c = pc['rho'] * u_c * self.Dh_cold / pc['mu']
                Pr_c = pc['mu'] * pc['cp'] / pc['lambda']

                Nu_c = TPMSCorrelations.get_correlations(self.TPMS_cold, Re_c, Pr_c, 'Gas')[0]
                h_c = Nu_c * pc['lambda'] / self.Dh_cold

                # Overall U
                U_vals[i] = 1 / (1/h_h + self.wall_thickness/self.k_wall + 1/h_c)

            # --- 3. Heat Transfer Calculation (Rate Equation) ---
            # Q = U * A * (Th_avg - Tc_avg)
            # We strictly separate wall heat transfer from enthalpy change.
            Q_raw = np.zeros(self.N_elements)

            for i in range(self.N_elements):
                # Driving force: Average temperature difference
                # Note: This allows bidirectional heat transfer (if Th < Tc, Q < 0)
                Th_avg = 0.5 * (self.Th[i] + self.Th[i+1])
                Tc_avg = 0.5 * (self.Tc[i] + self.Tc[i+1])

                Q_raw[i] = U_vals[i] * self.A_elem * (Th_avg - Tc_avg)

            # Relax Q
            Q = Q_old + relaxation * (Q_raw - Q_old)

            # --- 4. Enthalpy Integration (Energy Balance) ---
            hh = np.zeros(self.N_elements + 1)
            hc = np.zeros(self.N_elements + 1)

            # Hot Stream (0 -> N)
            # Inlet Enthalpy
            hh[0] = self._safe_get_prop(self.config['operating']['Th_in'],
                                        self.Ph[0], self.xh[0], False)['h']
            for i in range(self.N_elements):
                # h_out = h_in - Q / m
                hh[i+1] = hh[i] - Q[i] / mh

            # Cold Stream (N -> 0)
            # Inlet Enthalpy
            hc[-1] = self._safe_get_prop(self.config['operating']['Tc_in'],
                                         self.Pc[-1], None, True)['h']
            for i in range(self.N_elements - 1, -1, -1):
                # h_out = h_in + Q / m (Flowing backwards, Q added)
                hc[i] = hc[i+1] + Q[i] / mc

            # --- 5. Temperature Update (Invert Enthalpy) ---
            # Find T such that H(T, P, x) = hh

            # Hot Stream
            for i in range(len(hh)):
                # We must use the CURRENT xh[i] to correctly account for conversion heat
                def res_h(T):
                    return self._safe_get_prop(T, self.Ph[i], self.xh[i], False)['h'] - hh[i]

                guess = np.clip(self.Th[i], 20.0, 400.0)
                try:
                    sol = fsolve(res_h, guess)
                    self.Th[i] = np.clip(sol[0], 20.0, 400.0)
                except:
                    self.Th[i] = guess # Fallback

            # Cold Stream
            for i in range(len(hc)):
                def res_c(T):
                    return self._safe_get_prop(T, self.Pc[i], None, True)['h'] - hc[i]

                guess = np.clip(self.Tc[i], 4.0, 400.0)
                try:
                    sol = fsolve(res_c, guess)
                    self.Tc[i] = np.clip(sol[0], 4.0, 400.0)
                except:
                    self.Tc[i] = guess

            # --- Convergence Check ---
            err = np.max(np.abs(self.Th - Th_old)) + np.max(np.abs(self.Tc - Tc_old))

            if (iteration + 1) % 20 == 0 or iteration < 5:
                print(f'Iter {iteration + 1:3d}: Err={err:.4f} | Th_out={self.Th[-1]:.2f}K | xh_out={self.xh[-1]:.3f}')

            if err < tolerance:
                print(f"\nConverged in {iteration + 1} iterations.")
                self._print_results(Q)
                return True

        print("Max iterations reached.")
        self._print_results(Q)
        return False

    def _ortho_para_conversion(self, Th, Ph, xh, mh):
        """
        Calculate kinetics using Wilhelmsen (retuned) model or simple model.
        Supports bidirectional conversion.
        """
        xh_new = np.zeros_like(xh)
        xh_new[0] = xh[0]

        # Use Retuned Wilhelmsen parameters (Wijnans et al. 2024)
        # a=1.3246, b=34.76, c=-220.9, d=-20.65
        # kw = b + c*(T/Tc) + d*(P/Pc)
        Tc_H2 = 32.938
        Pc_H2 = 1.284e6

        for i in range(self.N_elements):
            T_avg = 0.5 * (Th[i] + Th[i + 1])
            P_avg = 0.5 * (Ph[i] + Ph[i + 1])
            x_avg = 0.5 * (xh[i] + xh[i + 1])
            x_eq = self.h2_props.get_equilibrium_fraction(T_avg)

            props = self._safe_get_prop(T_avg, P_avg, x_avg, False)
            rho = props['rho']
            C_H2 = rho / 0.002016 # mol/m3

            # Rate Constant (Wilhelmsen Retuned)
            kw = 34.76 - 220.9 * (T_avg / Tc_H2) - 20.65 * (P_avg / Pc_H2)

            # Rate Law
            # r = (kw / C_H) * ln[ (x/x_eq)^a * ((1-x_eq)/(1-x)) ]
            try:
                # Safety clips for log
                x_safe = np.clip(x_avg, 1e-4, 0.9999)
                x_eq_safe = np.clip(x_eq, 1e-4, 0.9999)

                term1 = (x_safe / x_eq_safe) ** 1.3246
                term2 = (1 - x_eq_safe) / (1 - x_safe)
                val = term1 * term2

                if val <= 0:
                    rate = 0.0
                else:
                    rate = (kw / C_H2) * np.log(val)
            except:
                rate = 0.0

            # Integration
            u = mh / (rho * self.Ac_hot)
            tau = self.L_elem / u
            dx = rate * tau

            xh_new[i + 1] = xh[i] + dx
            xh_new[i + 1] = np.clip(xh_new[i + 1], 0.0, 1.0)

        return xh_new

    def _print_results(self, Q):
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Hot Inlet (x=0):  {self.Th[0]:.2f} K, x={self.xh[0]:.3f}")
        print(f"Hot Outlet (x=L): {self.Th[-1]:.2f} K, x={self.xh[-1]:.3f}")
        print(f"Cold Inlet (x=L): {self.Tc[-1]:.2f} K")
        print(f"Cold Outlet (x=0): {self.Tc[0]:.2f} K")
        print(f"Total Heat Transferred (Wall): {np.sum(Q):.2f} W")

        # Energy Balance Check
        h_h_in = self._safe_get_prop(self.Th[0], self.Ph[0], self.xh[0], False)['h']
        h_h_out = self._safe_get_prop(self.Th[-1], self.Ph[-1], self.xh[-1], False)['h']
        Q_hot_loss = self.config['operating']['mh'] * (h_h_in - h_h_out)

        print(f"Hot Stream Enthalpy Drop: {Q_hot_loss:.2f} W")
        print(f" (Includes conversion heat and sensible cooling)")

def create_default_config():
    """Create default configuration dictionary"""
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
            'mh': 1e-3, 'mc': 2e-3,
            'xh_in': 0.452
        },
        'solver': {'n_elements': 20, 'max_iter': 100}
    }

if __name__ == "__main__":
    config = create_default_config()
    he = TPMSHeatExchanger(config)
    he.solve()