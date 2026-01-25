"""
TPMS Heat Exchanger for Ortho-Para Hydrogen Conversion Simulation

This module simulates heat transfer in a TPMS (Triply Periodic Minimal Surface) heat exchanger
specifically designed for ortho-para hydrogen conversion processes.

CORRECTION NOTE:
- Uses Enthalpy-Based Segmental Method (corrects internal heat generation physics).
- Added verbose logging and robust Epsilon-NTU initialization.
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
        Initialize temperatures using an analytic Epsilon-NTU estimate.
        """
        print("-" * 50)
        print("INITIALIZATION (Epsilon-NTU Estimator)")

        N = self.N_elements + 1
        Th_in = self.config['operating']['Th_in']
        Tc_in = self.config['operating']['Tc_in']

        # Estimate outlets using helper
        Th_out_est, Tc_out_est = self._estimate_outlets(Th_in, Tc_in)

        print(f"  Inputs: Th_in={Th_in:.2f} K, Tc_in={Tc_in:.2f} K")
        print(f"  Est. Outlets: Th_out={Th_out_est:.2f} K, Tc_out={Tc_out_est:.2f} K")
        print("-" * 50)

        # Set Linear Profiles
        self.Th = np.linspace(Th_in, Th_out_est, N)
        self.Tc = np.linspace(Tc_out_est, Tc_in, N) # Cold flows N->0

        self.Ph = np.full(N, self.config['operating']['Ph_in'])
        self.Pc = np.full(N, self.config['operating']['Pc_in'])

        # Initialize Para-hydrogen fraction
        # Use equilibrium at local estimated temperature for stability
        self.xh = np.zeros(N)
        for i in range(N):
            x_eq = self.h2_props.get_equilibrium_fraction(self.Th[i])
            # Blend inlet value with equilibrium
            w = i / (N - 1)
            self.xh[i] = self.config['operating']['xh_in'] * (1 - 0.5*w) + x_eq * (0.5*w)

        self.L_elem = self.L_HE / self.N_elements

    def _estimate_outlets(self, Th_in, Tc_in):
        """Helper to estimate outlets using standard Epsilon-NTU"""
        mh = self.config['operating']['mh']
        mc = self.config['operating']['mc']

        # 1. Properties at inlet guess
        ph = self._safe_get_prop(Th_in, self.config['operating']['Ph_in'], 0.5, False)
        pc = self._safe_get_prop(Tc_in, self.config['operating']['Pc_in'], None, True)

        # 2. Capacitance
        Ch = mh * ph['cp']
        Cc = mc * pc['cp']
        Cmin, Cmax = min(Ch, Cc), max(Ch, Cc)
        Cr = Cmin / Cmax

        # 3. Estimate U
        u_h = mh / (ph['rho'] * self.Ac_hot)
        Re_h = ph['rho'] * u_h * self.Dh_hot / ph['mu']
        Pr_h = ph['mu'] * ph['cp'] / ph['lambda']
        Nu_h, _ = TPMSCorrelations.get_correlations(self.TPMS_hot, Re_h, Pr_h, 'Gas')
        h_h = 1.2 * Nu_h * ph['lambda'] / self.Dh_hot

        u_c = mc / (pc['rho'] * self.Ac_cold)
        Re_c = pc['rho'] * u_c * self.Dh_cold / pc['mu']
        Pr_c = pc['mu'] * pc['cp'] / pc['lambda']
        Nu_c, _ = TPMSCorrelations.get_correlations(self.TPMS_cold, Re_c, Pr_c, 'Gas')
        h_c = Nu_c * pc['lambda'] / self.Dh_cold

        U = 1 / (1/h_h + self.wall_thickness/self.k_wall + 1/h_c)

        # 4. Effectiveness
        NTU = U * self.A_heat / Cmin
        if Cr < 1.0:
            eff = (1 - np.exp(-NTU * (1 - Cr))) / (1 - Cr * np.exp(-NTU * (1 - Cr)))
        else:
            eff = NTU / (1 + NTU)

        Q = eff * Cmin * (Th_in - Tc_in)

        Th_out = Th_in - Q/Ch
        Tc_out = Tc_in + Q/Cc
        return Th_out, Tc_out

    def _safe_get_prop(self, T, P, x=None, is_helium=False):
        """Safely get properties, handling crashes for T < 20K"""
        if np.isnan(T): T = 45.0
        try:
            if is_helium:
                p = self.h2_props.get_helium_properties(T, P)
            else:
                p = self.h2_props.get_properties(T, P, x)
            return p
        except Exception as e:
            return {
                'h': 1000.0, 'rho': 2.0,
                'cp': 14000.0 if not is_helium else 5200.0,
                'mu': 1.0e-5, 'lambda': 0.1, 'Pr': 0.7
            }

    def solve(self, max_iter=500, tolerance=1e-4):
        print("=" * 70)
        print("TPMS Heat Exchanger Solver (Stability Enhanced)")
        print("=" * 70)

        mh = self.config['operating']['mh']
        mc = self.config['operating']['mc']

        # Q = Heat transferred FROM hot TO cold via wall
        Q = np.zeros(self.N_elements)

        start_time = time.time()

        for iteration in range(max_iter):
            Th_old = self.Th.copy()
            Tc_old = self.Tc.copy()
            xh_old = self.xh.copy()
            Q_old = Q.copy()

            # --- STRATEGY 1: Kinetic Soft-Start ---
            # Ramp up reaction heat from 0% to 100% over first 50 iterations
            # This prevents "Thermal Shock" from the massive conversion heat
            kinetic_ramp = min(1.0, iteration / 50.0)

            # --- STRATEGY 2: Adaptive Relaxation ---
            # Start cautious (0.05), get aggressive (0.4) as solution stabilizes
            if iteration < 20:
                relax = 0.05
            else:
                relax = min(0.4, 0.05 + 0.01 * (iteration - 20))

            # --- 1. Kinetics (With Soft-Start) ---
            # Calculate raw conversion
            xh_calc = self._ortho_para_conversion(self.Th, self.Ph, self.xh, mh)

            # Apply ramp: Effective x is blended between "Frozen Flow" (old) and "Reacting Flow" (new)
            # dx_raw = xh_calc - self.xh
            # self.xh = self.xh + relax * kinetic_ramp * dx_raw
            # Simplified update with ramp applied to the change:
            self.xh = self.xh + relax * kinetic_ramp * (xh_calc - self.xh)
            self.xh = np.clip(self.xh, 0.0, 1.0)

            # --- 2. Heat Transfer Coefficients ---
            U_vals = np.zeros(self.N_elements)
            for i in range(self.N_elements):
                # Use averages for properties
                Th_avg = 0.5 * (self.Th[i] + self.Th[i + 1])
                Tc_avg = 0.5 * (self.Tc[i] + self.Tc[i + 1])
                xh_avg = 0.5 * (self.xh[i] + self.xh[i + 1])

                ph = self._safe_get_prop(Th_avg, self.Ph[i], xh_avg, False)
                u_h = mh / (ph['rho'] * self.Ac_hot)
                Re_h = ph['rho'] * u_h * self.Dh_hot / ph['mu']
                Pr_h = ph['mu'] * ph['cp'] / ph['lambda']
                Nu_h = TPMSCorrelations.get_correlations(self.TPMS_hot, Re_h, Pr_h, 'Gas')[0]
                h_h = 1.2 * Nu_h * ph['lambda'] / self.Dh_hot

                pc = self._safe_get_prop(Tc_avg, self.Pc[i], None, True)
                u_c = mc / (pc['rho'] * self.Ac_cold)
                Re_c = pc['rho'] * u_c * self.Dh_cold / pc['mu']
                Pr_c = pc['mu'] * pc['cp'] / pc['lambda']
                Nu_c = TPMSCorrelations.get_correlations(self.TPMS_cold, Re_c, Pr_c, 'Gas')[0]
                h_c = Nu_c * pc['lambda'] / self.Dh_cold

                U_vals[i] = 1 / (1 / h_h + self.wall_thickness / self.k_wall + 1 / h_c)

            # --- 3. Wall Heat Flux ---
            Q_raw = np.zeros(self.N_elements)
            for i in range(self.N_elements):
                Th_avg = 0.5 * (self.Th[i] + self.Th[i + 1])
                Tc_avg = 0.5 * (self.Tc[i] + self.Tc[i + 1])

                # Driving force: Hot - Cold
                delta_T = Th_avg - Tc_avg

                # Heat transfer rate
                Q_raw[i] = U_vals[i] * self.A_elem * delta_T

            # Relax Q
            Q = Q_old + relax * (Q_raw - Q_old)

            # --- 4. Enthalpy Balance ---
            hh = np.zeros(self.N_elements + 1)
            hc = np.zeros(self.N_elements + 1)

            # Hot (Forward): h_out = h_in - Q/m
            # Note: We must re-evaluate h_in using the *current* xh[0] to be consistent
            hh[0] = self._safe_get_prop(self.config['operating']['Th_in'],
                                        self.Ph[0], self.xh[0], False)['h']
            for i in range(self.N_elements):
                hh[i + 1] = hh[i] - Q[i] / mh

            # Cold (Backward): h_out = h_in + Q/m
            hc[-1] = self._safe_get_prop(self.config['operating']['Tc_in'],
                                         self.Pc[-1], None, True)['h']
            for i in range(self.N_elements - 1, -1, -1):
                hc[i] = hc[i + 1] + Q[i] / mc

            # --- 5. Temperature Update (Inversion) ---
            # Hot Stream
            for i in range(len(hh)):
                def res_h(T):
                    # Solving H(T, P, x) - H_target = 0
                    return self._safe_get_prop(T, self.Ph[i], self.xh[i], False)['h'] - hh[i]

                guess = np.clip(self.Th[i], 14.0, 400.0)
                try:
                    # Use fsolve with tight tolerance
                    sol = fsolve(res_h, guess, xtol=1e-5)
                    T_new = sol[0]
                    # Clamp result
                    T_new = np.clip(T_new, 14.0, 400.0)
                    self.Th[i] = 0.8 * self.Th[i] + 0.2 * T_new  # Heavy damping on T update
                except:
                    pass  # Keep old value if failed

            # Cold Stream
            for i in range(len(hc)):
                def res_c(T):
                    return self._safe_get_prop(T, self.Pc[i], None, True)['h'] - hc[i]

                guess = np.clip(self.Tc[i], 4.0, 400.0)
                try:
                    sol = fsolve(res_c, guess, xtol=1e-5)
                    T_new = sol[0]
                    T_new = np.clip(T_new, 4.0, 400.0)
                    self.Tc[i] = 0.8 * self.Tc[i] + 0.2 * T_new
                except:
                    pass

            # --- STRATEGY 3: Monotonicity Enforcement ---
            # Physics Check: Hot must cool down, Cold must heat up
            # This helps guide the solver out of unphysical local minima
            if iteration > 5:  # Allow initial adjustment
                for i in range(self.N_elements):
                    # If Hot downstream is hotter than upstream, clamp it
                    if self.Th[i + 1] > self.Th[i]:
                        self.Th[i + 1] = self.Th[i] - 1e-4

                    # If Cold upstream (i) is hotter than downstream (i+1) [Cold flows N->0]
                    # Wait, Cold flows N->0. So Tc[i] (outlet side) should be > Tc[i+1] (inlet side)
                    if self.Tc[i + 1] > self.Tc[i]:
                        self.Tc[i + 1] = self.Tc[i] - 1e-4

            # --- Monitoring ---
            err = np.max(np.abs(self.Th - Th_old)) + np.max(np.abs(self.Tc - Tc_old))

            if (iteration + 1) % 10 == 0 or iteration < 5:
                # Energy Balance Check
                h_h_in = hh[0];
                h_h_out = hh[-1]
                h_c_in = hc[-1];
                h_c_out = hc[0]
                Q_h_loss = mh * (h_h_in - h_h_out)
                Q_c_gain = mc * (h_c_out - h_c_in)
                imbalance = abs(Q_h_loss - Q_c_gain)

                print(f"Iter {iteration + 1:3d} | Err: {err:.4f} | Ramp: {kinetic_ramp:.2f} | Relax: {relax:.2f}")
                print(f"    Q_hot_loss: {Q_h_loss:.1f} W | Q_cold_gain: {Q_c_gain:.1f} W | Diff: {imbalance:.1f} W")
                print(
                    f"    T_hot: {self.Th[0]:.1f}->{self.Th[-1]:.1f} K | T_cold: {self.Tc[-1]:.1f}->{self.Tc[0]:.1f} K")

            if err < tolerance and kinetic_ramp >= 1.0:
                print(f"\n*** CONVERGED in {iteration + 1} iterations ***")
                self._print_results(Q)
                return True

        print("Max iterations reached.")
        self._print_results(Q)
        return False
    def _ortho_para_conversion(self, Th, Ph, xh, mh):
        """Wilhelmsen Retuned Kinetics"""
        xh_new = np.zeros_like(xh)
        xh_new[0] = xh[0]
        Tc_H2 = 32.938; Pc_H2 = 1.284e6

        for i in range(self.N_elements):
            T_avg = 0.5 * (Th[i] + Th[i+1])
            P_avg = 0.5 * (Ph[i] + Ph[i+1])
            x_avg = 0.5 * (xh[i] + xh[i+1])
            x_eq = self.h2_props.get_equilibrium_fraction(T_avg)

            props = self._safe_get_prop(T_avg, P_avg, x_avg, False)
            rho = props['rho']; C_H2 = rho / 0.002016

            kw = 34.76 - 220.9*(T_avg/Tc_H2) - 20.65*(P_avg/Pc_H2)

            try:
                term1 = (x_avg/x_eq)**1.3246
                term2 = (1-x_eq)/(1-x_avg+1e-9)
                val = term1 * term2
                rate = (kw/C_H2) * np.log(val) if val > 0 else 0
            except: rate = 0

            u = mh / (rho * self.Ac_hot)
            tau = self.L_elem / u
            xh_new[i+1] = np.clip(xh[i] + rate*tau, 0.0, 1.0)

        return xh_new

    def _print_results(self, Q):
        print("=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        print(f"Hot Inlet (x=0):  {self.Th[0]:.2f} K, x={self.xh[0]:.3f}")
        print(f"Hot Outlet (x=L): {self.Th[-1]:.2f} K, x={self.xh[-1]:.3f}")
        print(f"Cold Inlet (x=L): {self.Tc[-1]:.2f} K")
        print(f"Cold Outlet (x=0): {self.Tc[0]:.2f} K")
        print("-" * 30)
        print(f"Total Wall Heat Transfer: {np.sum(Q):.2f} W")

        # Detailed Energy Balance
        h_h_in = self._safe_get_prop(self.Th[0], self.Ph[0], self.xh[0], False)['h']
        h_h_out = self._safe_get_prop(self.Th[-1], self.Ph[-1], self.xh[-1], False)['h']
        Q_hot_loss = self.config['operating']['mh'] * (h_h_in - h_h_out)

        h_c_in = self._safe_get_prop(self.Tc[-1], self.Pc[-1], None, True)['h']
        h_c_out = self._safe_get_prop(self.Tc[0], self.Pc[0], None, True)['h']
        Q_cold_gain = self.config['operating']['mc'] * (h_c_out - h_c_in)

        print(f"Hot Stream Enthalpy Drop: {Q_hot_loss:.2f} W")
        print(f"Cold Stream Enthalpy Rise: {Q_cold_gain:.2f} W")
        print(f"Imbalance: {abs(Q_hot_loss - Q_cold_gain):.2f} W")

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
            'Ph_in': 2e6, 'Pc_in': 0.5e6,
            'mh': 10e-2, 'mc': 40e-2,
            'xh_in': 0.452
        },
        'solver': {'n_elements': 50, 'max_iter': 100}
    }

if __name__ == "__main__":
    config = create_default_config()
    he = TPMSHeatExchanger(config)
    he.solve()