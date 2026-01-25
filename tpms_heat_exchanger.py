"""
TPMS Heat Exchanger for Ortho-Para Hydrogen Conversion Simulation

This module simulates heat transfer in a TPMS (Triply Periodic Minimal Surface) heat exchanger
specifically designed for ortho-para hydrogen conversion processes.

CORRECTION NOTE:
- Uses Enthalpy-Based Segmental Method (corrects internal heat generation physics).
- Restores original physical boundary restrictions and safety clamps for robustness.
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
        """Safely get properties, handling crashes for T < 14K (Restored from original)"""
        # Hard clamp for safety
        if np.isnan(T): T = 20.0
        if T < 14.0: T = 14.0
        if T > 500.0: T = 500.0

        try:
            if is_helium:
                # Helium can go lower (down to ~2.2K lambda point)
                if T < 2.5: T = 2.5
                p = self.h2_props.get_helium_properties(T, P)
            else:
                # Add check for saturation pressure issues
                # Ensure temperature is not too close to saturation pressure
                try:
                    # First check if we're near saturation point
                    import CoolProp.CoolProp as CP
                    # Safely get saturation pressure
                    try:
                        psat = CP.PropsSI('P','T',T,'Q',0,'hydrogen')
                        if abs(P - psat) < 1e-4 * P:  # Within 1e-4 % of saturation pressure
                            # Adjust temperature slightly to avoid saturation region
                            T_adj = T + (0.1 if T < 30 else -0.1)
                            if T_adj >= 14.0 and T_adj <= 500.0:
                                T = T_adj
                    except:
                        pass  # If saturation calc fails, continue with original T

                    # Also ensure T is not below melting point at given pressure
                    try:
                        Tmelt = CP.PropsSI('T','P',P,'Q',1,'hydrogen')  # Melting point
                        if T < Tmelt:
                            T = Tmelt + 0.1  # Small buffer above melting point
                    except:
                        pass  # If melting point calc fails, continue with original T
                except:
                    pass  # If CoolProp import fails, continue with original T

                # Ensure x is valid if not None
                if x is not None:
                    x = np.clip(x, 0.0, 1.0)  # Clamp para fraction to valid range

                p = self.h2_props.get_properties(T, P, x)

                # Check if properties are valid
                if any(np.isnan(val) for val in [p.get('h', np.nan), p.get('rho', np.nan),
                                               p.get('cp', np.nan), p.get('mu', np.nan)]):
                    raise ValueError("Invalid property values returned")

            return p
        except Exception as e:
            # Print warning when property calculation fails
            print(f"Warning: Property calculation failed at T={T} K, P={P} Pa: {e}")
            # Fallback values if Refprop/Coolprop fails completely
            # MUST include mu, cp, lambda for Re/Pr calculations
            return {
                'h': 1000.0, 'rho': 2.0,
                'cp': 14000.0 if not is_helium else 5200.0,
                'mu': 1.0e-5, 'lambda': 0.1,
                'Pr': 0.7  # Just in case
            }

    def solve(self, max_iter=500, tolerance=1e-4, relaxation=0.2):
        print("=" * 70)
        print("TPMS Heat Exchanger (Enthalpy Balance with Robust Safety)")
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

            # --- Safety Check: Reset if Diverged ---
            if np.any(~np.isfinite(self.Th)) or np.any(~np.isfinite(self.Tc)):
                print(f"  Warning: Found non-finite temperatures at iter {iteration}, resetting...")
                self.Th = np.clip(self.Th, 14.0, 400.0)
                self.Tc = np.clip(self.Tc, 4.0, 400.0)
                # Apply heavier damping on Q to recover
                Q = Q * 0.5

            # --- 1. Kinetics Update ---
            # Update conversion based on current temperature profile
            xh_new = self._ortho_para_conversion(self.Th, self.Ph, self.xh, mh)
            self.xh = self.xh + 0.1 * (xh_new - self.xh) # Damped update
            self.xh = np.clip(self.xh, 0.0, 1.0) # Strict bounds

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
            Q_raw = np.zeros(self.N_elements)

            for i in range(self.N_elements):
                Th_avg = 0.5 * (self.Th[i] + self.Th[i+1])
                Tc_avg = 0.5 * (self.Tc[i] + self.Tc[i+1])

                # Q > 0 means Heat flows Hot -> Cold
                Q_raw[i] = U_vals[i] * self.A_elem * (Th_avg - Tc_avg)

            # Relax Q
            # Adaptive relaxation
            relax_Q = relaxation if iteration > 20 else 0.05
            Q = Q_old + relax_Q * (Q_raw - Q_old)

            # --- 4. Enthalpy Integration (Energy Balance) ---
            hh = np.zeros(self.N_elements + 1)
            hc = np.zeros(self.N_elements + 1)

            # Hot Stream (0 -> N)
            hh[0] = self._safe_get_prop(self.config['operating']['Th_in'],
                                        self.Ph[0], self.xh[0], False)['h']
            for i in range(self.N_elements):
                hh[i+1] = hh[i] - Q[i] / mh

            # Cold Stream (N -> 0)
            hc[-1] = self._safe_get_prop(self.config['operating']['Tc_in'],
                                         self.Pc[-1], None, True)['h']
            for i in range(self.N_elements - 1, -1, -1):
                hc[i] = hc[i+1] + Q[i] / mc

            # --- 5. Temperature Update (Robust Inversion) ---

            # Hot Stream
            for i in range(len(hh)):
                def res_h(T):
                    return self._safe_get_prop(T, self.Ph[i], self.xh[i], False)['h'] - hh[i]

                # Use current T as guess, clamped
                guess = max(14.0, min(self.Th[i], 400.0))

                try:
                    sol = fsolve(res_h, guess, full_output=True)
                    if sol[2] == 1: # Converged
                        new_temp = sol[0][0]
                        if 14.0 <= new_temp <= 400.0:
                            self.Th[i] = new_temp
                        else:
                            # Damped update if out of bounds (Restored logic)
                            self.Th[i] = 0.9 * self.Th[i] + 0.1 * guess
                    else:
                        self.Th[i] = 0.95 * self.Th[i] + 0.05 * guess
                except:
                    self.Th[i] = 0.95 * self.Th[i] + 0.05 * guess

            # Cold Stream
            for i in range(len(hc)):
                def res_c(T):
                    return self._safe_get_prop(T, self.Pc[i], None, True)['h'] - hc[i]

                guess = max(4.0, min(self.Tc[i], 400.0))

                try:
                    sol = fsolve(res_c, guess, full_output=True)
                    if sol[2] == 1:
                        new_temp = sol[0][0]
                        if 4.0 <= new_temp <= 400.0:
                            self.Tc[i] = new_temp
                        else:
                            self.Tc[i] = 0.9 * self.Tc[i] + 0.1 * guess
                    else:
                        self.Tc[i] = 0.95 * self.Tc[i] + 0.05 * guess
                except:
                    self.Tc[i] = 0.95 * self.Tc[i] + 0.05 * guess

            # --- Convergence Check ---
            err = np.max(np.abs(self.Th - Th_old)) + np.max(np.abs(self.Tc - Tc_old))

            if (iteration + 1) % 20 == 0 or iteration < 5:
                print(f'Iter {iteration + 1:3d}: Err={err:.4f} | Th_out={self.Th[-1]:.2f}K | xh_out={self.xh[-1]:.3f}')

            # Check for divergence and exit early if needed (Restored logic)
            if np.any(~np.isfinite(self.Th)) or np.any(~np.isfinite(Q)):
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
        Calculate kinetics using Wilhelmsen (retuned) model.
        Supports bidirectional conversion.
        """
        xh_new = np.zeros_like(xh)
        xh_new[0] = xh[0]

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

            # [cite_start]Rate Constant (Wilhelmsen Retuned [cite: 712])
            kw = 34.76 - 220.9 * (T_avg / Tc_H2) - 20.65 * (P_avg / Pc_H2)

            # Rate Law
            try:
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
            # Strictly apply physical boundaries
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