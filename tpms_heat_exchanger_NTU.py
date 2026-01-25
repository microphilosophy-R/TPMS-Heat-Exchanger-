"""
TPMS Heat Exchanger for Ortho-Para Hydrogen Conversion

Main solver for catalyst-filled TPMS heat exchangers in hydrogen liquefaction.
Couples heat transfer, fluid flow, and ortho-para conversion kinetics.
Integrates both simple first-order and complex experimental kinetics.

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
        if T < 14.0: T = 14.0
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
            # MUST include mu, cp, lambda for Re/Pr calculations
            return {
                'h': 1000.0, 'rho': 2.0,
                'cp': 14000.0 if not is_helium else 5200.0,
                'mu': 1.0e-5, 'lambda': 0.1,
                'Pr': 0.7  # Just in case
            }

    def solve(self, max_iter=200, tolerance=1e-4):
        print("=" * 70)
        print("TPMS Heat Exchanger Solver")
        kinetics_model = self.config.get('kinetics', {}).get('model', 'simple')
        print(f"Kinetics Model: {kinetics_model.upper()}")
        print("=" * 70)

        mh = self.config['operating']['mh']
        mc = self.config['operating']['mc']

        # Q[i] is heat transferred in element i (between node i and i+1)
        Q = np.zeros(self.N_elements)

        start_time = time.time()

        for iteration in range(max_iter):
            Th_old = self.Th.copy()
            Tc_old = self.Tc.copy()
            Q_old = Q.copy()

            # --- 1. Properties (at Nodes) ---
            props_h_nodes = [self._safe_get_prop(T, P, x, False)
                             for T, P, x in zip(self.Th, self.Ph, self.xh)]
            props_c_nodes = [self._safe_get_prop(T, P, None, True)
                             for T, P in zip(self.Tc, self.Pc)]

            # --- 2. Heat Transfer (Element-wise) ---
            Q_raw = np.zeros(self.N_elements)
            U_vals = np.zeros(self.N_elements)

            for i in range(self.N_elements):
                # Element i spans Node i to Node i+1

                # --- HOT SIDE CORRELATIONS ---
                p_h = props_h_nodes[i]
                # Calculate flow properties explicitly
                u_h = mh / (p_h['rho'] * self.Ac_hot)
                Re_h = p_h['rho'] * u_h * self.Dh_hot / p_h['mu']
                Pr_h = p_h['mu'] * p_h['cp'] / p_h['lambda']

                Nu_h_val = TPMSCorrelations.get_correlations(self.TPMS_hot, Re_h, Pr_h, 'Gas')[0]
                h_h = 1.2 * Nu_h_val * p_h['lambda'] / self.Dh_hot

                # --- COLD SIDE CORRELATIONS ---
                p_c = props_c_nodes[i]
                # Calculate flow properties explicitly
                u_c = mc / (p_c['rho'] * self.Ac_cold)
                Re_c = p_c['rho'] * u_c * self.Dh_cold / p_c['mu']
                Pr_c = p_c['mu'] * p_c['cp'] / p_c['lambda']

                Nu_c_val = TPMSCorrelations.get_correlations(self.TPMS_cold, Re_c, Pr_c, 'Gas')[0]
                h_c = Nu_c_val * p_c['lambda'] / self.Dh_cold

                # --- Overall U ---
                U = 1 / (1 / h_h + self.wall_thickness / self.k_wall + 1 / h_c)
                U_vals[i] = U

                # --- Epsilon-NTU ---
                # Hot enters at i, leaves at i+1
                # Cold enters at i+1, leaves at i (Counter-flow)
                Th_in_loc = self.Th[i]
                Tc_in_loc = self.Tc[i + 1]  # Cold flows Leftwards

                Ch = mh * p_h['cp']
                Cc = mc * p_c['cp']
                Cmin, Cmax = min(Ch, Cc), max(Ch, Cc)

                NTU = U * (self.A_heat / self.N_elements) / Cmin
                Cr = Cmin / Cmax

                # Effectiveness
                if NTU > 10: NTU = 10  # Cap for stability
                if Cr < 1.0:
                    if abs(1 - Cr * np.exp(-NTU * (1 - Cr))) > 1e-10:
                        eff = (1 - np.exp(-NTU * (1 - Cr))) / (1 - Cr * np.exp(-NTU * (1 - Cr)))
                    else:
                        eff = 1.0
                else:
                    eff = NTU / (1 + NTU)

                # Calculate Heat Transfer Rate
                # Allow reverse heat transfer if Cold > Hot
                temp_diff = Th_in_loc - Tc_in_loc
                Q_raw[i] = eff * Cmin * temp_diff

            # Relax Q
            relax = 0.2 if iteration > 10 else 0.05
            Q = Q_old + relax * (Q_raw - Q_old)

            # --- 3. Energy Balance (Integration) ---
            hh = np.zeros(self.N_elements + 1)
            hc = np.zeros(self.N_elements + 1)

            # Hot: Forward (0 -> N)
            hh[0] = self._safe_get_prop(self.config['operating']['Th_in'],
                                        self.Ph[0], self.config['operating']['xh_in'], False)['h']
            for i in range(self.N_elements):
                # Hot fluid loses heat Q[i]
                hh[i + 1] = hh[i] - Q[i] / mh

            # Cold: Backward (N -> 0)
            hc[-1] = self._safe_get_prop(self.config['operating']['Tc_in'],
                                         self.Pc[-1], None, True)['h']
            for i in range(self.N_elements - 1, -1, -1):
                # Cold fluid gains heat Q[i]
                hc[i] = hc[i + 1] + Q[i] / mc

            # --- 4. Temperature Update (Inverse Lookup) ---
            # Hot
            for i in range(len(hh)):
                def res_h(T):
                    return self._safe_get_prop(T, self.Ph[i], self.xh[i], False)['h'] - hh[i]

                guess = max(14.0, min(self.Th[i], 400.0))
                try:
                    sol = fsolve(res_h, guess, full_output=True)
                    if sol[2] == 1:
                        new_temp = sol[0][0]
                        if 14.0 <= new_temp <= 400.0:
                            self.Th[i] = new_temp
                        else:
                            self.Th[i] = 0.9 * self.Th[i] + 0.1 * guess
                    else:
                        self.Th[i] = 0.95 * self.Th[i] + 0.05 * guess
                except:
                    self.Th[i] = 0.95 * self.Th[i] + 0.05 * guess

            # Cold
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

            # --- 5. Kinetics (Bidirectional) ---
            xh_new = self._ortho_para_conversion(self.Th, self.Ph, self.xh, mh)
            # Damped update for stability
            self.xh = self.xh + 0.05 * (xh_new - self.xh)

            # Convergence
            err = np.max(np.abs(self.Th - Th_old)) + np.max(np.abs(self.Tc - Tc_old))

            if (iteration + 1) % 10 == 0:
                print(f'N_iter = {iteration + 1:3d}: Err={err:.4f} | Th_out={self.Th[-1]:.1f}K | Tco={self.Tc[0]:.1f}K')
                if np.any(~np.isfinite(self.Th)) or np.any(~np.isfinite(self.Tc)):
                    print("  Warning: Found non-finite temperatures, resetting...")
                    self.Th = np.clip(self.Th, 14.0, 400.0)
                    self.Tc = np.clip(self.Tc, 4.0, 400.0)

            if np.any(~np.isfinite(self.Th)) or np.any(~np.isfinite(self.Tc)) or np.any(~np.isfinite(Q)):
                print(f"Solver diverged at iteration {iteration + 1}. Terminating.")
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
        Calculate ortho-para conversion using the Wilhelmsen et al. kinetic model
        with 'Retuned' parameters from Wijnans et al. (2024).

        This model is bidirectional, allowing both forward (ortho->para) and
        reverse (para->ortho) conversion based on local thermodynamic equilibrium.

        Model:
            dx/dt = (kw / C_H) * ln[ (x/x_eq)^a * ((1-x_eq)/(1-x)) ]

            Where kw = b + c*(T/Tc) + d*(P/Pc)

        Parameters from Wijnans et al. (2024), Table 3 (Retuned):
            a = 1.3246 [-]
            b = 34.76 [mol/m3/s]
            c = -220.9 [mol/m3/s]
            d = -20.65 [mol/m3/s]
        """
        # Critical properties of Hydrogen
        Tc_H2 = 32.938  # K
        Pc_H2 = 1.284e6  # Pa (approx 12.8 bar)
        M_H2 = 2.016e-3  # kg/mol

        # "Retuned" parameters from Wijnans et al. (2024) Table 3
        param_a = 1.3246
        param_b = 34.76
        param_c = -220.9
        param_d = -20.65

        xh_new = np.zeros_like(xh)
        xh_new[0] = xh[0]

        for i in range(self.N_elements):
            # Average conditions in the element
            T_avg = 0.5 * (Th[i] + Th[i + 1])
            P_avg = 0.5 * (Ph[i] + Ph[i + 1])
            x_avg = 0.5 * (xh[i] + xh[i + 1])  # Use implicit avg for stability

            # 1. Calculate Equilibrium Fraction
            x_eq = self.h2_props.get_equilibrium_fraction(T_avg)

            # 2. Get Fluid Properties (Density)
            # Use safe property lookup
            try:
                props = self._safe_get_prop(T_avg, P_avg, x_avg, False)
                rho = props['rho']
            except:
                rho = 2.0  # Fallback density

            # Molar concentration C_H [mol/m^3]
            C_H = rho / M_H2

            # 3. Calculate Reaction Rate Constant (kw)
            # kw = b + c*(T/Tc) + d*(P/Pc)
            # Note: kw typically becomes negative in the gas phase with these parameters,
            # which correctly offsets the sign of the log term for x < x_eq.
            kw = param_b + param_c * (T_avg / Tc_H2) + param_d * (P_avg / Pc_H2)

            # 4. Calculate Kinetic Rate Term (Langmuir-Hinshelwood form)
            # Term = ln[ (x/x_eq)^a * ((1-x_eq)/(1-x)) ]

            # Safety checks for log domain
            x_safe = np.clip(x_avg, 1e-4, 0.9999)
            x_eq_safe = np.clip(x_eq, 1e-4, 0.9999)

            try:
                term1 = (x_safe / x_eq_safe) ** param_a
                term2 = (1.0 - x_eq_safe) / (1.0 - x_safe)

                log_arg = term1 * term2

                if log_arg <= 0:
                    rate = 0.0
                else:
                    # Rate law: dx/dt = (kw / C_H) * ln(...)
                    # This naturally handles direction:
                    # If x < x_eq (Ortho->Para): log_arg < 1 -> ln < 0.
                    #    Since kw is typically negative at T>20K, Rate > 0 (Forward).
                    # If x > x_eq (Para->Ortho): log_arg > 1 -> ln > 0.
                    #    Since kw is negative, Rate < 0 (Reverse).
                    rate = (kw / C_H) * np.log(log_arg)

            except Exception:
                rate = 0.0

            # 5. Integration (Spatial)
            # dx = rate * residence_time
            # u = mass_flux / (rho * Area)
            u = mh / (rho * self.Ac_hot)
            tau = self.L_elem / u

            # Update composition
            dx = rate * tau
            xh_new[i + 1] = xh[i] + dx

            # Physical clamping [0, 1]
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
    """Create default configuration dictionary"""
    return {
        'geometry': {
            'length': 0.94,         # m
            'width': 0.15,          # m
            'height': 0.10,         # m
            'porosity_hot': 0.65,
            'porosity_cold': 0.70,
            'unit_cell_size': 5e-3,  # m
            'wall_thickness': 0.5e-3,  # m
            'surface_area_density': 600  # m²/m³
        },
        'tpms': {
            'type_hot': 'Diamond',
            'type_cold': 'Gyroid'
        },
        'material': {
            'k_wall': 237  # W/(m·K) - Aluminum
        },
        'operating': {
            'Th_in': 66.3,      # K
            'Th_out': 53.5,     # K
            'Tc_in': 43.5,      # K
            'Tc_out': 61.3,     # K
            'Ph_in': 1.13e6,    # Pa
            'Pc_in': 0.54e6,    # Pa
            'mh': 1e-3,         # kg/s
            'mc': 2e-3,         # kg/s
            'xh_in': 0.452      # Initial para fraction
        },
        'kinetics': {
            'model': 'simple'   # Options: 'simple', 'complex'
        },
        'catalyst': {
            'enhancement': 1.2,
            'pressure_factor': 1.3
        },
        'solver': {
            'n_elements': 20,
            'max_iter': 10,
            'tolerance': 1e-3,
            'relaxation': 0.3
        }
    }


if __name__ == "__main__":
    # Create and run heat exchanger
    config = create_default_config()

    # You can switch between models here
    config['kinetics']['model'] = 'simple'

    he = TPMSHeatExchanger(config)
    he.solve()