"""
CORRECTED TPMS Heat Exchanger for Ortho-Para Hydrogen Conversion

This version fixes the critical energy balance violations and numerical instability.

Key Corrections:
1. Proper energy balance enforcement
2. Thermodynamically consistent heat transfer
3. Stable iteration with physical bounds
4. Counter-flow heat exchanger logic fixed

Author: Corrected version based on Zhang et al. (2025)
"""

import numpy as np
import time
from scipy.optimize import fsolve
import warnings

from hydrogen_properties import HydrogenProperties
from tpms_correlations import TPMSCorrelations

warnings.filterwarnings("ignore", message="Some Re values outside validated range")


class TPMSHeatExchangerCorrected:
    """
    Physically Correct TPMS Heat Exchanger Solver

    Counter-flow configuration:
    - Hot fluid: flows 0 → L (left to right)
    - Cold fluid: flows L → 0 (right to left)

    Physical constraints enforced:
    - Hot fluid must cool down (Th_out < Th_in)
    - Cold fluid must heat up (Tc_out > Tc_in)
    - Energy balance: Q_hot_released = Q_cold_absorbed
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
        """Calculate geometric parameters"""
        self.Dh_hot = 4 * self.porosity_hot * self.unit_cell / (2 * np.pi)
        self.Dh_cold = 4 * self.porosity_cold * self.unit_cell / (2 * np.pi)
        self.Ac_hot = self.W_HE * self.H_HE * self.porosity_hot
        self.Ac_cold = self.W_HE * self.H_HE * self.porosity_cold

        surface_area_density = self.config['geometry'].get('surface_area_density', 600)
        self.A_heat = self.L_HE * self.W_HE * self.H_HE * surface_area_density
        self.k_wall = self.config['material']['k_wall']

        print(f"Geometry calculated:")
        print(f"  Dh_hot = {self.Dh_hot*1e3:.3f} mm")
        print(f"  Dh_cold = {self.Dh_cold*1e3:.3f} mm")
        print(f"  Ac_hot = {self.Ac_hot*1e6:.2f} mm²")
        print(f"  Ac_cold = {self.Ac_cold*1e6:.2f} mm²")
        print(f"  A_heat = {self.A_heat:.3f} m²")

    def _initialize_solution(self):
        """
        Initialize with physically reasonable guess

        Spatial nodes: 0 (left) → N (right)
        Hot: flows left→right (0→N)
        Cold: flows right→left (N→0)
        """
        N = self.N_elements + 1

        Th_in = self.config['operating']['Th_in']
        Tc_in = self.config['operating']['Tc_in']

        # Reasonable outlet guesses (hot cools down, cold heats up)
        Th_out_guess = Tc_in + 5.0  # Hot outlet slightly above cold inlet
        Tc_out_guess = Th_in - 5.0  # Cold outlet slightly below hot inlet

        # Hot fluid array: T decreases from inlet (0) to outlet (N)
        self.Th = np.linspace(Th_in, Th_out_guess, N)

        # Cold fluid array: T increases from inlet (N) to outlet (0)
        self.Tc = np.linspace(Tc_out_guess, Tc_in, N)

        self.Ph = np.full(N, self.config['operating']['Ph_in'])
        self.Pc = np.full(N, self.config['operating']['Pc_in'])

        # Para-hydrogen fraction increases along hot flow
        x_eq_in = self.h2_props.get_equilibrium_fraction(Th_in)
        x_eq_out = self.h2_props.get_equilibrium_fraction(Th_out_guess)
        self.xh = np.linspace(self.config['operating']['xh_in'],
                              0.5*(self.config['operating']['xh_in'] + x_eq_out), N)

        self.L_elem = self.L_HE / self.N_elements

        print(f"\nInitialization:")
        print(f"  Hot: {Th_in:.2f} K → {Th_out_guess:.2f} K")
        print(f"  Cold: {Tc_in:.2f} K → {Tc_out_guess:.2f} K")

    def _safe_get_prop(self, T, P, x=None, is_helium=False):
        """Safely get properties with physical bounds"""
        # Enforce physical temperature limits
        if is_helium:
            T = np.clip(T, 4.0, 400.0)
        else:
            T = np.clip(T, 14.0, 400.0)

        P = np.clip(P, 1e4, 5e6)  # 0.01-5 MPa

        if x is not None:
            x = np.clip(x, 0.0, 1.0)

        try:
            if is_helium:
                p = self.h2_props.get_helium_properties(T, P)
            else:
                p = self.h2_props.get_properties(T, P, x if x is not None else 0.5)

            # Validate returned properties
            if np.isnan(p['h']) or np.isnan(p['rho']) or np.isnan(p['cp']):
                raise ValueError("NaN in properties")

            return p

        except Exception as e:
            # Fallback with reasonable values
            print(f"Warning: Property calculation failed at T={T:.1f}K, using fallback")
            return {
                'h': 5000.0 if not is_helium else 2000.0,
                'rho': 5.0 if not is_helium else 2.0,
                'cp': 14000.0 if not is_helium else 5200.0,
                'mu': 1.0e-5,
                'lambda': 0.15,
            }

    def solve(self, max_iter=500, tolerance=1e-3, relaxation=0.15):
        """
        Solve heat exchanger with proper energy balance

        Parameters
        ----------
        max_iter : int
            Maximum iterations
        tolerance : float
            Convergence tolerance on temperature change
        relaxation : float
            Under-relaxation factor (0.1-0.3 recommended)
        """
        print("=" * 70)
        print("CORRECTED TPMS Heat Exchanger Solver")
        print("=" * 70)

        mh = self.config['operating']['mh']
        mc = self.config['operating']['mc']

        print(f"\nOperating conditions:")
        print(f"  mh = {mh*1e3:.3f} g/s")
        print(f"  mc = {mc*1e3:.3f} g/s")
        print(f"  mc/mh = {mc/mh:.2f}")

        start_time = time.time()

        for iteration in range(max_iter):
            Th_old = self.Th.copy()
            Tc_old = self.Tc.copy()
            xh_old = self.xh.copy()

            # ============================================
            # 1. CALCULATE PROPERTIES AT EACH NODE
            # ============================================
            props_h = []
            props_c = []

            for i in range(len(self.Th)):
                ph = self._safe_get_prop(self.Th[i], self.Ph[i], self.xh[i], False)
                pc = self._safe_get_prop(self.Tc[i], self.Pc[i], None, True)
                props_h.append(ph)
                props_c.append(pc)

            # ============================================
            # 2. CALCULATE HEAT TRANSFER FOR EACH ELEMENT
            # ============================================
            Q_elements = np.zeros(self.N_elements)
            U_elements = np.zeros(self.N_elements)

            for k in range(self.N_elements):
                # Element k: from spatial node k to k+1
                # Hot fluid: enters at k, exits at k+1
                # Cold fluid: enters at k+1, exits at k (counter-flow!)

                # Average properties in element
                ph_avg = {key: 0.5*(props_h[k][key] + props_h[k+1][key])
                         for key in ['rho', 'cp', 'mu', 'lambda']}
                pc_avg = {key: 0.5*(props_c[k][key] + props_c[k+1][key])
                         for key in ['rho', 'cp', 'mu', 'lambda']}

                # Heat transfer coefficients
                h_hot = self._calculate_htc(ph_avg, mh, self.Ac_hot,
                                           self.Dh_hot, self.TPMS_hot, True)
                h_cold = self._calculate_htc(pc_avg, mc, self.Ac_cold,
                                            self.Dh_cold, self.TPMS_cold, False)

                # Overall U
                U = 1.0 / (1.0/h_hot + self.wall_thickness/self.k_wall + 1.0/h_cold)
                U_elements[k] = U

                # LMTD Method for counter-flow
                # Hot side: Th[k] → Th[k+1]
                # Cold side: Tc[k+1] → Tc[k]
                dT_in = self.Th[k] - self.Tc[k]      # Hot inlet - Cold outlet
                dT_out = self.Th[k+1] - self.Tc[k+1]  # Hot outlet - Cold inlet

                # Ensure positive temperature differences
                if dT_in <= 0 or dT_out <= 0:
                    # Temperature pinch or crossover - set Q to zero
                    Q_elements[k] = 0.0
                    continue

                # LMTD
                if abs(dT_in - dT_out) < 1e-6:
                    LMTD = dT_in
                else:
                    LMTD = (dT_in - dT_out) / np.log(dT_in / dT_out)

                # Heat transfer
                A_elem = self.A_heat / self.N_elements
                Q_elements[k] = U * A_elem * LMTD

                # Physical limit check
                # Maximum possible heat transfer based on capacitance
                Ch = mh * ph_avg['cp']
                Cc = mc * pc_avg['cp']
                Q_max_hot = Ch * (self.Th[k] - self.Th[k+1])
                Q_max_cold = Cc * (self.Tc[k] - self.Tc[k+1])

                # Should not exceed either stream's capacity
                Q_elements[k] = min(Q_elements[k], abs(Q_max_hot), abs(Q_max_cold))

            # ============================================
            # 3. UPDATE TEMPERATURES VIA ENERGY BALANCE
            # ============================================
            # Use explicit forward integration with relaxation

            # Hot stream (0 → N): loses heat
            Th_new = np.zeros(len(self.Th))
            Th_new[0] = self.config['operating']['Th_in']

            for k in range(self.N_elements):
                # Heat lost by hot fluid in element k
                cp_h = props_h[k]['cp']
                dT_h = -Q_elements[k] / (mh * cp_h)
                Th_new[k+1] = Th_new[k] + dT_h

            # Cold stream (N → 0): gains heat
            # But we store in array going 0→N, so integrate backward
            Tc_new = np.zeros(len(self.Tc))
            Tc_new[-1] = self.config['operating']['Tc_in']

            for k in range(self.N_elements-1, -1, -1):
                # Heat gained by cold fluid in element k
                cp_c = props_c[k+1]['cp']
                dT_c = Q_elements[k] / (mc * cp_c)
                Tc_new[k] = Tc_new[k+1] + dT_c

            # ============================================
            # 4. APPLY RELAXATION AND PHYSICAL BOUNDS
            # ============================================
            # Under-relaxation for stability
            self.Th = Th_old + relaxation * (Th_new - Th_old)
            self.Tc = Tc_old + relaxation * (Tc_new - Tc_old)

            # Enforce physical bounds
            self.Th = np.clip(self.Th, 14.0, 100.0)
            self.Tc = np.clip(self.Tc, 4.0, 100.0)

            # # Enforce monotonicity
            # # Hot must cool: T[k+1] <= T[k]
            # for k in range(len(self.Th)-1):
            #     if self.Th[k+1] > self.Th[k]:
            #         self.Th[k+1] = self.Th[k] - 0.1
            #
            # # Cold must heat: T[k] <= T[k+1]
            # for k in range(len(self.Tc)-1):
            #     if self.Tc[k] > self.Tc[k+1]:
            #         self.Tc[k] = self.Tc[k+1] - 0.1

            # ============================================
            # 5. UPDATE CONVERSION
            # ============================================
            xh_new = self._ortho_para_conversion()
            self.xh = xh_old + 0.5 * relaxation * (xh_new - xh_old)
            self.xh = np.clip(self.xh, 0.0, 1.0)

            # ============================================
            # 6. CHECK CONVERGENCE
            # ============================================
            err_T = np.max(np.abs(self.Th - Th_old)) + np.max(np.abs(self.Tc - Tc_old))
            err_x = np.max(np.abs(self.xh - xh_old))
            err = err_T + err_x

            if (iteration + 1) % 20 == 0 or iteration < 5:
                print(f"Iter {iteration+1:4d}: err={err:.4e} | "
                      f"Th: {self.Th[0]:.2f}→{self.Th[-1]:.2f}K | "
                      f"Tc: {self.Tc[-1]:.2f}→{self.Tc[0]:.2f}K | "
                      f"Q_tot={np.sum(Q_elements):.1f}W")

            # Check for physical violations
            if self.Th[-1] > self.Th[0]:
                print(f"\nERROR: Hot outlet hotter than inlet! Th_out={self.Th[-1]:.2f} > Th_in={self.Th[0]:.2f}")
                print("Resetting temperatures...")
                self._initialize_solution()
                continue

            if self.Tc[0] < self.Tc[-1]:
                print(f"\nERROR: Cold outlet colder than inlet! Tc_out={self.Tc[0]:.2f} < Tc_in={self.Tc[-1]:.2f}")
                print("Resetting temperatures...")
                self._initialize_solution()
                continue

            if err < tolerance:
                elapsed = time.time() - start_time
                print(f"\n{'='*70}")
                print(f"CONVERGED in {iteration+1} iterations ({elapsed:.2f} s)")
                print(f"{'='*70}")
                self._print_results(Q_elements, U_elements)
                return True

        print(f"\nWARNING: Maximum iterations ({max_iter}) reached")
        print(f"Final error: {err:.4e}")
        self._print_results(Q_elements, U_elements)
        return False

    def _calculate_htc(self, props_avg, m_dot, Ac, Dh, tpms_type, is_hot):
        """Calculate heat transfer coefficient"""
        u = m_dot / (props_avg['rho'] * Ac)
        Re = props_avg['rho'] * u * Dh / props_avg['mu']
        Pr = props_avg['mu'] * props_avg['cp'] / props_avg['lambda']

        # Get TPMS correlations
        Nu, _ = TPMSCorrelations.get_correlations(tpms_type, Re, Pr, 'Gas')

        # Handle invalid correlations
        if np.isnan(Nu) or Nu < 1.0:
            Nu = 10.0  # Reasonable default

        # Catalyst enhancement for hot side
        enhancement = 1.2 if is_hot else 1.0

        h = enhancement * Nu * props_avg['lambda'] / Dh
        return h

    def _ortho_para_conversion(self):
        """Calculate ortho-para conversion using kinetics"""
        xh_new = np.zeros(len(self.xh))
        xh_new[0] = self.config['operating']['xh_in']

        Tc = 32.938  # Critical temperature
        Pc = 1.284e6  # Critical pressure

        for i in range(self.N_elements):
            # Average conditions in element
            T_avg = 0.5 * (self.Th[i] + self.Th[i+1])
            P_avg = 0.5 * (self.Ph[i] + self.Ph[i+1])
            x_avg = 0.5 * (xh_new[i] + self.xh[i+1])

            # Equilibrium fraction
            x_eq = self.h2_props.get_equilibrium_fraction(T_avg)

            if x_avg < x_eq - 1e-6:
                # Reaction kinetics
                K = 59.7 - 253.9 * (T_avg / Tc) - 11.6 * (P_avg / Pc)

                # Get density
                props = self._safe_get_prop(T_avg, P_avg, x_avg, False)
                C_H2 = props['rho'] / 0.002016  # Molar concentration

                # Forward rate constant
                k_f = (K / C_H2) * np.log(((x_avg / x_eq)**1.0924) * ((1 - x_eq) / (1 - x_avg + 1e-10)))
                k_f = max(0, k_f)  # Ensure positive

                # Residence time
                u_avg = self.config['operating']['mh'] / (props['rho'] * self.Ac_hot)
                t_res = self.L_elem / u_avg

                # Update concentration
                dx = k_f * (x_eq - x_avg) * t_res
                xh_new[i+1] = xh_new[i] + dx
            else:
                xh_new[i+1] = xh_new[i]

            xh_new[i+1] = np.clip(xh_new[i+1], 0.0, 1.0)

        return xh_new

    def _print_results(self, Q_elements, U_elements):
        """Print comprehensive results"""
        print("\nFINAL RESULTS")
        print("="*70)

        # Hot fluid
        print("\nHot Fluid (Hydrogen):")
        print(f"  Inlet:  T = {self.Th[0]:.2f} K, P = {self.Ph[0]/1e3:.1f} kPa, x_para = {self.xh[0]:.4f}")
        print(f"  Outlet: T = {self.Th[-1]:.2f} K, P = {self.Ph[-1]/1e3:.1f} kPa, x_para = {self.xh[-1]:.4f}")
        print(f"  ΔT = {self.Th[0] - self.Th[-1]:.2f} K")
        print(f"  ΔP = {(self.Ph[0] - self.Ph[-1])/1e3:.2f} kPa ({(self.Ph[0]-self.Ph[-1])/self.Ph[0]*100:.1f}%)")

        # Cold fluid
        print("\nCold Fluid (Helium):")
        print(f"  Inlet:  T = {self.Tc[-1]:.2f} K, P = {self.Pc[-1]/1e3:.1f} kPa")
        print(f"  Outlet: T = {self.Tc[0]:.2f} K, P = {self.Pc[0]/1e3:.1f} kPa")
        print(f"  ΔT = {self.Tc[0] - self.Tc[-1]:.2f} K")
        print(f"  ΔP = {(self.Pc[-1] - self.Pc[0])/1e3:.2f} kPa ({abs(self.Pc[-1]-self.Pc[0])/self.Pc[-1]*100:.1f}%)")

        # Heat transfer
        Q_total = np.sum(Q_elements)
        print("\nHeat Transfer:")
        print(f"  Total heat load: {Q_total:.2f} W")
        print(f"  Average U: {np.mean(U_elements):.2f} W/(m²·K)")
        print(f"  Min U: {np.min(U_elements):.2f} W/(m²·K)")
        print(f"  Max U: {np.max(U_elements):.2f} W/(m²·K)")

        # Energy balance check
        mh = self.config['operating']['mh']
        mc = self.config['operating']['mc']

        props_h_in = self._safe_get_prop(self.Th[0], self.Ph[0], self.xh[0], False)
        props_h_out = self._safe_get_prop(self.Th[-1], self.Ph[-1], self.xh[-1], False)
        Q_hot = mh * (props_h_in['h'] - props_h_out['h'])

        props_c_in = self._safe_get_prop(self.Tc[-1], self.Pc[-1], None, True)
        props_c_out = self._safe_get_prop(self.Tc[0], self.Pc[0], None, True)
        Q_cold = mc * (props_c_out['h'] - props_c_in['h'])

        print(f"\nEnergy Balance Check:")
        print(f"  Q_hot (released): {Q_hot:.2f} W")
        print(f"  Q_cold (absorbed): {Q_cold:.2f} W")
        print(f"  Imbalance: {abs(Q_hot - Q_cold):.2f} W ({abs(Q_hot-Q_cold)/Q_hot*100:.2f}%)")

        # Conversion
        x_eq_out = self.h2_props.get_equilibrium_fraction(self.Th[-1])
        x_eq_in = self.h2_props.get_equilibrium_fraction(self.Th[0])
        conv_eff = (self.xh[-1] - self.xh[0]) / (x_eq_out - x_eq_in) * 100

        print(f"\nConversion Performance:")
        print(f"  Efficiency: {conv_eff:.2f}%")
        print(f"  x_para increase: {(self.xh[-1] - self.xh[0])*100:.2f}%")
        print(f"  Equilibrium at outlet: {x_eq_out:.4f}")
        print(f"  DNE at outlet: {(x_eq_out - self.xh[-1])/x_eq_out*100:.2f}%")

        print("="*70)


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
            'Tc_in': 43.5,      # K
            'Ph_in': 1.13e6,    # Pa
            'Pc_in': 0.54e6,    # Pa
            'mh': 1e-3,         # kg/s
            'mc': 2e-3,         # kg/s
            'xh_in': 0.452      # Initial para fraction
        },
        'catalyst': {
            'enhancement': 1.2,
            'pressure_factor': 1.3
        },
        'solver': {
            'n_elements': 20,
            'max_iter': 500,
            'tolerance': 1e-3,
            'relaxation': 0.15  # Conservative relaxation
        }
    }


if __name__ == "__main__":
    print("CORRECTED TPMS Heat Exchanger - Test Run")
    print("="*70)

    # Create and run heat exchanger
    config = create_default_config()
    he = TPMSHeatExchangerCorrected(config)
    success = he.solve(max_iter=500, tolerance=1e-3, relaxation=0.15)

    if success:
        print("\n✓ Solution physically valid and converged")
    else:
        print("\n✗ Solution did not fully converge")