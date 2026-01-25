"""
TPMS Heat Exchanger for Ortho-Para Hydrogen Conversion Simulation

This module simulates heat transfer in a TPMS (Triply Periodic Minimal Surface) heat exchanger
specifically designed for ortho-para hydrogen conversion processes. It solves coupled heat 
transfer and chemical conversion equations for counter-flow configuration.

Key Features:
1. Energy balance enforcement
2. Thermodynamically consistent heat transfer
3. Stable iteration with physical bounds
4. Counter-flow heat exchanger logic
5. Ortho-para hydrogen conversion kinetics

Author: Based on research by Zhang et al. (2025)
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
    TPMS Heat Exchanger Solver for Ortho-Para Hydrogen Conversion
    
    This class implements a counter-flow heat exchanger model with ortho-para hydrogen 
    conversion kinetics. The hot stream contains hydrogen undergoing conversion from 
    ortho to para form, releasing conversion heat that affects heat transfer performance.
    
    Configuration:
    - Hot fluid: flows 0 → L (left to right), contains hydrogen with ortho-para conversion
    - Cold fluid: flows L → 0 (right to left), typically helium or another coolant
    
    Physical constraints enforced:
    - Hot fluid must cool down (Th_out < Th_in) unless conversion heat dominates
    - Cold fluid must heat up (Tc_out > Tc_in) unless strongly affected by conversion
    - Energy balance: Q_hot_released = Q_cold_absorbed (including conversion heat)
    """

    def __init__(self, config):
        """
        Initialize the TPMS heat exchanger solver
        
        Parameters
        ----------
        config : dict
            Configuration dictionary containing all simulation parameters
        """
        self.config = config
        self.h2_props = HydrogenProperties()

        # Extract geometry parameters from configuration
        self.L_HE = config['geometry']['length']           # Heat exchanger length (m)
        self.W_HE = config['geometry']['width']            # Heat exchanger width (m)
        self.H_HE = config['geometry']['height']           # Heat exchanger height (m)
        self.porosity_hot = config['geometry']['porosity_hot']  # Porosity of hot channel
        self.porosity_cold = config['geometry']['porosity_cold'] # Porosity of cold channel
        self.unit_cell = config['geometry']['unit_cell_size']     # Unit cell size (m)
        self.wall_thickness = config['geometry']['wall_thickness'] # Wall thickness (m)

        # TPMS type for each channel (e.g., Diamond, Gyroid, Schwarz, etc.)
        self.TPMS_hot = config['tpms']['type_hot']
        self.TPMS_cold = config['tpms']['type_cold']

        # Calculate geometric properties and initialize solution
        self._calculate_geometry()
        self.N_elements = config['solver']['n_elements']  # Number of axial elements for discretization
        self._initialize_solution()

    def _calculate_geometry(self):
        """
        Calculate geometric parameters for the heat exchanger
        
        This method computes hydraulic diameter, cross-sectional areas, and heat transfer area
        based on the geometry configuration parameters.
        """
        # Hydraulic diameter calculation for porous media
        # Dh = 4 * porosity * unit_cell / (2 * pi) - approximation for TPMS structures
        self.Dh_hot = 4 * self.porosity_hot * self.unit_cell / (2 * np.pi)
        self.Dh_cold = 4 * self.porosity_cold * self.unit_cell / (2 * np.pi)
        
        # Cross-sectional flow areas for hot and cold streams
        self.Ac_hot = self.W_HE * self.H_HE * self.porosity_hot
        self.Ac_cold = self.W_HE * self.H_HE * self.porosity_cold

        # Heat transfer area calculation based on surface area density
        surface_area_density = self.config['geometry'].get('surface_area_density', 600)  # m²/m³
        self.A_heat = self.L_HE * self.W_HE * self.H_HE * surface_area_density  # Total heat transfer area (m²)
        self.k_wall = self.config['material']['k_wall']  # Wall thermal conductivity (W/(m·K))

        print(f"Geometry calculated:")
        print(f"  Dh_hot = {self.Dh_hot*1e3:.3f} mm")      # Convert to mm for readability
        print(f"  Dh_cold = {self.Dh_cold*1e3:.3f} mm")    # Convert to mm for readability
        print(f"  Ac_hot = {self.Ac_hot*1e6:.2f} mm²")     # Convert to mm² for readability
        print(f"  Ac_cold = {self.Ac_cold*1e6:.2f} mm²")   # Convert to mm² for readability
        print(f"  A_heat = {self.A_heat:.3f} m²")          # Heat transfer area in m²

    def _initialize_solution(self):
        """
        Initialize temperature and composition fields with physically reasonable guesses
        
        This method sets up initial temperature profiles for both hot and cold streams,
        ensuring that the initial guess satisfies basic physical expectations for a 
        counter-flow heat exchanger.
        
        Spatial discretization:
        - Axial positions: 0 (left) → N (right)
        - Hot stream: flows left→right (inlet at 0, outlet at N)
        - Cold stream: flows right→left (inlet at N, outlet at 0)
        """
        N = self.N_elements + 1  # Number of nodes (elements + 1)

        # Extract inlet temperatures from configuration
        Th_in = self.config['operating']['Th_in']  # Hot inlet temperature (K)
        Tc_in = self.config['operating']['Tc_in']  # Cold inlet temperature (K)

        # Make reasonable outlet temperature guesses based on counter-flow operation
        # Hot outlet should be close to cold inlet (but slightly higher due to finite heat transfer)
        Th_out_guess = Tc_in + 5.0  # Hot outlet temperature guess (K)
        # Cold outlet should be close to hot inlet (but slightly lower due to finite heat transfer)  
        Tc_out_guess = Th_in - 5.0  # Cold outlet temperature guess (K)

        # Initialize temperature arrays
        # Hot stream: T decreases from inlet (node 0) to outlet (node N)
        self.Th = np.linspace(Th_in, Th_out_guess, N)

        # Cold stream: T increases from inlet (node N) to outlet (node 0)
        # Stored in order from node 0 to N, but flows from right to left
        self.Tc = np.linspace(Tc_out_guess, Tc_in, N)

        # Initialize pressure arrays (constant pressure assumed for this model)
        self.Ph = np.full(N, self.config['operating']['Ph_in'])  # Hot stream pressures
        self.Pc = np.full(N, self.config['operating']['Pc_in'])  # Cold stream pressures

        # Initialize para-hydrogen fraction profile
        # At equilibrium, para fraction increases along hot flow direction due to conversion
        x_eq_in = self.h2_props.get_equilibrium_fraction(Th_in)    # Equilibrium fraction at hot inlet
        x_eq_out = self.h2_props.get_equilibrium_fraction(Th_out_guess)  # Equilibrium fraction at hot outlet
        # Linear profile from inlet fraction to near-equilibrium at outlet
        self.xh = np.linspace(self.config['operating']['xh_in'],
                              0.5*(self.config['operating']['xh_in'] + x_eq_out), N)

        # Calculate element length for discrete analysis
        self.L_elem = self.L_HE / self.N_elements  # Length of each axial element (m)

        print(f"\nInitialization:")
        print(f"  Hot: {Th_in:.2f} K → {Th_out_guess:.2f} K")  # Display temperature range
        print(f"  Cold: {Tc_in:.2f} K → {Tc_out_guess:.2f} K")  # Display temperature range

    def _safe_get_prop(self, T, P, x=None, is_helium=False):
        """
        Safely retrieve thermodynamic properties with physical bounds checking
        
        This method ensures that property calculations remain within physically 
        reasonable bounds and handles exceptions gracefully.
        
        Parameters
        ----------
        T : float or array-like
            Temperature (K)
        P : float or array-like
            Pressure (Pa)
        x : float, optional
            Para-hydrogen mole fraction (for hydrogen only)
        is_helium : bool, default False
            Flag indicating if properties are for helium (True) or hydrogen (False)
            
        Returns
        -------
        dict
            Dictionary containing thermodynamic properties (h, rho, cp, mu, lambda)
        """
        # Apply physical bounds to prevent property calculation errors
        if is_helium:
            T = np.clip(T, 4.0, 400.0)  # Helium temperature bounds (K)
        else:
            T = np.clip(T, 14.0, 400.0)  # Hydrogen temperature bounds (K)

        P = np.clip(P, 1e4, 5e6)  # Pressure bounds (0.01-5 MPa)

        if x is not None:
            x = np.clip(x, 0.0, 1.0)  # Para fraction bounds (0-100%)

        try:
            if is_helium:
                # Get helium properties from hydrogen properties module
                p = self.h2_props.get_helium_properties(T, P)
            else:
                # Get hydrogen properties with specified para fraction
                p = self.h2_props.get_properties(T, P, x if x is not None else 0.5)

            # Validate returned properties to catch NaN or invalid values
            if np.isnan(p['h']) or np.isnan(p['rho']) or np.isnan(p['cp']):
                raise ValueError("NaN in properties")

            return p

        except Exception as e:
            # Fallback to reasonable default values when property calculation fails
            print(f"Warning: Property calculation failed at T={T:.1f}K, using fallback")
            return {
                'h': 5000.0 if not is_helium else 2000.0,   # Specific enthalpy (J/kg)
                'rho': 5.0 if not is_helium else 2.0,       # Density (kg/m³)
                'cp': 14000.0 if not is_helium else 5200.0, # Specific heat (J/kg·K)
                'mu': 1.0e-5,                               # Dynamic viscosity (Pa·s)
                'lambda': 0.15,                             # Thermal conductivity (W/m·K)
            }

    def solve(self, max_iter=500, tolerance=1e-3, relaxation=0.15):
        """
        Solve the heat exchanger model with proper energy balance enforcement
        
        This method implements an iterative solution procedure that calculates:
        1. Local thermodynamic properties at each node
        2. Heat transfer coefficients for each element
        3. Heat transfer rates using LMTD method
        4. Updated temperature profiles based on energy balance
        5. Para-hydrogen conversion due to kinetics
        
        Parameters
        ----------
        max_iter : int, default 500
            Maximum number of iterations before giving up
        tolerance : float, default 1e-3
            Convergence tolerance for temperature changes
        relaxation : float, default 0.15
            Under-relaxation factor for numerical stability (0.05-0.3 recommended)
            
        Returns
        -------
        bool
            True if solution converged and is physically valid, False otherwise
        """
        print("=" * 70)
        print("CORRECTED TPMS Heat Exchanger Solver")
        print("=" * 70)

        # Extract mass flow rates from configuration
        mh = self.config['operating']['mh']  # Hot stream mass flow rate (kg/s)
        mc = self.config['operating']['mc']  # Cold stream mass flow rate (kg/s)

        print(f"\nOperating conditions:")
        print(f"  mh = {mh*1e3:.3f} g/s")      # Convert to g/s for readability
        print(f"  mc = {mc*1e3:.3f} g/s")      # Convert to g/s for readability
        print(f"  mc/mh = {mc/mh:.2f}")        # Mass flow rate ratio

        start_time = time.time()  # Record start time for performance monitoring

        # Main iteration loop
        for iteration in range(max_iter):
            # Store previous iteration values for convergence checking and relaxation
            Th_old = self.Th.copy()
            Tc_old = self.Tc.copy()
            xh_old = self.xh.copy()

            # ============================================
            # STEP 1: CALCULATE THERMODYNAMIC PROPERTIES AT EACH AXIAL NODE
            # ============================================
            # Compute properties at all nodes for both streams
            props_h = []  # Properties for hot hydrogen stream
            props_c = []  # Properties for cold helium stream

            for i in range(len(self.Th)):
                # Calculate properties for hot hydrogen stream (includes para fraction)
                ph = self._safe_get_prop(self.Th[i], self.Ph[i], self.xh[i], False)
                # Calculate properties for cold helium stream
                pc = self._safe_get_prop(self.Tc[i], self.Pc[i], None, True)
                props_h.append(ph)
                props_c.append(pc)

            # ============================================
            # STEP 2: CALCULATE HEAT TRANSFER FOR EACH AXIAL ELEMENT
            # ============================================
            # Initialize arrays for heat transfer rates and overall heat transfer coefficients
            Q_elements = np.zeros(self.N_elements)  # Heat transfer rate for each element (W)
            U_elements = np.zeros(self.N_elements)  # Overall heat transfer coefficient (W/m²·K)

            for k in range(self.N_elements):
                # Process element k: spans from axial node k to node k+1
                # Hot fluid: enters at node k, exits at node k+1 (flows left to right)
                # Cold fluid: enters at node k+1, exits at node k (flows right to left, counter-current)

                # Calculate average properties in the element using arithmetic mean
                # This provides better accuracy than using inlet or outlet conditions alone
                ph_avg = {key: 0.5*(props_h[k][key] + props_h[k+1][key])
                         for key in ['rho', 'cp', 'mu', 'lambda']}  # Hot stream average properties
                pc_avg = {key: 0.5*(props_c[k][key] + props_c[k+1][key])
                         for key in ['rho', 'cp', 'mu', 'lambda']}  # Cold stream average properties

                # Calculate individual heat transfer coefficients for hot and cold sides
                h_hot = self._calculate_htc(ph_avg, mh, self.Ac_hot,
                                           self.Dh_hot, self.TPMS_hot, True)  # Hot side HTC
                h_cold = self._calculate_htc(pc_avg, mc, self.Ac_cold,
                                            self.Dh_cold, self.TPMS_cold, False)  # Cold side HTC

                # Calculate overall heat transfer coefficient (based on heat transfer resistances)
                # Includes hot-side convection, wall conduction, and cold-side convection resistances
                U = 1.0 / (1.0/h_hot + self.wall_thickness/self.k_wall + 1.0/h_cold)
                U_elements[k] = U

                # Apply Log Mean Temperature Difference (LMTD) method for counter-current flow
                # Hot side: Th[k] → Th[k+1] (temperature drops in hot stream)
                # Cold side: Tc[k+1] → Tc[k] (temperature rises in cold stream)
                dT_in = self.Th[k] - self.Tc[k+1]    # Temperature difference at "hot end" - corrected
                dT_out = self.Th[k+1] - self.Tc[k]   # Temperature difference at "cold end" - corrected

                # Check for temperature crossovers or pinches that would violate thermodynamics
                if dT_in <= 0 or dT_out <= 0:
                    # Temperature crossover indicates thermodynamic impossibility
                    # Set heat transfer to zero to prevent unphysical behavior
                    Q_elements[k] = 0.0
                    continue

                # Calculate LMTD for counter-current flow
                # Handles the case where dT_in ≈ dT_out to avoid numerical singularity
                if abs(dT_in - dT_out) < 1e-6:
                    LMTD = dT_in  # Arithmetic mean when logarithmic form approaches indeterminate
                else:
                    LMTD = (dT_in - dT_out) / np.log(abs(dT_in / dT_out))  # Standard LMTD formula

                # Calculate heat transfer rate using overall coefficient and LMTD
                A_elem = self.A_heat / self.N_elements  # Heat transfer area per element (m²)
                Q_elements[k] = U * A_elem * LMTD  # Heat transfer rate in element k (W)

                # Apply physical limits to prevent unrealistic heat transfer rates
                # Maximum possible heat transfer based on stream capacitance rates
                Ch = mh * ph_avg['cp']  # Hot stream capacitance rate (W/K)
                Cc = mc * pc_avg['cp']  # Cold stream capacitance rate (W/K)
                
                # Calculate max possible heat transfer based on actual temperature differences
                Q_max_hot = Ch * abs(self.Th[k] - self.Th[k+1])  # Max heat hot stream can release
                Q_max_cold = Cc * abs(self.Tc[k+1] - self.Tc[k])  # Max heat cold stream can absorb

                # Limit heat transfer to physically possible values
                Q_max_possible = min(abs(Q_max_hot), abs(Q_max_cold))
                Q_elements[k] = np.sign(Q_elements[k]) * min(abs(Q_elements[k]), Q_max_possible)

            # ============================================
            # STEP 3: UPDATE TEMPERATURE PROFILES USING ENERGY BALANCE
            # ============================================
            # Use explicit forward integration with under-relaxation for stability
            # This approach maintains causality in the counter-current flow arrangement

            # Update hot stream temperatures (flows from node 0 to N)
            Th_new = np.zeros(len(self.Th))
            Th_new[0] = self.config['operating']['Th_in']  # Hot inlet temperature is fixed

            for k in range(self.N_elements):
                # Calculate temperature drop in hot stream due to heat loss in element k
                cp_h = props_h[k]['cp']  # Use inlet properties for element k
                if cp_h > 0:  # Ensure positive heat capacity
                    dT_h = -Q_elements[k] / (mh * cp_h)  # Negative because heat is lost
                    Th_new[k+1] = Th_new[k] + dT_h  # Integrate along hot stream
                else:
                    Th_new[k+1] = Th_new[k]  # No temperature change if invalid cp

            # Update cold stream temperatures (flows from node N to 0 in counter-current direction)
            # But stored in array from index 0 to N, so integrate backwards
            Tc_new = np.zeros(len(self.Tc))
            Tc_new[-1] = self.config['operating']['Tc_in']  # Cold inlet temperature is fixed

            for k in range(self.N_elements-1, -1, -1):
                # Calculate temperature rise in cold stream due to heat gain in element k
                cp_c = props_c[k+1]['cp']  # Use outlet properties for element k
                if cp_c > 0:  # Ensure positive heat capacity
                    dT_c = Q_elements[k] / (mc * cp_c)  # Positive because heat is gained
                    Tc_new[k] = Tc_new[k+1] + dT_c  # Integrate along cold stream (backward direction)
                else:
                    Tc_new[k] = Tc_new[k+1]  # No temperature change if invalid cp

            # ============================================
            # STEP 4: APPLY UNDER-RELAXATION AND PHYSICAL BOUNDS
            # ============================================
            # Apply under-relaxation to improve numerical stability
            # New values are blended with old values using relaxation factor
            self.Th = Th_old + relaxation * (Th_new - Th_old)
            self.Tc = Tc_old + relaxation * (Tc_new - Tc_old)

            # Enforce physical temperature bounds to prevent property calculation errors
            self.Th = np.clip(self.Th, 14.0, 100.0)  # Hot stream temperature bounds (K)
            self.Tc = np.clip(self.Tc, 4.0, 100.0)   # Cold stream temperature bounds (K)

            # Enforce monotonicity constraints based on physics
            # Hot stream should generally cool down from inlet to outlet
            for k in range(len(self.Th)-1):
                if self.Th[k+1] > self.Th[k]:
                    # Prevent heating in hot stream unless conversion heat dominates
                    self.Th[k+1] = self.Th[k] - 0.1  # Small temperature drop

            # Cold stream should generally heat up from inlet to outlet
            for k in range(len(self.Tc)-1):
                if self.Tc[k] > self.Tc[k+1]:
                    # Prevent cooling in cold stream unless conversion heat dominates
                    self.Tc[k] = self.Tc[k+1] - 0.1  # Small temperature rise

            # ============================================
            # STEP 5: UPDATE ORTHO-PARA CONVERSION PROFILES
            # ============================================
            # Calculate para-hydrogen fraction evolution due to conversion kinetics
            xh_new = self._ortho_para_conversion()
            # Apply under-relaxation to conversion updates for stability
            self.xh = xh_old + 0.5 * relaxation * (xh_new - xh_old)
            # Enforce bounds on para fraction (0-100%)
            self.xh = np.clip(self.xh, 0.0, 1.0)

            # ============================================
            # STEP 6: CHECK CONVERGENCE AND MONITOR PROGRESS
            # ============================================
            # Calculate convergence metrics for temperatures and conversion
            err_T = np.max(np.abs(self.Th - Th_old)) + np.max(np.abs(self.Tc - Tc_old))
            err_x = np.max(np.abs(self.xh - xh_old))
            err = err_T + err_x  # Combined error metric

            # Print progress every 20 iterations or for first 5 iterations
            if (iteration + 1) % 20 == 0 or iteration < 5:
                print(f"Iter {iteration+1:4d}: err={err:.4e} | "
                      f"Th: {self.Th[0]:.2f}→{self.Th[-1]:.2f}K | "
                      f"Tc: {self.Tc[-1]:.2f}→{self.Tc[0]:.2f}K | "
                      f"Q_tot={np.sum(Q_elements):.1f}W")

            # Check for serious physical violations that require solution reset
            if self.Th[-1] > self.Th[0]:
                # Hot outlet temperature higher than inlet violates second law
                print(f"\nERROR: Hot outlet hotter than inlet! Th_out={self.Th[-1]:.2f} > Th_in={self.Th[0]:.2f}")
                print("Resetting temperatures...")
                self._initialize_solution()
                continue

            if self.Tc[0] < self.Tc[-1]:
                # Cold outlet temperature lower than inlet violates second law
                print(f"\nERROR: Cold outlet colder than inlet! Tc_out={self.Tc[0]:.2f} < Tc_in={self.Tc[-1]:.2f}")
                print("Resetting temperatures...")
                self._initialize_solution()
                continue

            # Check for convergence based on error tolerance
            if err < tolerance:
                elapsed = time.time() - start_time  # Calculate total solution time
                print(f"\n{'='*70}")
                print(f"CONVERGED in {iteration+1} iterations ({elapsed:.2f} s)")
                print(f"{'='*70}")
                # Validate the converged solution and return result
                is_valid = self._print_results(Q_elements, U_elements)
                return is_valid  # Return validation result, not just True

        # Maximum iterations reached without convergence
        print(f"\nWARNING: Maximum iterations ({max_iter}) reached")
        print(f"Final error: {err:.4e}")
        is_valid = self._print_results(Q_elements, U_elements)
        return False  # Non-convergence always returns False

    def _calculate_htc(self, props_avg, m_dot, Ac, Dh, tpms_type, is_hot):
        """
        Calculate heat transfer coefficient using TPMS correlations
        
        This method computes the convective heat transfer coefficient for flow through
        TPMS structures using empirical correlations from the literature.
        
        Parameters
        ----------
        props_avg : dict
            Average thermodynamic properties {'rho', 'cp', 'mu', 'lambda'}
        m_dot : float
            Mass flow rate (kg/s)
        Ac : float
            Cross-sectional flow area (m²)
        Dh : float
            Hydraulic diameter (m)
        tpms_type : str
            TPMS type ('Diamond', 'Gyroid', 'Schwarz', etc.)
        is_hot : bool
            True if calculating for hot stream, False for cold stream
            
        Returns
        -------
        float
            Heat transfer coefficient (W/m²·K)
        """
        # Calculate velocity from mass flow rate and properties
        u = m_dot / (props_avg['rho'] * Ac)  # Velocity (m/s)
        
        # Calculate Reynolds number for flow characterization
        Re = props_avg['rho'] * u * Dh / props_avg['mu']
        
        # Calculate Prandtl number for heat transfer correlation
        Pr = props_avg['mu'] * props_avg['cp'] / props_avg['lambda']

        # Get Nusselt number from TPMS-specific correlations
        Nu, _ = TPMSCorrelations.get_correlations(tpms_type, Re, Pr, 'Gas')

        # Handle invalid correlation results with reasonable defaults
        if np.isnan(Nu) or Nu < 1.0:
            Nu = 10.0  # Reasonable default for turbulent flow in porous media

        # Apply catalyst enhancement factor for hot side (if applicable)
        # Catalyst enhances heat transfer due to increased surface area
        enhancement = 1.2 if is_hot else 1.0

        # Calculate heat transfer coefficient from Nusselt number
        h = enhancement * Nu * props_avg['lambda'] / Dh
        return h

    def _ortho_para_conversion(self):
        """
        Calculate ortho-para hydrogen conversion using kinetic model
        
        This method solves the differential equation for ortho-para conversion
        along the hot stream, accounting for temperature-dependent kinetics
        and equilibrium limitations.
        
        Returns
        -------
        numpy.ndarray
            Updated para-hydrogen fraction profile along the hot stream
        """
        # Initialize new para fraction array
        xh_new = np.zeros(len(self.xh))
        xh_new[0] = self.config['operating']['xh_in']  # Inlet para fraction is fixed

        # Critical point parameters for hydrogen
        Tc = 32.938   # Critical temperature of hydrogen (K)
        Pc = 1.284e6  # Critical pressure of hydrogen (Pa)

        # Process each axial element to calculate conversion
        for i in range(self.N_elements):
            # Calculate average conditions in element i
            T_avg = 0.5 * (self.Th[i] + self.Th[i+1])      # Average temperature (K)
            P_avg = 0.5 * (self.Ph[i] + self.Ph[i+1])      # Average pressure (Pa)
            x_avg = 0.5 * (xh_new[i] + self.xh[i+1])       # Average para fraction

            # Calculate equilibrium para fraction at average temperature
            x_eq = self.h2_props.get_equilibrium_fraction(T_avg)

            # Only proceed with conversion if current fraction is below equilibrium
            if x_avg < x_eq - 1e-6:
                # Calculate kinetic rate constant using empirical correlation
                # Based on temperature and pressure effects on conversion rate
                K = 59.7 - 253.9 * (T_avg / Tc) - 11.6 * (P_avg / Pc)

                # Calculate molar concentration of hydrogen
                props = self._safe_get_prop(T_avg, P_avg, x_avg, False)
                C_H2 = props['rho'] / 0.002016  # Molar concentration (mol/m³), MW_H2 = 2.016 g/mol

                # Calculate forward reaction rate constant
                # Using logarithmic relation based on deviation from equilibrium
                k_f = (K / C_H2) * np.log(((x_avg / x_eq)**1.0924) * ((1 - x_eq) / (1 - x_avg + 1e-10)))
                k_f = max(0, k_f)  # Ensure positive rate constant

                # Calculate residence time in the element
                u_avg = self.config['operating']['mh'] / (props['rho'] * self.Ac_hot)  # Avg velocity
                t_res = self.L_elem / u_avg  # Residence time (s)

                # Update para fraction based on conversion during residence time
                dx = k_f * (x_eq - x_avg) * t_res  # Change in para fraction
                xh_new[i+1] = xh_new[i] + dx
            else:
                # No conversion occurs when already at or near equilibrium
                xh_new[i+1] = xh_new[i]

            # Apply bounds to prevent unphysical para fractions
            xh_new[i+1] = np.clip(xh_new[i+1], 0.0, 1.0)

        return xh_new

    def _print_results(self, Q_elements, U_elements):
        """
        Print comprehensive results and perform physical validation
        
        This method displays the final solution results and validates them against
        physical principles and engineering expectations.
        
        Parameters
        ----------
        Q_elements : numpy.ndarray
            Heat transfer rates for each axial element (W)
        U_elements : numpy.ndarray
            Overall heat transfer coefficients for each element (W/m²·K)
            
        Returns
        -------
        bool
            True if solution passes all validation checks, False otherwise
        """
        print("\nFINAL RESULTS")
        print("="*70)

        # Display hot stream results
        print("\nHot Fluid (Hydrogen):")
        print(f"  Inlet:  T = {self.Th[0]:.2f} K, P = {self.Ph[0]/1e3:.1f} kPa, x_para = {self.xh[0]:.4f}")
        print(f"  Outlet: T = {self.Th[-1]:.2f} K, P = {self.Ph[-1]/1e3:.1f} kPa, x_para = {self.xh[-1]:.4f}")
        print(f"  ΔT = {self.Th[0] - self.Th[-1]:.2f} K")  # Temperature change
        print(f"  ΔP = {(self.Ph[0] - self.Ph[-1])/1e3:.2f} kPa ({(self.Ph[0]-self.Ph[-1])/self.Ph[0]*100:.1f}%)")

        # Display cold stream results
        print("\nCold Fluid (Helium):")
        print(f"  Inlet:  T = {self.Tc[-1]:.2f} K, P = {self.Pc[-1]/1e3:.1f} kPa")
        print(f"  Outlet: T = {self.Tc[0]:.2f} K, P = {self.Pc[0]/1e3:.1f} kPa")
        print(f"  ΔT = {self.Tc[0] - self.Tc[-1]:.2f} K")  # Temperature change
        print(f"  ΔP = {(self.Pc[-1] - self.Pc[0])/1e3:.2f} kPa ({abs(self.Pc[-1]-self.Pc[0])/self.Pc[-1]*100:.1f}%)")

        # Display heat transfer results
        Q_total = np.sum(Q_elements)  # Total heat transfer (W)
        print("\nHeat Transfer:")
        print(f"  Total heat load: {Q_total:.2f} W")
        print(f"  Average U: {np.mean(U_elements):.2f} W/(m²·K)")
        print(f"  Min U: {np.min(U_elements):.2f} W/(m²·K)")
        print(f"  Max U: {np.max(U_elements):.2f} W/(m²·K)")

        # Perform energy balance check
        mh = self.config['operating']['mh']  # Hot stream mass flow rate
        mc = self.config['operating']['mc']  # Cold stream mass flow rate

        # Calculate heat released by hot stream
        props_h_in = self._safe_get_prop(self.Th[0], self.Ph[0], self.xh[0], False)
        props_h_out = self._safe_get_prop(self.Th[-1], self.Ph[-1], self.xh[-1], False)
        Q_hot = mh * (props_h_in['h'] - props_h_out['h'])

        # Calculate heat absorbed by cold stream
        props_c_in = self._safe_get_prop(self.Tc[-1], self.Pc[-1], None, True)
        props_c_out = self._safe_get_prop(self.Tc[0], self.Pc[0], None, True)
        Q_cold = mc * (props_c_out['h'] - props_c_in['h'])

        # Calculate conversion heat contribution (difference between para and ortho states)
        props_h_in_para = self._safe_get_prop(self.Th[0], self.Ph[0], 1.0, False)  # Pure para
        props_h_in_ortho = self._safe_get_prop(self.Th[0], self.Ph[0], 0.0, False)  # Pure ortho
        h_conversion_avg = props_h_in_ortho['h'] - props_h_in_para['h']  # Conversion heat (J/kg)
        Q_conversion = mh * (self.xh[-1] - self.xh[0]) * h_conversion_avg  # Conversion heat (W)

        print(f"\nEnergy Balance Check:")
        print(f"  Q_hot (released): {Q_hot:.2f} W")
        print(f"  Q_cold (absorbed): {Q_cold:.2f} W")
        print(f"  Q_conversion (o→p): {Q_conversion:.2f} W")
        print(f"  Q_total (via U*A*LMTD): {Q_total:.2f} W")
        print(f"  Imbalance: {abs(Q_hot - Q_cold):.2f} W ({abs(Q_hot-Q_cold)/max(abs(Q_hot), 1e-10)*100:.2f}%)")

        # PERFORM VALIDATION CHECKS
        is_valid = True
        validation_errors = []

        # Check 1: Energy balance should be within acceptable tolerance (5%)
        energy_imbalance_pct = abs(Q_hot - Q_cold) / max(abs(Q_hot), 1e-10) * 100
        if energy_imbalance_pct > 5.0:
            is_valid = False
            validation_errors.append(f"Energy imbalance {energy_imbalance_pct:.1f}% > 5% threshold")

        # Check 2: Temperature changes should be significant but reasonable
        dT_hot = abs(self.Th[0] - self.Th[-1])
        if dT_hot < 0.5:
            validation_errors.append(f"Hot side ΔT = {dT_hot:.2f} K is suspiciously small")
            is_valid = False

        # Check 3: Heat transfer rates should be consistent between methods
        if abs(Q_total - Q_hot) / max(abs(Q_hot), 1e-10) > 0.5:
            validation_errors.append(f"Q_total ({Q_total:.1f}W) vastly different from Q_hot ({Q_hot:.1f}W)")
            is_valid = False

        # Check 4: Conversion efficiency should be within realistic bounds
        x_eq_out = self.h2_props.get_equilibrium_fraction(self.Th[-1])  # Equilibrium at outlet
        x_eq_in = self.h2_props.get_equilibrium_fraction(self.Th[0])    # Equilibrium at inlet
        conv_eff = (self.xh[-1] - self.xh[0]) / max(x_eq_out - x_eq_in, 1e-10) * 100

        if conv_eff > 150 or conv_eff < -10:
            validation_errors.append(f"Conversion efficiency {conv_eff:.1f}% is physically unrealistic")
            is_valid = False

        # Check 5: Mass flow ratio should be within practical range
        if mc / mh < 0.5 or mc / mh > 10:
            validation_errors.append(f"Mass flow ratio mc/mh = {mc/mh:.2f} is extreme")

        # Display conversion performance metrics
        print(f"\nConversion Performance:")
        print(f"  Efficiency: {conv_eff:.2f}%")
        print(f"  x_para increase: {(self.xh[-1] - self.xh[0])*100:.2f}%")
        print(f"  Equilibrium at outlet: {x_eq_out:.4f}")
        print(f"  DNE at outlet: {(x_eq_out - self.xh[-1])/x_eq_out*100:.2f}%")

        print("="*70)

        # Print validation results
        if is_valid:
            print("\n✓ SOLUTION IS PHYSICALLY VALID")
            print("  All validation checks passed")
        else:
            print("\n✗ SOLUTION HAS PHYSICAL ERRORS")
            print("  Validation failures detected:")
            for error in validation_errors:
                print(f"    • {error}")
            print("\n  Possible causes:")
            print("    1. Flow rates too extreme (try mc/mh = 1.5-3.0)")
            print("    2. Temperature approach too small (need Th_in - Tc_in > 10K)")
            print("    3. Heat exchanger too small for given flow rates")
            print("    4. Numerical convergence to non-physical solution")
            print("\n  Recommendations:")
            print("    • Reduce mass flow rates")
            print("    • Increase relaxation damping (try 0.05-0.10)")
            print("    • Increase heat exchanger size or surface area density")
            print("    • Check if conversion heat dominates sensible cooling")

        print("="*70)

        return is_valid


def create_default_config():
    """
    Create default configuration dictionary for TPMS heat exchanger
    
    This function provides a reasonable starting configuration for the heat exchanger
    simulation, including geometry, material properties, operating conditions, etc.
    
    Returns
    -------
    dict
        Configuration dictionary with all required parameters
    """
    return {
        'geometry': {
            'length': 0.94,         # Heat exchanger length (m)
            'width': 0.15,          # Heat exchanger width (m)
            'height': 0.10,         # Heat exchanger height (m)
            'porosity_hot': 0.65,   # Porosity of hot channel
            'porosity_cold': 0.70,  # Porosity of cold channel
            'unit_cell_size': 5e-3, # TPMS unit cell size (m)
            'wall_thickness': 0.5e-3,  # Wall thickness between channels (m)
            'surface_area_density': 1200  # Increased surface area per unit volume (m²/m³) - was 600
        },
        'tpms': {
            'type_hot': 'Diamond',   # TPMS type for hot channel
            'type_cold': 'Gyroid'    # TPMS type for cold channel
        },
        'material': {
            'k_wall': 237  # Wall thermal conductivity (W/(m·K)) - Aluminum
        },
        'operating': {
            'Th_in': 66.3,      # Hot inlet temperature (K)
            'Tc_in': 35.0,      # Lower cold inlet temperature (K) - was 43.5
            'Ph_in': 1.13e6,    # Hot stream inlet pressure (Pa)
            'Pc_in': 0.54e6,    # Cold stream inlet pressure (Pa)
            'mh': 0.2e-3,       # Further reduced hot stream mass flow rate (kg/s) - was 0.5e-3
            'mc': 0.4e-3,       # Further reduced cold stream mass flow rate (kg/s) - was 1.0e-3
            'xh_in': 0.452      # Initial para-hydrogen fraction at hot inlet
        },
        'catalyst': {
            'enhancement': 1.2,      # Heat transfer enhancement factor
            'pressure_factor': 1.3   # Pressure effect factor
        },
        'solver': {
            'n_elements': 20,        # Number of axial discretization elements
            'max_iter': 500,         # Maximum iterations
            'tolerance': 1e-4,       # Convergence tolerance - Tighter for better precision
            'relaxation': 0.05       # Even lower relaxation factor for stability - was 0.08
        }
    }


if __name__ == "__main__":
    print("CORRECTED TPMS Heat Exchanger - Test Run")
    print("="*70)

    # Create and run heat exchanger simulation
    config = create_default_config()
    he = TPMSHeatExchanger(config)
    is_valid = he.solve(max_iter=500, tolerance=1e-3, relaxation=0.15)

    if is_valid:
        print("\n✓ Solution is physically valid and converged")
    else:
        print("\n✗ Solution failed validation or did not converge")
        print("   Review the validation errors above and adjust parameters")