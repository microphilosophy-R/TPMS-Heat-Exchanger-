TPMS Heat Exchanger Simulation Framework (Python)

A comprehensive, unified numerical simulation framework for analyzing Triply Periodic Minimal Surface (TPMS) heat exchangers. This package is optimized for cryogenic applications, featuring coupled thermal-hydraulic solving and ortho-para hydrogen conversion kinetics.

Features

✨ Unified Solver: Coupled resolution of energy, momentum, and chemical kinetics with relaxation

🔍 Convergence Tracking: Real-time diagnostics of residuals, energy imbalance, and damping factors

🏗️ Multiple TPMS Structures: Gyroid, Diamond, Primitive, Neovius, FRD, FKS

📊 Academic Visualization: Publication-quality plots (Times New Roman) and formatted CSV exports

🔬 Real Fluid Properties: High-fidelity CoolProp (HEOS) integration for Hydrogen mixtures & Helium

🎯 Cryogenic Optimized: Validated for hydrogen liquefaction temperature ranges (20-80 K)

Installation

Prerequisites

# Python 3.8 or higher
python --version

# Install CoolProp
pip install CoolProp>=6.8.0

# Install dependencies
pip install numpy pandas matplotlib scipy


Quick Setup

# Clone the repository
git clone <repository-url>
cd tpms_simulation

# Run the main solver to test configuration
python tpms_thermo_hydraulic_calculator.py


Package Structure

tpms_simulation/
├── tpms_thermo_hydraulic_calculator.py          # MAIN ENTRY: Unified solver & simulation loop
├── convergence_tracker.py   # Numerical stability tracking & diagnostics
├── tpms_visualization.py    # Academic plotting & data export module
├── tpms_correlations.py     # Database of Nu-Re and f-Re correlations
├── hydrogen_properties.py   # CoolProp wrapper with enthalpy correction
└── README.md                # This file


Quick Start

Basic Usage

You can run the simulation directly using the default configuration embedded in the main script:

python tpms_thermo_hydraulic_calculator.py


Custom Scripting

To run a custom simulation programmatically:

from new_enhanced import TPMSHeatExchanger, create_default_config

# 1. Load default config
config = create_default_config()

# 2. Customize parameters
config['tpms']['type_hot'] = 'Diamond'
config['tpms']['type_cold'] = 'Gyroid'
config['operating']['mh'] = 0.025  # kg/s

# 3. Initialize and Solve
he = TPMSHeatExchanger(config)
is_converged = he.solve()

# 4. Finalize (Generates plots and CSVs)
if is_converged:
    he.finalize_simulation()


Modules Overview

1. new_enhanced.py (The Solver)

The core engine containing the TPMSHeatExchanger class.

Key Capabilities:

Smart Initialization: Uses $\epsilon-NTU$ estimates for initial temperature profiles.

Relaxation Schemes: Adaptive under-relaxation for thermal ($T$), hydraulic ($P$), and kinetic ($x_p$) updates.

Thermodynamic Guardrails: Prevents temperature crossovers using local enthalpy potential checks.

2. convergence_tracker.py

A utility that acts as the "black box" recorder for the simulation.

Features:

Tracks residuals for $T$, $P$, and $Q$ independently.

Monitors global energy imbalance (Hot Loss vs. Cold Gain).

Plots convergence history to output/tpms_convergence.png.

3. tpms_visualization.py

Generates "Academic Standard" figures suitable for papers/dissertations.

Output Files:

tpms_performance.png: 4-panel plot (Temp, Conversion, Pressure, Nu).

tpms_final_results.csv: spatially resolved data for all variables.

4. hydrogen_properties.py

Handles the complex thermodynamics of Hydrogen spin isomers.

Reference State: Saturated liquid para-hydrogen.

Conversion Heat: Calibrated to $\Delta h_{n\to p} \approx 527.138$ kJ/kg at 20 K.

Configuration

The simulation is controlled via a nested dictionary passed to the solver.

config = {
    'geometry': {
        'length': 0.94,          # m
        'width': 0.25,           # m
        'porosity_hot': 0.65,    # Void fraction
        'surface_area_density': 1600 # m²/m³
    },
    'tpms': {
        'type_hot': 'Diamond',   # Options: Diamond, Gyroid, FKS...
        'type_cold': 'Gyroid'
    },
    'operating': {
        'Th_in': 78.0,           # K (Inlet Hot)
        'Tc_in': 43.0,           # K (Inlet Cold)
        'Ph_in': 2.0e6,          # Pa
        'xh_in': 0.452           # Inlet Para-fraction
    },
    'solver': {
        'n_elements': 100,       # Grid size
        'relax_thermal': 0.15,   # Relaxation factor
        'tolerance': 1e-4
    }
}


TPMS Selection Guide

For Cryogenic Hydrogen (20-80 K)

TPMS Structure

Re Range

Nu Characteristics

f Characteristics

Recommendation

Diamond

800 - 9590

$\propto Re^{0.625}$

$\propto Re^{-0.194}$

Best Overall (High $Nu$, Moderate $f$)

Gyroid

2000 - 8170

$\propto Re^{0.700}$

$\propto Re^{-0.200}$

Good for high flow/turbulence

FKS

730 - 10230

$\propto Re^{0.610}$

$\propto Re^{-0.133}$

Specialized for low Reynolds

Primitive

N/A

Lower Performance

Higher Drag

Not recommended for H₂

Troubleshooting

1. Divergence Detected

Symptom: "!!! Divergence Detected !!!" printed in console.
Fix:

Increase damping: Set config['solver']['relax_thermal'] to 0.05.

Check Energy Balance: Ensure mh * Cp_h and mc * Cp_c are somewhat comparable.

2. CoolProp Errors

Symptom: ImportError or ValueError: fluid not found.
Fix:

Ensure CoolProp is installed: pip install CoolProp.

This solver uses the HEOS backend; ensure your CoolProp version supports it (v6.0+).

3. Zero Conversion

Symptom: Para-hydrogen fraction remains constant.
Fix:

Ensure fluid name contains "hydrogen" (e.g., 'hydrogen mixture').

Check relax_kinetics in solver config (should be > 0).

References

The correlations implemented in tpms_correlations.py are derived from the following experimental studies, as cited in the code:

[101] - Cryogenic Gas Correlations (Diamond, FKS)

[41] - Air Correlations (Gyroid, Diamond)

[46], [149] - Water Correlations (Gyroid)

[109] - Sheet-type TPMS Correlations

[119] - Low-Re Water Correlations

Quick Reference

# 1. Import
from new_enhanced import TPMSHeatExchanger, create_default_config

# 2. Config
cfg = create_default_config()
cfg['solver']['relax_thermal'] = 0.2

# 3. Solve
solver = TPMSHeatExchanger(cfg)
solver.solve()

# 4. Results
solver.finalize_simulation() 
# -> Check 'output/' folder for PNGs and CSVs


Last Updated: January 2026

Python: 3.8+

Dependencies: CoolProp, NumPy, Pandas, Matplotlib, SciPy

Happy Simulating! 🚀