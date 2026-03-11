# TPMS_HE — Cryogenic TPMS Heat Exchanger Framework

Integrated Python framework for the design, simulation, and analysis of
Triply Periodic Minimal Surface (TPMS) heat exchangers in cryogenic
hydrogen liquefaction applications.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Package Map](#2-package-map)
3. [Wizard Flow (UI)](#3-wizard-flow-ui)
4. [Class & Interface Reference](#4-class--interface-reference)
5. [Configuration Schema](#5-configuration-schema)
6. [Experiment Output Structure](#6-experiment-output-structure)
7. [Quick Start](#7-quick-start)
8. [Dependencies](#8-dependencies)

---

## 1. Architecture Overview

```
+======================================================================+
|                       TPMS_HE_main/                                  |
|                                                                       |
|  +-----------------------------------------------------------------+  |
|  |  Streamlit Wizard  (app.py -> ui/)                              |  |
|  |                                                                 |  |
|  |  Step 1        Step 2        Step 3       Step 4    Step 5     |  |
|  |  Geometry  -> Operating  -> Channels  -> Solver  -> Output     |  |
|  |  & channels               (model eq.)   config    paths       |  |
|  |                                                                 |  |
|  |              Step 6: Confirm & Run  ---------------------->    |  |
|  |              Step 7: Results display                           |  |
|  +---------------------------------+-------------------------------+  |
|                                    | build_solver_config()            |
|  +---------------------------------v-------------------------------+  |
|  |  solver/                                                         |  |
|  |  +------------------+    +-----------------------------------+  |  |
|  |  |  config.py       |    |  calculator.py                    |  |  |
|  |  |  normalize_      |--->|  TPMSHeatExchanger                |  |  |
|  |  |  config()        |    |    .solve()          -> bool      |  |  |
|  |  |  create_default_ |    |    .finalize_simulation()         |  |  |
|  |  |  config()        |    |    .get_channel_closure()         |  |  |
|  |  |  create_cpfhx_   |    +---------------+-------------------+  |  |
|  |  |  config()        |                    | per-element closure   |  |
|  |  |  make_run_dir()  |                    | dispatch              |  |
|  |  +------------------+                    |                       |  |
|  +------------------------------------------|-----------------------+  |
|                                             |                          |
|     +---------------------------------------+-------------------+      |
|     |                                       |                   |      |
|     v                                       v                   v      |
|  correlations/                    models/              properties/     |
|  ThermoHydraulic                  PackedBedTPMSModel   ThermalProper- |
|  Correlations                     .get_htc_and_        ties           |
|  .get_correlations()              friction()           .get_propert-  |
|  (bare: Nu, f)                    plate_fin_fin_       ies()          |
|                                   efficiency()         .get_equil-    |
|                                   (packed: Nu,f,htc)   ibrium_frac-  |
|                                                         tion()        |
|                                                                        |
|     +---------------------------------------------------------------+  |
|     |  Post-processing / experiments                                |  |
|     |  visualization/          design/          analysis/           |  |
|     |  TPMSVisualizer          TPMSDesigner     sensitivity.py      |  |
|     |  .plot_comprehensive()   .solve_          comparison.py       |  |
|     |  .export_results_csv()   geometry()       (per-run dirs)      |  |
|     +---------------------------------------------------------------+  |
+======================================================================+
```

### Data flow

```
config dict
    |
    +-> normalize_config()           schema normalisation + defaults
    |
    +-> TPMSHeatExchanger(cfg)       initialise N+1 nodal arrays
    |       |
    |       +-- for each element i:
    |       |       get_channel_closure(mode, structure)
    |       |         "bare"   -> ThermoHydraulicCorrelations.get_correlations()
    |       |         "packed" -> PackedBedTPMSModel.get_htc_and_friction()
    |       |
    |       +-- ThermalProperties.get_properties()   fluid state (CoolProp)
    |       |
    |       +-- iterative coupled loop:
    |               thermal   (T update, under-relaxed)
    |               hydraulic (dP via Ergun or TPMS f-Re)
    |               kinetics  (ortho->para conversion, if packed H2)
    |
    +-> TPMSVisualizer / CSV export  -> results/<run_dir>/
```

---

## 2. Package Map

```
TPMS_HE/
+-- TPMS_HE_main/
|   |
|   +-- correlations/
|   |   +-- __init__.py                    exports ThermoHydraulicCorrelations
|   |   +-- thermohydraulic_correlations.py
|   |       class ThermoHydraulicCorrelations
|   |           SUPPORTED_TPMS_TYPES   tuple[str]
|   |           .get_correlations(type, Re, Pr, fluid) -> (Nu, f)
|   |           .get_supported_tpms_types()             -> tuple[str]
|   |
|   +-- properties/
|   |   +-- __init__.py                    exports ThermalProperties
|   |   +-- hydrogen_properties.py
|   |       class ThermalProperties
|   |           .get_properties(T, P, species, x_para) -> dict{h,s,rho,cp,mu,lambda}
|   |           .get_equilibrium_fraction(T)             -> float  (para-H2 x_eq)
|   |
|   +-- models/
|   |   +-- __init__.py
|   |   +-- plate_fin.py
|   |   |   plate_fin_fin_efficiency(h, Hf, tf, k_wall, Af_Ah) -> float
|   |   +-- packed_bed.py
|   |       class PackedBedTPMSModel
|   |           .get_htc_and_friction(Re_p, Pr, ...) -> (Nu, f, htc, details)
|   |           .interval_estimate(Re_p, ...)          -> (lower, nominal, upper)
|   |           .friction_factor_ergun(Re_p, eps, psi) -> f_eq
|   |
|   +-- solver/
|   |   +-- __init__.py
|   |   +-- config.py
|   |   |   normalize_config(cfg)          -> dict
|   |   |   create_default_config()        -> dict   (H2/He, Diamond/Gyroid)
|   |   |   create_cpfhx_config(bp, r)    -> dict   (Wang 2024 CPFHX preset)
|   |   |   make_run_dir(run_name)         -> str    creates results/<run_name>/
|   |   |   _default_output_paths(subdir) -> dict   file path dict
|   |   +-- calculator.py
|   |       class TPMSHeatExchanger
|   |           .solve(max_iter, tol)             -> bool (converged?)
|   |           .get_channel_closure(mode, ...)   -> (Nu, f, htc, details)
|   |           .finalize_simulation()            -> None (plots + CSV)
|   |           Nodal arrays post-solve (length N+1):
|   |             .Th   hot temperature  [K]   index 0 = hot inlet
|   |             .Tc   cold temperature [K]   index N = cold inlet
|   |             .Ph   hot pressure     [Pa]
|   |             .Pc   cold pressure    [Pa]
|   |             .xh   para-H2 fraction [-]
|   |             .Q    heat duty        [W]   length N
|   |
|   +-- visualization/
|   |   +-- __init__.py
|   |   +-- plotter.py
|   |   |   class TPMSVisualizer
|   |   |       .plot_comprehensive(hx)
|   |   |       .plot_performance_evaluation()
|   |   |       .plot_resistance_pie()
|   |   |       .export_results_to_csv(hx, path)
|   |   +-- convergence.py
|   |       class ConvergenceTracker
|   |           .update(iter, errors, hx, relax)
|   |
|   +-- design/
|   |   +-- __init__.py
|   |   +-- designer.py
|   |       class TPMSDesigner
|   |           .solve_geometry(target_Q, ...) -> dict  (inverse solver)
|   |           .run_simulation()              -> bool
|   |           .post_process()
|   |
|   +-- analysis/           correlation-level experiments (no full solver loop)
|   |   +-- __init__.py
|   |   +-- sensitivity.py  -> results/sensitivity_YYYYMMDD_HHMMSS/
|   |   +-- comparison.py   -> results/packed_vs_bare_YYYYMMDD_HHMMSS/
|   |
|   +-- ui/                 Streamlit wizard modules
|   |   +-- state.py        init_ui_state, autosave, presets
|   |   +-- validation.py   validate_ui_state, build_solver_config
|   |   +-- components.py   STEP_DEFS, nav bar, channel summary strip
|   |   +-- step_geometry.py
|   |   +-- step_operating.py
|   |   +-- step_channels.py
|   |   +-- step_solver.py
|   |   +-- step_output.py
|   |   +-- step_results.py   run_simulation, render_run_result
|   |
|   +-- tests/
|   |   +-- test_correlations.py   11 assertions
|   |   +-- test_config.py          8 assertions
|   |   +-- test_properties.py     12 assertions
|   |   +-- test_solver.py          9 assertions (runs solver with n_elements=10)
|   |
|   +-- app.py              Streamlit entry point (~105 lines)
|
+-- results/                all solver and analysis outputs land here
|   +-- <run_name>/
|   |   +-- final_results.csv
|   |   +-- convergence_history.csv
|   |   +-- performance_profile.png
|   |   +-- convergence_diagnostics.png
|   |   +-- resistance_pie.png
|   |   +-- performance_evaluation.png
|   +-- sensitivity_YYYYMMDD_HHMMSS/
|   |   +-- pec_vs_re.{png,csv}
|   |   +-- pec_vs_porosity.{png,csv}
|   |   +-- pec_vs_cell_size.{png,csv}
|   |   +-- colburn_friction_map.png
|   +-- packed_vs_bare_YYYYMMDD_HHMMSS/
|       +-- htc_friction_comparison.{png,csv}
|       +-- overall_U_comparison.{png,csv}
|       +-- sensitivity_analysis.png
|       +-- resistance_breakdown.png
|       +-- tpms_type_comparison.csv
+-- results_packed_vs_bare/  (legacy, kept for backward compatibility)
```

---

## 3. Wizard Flow (UI)

Each step is validated before the user can proceed. Blocking issues disable
**Next ->**. All state is autosaved to `.streamlit/tpms_ui_state.json`.

```
Step 1 -- Geometry & Channels
  |-- HX dimensions: length, width, height [m]
  |-- TPMS geometry: unit_cell_size [m], wall_thickness [m]
  |-- Derived outputs (read-only, computed live):
  |     Dh_hot  = 4 * eps_hot  * a / (2*pi)  [m]
  |     Dh_cold = 4 * eps_cold * a / (2*pi)  [m]
  |     SAD     = estimated via Marching Cubes (scikit-image) or analytic formula
  |     V_fluid = W * H * L * eps            [m^3]
  |     eps     = porosity from structure geometry
  |-- Per-channel configuration (Hot / Cold):
  |     mode:       bare | packed
  |     structure:  Gyroid | Diamond | Primitive | Neovius | FRD | FKS
  |                 SmoothPlateFin | PlateFin
  |     surface_area_density [m^-1]  (editable; Geometry Estimator can fill)
  |     packed params: particle_diameter, bed_porosity, k_solid,
  |                    shape_factor, mode (lower/nominal/upper)
  +-- Channel summary strip (persistent top bar across all steps)

Step 2 -- Operating Conditions
  |-- Hot inlet:  Th_in [K], Ph_in [MPa], mass flow mh [kg/s], xh_in
  |-- Cold inlet: Tc_in [K], Pc_in [MPa], mass flow mc [kg/s]
  |-- Fluid species: "hydrogen mixture" | "normal hydrogen" | "helium" | "argon"
  +-- Validation: Th_in > Tc_in, pressures > 0, flows > 0

Step 3 -- Channels  (model selection + live equation display)
  |-- Hot channel card:
  |     mode selector     -> bare or packed
  |     structure picker  -> dropdown of SUPPORTED_TPMS_TYPES
  |     [v] Equation expander:
  |         bare mode:    Nu = f(Re, Pr) and f = g(Re) for selected structure
  |         packed mode:  ZBS/Martin-Nilles HTC + Dixon HTC + Ergun dP
  |                       with per-structure psi correction factors
  |-- Cold channel card: identical layout
  +-- Switching mode/structure updates the equation display live

Step 4 -- Solver Settings
  |-- n_elements, max_iter, tolerance
  |-- relax_thermal, relax_hydraulic, relax_kinetics, Q_damping
  +-- Adaptive Q_damping (auto-tuned from estimated NTU before first iteration)

Step 5 -- Output Paths
  +-- results_csv, convergence_csv, performance_plot, convergence_plot,
      resistance_pie, performance_eval_plot

Step 6 -- Confirm & Run
  |-- Structured summary table (all settings)
  |-- Global validation panel (blocking issues highlighted red)
  +-- [Run Simulation] -> run_simulation(state)
           -> TPMSHeatExchanger.solve()
           -> TPMSHeatExchanger.finalize_simulation()

Step 7 -- Results
  |-- Convergence status + iteration count
  |-- Key metrics: Q_total, eff_HX, dP_hot, dP_cold, Th_out, Tc_out
  |-- Figures: performance profile, convergence diagnostics
  |-- Resistance pie chart
  +-- CSV preview (first 20 rows)
```

---

## 4. Class & Interface Reference

### `ThermoHydraulicCorrelations`  (correlations/)

Stateless correlation database.  All methods are static.

| Method / Attribute | Signature | Description |
|---|---|---|
| `SUPPORTED_TPMS_TYPES` | `tuple[str]` | Gyroid, Diamond, Primitive, Neovius, FRD, FKS, SmoothPlateFin, PlateFin |
| `get_correlations` | `(type, Re, Pr, fluid) -> (Nu, f)` | Nusselt + Fanning friction; scalar or ndarray |
| `get_supported_tpms_types` | `() -> tuple[str]` | Alias for SUPPORTED_TPMS_TYPES |

`fluid_type` options: `'Water'`, `'Air'`, `'Gas'` (H2/He cryogenic), `'RP-3'`.

### `ThermalProperties`  (properties/)

CoolProp-backed fluid property engine with ortho-para H2 correction.

| Method | Signature | Description |
|---|---|---|
| `get_properties` | `(T, P, species, x_para) -> dict` | Returns h, s, rho, cp, mu, lambda and aux keys |
| `get_equilibrium_fraction` | `(T) -> float` | Para-H2 equilibrium fraction via quantum partition functions |

`species` options: `'hydrogen mixture'` (requires `x_para`), `'normal hydrogen'`,
`'helium'`, `'argon'`.

### `PackedBedTPMSModel`  (models/)

Two-level closure for packed-bed TPMS channels.

| Method | Signature | Description |
|---|---|---|
| `get_htc_and_friction` | `(Re_p, Pr, d_p, eps_bed, k_s, psi, mode, htc_model, ...) -> (Nu, f, htc, details)` | Primary closure |
| `interval_estimate` | `(Re_p, ...) -> (lower, nominal, upper)` | Bracketed HTC uncertainty band |
| `overall_htc_packed_side` | `(Re_p, ...) -> h_eff` | Effective HTC including TPMS fin enhancement |
| `friction_factor_ergun` | `(Re_p, eps_bed, psi) -> f_eq` | Modified Ergun friction factor |

`mode` in `('lower', 'nominal', 'upper')`.

### `TPMSHeatExchanger`  (solver/)

Main iterative nodal solver.  Counter-flow, N-element discretisation.

```python
hx = TPMSHeatExchanger(config)            # normalize_config() called internally
converged = hx.solve(max_iter=500, tolerance=1e-3)   # -> bool
hx.finalize_simulation()                  # writes CSV + figures to output paths
```

Post-solve nodal arrays (all length N+1 unless noted):

| Attribute | Units | Notes |
|---|---|---|
| `hx.Th` | K | Hot temperature; index 0 = hot inlet |
| `hx.Tc` | K | Cold temperature; index N = cold inlet |
| `hx.Ph` | Pa | Hot pressure |
| `hx.Pc` | Pa | Cold pressure |
| `hx.xh` | -- | Para-H2 mole fraction |
| `hx.Q`  | W  | Per-element heat duty (length N) |

Closure dispatch (`get_channel_closure`):

```
mode="bare"   -> ThermoHydraulicCorrelations.get_correlations(structure, Re, Pr, fluid)
mode="packed" -> PackedBedTPMSModel.get_htc_and_friction(Re_p, Pr, d_p, eps_bed, ...)
```

Ortho-para kinetics gate:

| Hot channel mode | Hot fluid | Kinetics |
|---|---|---|
| packed | hydrogen | ON |
| bare | hydrogen | OFF |
| any | helium / argon | OFF |

### `TPMSDesigner`  (design/)

Inverse geometry solver: given a target heat duty, finds HX dimensions.

```python
designer = TPMSDesigner(base_config)
result = designer.solve_geometry(target_Q=5000)   # W
designer.run_simulation()
designer.post_process()
```

### `TPMSVisualizer`  (visualization/)

```python
vis = TPMSVisualizer()
vis.set_academic_style()
vis.plot_comprehensive(hx)           # 6-panel figure
vis.plot_performance_evaluation()    # PEC, j, f panels
vis.plot_resistance_pie()            # thermal resistance breakdown
vis.export_results_to_csv(hx, path)
```

### Config helpers  (solver/config)

```python
from solver.config import (
    create_default_config,    # -> dict  (Diamond/Gyroid, H2/He, standard CPFHX dims)
    create_cpfhx_config,      # (back_pressure_MPa, flowrate_ratio_r) -> dict
    normalize_config,         # (raw_dict) -> canonical dict (fills all defaults)
    make_run_dir,             # (run_name) -> str  creates results/<run_name>/
    _default_output_paths,    # (run_subdir="") -> dict of output file paths
)
```

---

## 5. Configuration Schema

```python
config = {
    "geometry": {
        "length": 0.94,             # [m]  HX axial length
        "width":  0.25,             # [m]
        "height": 0.25,             # [m]
        "unit_cell_size": 5e-3,     # [m]  TPMS lattice parameter a
        "wall_thickness": 0.5e-3,   # [m]  solid wall between channels
        "plate_thickness": 1e-3,    # [m]  (PlateFin only)
        "porosity_hot":  0.65,      # eps_hot  (open volume fraction)
        "porosity_cold": 0.70,      # eps_cold
        "surface_area_density": 60, # [m^-1]  global SAD fallback
        # PlateFin / CPFHX fin geometry (Wang 2024, Table 2)
        "fin_height":    9.5e-3,
        "fin_spacing":   3.2e-3,
        "fin_thickness": 0.6e-3,
    },
    "channels": {
        "hot": {
            "mode":      "bare",     # "bare" | "packed"
            "structure": "Diamond",  # any SUPPORTED_TPMS_TYPES entry
            "surface_area_density": 1500,   # [m^-1]  per-channel override
            "geometry":  { ... },    # per-channel overrides (None = use global)
            "packed": {
                "particle_diameter": 1e-3,   # [m]
                "bed_porosity":      0.40,
                "k_solid":           10.0,   # [W/(m K)]
                "shape_factor":      1.0,
                "mode":              "nominal",       # "lower"|"nominal"|"upper"
                "htc_model":         "martin_nilles", # "martin_nilles"|"dixon"
            },
        },
        "cold": { ... },   # same structure
    },
    "operating": {
        "Th_in": 78,    "Ph_in": 2e6,   "mh": 0.02,  "xh_in": 0.452,
        "Tc_in": 43,    "Pc_in": 1.5e6, "mc": 0.06,
        "fluid_hot":  "hydrogen mixture",
        "fluid_cold": "helium",
    },
    "material": { "k_wall": 237 },   # [W/(m K)]  aluminium
    "solver": {
        "n_elements": 100,  "max_iter": 500,  "tolerance": 1e-3,
        "relax_thermal": 0.15,  "relax_hydraulic": 0.5,
        "relax_kinetics": 1.0,  "Q_damping": 0.5,
    },
    "output": {   # populated by _default_output_paths(run_subdir)
        "results_csv": "...",  "convergence_csv": "...",
        "performance_plot": "...",  ...
    },
}
```

Key geometric formulas:

```
Hydraulic diameter (bare TPMS):
  Dh = 4 * eps * a / (2*pi)

Heat transfer area per element (independent per channel):
  A_elem_h = L * W * H * alpha_h / N
  A_elem_c = L * W * H * alpha_c / N

Elemental conductance (Kays & London):
  UA_elem = 1 / (1/(h_h*A_h) + t_wall/(k_wall*A_h) + 1/(h_c*A_c))
  Q_elem  = UA_elem * (Th_i - Tc_i)
```

---

## 6. Experiment Output Structure

Every run creates an isolated subdirectory so results are never overwritten:

```
results/
+-- <run_name>/                      one per programmatic solve() call
|   +-- final_results.csv            nodal T, P, x, Q, Nu, f, Re, Pr, htc, U
|   +-- convergence_history.csv      error per iteration
|   +-- performance_profile.png      T, x, Q, P, Nu, f profiles
|   +-- convergence_diagnostics.png
|   +-- resistance_pie.png           thermal resistance breakdown
|   +-- performance_evaluation.png   j, f, PEC bar charts
|
+-- sensitivity_YYYYMMDD_HHMMSS/     python -m analysis.sensitivity
|   +-- pec_vs_re.{png,csv}
|   +-- pec_vs_porosity.{png,csv}
|   +-- pec_vs_cell_size.{png,csv}
|   +-- colburn_friction_map.png
|
+-- packed_vs_bare_YYYYMMDD_HHMMSS/  python -m analysis.comparison
    +-- htc_friction_comparison.{png,csv}
    +-- overall_U_comparison.{png,csv}
    +-- sensitivity_analysis.png
    +-- resistance_breakdown.png
    +-- tpms_type_comparison.csv
```

Programmatic run with named directory:

```python
from solver.config import create_default_config, make_run_dir, _default_output_paths
from solver.calculator import TPMSHeatExchanger

cfg = create_default_config()
cfg["channels"]["hot"]["mode"] = "packed"
cfg["output"] = _default_output_paths(run_subdir="my_experiment_001")

hx = TPMSHeatExchanger(cfg)
hx.solve()
hx.finalize_simulation()   # writes to results/my_experiment_001/
```

---

## 7. Quick Start

### Streamlit Wizard

```bash
cd TPMS_HE_main
python -m streamlit run app.py
```

### Programmatic (bare channels, default config)

```python
from solver import TPMSHeatExchanger, create_default_config

cfg = create_default_config()
hx  = TPMSHeatExchanger(cfg)
if hx.solve():
    hx.finalize_simulation()
```

### Programmatic (CPFHX packed-bed, Wang 2024)

```python
from solver import TPMSHeatExchanger, create_cpfhx_config
from solver.config import _default_output_paths

cfg = create_cpfhx_config(back_pressure_MPa=1.04, flowrate_ratio_r=2.7)
cfg["channels"]["hot"]["mode"] = "packed"
cfg["output"] = _default_output_paths("cpfhx_1.04MPa_r2.7")

hx = TPMSHeatExchanger(cfg)
hx.solve()
hx.finalize_simulation()
```

### Run sensitivity analysis experiment

```bash
cd TPMS_HE_main
python -m analysis.sensitivity
# -> results/sensitivity_YYYYMMDD_HHMMSS/
```

### Run bare-vs-packed comparison experiment

```bash
cd TPMS_HE_main
python -m analysis.comparison
# -> results/packed_vs_bare_YYYYMMDD_HHMMSS/
```

### Run tests

```bash
cd TPMS_HE_main
python -m pytest tests/ -v
# 40 tests, <10 seconds
```

---

## 8. Dependencies

```bash
pip install numpy scipy pandas matplotlib CoolProp streamlit scikit-image
```

| Package | Role |
|---|---|
| numpy / scipy | Numerics, optimisation (fsolve), root-finding (brentq) |
| pandas | CSV export and tabular output |
| matplotlib | All figures |
| CoolProp | Fluid properties (H2 isomers, He, Ar) via AbstractState |
| streamlit | Wizard UI |
| scikit-image | Marching Cubes for TPMS SAD estimation (optional) |
