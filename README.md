# TPMS_HE

Integrated cryogenic TPMS heat-exchanger framework with two-level architecture:

- Level 1: exchanger/channel orchestration (hot + cold channels, coupled solver loop)
- Level 2: per-channel closure dispatch (`bare` TPMS or `packed` TPMS)

This repository now merges the original two stages into one solver path while keeping backward compatibility with legacy configs.

## Recent Progress (March 2026)

- Completed two-level integration in the main solver:
  - Level 1 orchestration in `TPMSHeatExchanger`
  - Level 2 per-channel closure dispatch (`bare`/`packed`)
- Added canonical channel schema with backward-compatible normalization (`normalize_config`).
- Upgraded Streamlit UI to a guided 6-step wizard with:
  - persistent hot/cold channel status strip
  - autosave/reload/reset
  - strict blocking validation
  - final structured confirmation table
- Fixed runtime stability issues found during UI-driven runs:
  - Windows console encoding-safe export logs (`[OK] ...` messages)
  - mathtext parse issue in visualization summary (`Delta P_h` text rendering)
  - matplotlib `ParseException` in text annotation panels (see below)
- Added TPMS Geometry Estimator panel (Step 1 – Geometry):
  - Surface area density (SAD) estimation via Marching Cubes (scikit-image, optional)
    for Gyroid, Diamond, Primitive; empirical formulas for Neovius, FRD, FKS
  - Fluid cross-section area vs. axial position plot (voxel-counting, Matplotlib)
  - "Use this value" button writes estimated SAD back to `surface_area_density` input
  - Iso-value search via `scipy.optimize.brentq` in normalized [0, 2π] space
  - Graceful fallback to analytic-constant approximations when scikit-image absent
- Added equation expanders to Step 3 – Channels:
  - Bare mode: shows Nu/f correlations for each TPMS structure and fluid type
  - Packed mode: shows HTC model equations (ZBS/Martin-Nilles, Dixon) and Ergun
    pressure drop with per-structure ψ correction factors
- Added `SmoothPlateFin` as a built-in TPMS structure (see below)
- Removed `k_enhance` dispersion multiplier from packed-bed model
  (`overall_htc_packed_side`): the empirical 1.2×/1.5× conductivity boost for
  nominal/upper modes is removed. TPMS geometry effects on effective conductivity
  are now captured by `C_shape` (geometric path shortcut: 8→6→4) and the accurate
  `surface_area_density` from the Geometry Estimator. The two remaining enhancement
  mechanisms are `C_shape` and `area_factor` (fin efficiency).

## Repository Layout

```text
TPMS_HE/
|- TPMS_HE_main/
|  |- tpms_thermo_hydraulic_calculator.py   # Integrated solver + config normalization
|  |- tpms_correlations.py                  # Bare TPMS Nu-Re / f-Re closures
|  |- packed_bed_model.py                   # Packed-bed TPMS closure model
|  |- app.py                                # Streamlit wizard UI (autosave + validation)
|  |- hydrogen_properties.py
|  |- convergence_tracker.py
|  |- tpms_visualization.py
|  |- analysis_packed_vs_bare.py
|- results/
|- results_packed_vs_bare/
```

## Heat Transfer Area Model

Hot and cold channels may use **different TPMS structures** (e.g. Gyroid hot,
Diamond cold), each with a different specific surface area density α. The solver
therefore maintains **per-channel elemental areas**:

```
A_elem_h = L_HE * W * H * α_h / N_elements   [hot channel]
A_elem_c = L_HE * W * H * α_c / N_elements   [cold channel]
```

The heat transfer per element is computed in **conductance space** (W/K),
which is reference-area-independent and correct for any α_h / α_c combination:

```
G_hot    = h_h * A_elem_h          [W/K]  hot-side convective conductance
G_cold   = h_c * A_elem_c          [W/K]  cold-side convective conductance
G_wall   = k_wall * A_elem_h / t   [W/K]  wall conduction conductance

UA_elem  = 1 / (1/G_hot + 1/G_wall + 1/G_cold)   [W/K]  total conductance
Q_elem   = UA_elem * ΔT                            [W]
```

This is the standard Kays & London formulation:
`1/(UA) = 1/(h_h·A_h) + R_wall + 1/(h_c·A_c)`.
`UA` (W/K) is the only area-reference-independent quantity; `Q = UA·ΔT` requires
no explicit reference area choice.

**Hydraulic quantities** (velocity, Re, dP) remain per-channel and use their own
cross-sectional area and Dh independently of α:

```
Ac_h = W * H * ε_h    Dh_h = 4·ε_h·cell_size/(2π)
Ac_c = W * H * ε_c    Dh_c = 4·ε_c·cell_size/(2π)
```

### Per-channel α configuration

Each channel has its own `surface_area_density` key in the config:

```python
config["channels"]["hot"]["surface_area_density"] = 1500   # e.g. Gyroid [1/m]
config["channels"]["cold"]["surface_area_density"] = 1653  # e.g. Diamond [1/m]
```

If not set, both channels fall back to `config["geometry"]["surface_area_density"]`
(backward-compatible default = 60 m⁻¹). In the UI, each channel card in Step 3
has its own α input; the Geometry Estimator "Use this value" button writes to the
active channel's α.

`U[i]` stored in output CSV is referenced to the hot-side area (W/m²K × A_h = W/K).

### Packed-bed TPMS enhancement mechanisms

In packed mode (`overall_htc_packed_side`), TPMS geometry enhances heat transfer
through two mechanisms:

1. **C_shape** — geometric path shortcut: TPMS channels reduce the effective
   conduction length relative to a circular tube.
   - `lower`: C_shape = 8 (round-tube baseline)
   - `nominal`: C_shape = 6 (moderate TPMS shortcut)
   - `upper`: C_shape = 4 (aggressive shortcut)

2. **area_factor** — fin area enhancement: TPMS ligaments protrude into the packed
   bed, acting as fins with efficiency η computed from `tpms_fin_efficiency()`.

```
h_eff = area_factor / R_total
area_factor = 1 + f_fin * eta_fin    (lower: f=0, nominal: f=0.2, upper: f=0.4)
```

The `f_fin` fractions (0.2/0.4) are conservative placeholders. Future work will
derive these from the actual TPMS `surface_area_density` ratio `(α_TPMS/α_flat - 1)`
using the Geometry Estimator output.

`k_enhance` (a dispersion multiplier applied to `k_r_eff`) has been removed. The
geometric effect it approximated is now captured by `C_shape` and the accurate SAD.

`A_elem` itself is unchanged — `area_factor` multiplies only the effective HTC.

### SmoothPlateFin baseline structure

A plain parallel-plate channel (`SmoothPlateFin`) is supported as a first-class
"TPMS" structure alongside Gyroid, Diamond, etc. It uses:

- **Nu**: Dittus-Boelter — `Nu = 0.023 Re^0.8 Pr^n` (n=0.4 heating, n=0.3 cooling);
  laminar fallback `Nu = 3.66` for Re < 2300
- **f**: Petukhov-Filonenko Fanning — `f = (0.790 ln Re − 1.64)^−2 / 4`;
  laminar fallback `f = 16/Re`
- **surface_area_density**: `2/H` (two flat walls, channel height H); no TPMS
  multiplier
- **hydraulic diameter**: `2·W·H·ε / (W + H·ε)` (rectangular duct)

Setting one or both channels to `SmoothPlateFin` gives the classical plate-fin
result under identical geometry and flow conditions, enabling direct PEC comparison:

```
PEC = (Q_TPMS / Q_plateFin) / (ΔP_TPMS / ΔP_plateFin)^(1/3)
```

This ratio is reported in the output CSV and shown in the Step 6 Run Result panel.

## Two-Level Integration Design

### Level 1: Exchanger orchestration

Implemented in `TPMSHeatExchanger`:

- hot/cold channel states and marching order
- coupled energy + pressure + kinetics loops
- calls Level 2 closure per element and per channel
- existing output pipeline preserved (CSV + plots)

### Level 2: Correlation/closure registry

Implemented via `get_channel_closure(...)` dispatch:

- `mode="bare"` -> `TPMSCorrelations.get_correlations(...)`
- `mode="packed"` -> `PackedBedTPMSModel.get_htc_and_friction(...)`
- returns unified `(Nu, f, htc, details)` contract

## Ortho-Para Conversion Gating

Ortho-to-para hydrogen conversion is now tied to **hot channel mode**:

- hot channel `packed` + hydrogen hot fluid -> conversion kinetics **ON**
- hot channel `bare` + hydrogen hot fluid -> conversion kinetics **OFF** (`rate = 0`)
- non-hydrogen hot fluid -> conversion kinetics **OFF**

Implementation details:

- `xh` is initialized as a flat inlet profile (no artificial conversion ramp).
- In bare hot-channel cases, solver enforces constant `xh` profile each iteration.

## Cross-Temperature Phenomenon (Tc_out > Th_in)

When the hot channel is packed with an ortho-para conversion catalyst, the
H2 stream contains an **internal distributed heat source** — the exothermic
ortho→para conversion (~703 kJ/kg at cryogenic temperatures).  This decouples
the system from standard single-phase HX thermodynamics:

```
Standard HX (no internal source):    Tc_out ≤ Th_in  always
Packed-bed H2 HX (conversion ON):    Tc_out > Th_in  is physically valid
```

**Physical mechanism:**  Near the cold end of the exchanger the H2 has already
been cooled significantly and its ortho-para conversion accelerates (higher
equilibrium para-fraction at lower T).  The conversion heat released into the H2
stream can locally exceed the sensible cooling rate, so the H2 temperature
*rises* in that region.  The cold He stream continues to absorb heat from what
is now a locally warmer H2, driving Tc_out above Th_in (the H2 *inlet*
temperature).

**Consequence for effectiveness:**  The standard NTU-ε min-capacity-rate formula
assumes no internal sources and would yield ε > 100% if applied naively.  Two
corrections are applied in this solver:

1. **Q_actual from enthalpy states** — `Q = mh × [h(Th_in, xh_in) − h(Th_out, xh_out)]`
   (inlet/outlet enthalpy difference, not `Σ UA·ΔT`).  This correctly accounts
   for conversion heat deposited into the stream.

2. **Q_max from full conversion to equilibrium** — when conversion is active,
   `Q_max = mh × [h(Th_in, xh_in) − h(Tc_in, x_eq(Tc_in))]`.  This is the
   maximum enthalpy the H2 could release: sensible cooling *plus* complete
   ortho→para conversion at the cold-end temperature.  Because this bound always
   exceeds Q_actual, effectiveness is guaranteed ≤ 100%.

   In bare mode (no conversion), the standard rule applies:
   `Q_max = min(Q_max_hot, Q_max_cold)`.

**Code location:** `_initialize_state` in
`tpms_thermo_hydraulic_calculator.py` (lines ~475–504); `_print_results` and
`tpms_visualization.py` use enthalpy-based Q throughout.

## New Canonical Config Schema

```python
config["channels"]["hot"]["mode"] = "bare" or "packed"
config["channels"]["hot"]["structure"] = "Gyroid" | "Diamond" | "Primitive" | "Neovius" | "FRD" | "FKS"
config["channels"]["hot"]["packed"] = {
    "particle_diameter": ...,
    "bed_porosity": ...,
    "k_solid": ...,
    "shape_factor": ...,
    "mode": "lower" | "nominal" | "upper",
}

config["channels"]["cold"][...]  # same keys
```

### Backward compatibility

Legacy keys are still accepted:

- `tpms.type_hot`, `tpms.type_cold`
- global `catalyst.*`
- solver `relax` (mapped to `relax_thermal` if needed)

`normalize_config(...)` auto-builds canonical `channels.*` from legacy config.

## Run the Integrated Solver

```bash
cd TPMS_HE_main
python tpms_thermo_hydraulic_calculator.py
```

Programmatic entry:

```python
from tpms_thermo_hydraulic_calculator import TPMSHeatExchanger, create_default_config

cfg = create_default_config()
cfg["channels"]["hot"]["mode"] = "packed"
cfg["channels"]["cold"]["mode"] = "bare"

he = TPMSHeatExchanger(cfg)
he.solve(max_iter=cfg["solver"]["max_iter"], tolerance=cfg["solver"]["tolerance"])
he.finalize_simulation()
```

## Streamlit UI Controller (Wizard)

Run:

```bash
cd TPMS_HE_main
python -m streamlit run app.py
```

Wizard steps:

- 1) Geometry
- 2) Operating
- 3) Channels (hot/cold each with explicit `bare/packed` + TPMS structure)
- 4) Solver
- 5) Output
- 6) Confirm & Run (structured summary table)

UI behavior:

- persistent channel summary strip: `Hot: mode / structure / packed-mode`, `Cold: ...`
- autosave to `TPMS_HE_main/.streamlit/tpms_ui_state.json`
- strict validation gate (blocking errors disable `Next`/`Run`)
- sliders for bounded core fields (porosity, relax factors, `xh_in`, packed `bed_porosity`, `shape_factor`)
- `Reload Autosave` and `Reset Defaults` controls

Outputs shown in UI:

- convergence status and key metrics
- generated file paths
- performance/convergence figures
- CSV preview

Troubleshooting:

- if `streamlit` command is not found, always use:
  - `python -m streamlit run app.py`
- if autosave causes unexpected form values, use `Reload Autosave` or `Reset Defaults`
- if you see `ParseException` during plotting, see the section below

### matplotlib ParseException in text annotation panels

**Symptom:**

```
ParseException: Expected end of text, found '$' (at char 0), (line:1, col:1)
```

Matplotlib prints each character of the failing string to the terminal before
raising the exception, so you will see output like:

```
T h e r m o − H y d r a u l i c   P e r f o r m a n c e
ParseException: Expected end of text, found '$'...
```

**Root cause:**

Matplotlib's mathtext parser is triggered whenever a text string contains `$`
delimiters.  Inside `$...$`, a hyphen `-` is treated as a **minus operator**,
so strings like `$\bf{Thermo-Hydraulic\ Summary}$` cause a parse error because
`Thermo` and `Hydraulic` are parsed as variables with `-` (subtraction) between
them.  Additionally, mixed strings of the form `r"$\bf{...}$" + f"...plain..."`
confuse the parser in some matplotlib versions.

**Fix applied (March 2026):**

All `$\bf{...}$` and `$\it{...}$` mathtext markup was removed from the
multi-line text annotation strings in `tpms_visualization.py`.  Headers now use
plain `"--- Header ---"` style text; section labels use plain strings.  Axis
titles and labels that use `$...$` only for single symbols (e.g. `$\xi$`,
`$\Delta P$`) are unaffected — those contain no hyphens and parse correctly.

**Rule going forward:**

Never use `$\bf{text-with-hyphens}$` or `$\it{text-with-hyphens}$` in
`ax.text()` / `ax.annotate()` calls.  Use plain Unicode text for annotation
boxes; reserve `$...$` for axis labels and titles where only math symbols
(Greek letters, fractions, superscripts) are needed.

## Mode Combination Examples

All 4 hot/cold combinations are supported:

```python
# 1) bare / bare
cfg["channels"]["hot"]["mode"] = "bare"
cfg["channels"]["cold"]["mode"] = "bare"
# hydrogen conversion: OFF (rate=0)

# 2) packed / bare
cfg["channels"]["hot"]["mode"] = "packed"
cfg["channels"]["cold"]["mode"] = "bare"
# hydrogen conversion: ON

# 3) bare / packed
cfg["channels"]["hot"]["mode"] = "bare"
cfg["channels"]["cold"]["mode"] = "packed"
# hydrogen conversion: OFF (hot channel controls kinetics)

# 4) packed / packed
cfg["channels"]["hot"]["mode"] = "packed"
cfg["channels"]["cold"]["mode"] = "packed"
# hydrogen conversion: ON
```

## Stage 2 Chinese Notation Note

Some packed-bed comments/docstrings remain in Chinese. Common symbols:

- `d_p`: particle diameter
- `eps_bed`: bed porosity
- `k_s`: solid thermal conductivity
- `D_h`: TPMS hydraulic diameter
- `Re_p`: particle Reynolds number
- `h_eff`: effective packed-side heat-transfer coefficient
- `f_equiv`: equivalent Fanning friction factor
- `psi`: TPMS pressure-drop correction factor

## Dependencies

```bash
pip install numpy scipy pandas matplotlib CoolProp streamlit
```
