# TPMS_HE

Integrated cryogenic TPMS heat-exchanger framework with two-level architecture:

- Level 1: exchanger/channel orchestration (hot + cold channels, coupled solver loop)
- Level 2: per-channel closure dispatch (`bare` TPMS or `packed` TPMS)

This repository now merges the original two stages into one solver path while keeping backward compatibility with legacy configs.

## Repository Layout

```text
TPMS_HE/
|- TPMS_HE_main/
|  |- tpms_thermo_hydraulic_calculator.py   # Integrated solver + config normalization
|  |- tpms_correlations.py                  # Bare TPMS Nu-Re / f-Re closures
|  |- packed_bed_model.py                   # Packed-bed TPMS closure model
|  |- app.py                                # Streamlit UI controller (MVP)
|  |- hydrogen_properties.py
|  |- convergence_tracker.py
|  |- tpms_visualization.py
|  |- analysis_packed_vs_bare.py
|- results/
|- results_packed_vs_bare/
```

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

## Mode Combination Examples

All 4 hot/cold combinations are supported:

```python
# 1) bare / bare
cfg["channels"]["hot"]["mode"] = "bare"
cfg["channels"]["cold"]["mode"] = "bare"

# 2) packed / bare
cfg["channels"]["hot"]["mode"] = "packed"
cfg["channels"]["cold"]["mode"] = "bare"

# 3) bare / packed
cfg["channels"]["hot"]["mode"] = "bare"
cfg["channels"]["cold"]["mode"] = "packed"

# 4) packed / packed
cfg["channels"]["hot"]["mode"] = "packed"
cfg["channels"]["cold"]["mode"] = "packed"
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
