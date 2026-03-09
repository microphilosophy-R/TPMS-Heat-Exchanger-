# TPMS_HE_main

Integrated TPMS heat-exchanger solver package (Stage-1 + Stage-2 merged).

## What Is Included

- `tpms_thermo_hydraulic_calculator.py`: main coupled solver
- `tpms_correlations.py`: bare TPMS Nu-Re and f-Re correlations
- `packed_bed_model.py`: packed-bed closure model for TPMS channels
- `app.py`: Streamlit wizard UI
- `hydrogen_properties.py`: CoolProp-backed property model
- `convergence_tracker.py`: convergence diagnostics
- `tpms_visualization.py`: result plotting + CSV export

## Two-Level Architecture

Level 1 (orchestration):

- manages hot/cold channel marching
- solves coupled thermal-hydraulic + conversion loops

Level 2 (channel closure):

- `bare` mode -> `TPMSCorrelations`
- `packed` mode -> `PackedBedTPMSModel`

## Channel Configuration

```python
config["channels"]["hot"]["mode"] = "bare" or "packed"
config["channels"]["hot"]["structure"] = "Gyroid" | "Diamond" | "Primitive" | "Neovius" | "FRD" | "FKS"
config["channels"]["hot"]["packed"] = {
    "particle_diameter": 1e-3,
    "bed_porosity": 0.40,
    "k_solid": 10.0,
    "shape_factor": 1.0,
    "mode": "nominal",  # lower / nominal / upper
}
```

The same schema applies to `channels["cold"]`.

## Ortho-Para Conversion Rule

Conversion is gated by **hot channel mode**:

- hot packed + hydrogen hot fluid -> conversion ON
- hot bare + hydrogen hot fluid -> conversion OFF (`rate = 0`)
- non-hydrogen hot fluid -> conversion OFF

Notes:

- `xh` starts from a flat inlet profile.
- In hot-bare mode, the solver keeps `xh` constant each iteration.

Example (conversion OFF):

```python
cfg["operating"]["fluid_hot"] = "hydrogen mixture"
cfg["channels"]["hot"]["mode"] = "bare"
```

Example (conversion ON):

```python
cfg["operating"]["fluid_hot"] = "hydrogen mixture"
cfg["channels"]["hot"]["mode"] = "packed"
```

## Run

Command-line solver:

```bash
python tpms_thermo_hydraulic_calculator.py
```

Streamlit UI:

```bash
python -m streamlit run app.py
```

## Streamlit Wizard Highlights

- step-by-step setup (Geometry -> Operating -> Channels -> Solver -> Output -> Confirm)
- explicit hot/cold mode + TPMS structure selectors
- autosave file: `.streamlit/tpms_ui_state.json`
- strict validation gate before run

## Dependencies

```bash
pip install numpy scipy pandas matplotlib CoolProp streamlit
```
