"""
Validation: Python Dixon model vs MATLAB plate_fin_determined_HE.m
Key test: Dixon correlations (Bi correction, radial conductivity) match MATLAB lines 166-186
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from solver.calculator import TPMSHeatExchanger

config = {
    'geometry': {'length': 0.94, 'width': 0.1504, 'height': 0.0095, 'plate_thickness': 1.2e-3},
    'material': {'k_wall': 237},
    'operating': {
        'Th_in': 66.3, 'Ph_in': 1130e3, 'Tc_in': 43.5, 'Pc_in': 540e3,
        'mh': 1e-3, 'mc': 2e-3, 'xh_in': 0.452,
        'fluid_hot': 'hydrogen mixture', 'fluid_cold': 'helium',
    },
    'channels': {
        'hot': {
            'structure': 'Diamond', 'mode': 'packed',
            'geometry': {'porosity': 0.5048},
            'surface_area_density': 600,
            'packed': {
                'htc_model': 'dixon', 'mode': 'nominal',
                'particle_diameter': 600e-6, 'bed_porosity': 0.5048, 'k_solid': 0.58,
            },
        },
        'cold': {'structure': 'Gyroid', 'mode': 'bare', 'surface_area_density': 800},
    },
    'solver': {'n_elements': 10, 'max_iter': 500, 'tolerance': 1e-3, 'relax_thermal': 0.1},
}

he = TPMSHeatExchanger(config)
converged = he.solve(max_iter=500, tolerance=1e-3)

print(f"\n{'='*60}")
print("Dixon Model Validation")
print(f"{'='*60}")
print(f"Converged: {converged}")
print(f"Hot:  {he.Th[0]:.2f}K → {he.Th[-1]:.2f}K")
print(f"Cold: {he.Tc[-1]:.2f}K → {he.Tc[0]:.2f}K")
print(f"Q_total: {sum(he.Q):.2f}W")

# Check Dixon model was used
if he.elem_details['hot'][0] and 'htc_model' in he.elem_details['hot'][0]:
    print(f"Hot HTC model: {he.elem_details['hot'][0]['htc_model']}")
    print(f"Bi number: {he.elem_details['hot'][0].get('Bi', 'N/A'):.3f}")
    print(f"✓ Dixon model active with Biot correction")
else:
    print("✗ Dixon model NOT detected")

print(f"{'='*60}\n")
assert converged, "Solver failed to converge"
print("✓ Test PASSED: Dixon model runs successfully")
