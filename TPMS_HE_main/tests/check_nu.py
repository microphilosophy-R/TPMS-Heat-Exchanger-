"""Quick check of hot side Nu values"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from solver.calculator import TPMSHeatExchanger

# Case 1 from Wang data
config = {
    'geometry': {'length': 0.94, 'width': 0.1504, 'plate_thickness': 1.2e-3},
    'material': {'k_wall': 237},
    'operating': {
        'Th_in': 63.8, 'Ph_in': 1.04e6,
        'Tc_in': 42.8, 'Pc_in': 0.54e6,
        'mh': 1e-3, 'mc': 2.4e-3, 'xh_in': 0.452,
        'fluid_hot': 'hydrogen mixture', 'fluid_cold': 'helium',
    },
    'channels': {
        'hot': {
            'structure': 'PlateFin', 'mode': 'packed',
            'geometry': {
                'height': 9.5e-3,
                'fin_height': 9.5e-3,
                'fin_spacing': 3.2e-3,
                'fin_thickness': 0.6e-3,
            },
            'surface_area_density': 760,
            'packed': {
                'htc_model': 'dixon', 'mode': 'nominal',
                'particle_diameter': 600e-6, 'bed_porosity': 0.5048, 'k_solid': 0.58,
            },
        },
        'cold': {
            'structure': 'PlateFin', 'mode': 'bare',
            'geometry': {
                'height': 9.5e-3,
                'n_layers': 2.0,
                'fin_height': 9.5e-3,
                'fin_spacing': 1.0e-3,
                'fin_thickness': 0.2e-3,
            },
            'surface_area_density': 2100,
        },
    },
    'solver': {'n_elements': 10, 'max_iter': 500, 'tolerance': 1e-3, 'relax_thermal': 0.1},
}

he = TPMSHeatExchanger(config)
he.solve(max_iter=500, tolerance=1e-3)

print(f"\nHot side Nu values (experimental range: 7-10):")
print(f"  Mean: {he.elem_h['Nu'].mean():.2f}")
print(f"  Min:  {he.elem_h['Nu'].min():.2f}")
print(f"  Max:  {he.elem_h['Nu'].max():.2f}")
print(f"\nHot side Re values:")
print(f"  Mean: {he.elem_h['Re'].mean():.2f}")
