"""
Validation against Wang et al. CPFHX experimental data (Table 6)
Tests all 6 operating conditions with Dixon model
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from solver.calculator import TPMSHeatExchanger

# Wang et al. Table 6 experimental data
WANG_DATA = [
    # P_back, m_ratio, Th_in, Th_out, Tc_in, Tc_out
    (1.04, 2.4, 63.8, 55.9, 42.8, 61.7),
    (1.04, 2.7, 63.1, 55.1, 42.8, 60.8),
    (1.04, 3.0, 62.3, 54.1, 42.7, 59.7),
    (1.13, 2.4, 65.8, 57.4, 43.8, 64.2),
    (1.13, 2.7, 65.0, 56.5, 44.2, 63.1),
    (1.13, 3.0, 64.5, 55.9, 44.8, 62.2),
]

def run_case(case_id, P_back, m_ratio, Th_in, Th_out_exp, Tc_in, Tc_out_exp):
    """Run single validation case"""
    mh = 1e-3  # 1 g/s baseline
    mc = mh * m_ratio

    # MATLAB geometry: hot 47 fins×1 layer, cold 150.4 fins×2 layers
    config = {
        'geometry': {'length': 0.94, 'width': 0.1504, 'plate_thickness': 1.2e-3},
        'material': {'k_wall': 237},
        'operating': {
            'Th_in': Th_in, 'Ph_in': P_back*1e6,
            'Tc_in': Tc_in, 'Pc_in': 0.54e6,
            'mh': mh, 'mc': mc, 'xh_in': 0.452,
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
                'surface_area_density': 760,  # Tuned to match MATLAB
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
                'surface_area_density': 2100,  # Tuned to match MATLAB
            },
        },
        'solver': {'n_elements': 10, 'max_iter': 500, 'tolerance': 1e-3, 'relax_thermal': 0.1},
    }

    he = TPMSHeatExchanger(config)
    converged = he.solve(max_iter=500, tolerance=1e-3)

    Th_out_sim = he.Th[-1]
    Tc_out_sim = he.Tc[0]

    err_h = abs(Th_out_sim - Th_out_exp)
    err_c = abs(Tc_out_sim - Tc_out_exp)

    return converged, Th_out_sim, Tc_out_sim, err_h, err_c

print(f"\n{'='*80}")
print("Wang et al. CPFHX Experimental Validation (Table 6)")
print(f"{'='*80}")
print(f"{'Case':<6} {'P(MPa)':<8} {'m_r':<6} {'Th_out_exp':<12} {'Th_out_sim':<12} {'ΔT_h':<8} {'Tc_out_exp':<12} {'Tc_out_sim':<12} {'ΔT_c':<8}")
print(f"{'-'*80}")

results = []
for i, (P, mr, Thi, Tho_exp, Tci, Tco_exp) in enumerate(WANG_DATA, 1):
    conv, Tho_sim, Tco_sim, eh, ec = run_case(i, P, mr, Thi, Tho_exp, Tci, Tco_exp)
    results.append((eh, ec))
    status = "✓" if conv else "✗"
    print(f"{status} {i:<4} {P:<8.2f} {mr:<6.1f} {Tho_exp:<12.1f} {Tho_sim:<12.2f} {eh:<8.2f} {Tco_exp:<12.1f} {Tco_sim:<12.2f} {ec:<8.2f}")

print(f"{'-'*80}")
avg_err_h = sum(r[0] for r in results) / len(results)
avg_err_c = sum(r[1] for r in results) / len(results)
max_err_h = max(r[0] for r in results)
max_err_c = max(r[1] for r in results)

print(f"Average error: Hot={avg_err_h:.2f}K, Cold={avg_err_c:.2f}K")
print(f"Maximum error: Hot={max_err_h:.2f}K, Cold={max_err_c:.2f}K")
print(f"{'='*80}\n")

if max_err_h < 5.0 and max_err_c < 10.0:
    print("✓ Validation PASSED: Errors within acceptable range")
else:
    print("✗ Validation FAILED: Errors exceed threshold")
