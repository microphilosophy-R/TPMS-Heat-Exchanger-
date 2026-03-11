"""
Plate-fin fin efficiency model (Wang et al. 2024, Eqs. 10-12).

Extracted from tpms_thermo_hydraulic_calculator.py to break the
circular dependency between models/ and solver/.
"""

import numpy as np


def plate_fin_fin_efficiency(h, Hf, tf, k_wall, Af_Ah_ratio):
    """
    Weighted overall fin efficiency for perforated plate fins (Wang et al. 2024, Eqs. 10-12).

        eta_f = tanh(m*Hf) / (m*Hf)     where m = sqrt(2h / (k_wall * tf))
        eta_h = 1 - (Af/Ah) * (1 - eta_f)

    Parameters
    ----------
    h : float           Local convective HTC [W/m2*K]
    Hf : float          Fin height [m]
    tf : float          Fin thickness [m]
    k_wall : float      Fin material thermal conductivity [W/m*K]
    Af_Ah_ratio : float Secondary (fin) area / total area ratio [-]

    Returns
    -------
    eta_h : float       Overall weighted fin efficiency [-]
    """
    if h <= 0 or k_wall <= 0 or tf <= 0:
        return 1.0
    m  = np.sqrt(2.0 * h / (k_wall * tf))
    mL = m * Hf
    if mL < 0.01:
        eta_f = 1.0
    elif mL > 20.0:
        eta_f = 1.0 / mL
    else:
        eta_f = np.tanh(mL) / mL
    return 1.0 - Af_Ah_ratio * (1.0 - eta_f)


if __name__ == "__main__":
    eta = plate_fin_fin_efficiency(h=500, Hf=9.5e-3, tf=0.6e-3, k_wall=237.0, Af_Ah_ratio=0.89)
    print(f"plate_fin_fin_efficiency smoke test: eta_h = {eta:.4f}")
    assert 0 < eta <= 1.0
    print("OK")
