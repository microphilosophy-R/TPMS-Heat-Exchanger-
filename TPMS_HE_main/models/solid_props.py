"""Minimal solid thermal conductivity lookup for packed bed catalysts."""
import os
import numpy as np
from scipy.interpolate import interp1d

_DATA_DIR = os.path.join(os.path.dirname(__file__), 'solid_data')

MATERIAL_MAP = {
    'AL5083': 'ALUMINUM_ALLOY_5083-T0.csv',
    'AL6061': 'ALUMINUM_ALLOY_6061-T6.csv',
    'AL': 'ALUMINUM_PURE.csv',
    'ALUMINUM': 'ALUMINUM_PURE.csv',
    'AU': 'GOLD_PURE.csv',
    'GOLD': 'GOLD_PURE.csv',
    'BRASS': 'COPPER_ZINC_90_10.csv',
    'CU': 'COPPER_PURE.csv',
    'COPPER': 'COPPER_PURE.csv',
    'EPOXY': 'EPOXY.csv',
    'INVAR': 'INVAR-36.csv',
    'KAPTON': 'KAPTON.csv',
    'PB': 'LEAD_PURE.csv',
    'LEAD': 'LEAD_PURE.csv',
    'SAPPHIRE': 'SAPPHIRE_PURE.csv',
    'SS304L': 'STAINLESS_STEEL_304L.csv',
    'SS310L': 'STAINLESS_STEEL_310L.csv',
    'TI': 'TITANIUM_PURE.csv',
    'TITANIUM': 'TITANIUM_PURE.csv',
}

def get_k_solid(material, T):
    """Get thermal conductivity for material at temperature T [K].

    Parameters
    ----------
    material : str
        Material name (see MATERIAL_MAP for supported materials)
    T : float
        Temperature [K]

    Returns
    -------
    k : float
        Thermal conductivity [W/m·K]
    """
    mat_upper = material.upper()
    if mat_upper not in MATERIAL_MAP:
        raise ValueError(f"Material '{material}' not supported. Available: {list(MATERIAL_MAP.keys())}")

    data_file = os.path.join(_DATA_DIR, MATERIAL_MAP[mat_upper])
    data = np.loadtxt(data_file, delimiter=',', skiprows=1)
    T_data = data[:, 0]
    k_data = data[:, 3]
    interp = interp1d(T_data, k_data, kind='linear', bounds_error=False, fill_value='extrapolate')
    return float(interp(T))

