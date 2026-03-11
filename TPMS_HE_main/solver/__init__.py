"""
Solver package — iterative thermo-hydraulic solver for TPMS and plate-fin heat exchangers.
"""
from solver.config import (
    normalize_config,
    create_default_config,
    create_cpfhx_config,
    make_run_dir,
    _default_output_paths,
)
from solver.calculator import TPMSHeatExchanger

__all__ = [
    "TPMSHeatExchanger",
    "normalize_config",
    "create_default_config",
    "create_cpfhx_config",
    "make_run_dir",
    "_default_output_paths",
]
