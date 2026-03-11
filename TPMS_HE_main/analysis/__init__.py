"""
Analysis package — correlation-level sensitivity studies and packed-bed comparisons.

Usage as scripts (run from TPMS_HE_main/):
    python -m analysis.sensitivity
    python -m analysis.comparison
"""
from analysis.sensitivity import main as run_sensitivity_analysis
from analysis.comparison import main as run_packed_vs_bare

__all__ = ["run_sensitivity_analysis", "run_packed_vs_bare"]
