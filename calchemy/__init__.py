"""
Calchemy — A declarative DSL for DataFrame column calculations.

Every DataFrame has gold in it. Calchemy helps you extract it.
"""

__version__ = "0.1.0"

from calchemy.helpers import calc_div, calc_add, calc_sub, calc_mul, calc_pow, calc_abs, calc_log, calc_sqrt, calc_root
from calchemy.calc import calc
from calchemy.parse import _BINOP_MAP, _UNARYOP_MAP
from calchemy.types import CalcStep, CalcResult
from calchemy.chain import Calchemy

__all__ = [
    "calc_div",
    "calc_add",
    "calc_sub",
    "calc_mul",
    "calc",
    "calc_pow",
    "calc_abs",
    "calc_log",
    "calc_sqrt",
    "calc_root",
    "Calchemy",
    "CalcStep",
    "CalcResult",
    "_BINOP_MAP",
    "_UNARYOP_MAP",
]
