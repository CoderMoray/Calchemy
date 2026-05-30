"""
Calchemy — A declarative DSL for DataFrame column calculations.

Every DataFrame has gold in it. Calchemy helps you extract it.
"""

from calchemy.calchemy import calc_div, calc_add, calc_sub, calc_mul, calc
from calchemy.calchemy import _BINOP_MAP, _UNARYOP_MAP

__all__ = [
    "calc_div",
    "calc_add",
    "calc_sub",
    "calc_mul",
    "calc",
    "_BINOP_MAP",
    "_UNARYOP_MAP",
]
