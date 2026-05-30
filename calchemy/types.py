"""
Calchemy — A declarative DSL for DataFrame column calculations.

Every DataFrame has gold in it. Calchemy helps you extract it.
"""

import pandas as pd
from dataclasses import dataclass, field

import pandas as pd


# ──────────────────────────────────────────────
# 数据类
# ──────────────────────────────────────────────

@dataclass
class CalcStep:
    """拆解引擎中的单步计算记录。

    Attributes
    ----------
    expression : str
        本步的表达式，如 ``"revenue - cogs"``。
    result_col : str
        本步结果列名，如 ``"__calc_tmp_1"`` 或 ``"gm_rate"``。
    operator : str
        运算符，如 ``"+"``, ``"-"``, ``"*"``, ``"/"``, ``"- (unary)"``。
    """
    expression: str
    result_col: str
    operator: str


@dataclass
class CalcResult:
    """拆解引擎的完整返回结果。

    Attributes
    ----------
    df : pd.DataFrame
        结果 DataFrame（与输入为同一实例）。
    steps : list[CalcStep]
        拆解步骤列表，按执行顺序排列。
    tmp_columns : list[str]
        临时变量列名列表。
    """
    df: pd.DataFrame
    steps: list[CalcStep] = field(default_factory=list)
    tmp_columns: list[str] = field(default_factory=list)
