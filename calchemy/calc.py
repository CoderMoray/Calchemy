"""
Calchemy — A declarative DSL for DataFrame column calculations.

Every DataFrame has gold in it. Calchemy helps you extract it.
"""

import ast

import pandas as pd

from calchemy.types import CalcResult, CalcStep
from calchemy.utils import _parse_format_suffix, _validate_errors, _apply_format
from calchemy.parse import _decompose_ast


# ──────────────────────────────────────────────
# 混合运算引擎
# ──────────────────────────────────────────────

def _calc_decompose(
    df: pd.DataFrame,
    expr: str,
    rounding: int = 2,
    format: str | None = None,
    errors: str = 'coerce',
    keep_tmp: bool = False,
) -> CalcResult:
    """拆解混合运算表达式，返回 CalcResult 对象（含步骤记录）。

    与 ``calc()`` 相同的计算逻辑，但额外记录每步拆解过程，
    并可选择保留中间临时列。

    Parameters
    ----------
    df : pd.DataFrame
        待处理的 DataFrame（原地修改）。
    expr : str
        计算表达式，格式与 ``calc()`` 一致。
    rounding : int, default 2
        保留的小数位数。
    format : str, optional
        输出格式（'percent'/'pct'/'%'/'percentage' 等）。
    errors : {'coerce', 'raise', 'ignore'}, default 'coerce'
        异常数据处理方式。
    keep_tmp : bool, default False
        是否保留 ``__calc_tmp_*`` 临时列。

    Returns
    -------
    CalcResult
        包含结果 DataFrame、拆解步骤和临时列信息的对象。
    """
    # 1. 解析格式后缀 & 校验 errors
    expr, format = _parse_format_suffix(expr, format)
    _validate_errors(errors)

    # 2. 分离新列名和计算表达式
    if expr.count('=') != 1:
        raise ValueError(
            f"格式错误，期望 'new = expression'，收到：'{expr}'，原因：'=' 数量不为 1。"
        )

    new_col = expr.split('=')[0].strip()
    calc_expr = expr.split('=')[1].strip()

    if not new_col:
        raise ValueError(
            f"格式错误，新列名为空，期望 'new = expression'，收到：'{expr}'。"
        )
    if not calc_expr:
        raise ValueError(
            f"格式错误，计算表达式为空，期望 'new = expression'，收到：'{expr}'。"
        )

    # 3. 用 ast.parse 解析表达式（受限子集）
    try:
        tree = ast.parse(calc_expr, mode='eval')
    except SyntaxError as e:
        raise ValueError(
            f"表达式语法错误：'{calc_expr}'，原因：{e.msg}。"
            f"仅支持列名、数字常量和 +、-、*、/ 四则运算。"
        )

    root = tree.body

    # 4. 递归拆解 AST
    collected_cols: list[str] = []
    tmp_counter: list[int] = [0]
    steps: list[CalcStep] = []
    result_ref = _decompose_ast(root, df, errors, collected_cols, tmp_counter, steps)

    # 5. 写入 DataFrame
    is_multi_index = isinstance(df.index, pd.MultiIndex)

    if isinstance(result_ref, str):
        # result_ref 是列名字符串（临时列或原始列）
        if is_multi_index:
            df[new_col] = df[result_ref].values
        else:
            df.loc[:, new_col] = df[result_ref].values
        # 不在这里删除临时结果列，统一在步骤 7 处理（keep_tmp 控制）
    else:
        # 标量结果
        if is_multi_index:
            df[new_col] = result_ref
        else:
            df.loc[:, new_col] = result_ref

    # 6. 格式化输出
    _apply_format(df, new_col, format, rounding, is_multi_index)

    # 7. 收集并清理临时列
    tmp_columns = [c for c in df.columns if c.startswith("__calc_tmp_")]
    if not keep_tmp:
        for col in tmp_columns:
            del df[col]
        tmp_columns = []

    return CalcResult(df=df, steps=steps, tmp_columns=tmp_columns)


def calc(
    df: pd.DataFrame,
    expr: str,
    rounding: int = 2,
    format: str | None = None,
    errors: str = 'coerce',
) -> pd.DataFrame:
    """
    在 DataFrame 中新增一列，其值为混合四则运算表达式的结果，支持括号和多个操作数。

    Parameters
    ----------
    df : pd.DataFrame
        待处理的 DataFrame。
    expr : str
        计算表达式，支持三种写法：
        1) 简单运算：
           "new_col = col_a + col_b"
        2) 混合运算（含括号）：
           "gm_rate = (revenue - cogs) / revenue"
        3) 带格式后缀：
           "gm_rate = (revenue - cogs) / revenue >>> %"
        空格可随意添加。运算符支持 +、-、*、/，支持括号改变优先级。
        操作数可以是列名或数字常量。
    rounding : int, default 2
        保留的小数位数（百分比模式下同样适用）。
    format : str, optional
        显式指定返回格式，可选值：
        None / 'float'          → 普通浮点，先 round 再返回
        'percent'/'pct'/'%'/'percentage'/... → 转换为百分比字符串并保留指定小数位
        若 expr 中已用 >>> 指定格式，则以 expr 为准。
    errors : {'coerce', 'raise', 'ignore'}, default 'coerce'
        异常数据处理方式：
        - 'coerce'：问题数据 → NaN 或 0（见 Notes）；NaN 输入 → NaN
        - 'raise'：遇到零分母或 NaN 输入行立即抛出 ValueError，消息包含行索引
        - 'ignore'：跳过问题行，保留该行新列的原始值（NaN）

    Returns
    -------
    pd.DataFrame
        在原表基础上新增一列后的同一个 DataFrame 实例。

    Notes
    -----
    - 除法语义与 calc_div 一致：
      分母=0 且分子=0 → NaN；分母=0 且分子≠0 → 强制 0。
    - 加减乘的 NaN 处理：任一操作数为 NaN → 结果 NaN。
    - 安全性：使用 ast.parse 的受限子集解析表达式，**禁止 eval()**，
      不允许函数调用、属性访问、下标等操作。

    Examples
    --------
    >>> df = pd.DataFrame({"revenue": [100, 200], "cogs": [40, 80]})
    >>> calc(df, "gm = revenue - cogs")
    >>> df["gm"]
    0    60.0
    1    120.0
    Name: gm, dtype: float64

    >>> calc(df, "gm_rate = (revenue - cogs) / revenue >>> %")
    >>> df["gm_rate"]
    0    60.00%
    1    60.00%
    Name: gm_rate, dtype: object

    >>> calc(df, "tax = revenue * 0.13")
    >>> df["tax"]
    0    13.0
    1    26.0
    Name: tax, dtype: float64
    """
    result = _calc_decompose(df, expr, rounding=rounding, format=format, errors=errors)
    return result.df
