"""
Calchemy — A declarative DSL for DataFrame column calculations.

Every DataFrame has gold in it. Calchemy helps you extract it.
"""

import ast
from dataclasses import dataclass, field
from typing import Union

import pandas as pd
import numpy as np


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


# ──────────────────────────────────────────────
# 内部工具函数（不对外暴露）
# ──────────────────────────────────────────────

def _parse_format_suffix(expr: str, format: str | None) -> tuple[str, str | None]:
    """解析 >>> 格式后缀，返回 (清理后的表达式, 格式标识)。

    若 expr 中包含 >>>，则以 >>> 后的为准，忽略 format 参数。
    """
    if '>>>' in expr:
        format = expr.split('>>>')[1].strip()
        expr = expr.split('>>>')[0].strip()
    return expr, format


def _validate_errors(errors: str) -> None:
    """校验 errors 参数合法性。"""
    if errors not in ('coerce', 'raise', 'ignore'):
        raise ValueError(f"errors 参数须为 'coerce'/'raise'/'ignore'，收到：'{errors}'")


def _validate_columns(df: pd.DataFrame, *cols: str) -> None:
    """校验列名是否存在于 DataFrame 中，不存在则抛出友好 KeyError。"""
    for col in cols:
        if col not in df.columns:
            raise KeyError(f"列 '{col}' 不存在于 DataFrame 中，可用列：{list(df.columns)}")


def _apply_format(df: pd.DataFrame, new_col: str, format: str | None,
                  rounding: int, is_multi_index: bool) -> None:
    """根据 format 参数对目标列做格式化（原地修改 df）。"""
    if format is None:
        if is_multi_index:
            df[new_col] = df[new_col].astype(float).round(rounding)
        else:
            df.loc[:, new_col] = df[new_col].astype(float).round(rounding)

    elif format.lower() in ['percent', 'pct', '%', 'percentage', '百分比']:
        mask_not_na = df[new_col].notna()
        rounded = df.loc[mask_not_na, new_col].astype(float).round(rounding)
        formatted = rounded.fillna(0.0).map(("{:." + str(rounding) + "%}").format)

        df[new_col] = df[new_col].astype(object)

        if is_multi_index:
            result = df[new_col].copy()
            result[mask_not_na] = formatted
            df[new_col] = result
        else:
            df.loc[mask_not_na, new_col] = formatted


# ──────────────────────────────────────────────
# 除法
# ──────────────────────────────────────────────

def calc_div(
    df: pd.DataFrame,
    expr: str,
    rounding: int = 2,
    format: str | None = None,
    errors: str = 'coerce',
) -> pd.DataFrame:
    """
    在 DataFrame 中新增一列，其值为两列的除法结果，支持零值保护及多种输出格式。

    Parameters
    ----------
    df : pd.DataFrame
        待处理的 DataFrame。
    expr : str
        计算表达式，支持两种写法：
        1) 标准格式：
           "new_col = col_a / col_b"
        2) 带格式后缀：
           "new_col = col_a / col_b >>> percentage"
        空格可随意添加，会自动压缩。
    rounding : int, default 2
        保留的小数位数（百分比模式下同样适用）。
    format : str, optional
        显式指定返回格式，可选值：
        None / 'float'          → 普通浮点，先 round 再返回
        'percent'/'pct'/'%'/'percentage'/... → 转换为百分比字符串并保留指定小数位
        若 expr 中已用 >>> 指定格式，则以 expr 为准。
    errors : {'coerce', 'raise', 'ignore'}, default 'coerce'
        异常数据处理方式：
        - 'coerce'：零分母 → NaN（分子也为 0 时）或 0（分子非 0 时）；NaN 输入 → NaN
        - 'raise'：遇到零分母或 NaN 输入行立即抛出 ValueError，消息包含行索引
        - 'ignore'：跳过问题行，保留该行新列的原始值（NaN）

    Returns
    -------
    pd.DataFrame
        在原表基础上新增一列后的同一个 DataFrame 实例。

    Notes
    -----
    - 分母为 0 且分子也为 0 时，结果设为 NaN（0/0 无意义）。
    - 分子非 0 且分母为 0 的脏数据会被强制设为 0。
    - 任一操作数为 NaN 时，结果为 NaN。
    - 百分比格式返回的是**字符串**（如 "65.43%"），方便直接展示。

    Examples
    --------
    >>> df = pd.DataFrame({"revenue": [100, 200], "cost": [50, 40]})
    >>> calc_div(df, "ratio = revenue / cost")
    >>> df["ratio"]
    0    2.0
    1    5.0
    Name: ratio, dtype: float64

    >>> calc_div(df, "pct = revenue / cost >>> %")
    """
    # 1. 解析格式后缀 & 校验 errors
    expr, format = _parse_format_suffix(expr, format)
    _validate_errors(errors)

    # 2. 基础校验
    if expr.count('=') != 1:
        raise ValueError(
            f"格式错误，期望 'new = col1 / col2'，收到：'{expr}'，原因：'=' 数量不为 1。"
        )
    if expr.split('=')[1].count('/') != 1:
        raise ValueError(
            f"格式错误，期望 'new = col1 / col2'，收到：'{expr}'，原因：右侧存在多个 '/'。"
        )

    # 3. 提取列名
    expr = expr.strip().replace(' = ', '=').replace(' / ', '/')
    新列名 = expr.split('=')[0].strip()
    计算部分 = expr.split('=')[1]
    分子列名 = 计算部分.split('/')[0].strip()
    分母列名 = 计算部分.split('/')[1].strip()

    _validate_columns(df, 分子列名, 分母列名)

    # 4. raise 模式：先检查是否有问题行
    is_multi_index = isinstance(df.index, pd.MultiIndex)

    mask_nan_input = df[分子列名].isna() | df[分母列名].isna()
    mask_zero_denom = df[分母列名] == 0

    if errors == 'raise':
        problem_rows = df.index[mask_nan_input | mask_zero_denom]
        if len(problem_rows) > 0:
            raise ValueError(
                f"列 '{分子列名}' 或 '{分母列名}' 在以下行存在 NaN 或零分母数据"
                f"（索引）：{problem_rows.tolist()[:10]}..."
            )

    # 5. 核心计算（向量化操作）
    if is_multi_index:
        df[新列名] = np.nan
    else:
        df.loc[:, 新列名] = np.nan

    # raise 模式已通过校验，所有行都是有效的
    # ignore 模式下只处理无问题的行
    if errors == 'ignore':
        mask_valid = (~mask_nan_input) & (~mask_zero_denom)
    else:
        # coerce 或 raise（raise 已确保无问题行）
        mask_valid = ~mask_zero_denom

    mask_dirty = (df[分子列名] != 0) & mask_zero_denom

    if errors != 'ignore':
        # 计算有效行
        df.loc[mask_valid, 新列名] = (
            df.loc[mask_valid, 分子列名] / df.loc[mask_valid, 分母列名]
        )
        # 脏数据（分子≠0 且分母=0）→ 强制 0
        df.loc[mask_dirty & ~mask_nan_input, 新列名] = 0
    else:
        # ignore 模式：只计算合法行
        df.loc[mask_valid, 新列名] = (
            df.loc[mask_valid, 分子列名] / df.loc[mask_valid, 分母列名]
        )

    # 6. 格式化输出
    _apply_format(df, 新列名, format, rounding, is_multi_index)

    return df


# ──────────────────────────────────────────────
# 加法
# ──────────────────────────────────────────────

def calc_add(
    df: pd.DataFrame,
    expr: str,
    rounding: int = 2,
    format: str | None = None,
    errors: str = 'coerce',
) -> pd.DataFrame:
    """
    在 DataFrame 中新增一列，其值为两列的加法结果。

    Parameters
    ----------
    df : pd.DataFrame
        待处理的 DataFrame。
    expr : str
        计算表达式，支持两种写法：
        1) 标准格式：
           "new_col = col_a + col_b"
        2) 带格式后缀：
           "new_col = col_a + col_b >>> percentage"
        空格可随意添加，会自动压缩。
    rounding : int, default 2
        保留的小数位数（百分比模式下同样适用）。
    format : str, optional
        显式指定返回格式，可选值：
        None / 'float'          → 普通浮点，先 round 再返回
        'percent'/'pct'/'%'/'percentage'/... → 转换为百分比字符串并保留指定小数位
        若 expr 中已用 >>> 指定格式，则以 expr 为准。
    errors : {'coerce', 'raise', 'ignore'}, default 'coerce'
        异常数据处理方式：
        - 'coerce'：任一操作数为 NaN → 结果 NaN
        - 'raise'：遇到 NaN 输入行立即抛出 ValueError，消息包含行索引
        - 'ignore'：跳过 NaN 行，保留该行新列的原始值（NaN）

    Returns
    -------
    pd.DataFrame
        在原表基础上新增一列后的同一个 DataFrame 实例。

    Notes
    -----
    - 任一操作数为 NaN 时，结果为 NaN（coerce 默认行为）。
    - 加法无零值保护需求，0 是合法操作数。

    Examples
    --------
    >>> df = pd.DataFrame({"a": [10, 20], "b": [3, 4]})
    >>> calc_add(df, "c = a + b")
    >>> df["c"]
    0    13.0
    1    24.0
    Name: c, dtype: float64
    """
    # 1. 解析格式后缀 & 校验 errors
    expr, format = _parse_format_suffix(expr, format)
    _validate_errors(errors)

    # 2. 基础校验
    if expr.count('=') != 1:
        raise ValueError(
            f"格式错误，期望 'new = col1 + col2'，收到：'{expr}'，原因：'=' 数量不为 1。"
        )
    if expr.split('=')[1].count('+') != 1:
        raise ValueError(
            f"格式错误，期望 'new = col1 + col2'，收到：'{expr}'，原因：右侧存在多个 '+'。"
        )

    # 3. 提取列名
    expr = expr.strip().replace(' = ', '=').replace(' + ', '+')
    新列名 = expr.split('=')[0].strip()
    计算部分 = expr.split('=')[1]
    列名1 = 计算部分.split('+')[0].strip()
    列名2 = 计算部分.split('+')[1].strip()

    _validate_columns(df, 列名1, 列名2)

    # 4. 检测问题行
    is_multi_index = isinstance(df.index, pd.MultiIndex)
    mask_nan_input = df[列名1].isna() | df[列名2].isna()

    if errors == 'raise':
        problem_rows = df.index[mask_nan_input]
        if len(problem_rows) > 0:
            raise ValueError(
                f"列 '{列名1}' 或 '{列名2}' 在以下行存在 NaN 数据"
                f"（索引）：{problem_rows.tolist()[:10]}..."
            )

    # 5. 核心计算（向量化操作）
    mask_valid = ~mask_nan_input

    if is_multi_index:
        df[新列名] = np.nan
    else:
        df.loc[:, 新列名] = np.nan

    df.loc[mask_valid, 新列名] = df.loc[mask_valid, 列名1] + df.loc[mask_valid, 列名2]

    # 6. 格式化输出
    _apply_format(df, 新列名, format, rounding, is_multi_index)

    return df


# ──────────────────────────────────────────────
# 减法
# ──────────────────────────────────────────────

def calc_sub(
    df: pd.DataFrame,
    expr: str,
    rounding: int = 2,
    format: str | None = None,
    errors: str = 'coerce',
) -> pd.DataFrame:
    """
    在 DataFrame 中新增一列，其值为两列的减法结果。

    Parameters
    ----------
    df : pd.DataFrame
        待处理的 DataFrame。
    expr : str
        计算表达式，支持两种写法：
        1) 标准格式：
           "new_col = col_a - col_b"
        2) 带格式后缀：
           "new_col = col_a - col_b >>> percentage"
        空格可随意添加，会自动压缩。
    rounding : int, default 2
        保留的小数位数（百分比模式下同样适用）。
    format : str, optional
        显式指定返回格式，可选值：
        None / 'float'          → 普通浮点，先 round 再返回
        'percent'/'pct'/'%'/'percentage'/... → 转换为百分比字符串并保留指定小数位
        若 expr 中已用 >>> 指定格式，则以 expr 为准。
    errors : {'coerce', 'raise', 'ignore'}, default 'coerce'
        异常数据处理方式：
        - 'coerce'：任一操作数为 NaN → 结果 NaN
        - 'raise'：遇到 NaN 输入行立即抛出 ValueError，消息包含行索引
        - 'ignore'：跳过 NaN 行，保留该行新列的原始值（NaN）

    Returns
    -------
    pd.DataFrame
        在原表基础上新增一列后的同一个 DataFrame 实例。

    Notes
    -----
    - 任一操作数为 NaN 时，结果为 NaN（coerce 默认行为）。
    - 减法无零值保护需求，0 是合法操作数。
    - 减法允许负数结果。

    Examples
    --------
    >>> df = pd.DataFrame({"revenue": [100, 200], "cost": [50, 40]})
    >>> calc_sub(df, "margin = revenue - cost")
    >>> df["margin"]
    0    50.0
    1    160.0
    Name: margin, dtype: float64
    """
    # 1. 解析格式后缀 & 校验 errors
    expr, format = _parse_format_suffix(expr, format)
    _validate_errors(errors)

    # 2. 基础校验
    if expr.count('=') != 1:
        raise ValueError(
            f"格式错误，期望 'new = col1 - col2'，收到：'{expr}'，原因：'=' 数量不为 1。"
        )
    if expr.split('=')[1].count('-') != 1:
        raise ValueError(
            f"格式错误，期望 'new = col1 - col2'，收到：'{expr}'，原因：右侧存在多个 '-'。"
        )

    # 3. 提取列名
    expr = expr.strip().replace(' = ', '=').replace(' - ', '-')
    新列名 = expr.split('=')[0].strip()
    计算部分 = expr.split('=')[1]
    列名1 = 计算部分.split('-')[0].strip()
    列名2 = 计算部分.split('-')[1].strip()

    _validate_columns(df, 列名1, 列名2)

    # 4. 检测问题行
    is_multi_index = isinstance(df.index, pd.MultiIndex)
    mask_nan_input = df[列名1].isna() | df[列名2].isna()

    if errors == 'raise':
        problem_rows = df.index[mask_nan_input]
        if len(problem_rows) > 0:
            raise ValueError(
                f"列 '{列名1}' 或 '{列名2}' 在以下行存在 NaN 数据"
                f"（索引）：{problem_rows.tolist()[:10]}..."
            )

    # 5. 核心计算（向量化操作）
    mask_valid = ~mask_nan_input

    if is_multi_index:
        df[新列名] = np.nan
    else:
        df.loc[:, 新列名] = np.nan

    df.loc[mask_valid, 新列名] = df.loc[mask_valid, 列名1] - df.loc[mask_valid, 列名2]

    # 6. 格式化输出
    _apply_format(df, 新列名, format, rounding, is_multi_index)

    return df


# ──────────────────────────────────────────────
# 乘法
# ──────────────────────────────────────────────

def calc_mul(
    df: pd.DataFrame,
    expr: str,
    rounding: int = 2,
    format: str | None = None,
    errors: str = 'coerce',
) -> pd.DataFrame:
    """
    在 DataFrame 中新增一列，其值为两列的乘法结果。

    Parameters
    ----------
    df : pd.DataFrame
        待处理的 DataFrame。
    expr : str
        计算表达式，支持两种写法：
        1) 标准格式：
           "new_col = col_a * col_b"
        2) 带格式后缀：
           "new_col = col_a * col_b >>> percentage"
        空格可随意添加，会自动压缩。
    rounding : int, default 2
        保留的小数位数（百分比模式下同样适用）。
    format : str, optional
        显式指定返回格式，可选值：
        None / 'float'          → 普通浮点，先 round 再返回
        'percent'/'pct'/'%'/'percentage'/... → 转换为百分比字符串并保留指定小数位
        若 expr 中已用 >>> 指定格式，则以 expr 为准。
    errors : {'coerce', 'raise', 'ignore'}, default 'coerce'
        异常数据处理方式：
        - 'coerce'：任一操作数为 NaN → 结果 NaN（包括 0 × NaN = NaN）
        - 'raise'：遇到 NaN 输入行立即抛出 ValueError，消息包含行索引
        - 'ignore'：跳过 NaN 行，保留该行新列的原始值（NaN）

    Returns
    -------
    pd.DataFrame
        在原表基础上新增一列后的同一个 DataFrame 实例。

    Notes
    -----
    - 任一操作数为 NaN 时，结果为 NaN，包括 0 × NaN 的情况。
      即 0 × NaN = NaN，不做隐式填 0。
    - 乘法无零值保护需求，0 是合法操作数（0 × 任何有限值 = 0）。

    Examples
    --------
    >>> df = pd.DataFrame({"qty": [10, 20], "price": [3.5, 4.0]})
    >>> calc_mul(df, "total = qty * price")
    >>> df["total"]
    0    35.0
    1    80.0
    Name: total, dtype: float64
    """
    # 1. 解析格式后缀 & 校验 errors
    expr, format = _parse_format_suffix(expr, format)
    _validate_errors(errors)

    # 2. 基础校验
    if expr.count('=') != 1:
        raise ValueError(
            f"格式错误，期望 'new = col1 * col2'，收到：'{expr}'，原因：'=' 数量不为 1。"
        )
    if expr.split('=')[1].count('*') != 1:
        raise ValueError(
            f"格式错误，期望 'new = col1 * col2'，收到：'{expr}'，原因：右侧存在多个 '*'。"
        )

    # 3. 提取列名
    expr = expr.strip().replace(' = ', '=').replace(' * ', '*')
    新列名 = expr.split('=')[0].strip()
    计算部分 = expr.split('=')[1]
    列名1 = 计算部分.split('*')[0].strip()
    列名2 = 计算部分.split('*')[1].strip()

    _validate_columns(df, 列名1, 列名2)

    # 4. 检测问题行
    is_multi_index = isinstance(df.index, pd.MultiIndex)
    mask_nan_input = df[列名1].isna() | df[列名2].isna()

    if errors == 'raise':
        problem_rows = df.index[mask_nan_input]
        if len(problem_rows) > 0:
            raise ValueError(
                f"列 '{列名1}' 或 '{列名2}' 在以下行存在 NaN 数据"
                f"（索引）：{problem_rows.tolist()[:10]}..."
            )

    # 5. 核心计算（向量化操作）
    mask_valid = ~mask_nan_input

    if is_multi_index:
        df[新列名] = np.nan
    else:
        df.loc[:, 新列名] = np.nan

    df.loc[mask_valid, 新列名] = df.loc[mask_valid, 列名1] * df.loc[mask_valid, 列名2]

    # 6. 格式化输出
    _apply_format(df, 新列名, format, rounding, is_multi_index)

    return df


# ──────────────────────────────────────────────
# 内部 AST 求值器（calc() 使用）
# ──────────────────────────────────────────────

# 允许的二元运算符映射
_BINOP_MAP: dict[type, str] = {
    ast.Add: '+',
    ast.Sub: '-',
    ast.Mult: '*',
    ast.Div: '/',
}

# 允许的一元运算符映射
_UNARYOP_MAP: dict[type, str] = {
    ast.UAdd: '+',
    ast.USub: '-',
}


def _eval_ast_node(
    node: ast.AST,
    df: pd.DataFrame,
    errors: str,
    collected_cols: list[str],
) -> Union[pd.Series, float, int]:
    """递归求值 AST 节点，返回 Series 或标量。

    Parameters
    ----------
    node : ast.AST
        要求值的 AST 节点。
    df : pd.DataFrame
        数据源。
    errors : str
        异常处理模式。
    collected_cols : list[str]
        收集到的所有列名（用于 raise 模式下的错误消息）。

    Returns
    -------
    pd.Series | float | int
        求值结果。

    Raises
    ------
    ValueError
        遇到不允许的 AST 节点类型。
    """
    # 常量：数字字面量
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(
            f"不支持的字面量类型：{type(node.value).__name__}（值：{node.value!r}），"
            f"仅支持整数和浮点数。"
        )

    # 名称：列名引用
    if isinstance(node, ast.Name):
        col_name = node.id
        collected_cols.append(col_name)
        if col_name not in df.columns:
            raise KeyError(
                f"列 '{col_name}' 不存在于 DataFrame 中，可用列：{list(df.columns)}"
            )
        return df[col_name]

    # 二元运算
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BINOP_MAP:
            raise ValueError(
                f"不支持的运算符：{type(node.op).__name__}，"
                f"仅支持 +、-、*、/ 四则运算。"
            )

        left = _eval_ast_node(node.left, df, errors, collected_cols)
        right = _eval_ast_node(node.right, df, errors, collected_cols)

        op_symbol = _BINOP_MAP[op_type]

        # 非 '/' 运算：raise 模式检查 NaN 输入
        if errors == 'raise' and op_symbol != '/':
            mask_nan = pd.Series(False, index=df.index)
            if isinstance(left, pd.Series):
                mask_nan = mask_nan | left.isna()
            if isinstance(right, pd.Series):
                mask_nan = mask_nan | right.isna()
            if mask_nan.any():
                problem_rows = df.index[mask_nan]
                cols_str = '、'.join(f"'{c}'" for c in collected_cols)
                raise ValueError(
                    f"表达式中的列 {cols_str} 在以下行存在 NaN 数据"
                    f"（索引）：{problem_rows.tolist()[:10]}..."
                )

        if op_symbol == '+':
            result = left + right
        elif op_symbol == '-':
            result = left - right
        elif op_symbol == '*':
            result = left * right
        elif op_symbol == '/':
            # 除法需要零值保护，与 calc_div 语义一致
            left_is_series = isinstance(left, pd.Series)
            right_is_series = isinstance(right, pd.Series)

            if right_is_series:
                mask_zero_denom = (right == 0)
                if left_is_series:
                    mask_nan_input = left.isna() | right.isna()
                else:
                    mask_nan_input = right.isna() if left != left else pd.Series(True, index=df.index)

                # raise 模式
                if errors == 'raise':
                    problem_mask = mask_zero_denom | mask_nan_input
                    if problem_mask.any():
                        problem_rows = df.index[problem_mask]
                        cols_str = '、'.join(f"'{c}'" for c in collected_cols)
                        raise ValueError(
                            f"表达式中的列 {cols_str} 在以下行存在 NaN 或零分母数据"
                            f"（索引）：{problem_rows.tolist()[:10]}..."
                        )

                # 核心除法（向量化）
                result = np.nan * np.ones(len(df)) if left_is_series else np.nan
                if left_is_series:
                    result = left.copy().astype(float)
                    result[:] = np.nan

                mask_valid = ~mask_zero_denom
                if errors == 'ignore':
                    mask_valid = mask_valid & (~mask_nan_input)

                if left_is_series:
                    result = pd.Series(np.nan, index=df.index, dtype=float)
                    result[mask_valid] = left[mask_valid] / right[mask_valid]
                else:
                    # left 是标量
                    result = pd.Series(np.nan, index=df.index, dtype=float)
                    result[mask_valid] = left / right[mask_valid]

                # 脏数据：分子≠0 且分母=0 → 强制 0
                if left_is_series:
                    mask_dirty = (left != 0) & mask_zero_denom & (~mask_nan_input)
                else:
                    mask_dirty = mask_zero_denom & (~mask_nan_input) if left != 0 else pd.Series(False, index=df.index)

                if errors != 'ignore':
                    result[mask_dirty] = 0

            else:
                # right 是标量
                if right == 0:
                    if left_is_series:
                        mask_nan_input = left.isna()
                        if errors == 'raise' and mask_nan_input.any():
                            problem_rows = df.index[mask_nan_input]
                            raise ValueError(
                                f"表达式中的列在以下行存在 NaN 数据"
                                f"（索引）：{problem_rows.tolist()[:10]}..."
                            )
                        result = pd.Series(np.nan, index=df.index, dtype=float)
                        if left_is_series:
                            if errors == 'ignore':
                                mask_ok = ~mask_nan_input
                                result[mask_ok] = 0
                            else:
                                result[:] = 0
                            result[mask_nan_input] = np.nan
                    else:
                        result = np.nan if left == 0 else 0
                else:
                    result = left / right

            return result

        return result

    # 一元运算（正号 / 负号）
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARYOP_MAP:
            raise ValueError(
                f"不支持的一元运算符：{type(node.op).__name__}，"
                f"仅支持正号(+)和负号(-)。"
            )

        operand = _eval_ast_node(node.operand, df, errors, collected_cols)
        op_symbol = _UNARYOP_MAP[op_type]

        if op_symbol == '+':
            return operand
        elif op_symbol == '-':
            return -operand

    # 不允许的节点类型
    raise ValueError(
        f"不支持的表达式元素：{type(node).__name__}。"
        f"仅支持列名、数字常量和 +、-、*、/ 四则运算。"
    )


def _decompose_ast(
    node: ast.AST,
    df: pd.DataFrame,
    errors: str,
    collected_cols: list[str],
    tmp_counter: list[int],
    steps: list[CalcStep],
) -> Union[str, float, int]:
    """递归拆解 AST 节点，生成逐步计算记录。

    与 ``_eval_ast_node`` 不同，本函数返回**列名字符串**或**标量值**，
    而非 Series。每个二元运算生成一个临时列 ``__calc_tmp_N``，
    并通过 helper 函数完成实际计算。

    Parameters
    ----------
    node : ast.AST
        要拆解的 AST 节点。
    df : pd.DataFrame
        数据源（原地修改）。
    errors : str
        异常处理模式（'coerce' / 'raise' / 'ignore'）。
    collected_cols : list[str]
        收集到的所有列名（用于错误消息）。
    tmp_counter : list[int]
        临时列计数器，长度为 1，通过 ``tmp_counter[0]`` 递增。
    steps : list[CalcStep]
        拆解步骤列表，每步追加一条记录。

    Returns
    -------
    str | float | int
        列名字符串（表示临时列或原始列）或标量值。

    Raises
    ------
    ValueError
        遇到不允许的 AST 节点类型或运算符。
    KeyError
        引用的列名不存在。
    """
    # 常量：数字字面量 → 返回标量值
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(
            f"不支持的字面量类型：{type(node.value).__name__}（值：{node.value!r}），"
            f"仅支持整数和浮点数。"
        )

    # 名称：列名引用 → 返回列名字符串（不返回 Series）
    if isinstance(node, ast.Name):
        col_name = node.id
        collected_cols.append(col_name)
        if col_name not in df.columns:
            raise KeyError(
                f"列 '{col_name}' 不存在于 DataFrame 中，可用列：{list(df.columns)}"
            )
        return col_name

    # 二元运算
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BINOP_MAP:
            raise ValueError(
                f"不支持的运算符：{type(node.op).__name__}，"
                f"仅支持 +、-、*、/ 四则运算。"
            )
        op_symbol = _BINOP_MAP[op_type]

        left_ref = _decompose_ast(
            node.left, df, errors, collected_cols, tmp_counter, steps
        )
        right_ref = _decompose_ast(
            node.right, df, errors, collected_cols, tmp_counter, steps
        )

        # 两个操作数都是标量 → 直接用 Python 算，不走 helper
        if not isinstance(left_ref, str) and not isinstance(right_ref, str):
            if op_symbol == '+':
                return left_ref + right_ref
            elif op_symbol == '-':
                return left_ref - right_ref
            elif op_symbol == '*':
                return left_ref * right_ref
            elif op_symbol == '/':
                if right_ref == 0:
                    return float('nan') if left_ref == 0 else 0
                return left_ref / right_ref

        # 标量操作数 → 先添加为临时列
        if not isinstance(left_ref, str):
            tmp_counter[0] += 1
            scalar_col = f"__calc_tmp_{tmp_counter[0]}"
            df[scalar_col] = float(left_ref)
            left_ref = scalar_col

        if not isinstance(right_ref, str):
            tmp_counter[0] += 1
            scalar_col = f"__calc_tmp_{tmp_counter[0]}"
            df[scalar_col] = float(right_ref)
            right_ref = scalar_col

        # 现在 left_ref 和 right_ref 都是列名字符串
        tmp_counter[0] += 1
        tmp_col = f"__calc_tmp_{tmp_counter[0]}"
        expr_str = f"{tmp_col} = {left_ref} {op_symbol} {right_ref}"

        # 调用对应 helper（中间步骤不格式化，高精度 rounding 避免精度损失）
        if op_symbol == '+':
            calc_add(df, expr_str, rounding=15, format=None, errors=errors)
        elif op_symbol == '-':
            calc_sub(df, expr_str, rounding=15, format=None, errors=errors)
        elif op_symbol == '*':
            calc_mul(df, expr_str, rounding=15, format=None, errors=errors)
        elif op_symbol == '/':
            calc_div(df, expr_str, rounding=15, format=None, errors=errors)

        steps.append(
            CalcStep(expression=expr_str, result_col=tmp_col, operator=op_symbol)
        )
        return tmp_col

    # 一元运算
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARYOP_MAP:
            raise ValueError(
                f"不支持的一元运算符：{type(node.op).__name__}，"
                f"仅支持正号(+)和负号(-)。"
            )
        op_symbol = _UNARYOP_MAP[op_type]

        operand_ref = _decompose_ast(
            node.operand, df, errors, collected_cols, tmp_counter, steps
        )

        if op_symbol == '+':
            return operand_ref
        elif op_symbol == '-':
            if isinstance(operand_ref, str):
                # 列名 → 取反并生成临时列
                if errors == 'raise' and df[operand_ref].isna().any():
                    problem_rows = df.index[df[operand_ref].isna()]
                    raise ValueError(
                        f"列 '{operand_ref}' 在以下行存在 NaN 数据"
                        f"（索引）：{problem_rows.tolist()[:10]}..."
                    )
                tmp_counter[0] += 1
                tmp_col = f"__calc_tmp_{tmp_counter[0]}"
                df[tmp_col] = -df[operand_ref]
                steps.append(CalcStep(
                    expression=f"{tmp_col} = -{operand_ref}",
                    result_col=tmp_col,
                    operator="- (unary)",
                ))
                return tmp_col
            else:
                # 标量 → 直接取反
                return -operand_ref

    # 不允许的节点类型
    raise ValueError(
        f"不支持的表达式元素：{type(node).__name__}。"
        f"仅支持列名、数字常量和 +、-、*、/ 四则运算。"
    )


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
