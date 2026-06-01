"""
Calchemy — A declarative DSL for DataFrame column calculations.

Every DataFrame has gold in it. Calchemy helps you extract it.
"""

import re

import pandas as pd
import numpy as np

from calchemy.utils import _parse_format_suffix, _validate_errors, _validate_columns, _apply_format


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
# 扩展运算符（Phase 4.5）
# ──────────────────────────────────────────────

def calc_pow(
    df: pd.DataFrame,
    expr: str,
    rounding: int = 2,
    format: str | None = None,
    errors: str = 'coerce',
) -> pd.DataFrame:
    """指数运算。

    支持 ``**``（Python 风格）和 ``^``（Excel 风格）两种写法，
    底层统一为 ``np.power`` 向量化计算。

    Parameters
    ----------
    df : pd.DataFrame
        数据源。
    expr : str
        表达式，如 ``"A = B ** 2"`` 或 ``"A = B ^ C"``。
    rounding : int, default 2
        保留小数位数。
    format : str, optional
        输出格式。
    errors : {'coerce', 'raise', 'ignore'}, default 'coerce'
        异常处理模式。

    Returns
    -------
    pd.DataFrame
        修改后的 DataFrame。
    """
    expr, format = _parse_format_suffix(expr, format)
    _validate_errors(errors)

    if expr.count('=') != 1:
        raise ValueError(f"格式错误，期望 'new = base ** exp'，收到：'{expr}'")

    新列名 = expr.split('=')[0].strip()
    计算部分 = expr.split('=')[1].strip()

    # 尝试匹配 ** 或 ^
    if '**' in 计算部分:
        操作数列表 = 计算部分.split('**')
        op = '**'
    elif '^' in 计算部分:
        操作数列表 = 计算部分.split('^')
        op = '^'
    else:
        raise ValueError(f"表达式 '{expr}' 中未找到指数运算符 ** 或 ^")

    if len(操作数列表) != 2:
        raise ValueError(f"格式错误，期望 'base {op} exp'，收到：'{计算部分}'")

    底数 = 操作数列表[0].strip()
    指数 = 操作数列表[1].strip()

    cols = [c for c in [底数, 指数] if c in df.columns]
    _validate_columns(df, *cols)

    # 解析操作数
    def _get_operand(val):
        if val in df.columns:
            return df[val]
        try:
            return float(val)
        except ValueError:
            raise ValueError(f"无法解析操作数 '{val}'")

    底数值 = _get_operand(底数)
    指数值 = _get_operand(指数)

    is_multi_index = isinstance(df.index, pd.MultiIndex)

    # raise 模式检查 NaN
    if errors == 'raise':
        for name, data in [(底数, 底数值), (指数, 指数值)]:
            if hasattr(data, 'isna') and data.isna().any():
                problem_rows = df.index[data.isna()]
                raise ValueError(
                    f"calc_pow: 列 '{name}' 在以下行存在 NaN 数据"
                    f"（索引）：{problem_rows.tolist()[:10]}..."
                )

    # 计算
    if hasattr(底数值, '__iter__') and hasattr(指数值, '__iter__'):
        result = np.power(底数值, 指数值)
        result[(底数值 == 0) & (指数值 == 0)] = np.nan
    elif hasattr(底数值, '__iter__'):
        result = np.power(底数值, 指数值)
        if 指数值 == 0:
            result[底数值 == 0] = np.nan
    elif hasattr(指数值, '__iter__'):
        result = np.power(底数值, 指数值)
        if 底数值 == 0:
            result[指数值 == 0] = np.nan
    else:
        if 底数值 == 0 and 指数值 == 0:
            result = np.nan
        else:
            result = 底数值 ** 指数值

    if is_multi_index:
        df[新列名] = result
    else:
        df.loc[:, 新列名] = result

    _apply_format(df, 新列名, format, rounding, is_multi_index)
    return df


def calc_abs(
    df: pd.DataFrame,
    expr: str,
    rounding: int = 2,
    format: str | None = None,
    errors: str = 'coerce',
) -> pd.DataFrame:
    """绝对值运算。

    expr 格式: ``"A = abs(B)"``
    """
    expr, format = _parse_format_suffix(expr, format)
    _validate_errors(errors)

    if expr.count('=') != 1:
        raise ValueError(f"格式错误，期望 'new = abs(col)'，收到：'{expr}'")

    新列名 = expr.split('=')[0].strip()
    计算部分 = expr.split('=')[1].strip()

    match = re.match(r'abs\s*\(\s*(\w+)\s*\)', 计算部分)
    if not match:
        raise ValueError(f"calc_abs: 表达式格式错误，期望 'A = abs(B)'，收到: '{expr}'")
    源列 = match.group(1)

    _validate_columns(df, 源列)

    is_multi_index = isinstance(df.index, pd.MultiIndex)

    if errors == 'raise' and df[源列].isna().any():
        problem_rows = df.index[df[源列].isna()]
        raise ValueError(
            f"calc_abs: 列 '{源列}' 在以下行存在 NaN 数据"
            f"（索引）：{problem_rows.tolist()[:10]}..."
        )

    result = df[源列].abs()

    if is_multi_index:
        df[新列名] = result
    else:
        df.loc[:, 新列名] = result

    _apply_format(df, 新列名, format, rounding, is_multi_index)
    return df


def calc_log(
    df: pd.DataFrame,
    expr: str,
    rounding: int = 2,
    format: str | None = None,
    errors: str = 'coerce',
) -> pd.DataFrame:
    """对数运算。

    expr 格式: ``"A = log(B)"``（自然对数）或 ``"A = log(B, 10)"``（指定底数）
    """
    expr, format = _parse_format_suffix(expr, format)
    _validate_errors(errors)

    if expr.count('=') != 1:
        raise ValueError(f"格式错误，期望 'new = log(col)' 或 'new = log(col, base)'，收到：'{expr}'")

    新列名 = expr.split('=')[0].strip()
    计算部分 = expr.split('=')[1].strip()

    match = re.match(r'log\s*\(\s*(\w+)\s*(?:,\s*(\S+)\s*)?\)', 计算部分)
    if not match:
        raise ValueError(f"calc_log: 表达式格式错误，期望 'A = log(B)' 或 'A = log(B, base)'，收到: '{expr}'")
    源列 = match.group(1)
    底数字符串 = match.group(2)

    _validate_columns(df, 源列)

    is_multi_index = isinstance(df.index, pd.MultiIndex)

    if errors == 'raise' and df[源列].isna().any():
        problem_rows = df.index[df[源列].isna()]
        raise ValueError(
            f"calc_log: 列 '{源列}' 在以下行存在 NaN 数据"
            f"（索引）：{problem_rows.tolist()[:10]}..."
        )

    if errors == 'raise' and (df[源列] <= 0).any():
        bad_rows = df.index[df[源列] <= 0]
        raise ValueError(
            f"calc_log: 列 '{源列}' 包含非正值（log 定义域要求 > 0），"
            f"行索引: {bad_rows.tolist()[:10]}..."
        )

    if 底数字符串:
        底数 = float(底数字符串)
        result = np.log(df[源列]) / np.log(底数)
    else:
        result = np.log(df[源列])

    if is_multi_index:
        df[新列名] = result
    else:
        df.loc[:, 新列名] = result

    _apply_format(df, 新列名, format, rounding, is_multi_index)
    return df


def calc_sqrt(
    df: pd.DataFrame,
    expr: str,
    rounding: int = 2,
    format: str | None = None,
    errors: str = 'coerce',
) -> pd.DataFrame:
    """平方根运算。

    expr 格式: ``"A = sqrt(B)"``
    """
    expr, format = _parse_format_suffix(expr, format)
    _validate_errors(errors)

    if expr.count('=') != 1:
        raise ValueError(f"格式错误，期望 'new = sqrt(col)'，收到：'{expr}'")

    新列名 = expr.split('=')[0].strip()
    计算部分 = expr.split('=')[1].strip()

    match = re.match(r'sqrt\s*\(\s*(\w+)\s*\)', 计算部分)
    if not match:
        raise ValueError(f"calc_sqrt: 表达式格式错误，期望 'A = sqrt(B)'，收到: '{expr}'")
    源列 = match.group(1)

    _validate_columns(df, 源列)

    is_multi_index = isinstance(df.index, pd.MultiIndex)

    if errors == 'raise' and df[源列].isna().any():
        problem_rows = df.index[df[源列].isna()]
        raise ValueError(
            f"calc_sqrt: 列 '{源列}' 在以下行存在 NaN 数据"
            f"（索引）：{problem_rows.tolist()[:10]}..."
        )

    if errors == 'raise' and (df[源列] < 0).any():
        bad_rows = df.index[df[源列] < 0]
        raise ValueError(
            f"calc_sqrt: 列 '{源列}' 包含负数，行索引: {bad_rows.tolist()[:10]}..."
        )

    result = np.sqrt(df[源列])

    if is_multi_index:
        df[新列名] = result
    else:
        df.loc[:, 新列名] = result

    _apply_format(df, 新列名, format, rounding, is_multi_index)
    return df


def calc_root(
    df: pd.DataFrame,
    expr: str,
    rounding: int = 2,
    format: str | None = None,
    errors: str = 'coerce',
) -> pd.DataFrame:
    """n 次方根运算。

    expr 格式: ``"A = root(B, 3)"``
    """
    expr, format = _parse_format_suffix(expr, format)
    _validate_errors(errors)

    if expr.count('=') != 1:
        raise ValueError(f"格式错误，期望 'new = root(col, n)'，收到：'{expr}'")

    新列名 = expr.split('=')[0].strip()
    计算部分 = expr.split('=')[1].strip()

    match = re.match(r'root\s*\(\s*(\w+)\s*,\s*(\S+)\s*\)', 计算部分)
    if not match:
        raise ValueError(f"calc_root: 表达式格式错误，期望 'A = root(B, n)'，收到: '{expr}'")
    源列 = match.group(1)
    n字符串 = match.group(2)

    _validate_columns(df, 源列)

    n = float(n字符串)

    is_multi_index = isinstance(df.index, pd.MultiIndex)

    if errors == 'raise' and df[源列].isna().any():
        problem_rows = df.index[df[源列].isna()]
        raise ValueError(
            f"calc_root: 列 '{源列}' 在以下行存在 NaN 数据"
            f"（索引）：{problem_rows.tolist()[:10]}..."
        )

    result = np.power(df[源列], 1.0 / n)

    if is_multi_index:
        df[新列名] = result
    else:
        df.loc[:, 新列名] = result

    _apply_format(df, 新列名, format, rounding, is_multi_index)
    return df
