import pandas as pd
import numpy as np


# ──────────────────────────────────────────────
# 内部工具函数（不对外暴露）
# ──────────────────────────────────────────────

def _parse_format_suffix(expr_string: str, format: str | None) -> tuple[str, str | None]:
    """解析 >>> 格式后缀，返回 (清理后的表达式, 格式标识)。

    若 expr_string 中包含 >>>，则以 >>> 后的为准，忽略 format 参数。
    """
    if '>>>' in expr_string:
        format = expr_string.split('>>>')[1].strip()
        expr_string = expr_string.split('>>>')[0].strip()
    return expr_string, format


def _validate_errors(errors: str) -> None:
    """校验 errors 参数合法性。"""
    if errors not in ('coerce', 'raise', 'ignore'):
        raise ValueError(f"errors 参数须为 'coerce'/'raise'/'ignore'，收到：'{errors}'")


def _validate_columns(df: pd.DataFrame, *cols: str) -> None:
    """校验列名是否存在于 DataFrame 中，不存在则抛出友好 KeyError。"""
    for col in cols:
        if col not in df.columns:
            raise KeyError(f"列 '{col}' 不存在于 DataFrame 中，可用列：{list(df.columns)}")


def _apply_format(df: pd.DataFrame, new_column_name: str, format: str | None,
                  rounding: int, is_multi_index: bool) -> None:
    """根据 format 参数对目标列做格式化（原地修改 df）。"""
    if format is None:
        if is_multi_index:
            df[new_column_name] = df[new_column_name].astype(float).round(rounding)
        else:
            df.loc[:, new_column_name] = df[new_column_name].astype(float).round(rounding)

    elif format.lower() in ['percent', 'pct', '%', 'percentage', '百分比']:
        mask_not_na = df[new_column_name].notna()
        rounded = df.loc[mask_not_na, new_column_name].astype(float).round(rounding)
        formatted = rounded.fillna(0.0).map(("{:." + str(rounding) + "%}").format)

        df[new_column_name] = df[new_column_name].astype(object)

        if is_multi_index:
            result = df[new_column_name].copy()
            result[mask_not_na] = formatted
            df[new_column_name] = result
        else:
            df.loc[mask_not_na, new_column_name] = formatted


# ──────────────────────────────────────────────
# 除法
# ──────────────────────────────────────────────

def pandas_divided_helper(
    df: pd.DataFrame,
    divid_string: str,
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
    divid_string : str
        计算表达式，支持两种写法：
        1) 标准格式：
           "new_column_name = column_name1 / column_name2"
        2) 带格式后缀：
           "new_column_name = column_name1 / column_name2 >>> percentage"
        空格可随意添加，会自动压缩。
    rounding : int, default 2
        保留的小数位数（百分比模式下同样适用）。
    format : str, optional
        显式指定返回格式，可选值：
        None / 'float'          → 普通浮点，先 round 再返回
        'percent'/'pct'/'%'/'percentage'/... → 转换为百分比字符串并保留指定小数位
        若 divid_string 中已用 >>> 指定格式，则以 divid_string 为准。
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
    >>> pandas_divided_helper(df, "ratio = revenue / cost")
    >>> df["ratio"]
    0    2.0
    1    5.0
    Name: ratio, dtype: float64

    >>> pandas_divided_helper(df, "pct = revenue / cost >>> %")
    """
    # 1. 解析格式后缀 & 校验 errors
    divid_string, format = _parse_format_suffix(divid_string, format)
    _validate_errors(errors)

    # 2. 基础校验
    if divid_string.count('=') != 1:
        raise ValueError(
            f"格式错误，期望 'new = col1 / col2'，收到：'{divid_string}'，原因：'=' 数量不为 1。"
        )
    if divid_string.split('=')[1].count('/') != 1:
        raise ValueError(
            f"格式错误，期望 'new = col1 / col2'，收到：'{divid_string}'，原因：右侧存在多个 '/'。"
        )

    # 3. 提取列名
    divid_string = divid_string.strip().replace(' = ', '=').replace(' / ', '/')
    new_column_name = divid_string.split('=')[0].strip()
    calculating_cols = divid_string.split('=')[1]
    column_name1 = calculating_cols.split('/')[0].strip()
    column_name2 = calculating_cols.split('/')[1].strip()

    _validate_columns(df, column_name1, column_name2)

    # 4. raise 模式：先检查是否有问题行
    is_multi_index = isinstance(df.index, pd.MultiIndex)

    mask_nan_input = df[column_name1].isna() | df[column_name2].isna()
    mask_zero_denom = df[column_name2] == 0

    if errors == 'raise':
        problem_rows = df.index[mask_nan_input | mask_zero_denom]
        if len(problem_rows) > 0:
            raise ValueError(
                f"列 '{column_name1}' 或 '{column_name2}' 在以下行存在 NaN 或零分母数据"
                f"（索引）：{problem_rows.tolist()[:10]}..."
            )

    # 5. 核心计算（向量化操作）
    if is_multi_index:
        df[new_column_name] = np.nan
    else:
        df.loc[:, new_column_name] = np.nan

    # raise 模式已通过校验，所有行都是有效的
    # ignore 模式下只处理无问题的行
    if errors == 'ignore':
        mask_valid = (~mask_nan_input) & (~mask_zero_denom)
    else:
        # coerce 或 raise（raise 已确保无问题行）
        mask_valid = ~mask_zero_denom

    mask_dirty = (df[column_name1] != 0) & mask_zero_denom

    if errors != 'ignore':
        # 计算有效行
        if is_multi_index:
            df.loc[mask_valid, new_column_name] = (
                df.loc[mask_valid, column_name1] / df.loc[mask_valid, column_name2]
            )
        else:
            df.loc[mask_valid, new_column_name] = (
                df.loc[mask_valid, column_name1] / df.loc[mask_valid, column_name2]
            )
        # 脏数据（分子≠0 且分母=0）→ 强制 0
        df.loc[mask_dirty & ~mask_nan_input, new_column_name] = 0
    else:
        # ignore 模式：只计算合法行
        if is_multi_index:
            df.loc[mask_valid, new_column_name] = (
                df.loc[mask_valid, column_name1] / df.loc[mask_valid, column_name2]
            )
        else:
            df.loc[mask_valid, new_column_name] = (
                df.loc[mask_valid, column_name1] / df.loc[mask_valid, column_name2]
            )

    # 6. 格式化输出
    _apply_format(df, new_column_name, format, rounding, is_multi_index)

    return df


# ──────────────────────────────────────────────
# 加法
# ──────────────────────────────────────────────

def pandas_add_helper(
    df: pd.DataFrame,
    add_string: str,
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
    add_string : str
        计算表达式，支持两种写法：
        1) 标准格式：
           "new_column_name = column_name1 + column_name2"
        2) 带格式后缀：
           "new_column_name = column_name1 + column_name2 >>> percentage"
        空格可随意添加，会自动压缩。
    rounding : int, default 2
        保留的小数位数（百分比模式下同样适用）。
    format : str, optional
        显式指定返回格式，可选值：
        None / 'float'          → 普通浮点，先 round 再返回
        'percent'/'pct'/'%'/'percentage'/... → 转换为百分比字符串并保留指定小数位
        若 add_string 中已用 >>> 指定格式，则以 add_string 为准。
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
    >>> pandas_add_helper(df, "c = a + b")
    >>> df["c"]
    0    13.0
    1    24.0
    Name: c, dtype: float64
    """
    # 1. 解析格式后缀 & 校验 errors
    add_string, format = _parse_format_suffix(add_string, format)
    _validate_errors(errors)

    # 2. 基础校验
    if add_string.count('=') != 1:
        raise ValueError(
            f"格式错误，期望 'new = col1 + col2'，收到：'{add_string}'，原因：'=' 数量不为 1。"
        )
    if add_string.split('=')[1].count('+') != 1:
        raise ValueError(
            f"格式错误，期望 'new = col1 + col2'，收到：'{add_string}'，原因：右侧存在多个 '+'。"
        )

    # 3. 提取列名
    add_string = add_string.strip().replace(' = ', '=').replace(' + ', '+')
    new_column_name = add_string.split('=')[0].strip()
    calculating_cols = add_string.split('=')[1]
    column_name1 = calculating_cols.split('+')[0].strip()
    column_name2 = calculating_cols.split('+')[1].strip()

    _validate_columns(df, column_name1, column_name2)

    # 4. 检测问题行
    is_multi_index = isinstance(df.index, pd.MultiIndex)
    mask_nan_input = df[column_name1].isna() | df[column_name2].isna()

    if errors == 'raise':
        problem_rows = df.index[mask_nan_input]
        if len(problem_rows) > 0:
            raise ValueError(
                f"列 '{column_name1}' 或 '{column_name2}' 在以下行存在 NaN 数据"
                f"（索引）：{problem_rows.tolist()[:10]}..."
            )

    # 5. 核心计算（向量化操作）
    if is_multi_index:
        df[new_column_name] = np.nan
    else:
        df.loc[:, new_column_name] = np.nan

    if errors == 'ignore':
        mask_valid = ~mask_nan_input
    else:
        mask_valid = ~mask_nan_input  # coerce: NaN 行保留 NaN，只算有效行

    if is_multi_index:
        df.loc[mask_valid, new_column_name] = (
            df.loc[mask_valid, column_name1] + df.loc[mask_valid, column_name2]
        )
    else:
        df.loc[mask_valid, new_column_name] = (
            df.loc[mask_valid, column_name1] + df.loc[mask_valid, column_name2]
        )

    # 6. 格式化输出
    _apply_format(df, new_column_name, format, rounding, is_multi_index)

    return df


# ──────────────────────────────────────────────
# 减法
# ──────────────────────────────────────────────

def pandas_subtract_helper(
    df: pd.DataFrame,
    subtract_string: str,
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
    subtract_string : str
        计算表达式，支持两种写法：
        1) 标准格式：
           "new_column_name = column_name1 - column_name2"
        2) 带格式后缀：
           "new_column_name = column_name1 - column_name2 >>> percentage"
        空格可随意添加，会自动压缩。
    rounding : int, default 2
        保留的小数位数（百分比模式下同样适用）。
    format : str, optional
        显式指定返回格式，可选值：
        None / 'float'          → 普通浮点，先 round 再返回
        'percent'/'pct'/'%'/'percentage'/... → 转换为百分比字符串并保留指定小数位
        若 subtract_string 中已用 >>> 指定格式，则以 subtract_string 为准。
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
    >>> pandas_subtract_helper(df, "margin = revenue - cost")
    >>> df["margin"]
    0    50.0
    1    160.0
    Name: margin, dtype: float64
    """
    # 1. 解析格式后缀 & 校验 errors
    subtract_string, format = _parse_format_suffix(subtract_string, format)
    _validate_errors(errors)

    # 2. 基础校验
    if subtract_string.count('=') != 1:
        raise ValueError(
            f"格式错误，期望 'new = col1 - col2'，收到：'{subtract_string}'，原因：'=' 数量不为 1。"
        )
    if subtract_string.split('=')[1].count('-') != 1:
        raise ValueError(
            f"格式错误，期望 'new = col1 - col2'，收到：'{subtract_string}'，原因：右侧存在多个 '-'。"
        )

    # 3. 提取列名
    subtract_string = subtract_string.strip().replace(' = ', '=').replace(' - ', '-')
    new_column_name = subtract_string.split('=')[0].strip()
    calculating_cols = subtract_string.split('=')[1]
    column_name1 = calculating_cols.split('-')[0].strip()
    column_name2 = calculating_cols.split('-')[1].strip()

    _validate_columns(df, column_name1, column_name2)

    # 4. 检测问题行
    is_multi_index = isinstance(df.index, pd.MultiIndex)
    mask_nan_input = df[column_name1].isna() | df[column_name2].isna()

    if errors == 'raise':
        problem_rows = df.index[mask_nan_input]
        if len(problem_rows) > 0:
            raise ValueError(
                f"列 '{column_name1}' 或 '{column_name2}' 在以下行存在 NaN 数据"
                f"（索引）：{problem_rows.tolist()[:10]}..."
            )

    # 5. 核心计算（向量化操作）
    if is_multi_index:
        df[new_column_name] = np.nan
    else:
        df.loc[:, new_column_name] = np.nan

    if errors == 'ignore':
        mask_valid = ~mask_nan_input
    else:
        mask_valid = ~mask_nan_input

    if is_multi_index:
        df.loc[mask_valid, new_column_name] = (
            df.loc[mask_valid, column_name1] - df.loc[mask_valid, column_name2]
        )
    else:
        df.loc[mask_valid, new_column_name] = (
            df.loc[mask_valid, column_name1] - df.loc[mask_valid, column_name2]
        )

    # 6. 格式化输出
    _apply_format(df, new_column_name, format, rounding, is_multi_index)

    return df


# ──────────────────────────────────────────────
# 乘法
# ──────────────────────────────────────────────

def pandas_multiply_helper(
    df: pd.DataFrame,
    multiply_string: str,
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
    multiply_string : str
        计算表达式，支持两种写法：
        1) 标准格式：
           "new_column_name = column_name1 * column_name2"
        2) 带格式后缀：
           "new_column_name = column_name1 * column_name2 >>> percentage"
        空格可随意添加，会自动压缩。
    rounding : int, default 2
        保留的小数位数（百分比模式下同样适用）。
    format : str, optional
        显式指定返回格式，可选值：
        None / 'float'          → 普通浮点，先 round 再返回
        'percent'/'pct'/'%'/'percentage'/... → 转换为百分比字符串并保留指定小数位
        若 multiply_string 中已用 >>> 指定格式，则以 multiply_string 为准。
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
    >>> pandas_multiply_helper(df, "total = qty * price")
    >>> df["total"]
    0    35.0
    1    80.0
    Name: total, dtype: float64
    """
    # 1. 解析格式后缀 & 校验 errors
    multiply_string, format = _parse_format_suffix(multiply_string, format)
    _validate_errors(errors)

    # 2. 基础校验
    if multiply_string.count('=') != 1:
        raise ValueError(
            f"格式错误，期望 'new = col1 * col2'，收到：'{multiply_string}'，原因：'=' 数量不为 1。"
        )
    if multiply_string.split('=')[1].count('*') != 1:
        raise ValueError(
            f"格式错误，期望 'new = col1 * col2'，收到：'{multiply_string}'，原因：右侧存在多个 '*'。"
        )

    # 3. 提取列名
    multiply_string = multiply_string.strip().replace(' = ', '=').replace(' * ', '*')
    new_column_name = multiply_string.split('=')[0].strip()
    calculating_cols = multiply_string.split('=')[1]
    column_name1 = calculating_cols.split('*')[0].strip()
    column_name2 = calculating_cols.split('*')[1].strip()

    _validate_columns(df, column_name1, column_name2)

    # 4. 检测问题行
    is_multi_index = isinstance(df.index, pd.MultiIndex)
    mask_nan_input = df[column_name1].isna() | df[column_name2].isna()

    if errors == 'raise':
        problem_rows = df.index[mask_nan_input]
        if len(problem_rows) > 0:
            raise ValueError(
                f"列 '{column_name1}' 或 '{column_name2}' 在以下行存在 NaN 数据"
                f"（索引）：{problem_rows.tolist()[:10]}..."
            )

    # 5. 核心计算（向量化操作）
    if is_multi_index:
        df[new_column_name] = np.nan
    else:
        df.loc[:, new_column_name] = np.nan

    if errors == 'ignore':
        mask_valid = ~mask_nan_input
    else:
        mask_valid = ~mask_nan_input

    if is_multi_index:
        df.loc[mask_valid, new_column_name] = (
            df.loc[mask_valid, column_name1] * df.loc[mask_valid, column_name2]
        )
    else:
        df.loc[mask_valid, new_column_name] = (
            df.loc[mask_valid, column_name1] * df.loc[mask_valid, column_name2]
        )

    # 6. 格式化输出
    _apply_format(df, new_column_name, format, rounding, is_multi_index)

    return df
