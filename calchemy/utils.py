"""
Calchemy — A declarative DSL for DataFrame column calculations.

Every DataFrame has gold in it. Calchemy helps you extract it.
"""

import pandas as pd


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
