"""
Calchemy — A declarative DSL for DataFrame column calculations.

Every DataFrame has gold in it. Calchemy helps you extract it.
"""

import ast
from typing import Union

import pandas as pd
import numpy as np

from calchemy.types import CalcStep
from calchemy.helpers import calc_add, calc_sub, calc_mul, calc_div, calc_pow, calc_abs, calc_log, calc_sqrt, calc_root


# ──────────────────────────────────────────────
# 内部 AST 求值器（calc() 使用）
# ──────────────────────────────────────────────

# 允许的二元运算符映射
_BINOP_MAP: dict[type, str] = {
    ast.Add: '+',
    ast.Sub: '-',
    ast.Mult: '*',
    ast.Div: '/',
    ast.Pow: '**',
    ast.BitXor: '^',  # Excel 风格指数，Python 中 ^ 是按位异或，但 DSL 中解释为指数
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
    # 函数调用
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError(
                f"仅支持简单函数调用（如 abs(B)），不支持 {type(node.func).__name__}"
            )
        func_name = node.func.id
        allowed_funcs = {'abs', 'log', 'sqrt', 'root'}
        if func_name not in allowed_funcs:
            raise ValueError(
                f"不支持的函数 '{func_name}'。支持的函数：{', '.join(sorted(allowed_funcs))}"
            )

        args = []
        for arg in node.args:
            args.append(_eval_ast_node(arg, df, errors, collected_cols))

        import math, numpy as np
        if func_name == 'abs':
            if hasattr(args[0], 'abs'):
                return args[0].abs()
            return abs(args[0])
        elif func_name == 'log':
            if hasattr(args[0], '__iter__'):
                if len(args) == 1:
                    return np.log(args[0])
                return np.log(args[0]) / np.log(args[1])
            if len(args) == 1:
                return math.log(args[0])
            return math.log(args[0], args[1])
        elif func_name == 'sqrt':
            if hasattr(args[0], '__iter__'):
                return np.sqrt(args[0])
            return math.sqrt(args[0])
        elif func_name == 'root':
            if len(args) < 2:
                raise ValueError("root() 需要两个参数：root(值, n次方)")
            if hasattr(args[0], '__iter__'):
                return np.power(args[0], 1.0 / args[1])
            return args[0] ** (1.0 / args[1])

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
                f"仅支持 +、-、*、/、**、^ 运算。"
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

        elif isinstance(node.op, (ast.Pow, ast.BitXor)):
            return left ** right

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
        f"仅支持列名、数字常量、+、-、*、/、**、^ 运算以及 abs()、log()、sqrt()、root() 函数。"
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
    # 函数调用
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError(
                f"仅支持简单函数调用（如 abs(B)），不支持 {type(node.func).__name__}"
            )
        func_name = node.func.id
        allowed_funcs = {'abs', 'log', 'sqrt', 'root'}
        if func_name not in allowed_funcs:
            raise ValueError(
                f"不支持的函数 '{func_name}'。支持的函数：{', '.join(sorted(allowed_funcs))}"
            )

        # 递归处理参数
        arg_results = []
        for arg in node.args:
            arg_result = _decompose_ast(
                arg, df, errors, collected_cols, tmp_counter, steps
            )
            arg_results.append(arg_result)

        tmp_counter[0] += 1
        tmp_col = f"__calc_tmp_{tmp_counter[0]}"
        arg_str = ', '.join(str(a) for a in arg_results)
        expr_str = f"{tmp_col} = {func_name}({arg_str})"

        if func_name == 'abs':
            calc_abs(df, expr_str, rounding=15, format=None, errors=errors)
        elif func_name == 'log':
            calc_log(df, expr_str, rounding=15, format=None, errors=errors)
        elif func_name == 'sqrt':
            calc_sqrt(df, expr_str, rounding=15, format=None, errors=errors)
        elif func_name == 'root':
            calc_root(df, expr_str, rounding=15, format=None, errors=errors)

        steps.append(
            CalcStep(expression=expr_str, result_col=tmp_col, operator=func_name)
        )
        return tmp_col

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
                f"仅支持 +、-、*、/、**、^ 运算。"
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
            elif op_symbol in ('**', '^'):
                if left_ref == 0 and right_ref == 0:
                    return float('nan')
                return left_ref ** right_ref

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
        elif op_symbol in ('**', '^'):
            calc_pow(df, expr_str, rounding=15, format=None, errors=errors)

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
        f"仅支持列名、数字常量、+、-、*、/、**、^ 运算以及 abs()、log()、sqrt()、root() 函数。"
    )
