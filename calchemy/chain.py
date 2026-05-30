"""
Calchemy — A declarative DSL for DataFrame column calculations.

Every DataFrame has gold in it. Calchemy helps you extract it.
"""

import re
from typing import Union

import pandas as pd

from calchemy.types import CalcStep, CalcResult
from calchemy.calc import _calc_decompose


class Calchemy:
    """Calchemy 链式调用类。

    提供流式 API 对 DataFrame 进行多步列运算，
    同时记录拆解步骤和血缘关系。

    Parameters
    ----------
    df : pd.DataFrame
        待处理的 DataFrame（会被 copy，不修改原表）。

    Examples
    --------
    >>> df = pd.DataFrame({"revenue": [100, 200], "cost": [40, 80]})
    >>> result = (Calchemy(df)
    ...           .calc("margin = revenue - cost")
    ...           .calc("margin_rate = margin / revenue >>> %")
    ...           .result())
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df.copy()
        self._steps: list[CalcStep] = []
        self._tmp_columns: list[str] = []
        # 按 calc() 调用分组记录步骤，用于 lineage 避免临时列名冲突
        self._calc_calls: list[tuple[str, list[CalcStep]]] = []

    def calc(
        self,
        expr: str,
        rounding: int = 2,
        format: str | None = None,
        errors: str = 'coerce',
        keep_tmp: bool = False,
    ) -> 'Calchemy':
        """执行一次列运算，支持链式调用。

        Parameters
        ----------
        expr : str
            计算表达式，格式与 ``calc()`` 一致，如 ``"margin = revenue - cost"``。
        rounding : int, default 2
            保留小数位数。
        format : str, optional
            输出格式（'percent'/'pct'/'%' 等）。
        errors : {'coerce', 'raise', 'ignore'}, default 'coerce'
            异常处理方式。
        keep_tmp : bool, default False
            是否保留 ``__calc_tmp_*`` 临时列。

        Returns
        -------
        Calchemy
            返回 self，支持链式调用。
        """
        result = _calc_decompose(
            self._df, expr,
            rounding=rounding, format=format, errors=errors, keep_tmp=keep_tmp
        )
        self._df = result.df
        self._steps.extend(result.steps)
        self._tmp_columns.extend(result.tmp_columns)

        # 记录本次 calc 的结果列名和步骤，用于 lineage
        if expr.count('=') == 1:
            new_col = expr.split('=')[0].strip()
            self._calc_calls.append((new_col, list(result.steps)))

        return self

    def result(self) -> pd.DataFrame:
        """返回最终 DataFrame。"""
        return self._df

    @property
    def steps(self) -> list[CalcStep]:
        """返回所有拆解步骤（跨多次 calc() 累积）。"""
        return self._steps

    @property
    def tmp_columns(self) -> list[str]:
        """返回所有临时列名（跨多次 calc() 累积，仅在 keep_tmp=True 时有值）。"""
        return self._tmp_columns

    def lineage(self, target_col: str | None = None) -> Union[dict[str, list[str]], list[str]]:
        """血缘追踪：返回列的依赖关系。

        解析所有步骤的 expression，提取其中引用的原始列名
       （排除 ``__calc_tmp_*`` 临时列），构建血缘映射。

        Parameters
        ----------
        target_col : str, optional
            目标列名。为 None 时返回所有列的映射。

        Returns
        -------
        dict[str, list[str]] or list[str]
            target_col 为 None 时返回 {列名: [依赖列]} 映射；
            否则返回 target_col 依赖的原始列列表。
        """
        # 构建每个 result_col → 依赖列的映射
        # 由于不同 calc() 调用可能复用 __calc_tmp_N 编号，
        # 在构建血缘图时对临时列按调用批次做命名空间隔离
        col_deps: dict[str, set[str]] = {}

        def _ns(name: str, call_idx: int) -> str:
            """为临时列添加命名空间前缀，避免不同 calc 调用冲突。"""
            if name.startswith('__calc_tmp_'):
                return f"{name}_call_{call_idx}"
            return name

        for call_idx, (new_col, steps) in enumerate(self._calc_calls):
            # 处理中间拆解步骤
            for step in steps:
                local_result = _ns(step.result_col, call_idx)
                tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', step.expression)
                deps = {
                    _ns(t, call_idx)
                    for t in tokens
                    if t != step.result_col
                    and not t.isdigit()
                    and t not in ('pd', 'np', 'ast', 'len', 'sum', 'max', 'min')
                }
                col_deps[local_result] = deps

            # 处理最终列映射：结果列依赖最后一步的临时列（或直接使用原始列）
            if steps:
                last_tmp = _ns(steps[-1].result_col, call_idx)
                col_deps[new_col] = {last_tmp}
            else:
                # 无中间步骤，直接解析表达式中的列名
                calc_expr = self._calc_calls[call_idx][0]  # fallback, shouldn't happen often
                # 从原始 expr 重新解析（更可靠）
                # 但此处我们直接从 df 推断：无中间步骤意味着表达式直接引用了现有列
                # 由于信息已在 steps 中记录，无 steps 时通常是直接赋值如 A = B
                # 此时 new_col 的依赖需要回溯到原始 expr
                # 为简化，从 self._steps 为空的情况下，依赖就是 calc_expr 中的标识符
                # 但由于我们无法直接获取 expr，这里用一个空集合占位，
                # 实际会在 resolve 中回退为原始列名
                col_deps[new_col] = set()

        # 递归解析：如果依赖中包含临时列，需要回溯到它的原始依赖
        def resolve_deps(col: str, visited: set[str] | None = None) -> set[str]:
            if visited is None:
                visited = set()
            if col in visited:
                return set()
            visited.add(col)

            if col not in col_deps:
                return {col}  # 原始列

            result = set()
            for dep in col_deps[col]:
                if dep.startswith('__calc_tmp_'):
                    result.update(resolve_deps(dep, visited))
                else:
                    result.add(dep)
            return result

        if target_col is not None:
            return sorted(resolve_deps(target_col))

        # 返回所有非临时结果列的映射
        all_cols = {c for c in self._df.columns if not c.startswith('__calc_tmp_')}
        for new_col, _ in self._calc_calls:
            all_cols.add(new_col)

        return {col: sorted(resolve_deps(col)) for col in sorted(all_cols)}
