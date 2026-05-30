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
        self._lineage_map: dict[str, list[str]] = {}

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
            计算表达式，格式与 ``calc()`` 一致。
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

        # 记录本次 calc 的血缘映射
        new_col = expr.split('=')[0].strip()
        self._lineage_map[new_col] = self._resolve_deps(result.steps)

        return self

    def _resolve_deps(self, steps: list[CalcStep]) -> list[str]:
        """从 steps 中递归解析原始列依赖（回溯临时列）。"""
        if not steps:
            return []

        # 构建 result_col → 所有依赖标识符的映射（不过滤临时列）
        col_deps: dict[str, set[str]] = {}
        for step in steps:
            tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', step.expression)
            deps = {
                t for t in tokens
                if not t.isdigit()
                and t not in ('pd', 'np', 'ast', 'len', 'sum', 'max', 'min')
            }
            col_deps[step.result_col] = deps

        def resolve(col: str, visited: set[str] | None = None) -> set[str]:
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
                    result.update(resolve(dep, visited))
                else:
                    result.add(dep)
            return result

        # 最后一个 step 的 result_col 是本次计算的最终来源
        last_col = steps[-1].result_col
        return sorted(resolve(last_col))

    def result(self) -> pd.DataFrame:
        """返回最终 DataFrame。"""
        return self._df

    @property
    def steps(self) -> list[CalcStep]:
        """返回所有拆解步骤（跨多次 calc() 累积）。"""
        return self._steps

    @property
    def tmp_columns(self) -> list[str]:
        """返回所有临时列名（仅在 keep_tmp=True 时有值）。"""
        return self._tmp_columns

    def lineage(self, target_col: str | None = None) -> Union[dict[str, list[str]], list[str]]:
        """血缘追踪：返回列的依赖关系。

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
        if target_col is not None:
            return self._lineage_map.get(target_col, [])
        return dict(self._lineage_map)
