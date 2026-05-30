"""
Calchemy 核心模块测试

覆盖全部四个 helper 函数：calc_add / calc_sub / calc_mul / calc_div
测试维度：正常计算、NaN 输入、零值、百分比格式、errors 三种模式、列不存在、MultiIndex、格式错误
"""

import pytest
import pandas as pd
import numpy as np

from calchemy import calc_add, calc_sub, calc_mul, calc_div, calc_pow, calc_abs, calc_log, calc_sqrt, calc_root, calc, Calchemy
from calchemy.calc import _calc_decompose
from calchemy.types import CalcResult, CalcStep


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def df_basic():
    """基础测试 DataFrame：含正常值、零值、NaN"""
    return pd.DataFrame({
        'a': [10.0, 20.0, 30.0, 0.0, np.nan],
        'b': [5.0, 4.0, 0.0, 0.0, 3.0],
    })


@pytest.fixture
def df_multiindex():
    """MultiIndex 测试 DataFrame"""
    arrays = [[1, 1, 2, 2], ['a', 'b', 'a', 'b']]
    index = pd.MultiIndex.from_arrays(arrays, names=['first', 'second'])
    return pd.DataFrame({'a': [10.0, 20.0, 30.0, 40.0], 'b': [5.0, 4.0, 3.0, 2.0]}, index=index)


# ──────────────────────────────────────────────
# 加法测试
# ──────────────────────────────────────────────

class TestCalcAdd:
    """calc_add 测试套件"""

    def test_normal_calculation(self, df_basic):
        """正常加法计算"""
        result = calc_add(df_basic.copy(), 'sum = a + b')
        assert result['sum'].iloc[0] == 15.0
        assert result['sum'].iloc[1] == 24.0
        assert result['sum'].iloc[2] == 30.0
        assert result['sum'].iloc[3] == 0.0

    def test_nan_input_coerce(self, df_basic):
        """NaN 输入 → 结果 NaN（coerce 默认行为）"""
        result = calc_add(df_basic.copy(), 'sum = a + b')
        assert pd.isna(result['sum'].iloc[4])

    def test_zero_operands(self):
        """0 + 0 = 0"""
        df = pd.DataFrame({'a': [0.0], 'b': [0.0]})
        result = calc_add(df, 'sum = a + b')
        assert result['sum'].iloc[0] == 0.0

    def test_percentage_format(self, df_basic):
        """百分比格式后缀"""
        result = calc_add(df_basic.copy(), 'sum = a + b >>> %')
        assert result['sum'].iloc[0] == '1500.00%'
        # NaN 行应保持 NaN
        assert pd.isna(result['sum'].iloc[4])

    def test_errors_raise(self, df_basic):
        """errors='raise' → NaN 输入行抛出 ValueError"""
        with pytest.raises(ValueError, match='NaN'):
            calc_add(df_basic.copy(), 'sum = a + b', errors='raise')

    def test_errors_raise_no_nan(self):
        """errors='raise' → 无 NaN 时不抛异常"""
        df = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
        result = calc_add(df, 'sum = a + b', errors='raise')
        assert result['sum'].iloc[0] == 4.0

    def test_errors_ignore(self, df_basic):
        """errors='ignore' → NaN 行保留原值（NaN），其他行正常计算"""
        result = calc_add(df_basic.copy(), 'sum = a + b', errors='ignore')
        assert result['sum'].iloc[0] == 15.0
        assert pd.isna(result['sum'].iloc[4])

    def test_column_not_found(self, df_basic):
        """列不存在 → 友好 KeyError"""
        with pytest.raises(KeyError, match='x'):
            calc_add(df_basic.copy(), 'sum = x + y')

    def test_multiindex(self, df_multiindex):
        """MultiIndex DataFrame 正常计算"""
        result = calc_add(df_multiindex.copy(), 'sum = a + b')
        assert result['sum'].iloc[0] == 15.0
        assert result['sum'].iloc[3] == 42.0

    def test_format_error_no_equals(self, df_basic):
        """格式错误：无 = 号"""
        with pytest.raises(ValueError, match="'='"):
            calc_add(df_basic.copy(), 'bad format')

    def test_format_error_multiple_plus(self, df_basic):
        """格式错误：多个 + 号"""
        with pytest.raises(ValueError, match="多个"):
            calc_add(df_basic.copy(), 'bad = a + b + c')

    def test_errors_invalid_value(self, df_basic):
        """非法 errors 参数"""
        with pytest.raises(ValueError, match="coerce"):
            calc_add(df_basic.copy(), 'sum = a + b', errors='bad')

    def test_rounding(self):
        """rounding 参数生效"""
        df = pd.DataFrame({'a': [1.111], 'b': [2.222]})
        result = calc_add(df, 'sum = a + b', rounding=1)
        assert result['sum'].iloc[0] == 3.3

    def test_returns_same_dataframe(self, df_basic):
        """返回的是同一个 DataFrame 实例"""
        df = df_basic.copy()
        result = calc_add(df, 'sum = a + b')
        assert result is df


# ──────────────────────────────────────────────
# 减法测试
# ──────────────────────────────────────────────

class TestCalcSub:
    """calc_sub 测试套件"""

    def test_normal_calculation(self, df_basic):
        """正常减法计算"""
        result = calc_sub(df_basic.copy(), 'diff = a - b')
        assert result['diff'].iloc[0] == 5.0
        assert result['diff'].iloc[1] == 16.0
        assert result['diff'].iloc[2] == 30.0
        assert result['diff'].iloc[3] == 0.0

    def test_negative_result(self):
        """减法允许负数结果"""
        df = pd.DataFrame({'a': [5.0], 'b': [10.0]})
        result = calc_sub(df, 'diff = a - b')
        assert result['diff'].iloc[0] == -5.0

    def test_nan_input_coerce(self, df_basic):
        """NaN 输入 → 结果 NaN"""
        result = calc_sub(df_basic.copy(), 'diff = a - b')
        assert pd.isna(result['diff'].iloc[4])

    def test_zero_operands(self):
        """0 - 0 = 0"""
        df = pd.DataFrame({'a': [0.0], 'b': [0.0]})
        result = calc_sub(df, 'diff = a - b')
        assert result['diff'].iloc[0] == 0.0

    def test_percentage_format(self, df_basic):
        """百分比格式后缀"""
        result = calc_sub(df_basic.copy(), 'diff = a - b >>> %')
        assert result['diff'].iloc[0] == '500.00%'

    def test_errors_raise(self, df_basic):
        """errors='raise' → NaN 行抛异常"""
        with pytest.raises(ValueError, match='NaN'):
            calc_sub(df_basic.copy(), 'diff = a - b', errors='raise')

    def test_errors_ignore(self, df_basic):
        """errors='ignore' → NaN 行保留，其他行正常"""
        result = calc_sub(df_basic.copy(), 'diff = a - b', errors='ignore')
        assert result['diff'].iloc[0] == 5.0
        assert pd.isna(result['diff'].iloc[4])

    def test_column_not_found(self, df_basic):
        """列不存在 → KeyError"""
        with pytest.raises(KeyError, match='nonexistent'):
            calc_sub(df_basic.copy(), 'diff = nonexistent - a')

    def test_multiindex(self, df_multiindex):
        """MultiIndex 正常计算"""
        result = calc_sub(df_multiindex.copy(), 'diff = a - b')
        assert result['diff'].iloc[0] == 5.0
        assert result['diff'].iloc[3] == 38.0

    def test_format_error_no_equals(self, df_basic):
        """格式错误：无 = 号"""
        with pytest.raises(ValueError, match="'='"):
            calc_sub(df_basic.copy(), 'bad format')

    def test_format_error_multiple_minus(self, df_basic):
        """格式错误：多个 - 号"""
        with pytest.raises(ValueError, match="多个"):
            calc_sub(df_basic.copy(), 'bad = a - b - c')

    def test_returns_same_dataframe(self, df_basic):
        """返回同一个 DataFrame 实例"""
        df = df_basic.copy()
        result = calc_sub(df, 'diff = a - b')
        assert result is df


# ──────────────────────────────────────────────
# 乘法测试
# ──────────────────────────────────────────────

class TestCalcMul:
    """calc_mul 测试套件"""

    def test_normal_calculation(self, df_basic):
        """正常乘法计算"""
        result = calc_mul(df_basic.copy(), 'product = a * b')
        assert result['product'].iloc[0] == 50.0
        assert result['product'].iloc[1] == 80.0

    def test_multiply_by_zero(self, df_basic):
        """非零 × 0 = 0"""
        result = calc_mul(df_basic.copy(), 'product = a * b')
        assert result['product'].iloc[2] == 0.0

    def test_zero_multiply_zero(self):
        """0 × 0 = 0"""
        df = pd.DataFrame({'a': [0.0], 'b': [0.0]})
        result = calc_mul(df, 'product = a * b')
        assert result['product'].iloc[0] == 0.0

    def test_zero_multiply_nan(self):
        """0 × NaN = NaN（不做隐式填 0）"""
        df = pd.DataFrame({'a': [0.0], 'b': [np.nan]})
        result = calc_mul(df, 'product = a * b')
        assert pd.isna(result['product'].iloc[0])

    def test_nan_input_coerce(self, df_basic):
        """NaN 输入 → 结果 NaN"""
        result = calc_mul(df_basic.copy(), 'product = a * b')
        assert pd.isna(result['product'].iloc[4])

    def test_percentage_format(self, df_basic):
        """百分比格式后缀"""
        result = calc_mul(df_basic.copy(), 'product = a * b >>> %')
        assert result['product'].iloc[0] == '5000.00%'

    def test_errors_raise(self, df_basic):
        """errors='raise' → NaN 行抛异常"""
        with pytest.raises(ValueError, match='NaN'):
            calc_mul(df_basic.copy(), 'product = a * b', errors='raise')

    def test_errors_raise_no_nan(self):
        """errors='raise' → 无 NaN 时不抛异常"""
        df = pd.DataFrame({'a': [2.0], 'b': [3.0]})
        result = calc_mul(df, 'product = a * b', errors='raise')
        assert result['product'].iloc[0] == 6.0

    def test_errors_ignore(self, df_basic):
        """errors='ignore' → NaN 行保留，其他行正常"""
        result = calc_mul(df_basic.copy(), 'product = a * b', errors='ignore')
        assert result['product'].iloc[0] == 50.0
        assert pd.isna(result['product'].iloc[4])

    def test_column_not_found(self, df_basic):
        """列不存在 → KeyError"""
        with pytest.raises(KeyError, match='xyz'):
            calc_mul(df_basic.copy(), 'product = xyz * a')

    def test_multiindex(self, df_multiindex):
        """MultiIndex 正常计算"""
        result = calc_mul(df_multiindex.copy(), 'product = a * b')
        assert result['product'].iloc[0] == 50.0
        assert result['product'].iloc[3] == 80.0

    def test_format_error_multiple_star(self, df_basic):
        """格式错误：多个 * 号"""
        with pytest.raises(ValueError, match="多个"):
            calc_mul(df_basic.copy(), 'bad = a * b * c')

    def test_returns_same_dataframe(self, df_basic):
        """返回同一个 DataFrame 实例"""
        df = df_basic.copy()
        result = calc_mul(df, 'product = a * b')
        assert result is df


# ──────────────────────────────────────────────
# 除法测试
# ──────────────────────────────────────────────

class TestCalcDiv:
    """calc_div 测试套件（含 errors 参数）"""

    def test_normal_calculation(self, df_basic):
        """正常除法计算"""
        result = calc_div(df_basic.copy(), 'ratio = a / b')
        assert result['ratio'].iloc[0] == 2.0
        assert result['ratio'].iloc[1] == 5.0

    def test_zero_numerator_zero_denominator(self, df_basic):
        """0 / 0 → NaN"""
        result = calc_div(df_basic.copy(), 'ratio = a / b')
        assert pd.isna(result['ratio'].iloc[3])

    def test_nonzero_numerator_zero_denominator(self, df_basic):
        """非零 / 0 → 脏数据强制 0"""
        result = calc_div(df_basic.copy(), 'ratio = a / b')
        assert result['ratio'].iloc[2] == 0.0

    def test_nan_input_coerce(self, df_basic):
        """NaN 输入 → 结果 NaN"""
        result = calc_div(df_basic.copy(), 'ratio = a / b')
        assert pd.isna(result['ratio'].iloc[4])

    def test_percentage_format(self, df_basic):
        """百分比格式后缀"""
        result = calc_div(df_basic.copy(), 'pct = a / b >>> %')
        assert result['pct'].iloc[0] == '200.00%'

    def test_errors_raise_with_nan(self, df_basic):
        """errors='raise' → NaN 输入行抛异常"""
        with pytest.raises(ValueError, match='NaN'):
            calc_div(df_basic.copy(), 'ratio = a / b', errors='raise')

    def test_errors_raise_with_zero_denom(self):
        """errors='raise' → 零分母行抛异常"""
        df = pd.DataFrame({'a': [10.0, 0.0], 'b': [2.0, 0.0]})
        with pytest.raises(ValueError, match='零分母'):
            calc_div(df, 'ratio = a / b', errors='raise')

    def test_errors_raise_no_problems(self):
        """errors='raise' → 无问题时正常计算"""
        df = pd.DataFrame({'a': [10.0], 'b': [2.0]})
        result = calc_div(df, 'ratio = a / b', errors='raise')
        assert result['ratio'].iloc[0] == 5.0

    def test_errors_ignore(self, df_basic):
        """errors='ignore' → 问题行保留 NaN，其他行正常"""
        result = calc_div(df_basic.copy(), 'ratio = a / b', errors='ignore')
        assert result['ratio'].iloc[0] == 2.0
        assert pd.isna(result['ratio'].iloc[2])  # 零分母行 → NaN
        assert pd.isna(result['ratio'].iloc[3])  # 0/0 → NaN
        assert pd.isna(result['ratio'].iloc[4])  # NaN 输入 → NaN

    def test_column_not_found(self, df_basic):
        """列不存在 → KeyError"""
        with pytest.raises(KeyError, match='missing'):
            calc_div(df_basic.copy(), 'ratio = missing / a')

    def test_multiindex(self, df_multiindex):
        """MultiIndex 正常计算"""
        result = calc_div(df_multiindex.copy(), 'ratio = a / b')
        assert result['ratio'].iloc[0] == 2.0

    def test_format_error_multiple_slash(self, df_basic):
        """格式错误：多个 / 号"""
        with pytest.raises(ValueError, match="多个"):
            calc_div(df_basic.copy(), 'bad = a / b / c')

    def test_errors_default_is_coerce(self, df_basic):
        """errors 默认值 = coerce（向后兼容）"""
        result_default = calc_div(df_basic.copy(), 'ratio = a / b')
        result_coerce = calc_div(df_basic.copy(), 'ratio = a / b', errors='coerce')
        pd.testing.assert_series_equal(result_default['ratio'], result_coerce['ratio'])

    def test_returns_same_dataframe(self, df_basic):
        """返回同一个 DataFrame 实例"""
        df = df_basic.copy()
        result = calc_div(df, 'ratio = a / b')
        assert result is df


# ──────────────────────────────────────────────
# 通用边界测试（跨函数）
# ──────────────────────────────────────────────

class TestCrossFunctionEdgeCases:
    """跨函数通用边界测试"""

    def test_all_functions_with_pure_numbers(self):
        """四个函数在纯数值 DataFrame 上结果正确"""
        df = pd.DataFrame({'a': [6.0], 'b': [3.0]})
        assert calc_add(df.copy(), 'r = a + b')['r'].iloc[0] == 9.0
        assert calc_sub(df.copy(), 'r = a - b')['r'].iloc[0] == 3.0
        assert calc_mul(df.copy(), 'r = a * b')['r'].iloc[0] == 18.0
        assert calc_div(df.copy(), 'r = a / b')['r'].iloc[0] == 2.0

    def test_all_functions_reject_invalid_errors(self):
        """四个函数统一拒绝非法 errors 参数"""
        df = pd.DataFrame({'a': [1.0], 'b': [2.0]})
        for func in [calc_add, calc_sub, calc_mul, calc_div]:
            with pytest.raises(ValueError, match="coerce"):
                func(df.copy(), 'r = a + b' if func != calc_sub else 'r = a - b',
                     errors='invalid')

    def test_spaces_in_expression(self):
        """表达式中空格随意添加"""
        df = pd.DataFrame({'a': [10.0], 'b': [5.0]})
        # 紧凑写法
        r1 = calc_add(df.copy(), 'sum=a+b')
        # 松散写法
        r2 = calc_add(df.copy(), 'sum = a + b')
        # 超松散写法
        r3 = calc_add(df.copy(), '  sum   =   a   +   b  ')
        assert r1['sum'].iloc[0] == 15.0
        assert r2['sum'].iloc[0] == 15.0
        assert r3['sum'].iloc[0] == 15.0


# ──────────────────────────────────────────────
# 混合运算引擎 calc() 测试
# ──────────────────────────────────────────────

class TestCalc:
    """calc() 混合运算引擎测试套件"""

    # ── 简单运算（等价于 helper） ──

    def test_simple_addition(self):
        """简单加法：等价于 calc_add"""
        df = pd.DataFrame({'a': [10.0, 20.0], 'b': [5.0, 4.0]})
        result = calc(df, 'sum = a + b')
        assert result['sum'].iloc[0] == 15.0
        assert result['sum'].iloc[1] == 24.0

    def test_simple_subtraction(self):
        """简单减法：等价于 calc_sub"""
        df = pd.DataFrame({'revenue': [100.0], 'cost': [40.0]})
        result = calc(df, 'margin = revenue - cost')
        assert result['margin'].iloc[0] == 60.0

    def test_simple_multiplication(self):
        """简单乘法：等价于 calc_mul"""
        df = pd.DataFrame({'qty': [10.0], 'price': [3.5]})
        result = calc(df, 'total = qty * price')
        assert result['total'].iloc[0] == 35.0

    def test_simple_division(self):
        """简单除法：等价于 calc_div"""
        df = pd.DataFrame({'a': [10.0], 'b': [5.0]})
        result = calc(df, 'ratio = a / b')
        assert result['ratio'].iloc[0] == 2.0

    # ── 混合运算（括号 + 多操作数） ──

    def test_parentheses_expression(self):
        """括号改变优先级：毛利率 = (revenue - cogs) / revenue"""
        df = pd.DataFrame({'revenue': [100.0, 200.0], 'cogs': [40.0, 80.0]})
        result = calc(df, 'gm_rate = (revenue - cogs) / revenue')
        assert result['gm_rate'].iloc[0] == 0.6
        assert result['gm_rate'].iloc[1] == 0.6

    def test_parentheses_with_format(self):
        """括号 + 百分比格式"""
        df = pd.DataFrame({'revenue': [100.0], 'cogs': [40.0]})
        result = calc(df, 'gm_pct = (revenue - cogs) / revenue >>> %')
        assert result['gm_pct'].iloc[0] == '60.00%'

    def test_complex_expression(self):
        """复杂表达式：(a - b) * c"""
        df = pd.DataFrame({'a': [10.0, 20.0], 'b': [5.0, 4.0], 'c': [2.0, 3.0]})
        result = calc(df, 'result = (a - b) * c')
        assert result['result'].iloc[0] == 10.0
        assert result['result'].iloc[1] == 48.0

    def test_operator_precedence(self):
        """运算优先级：a + b * c（先乘后加）"""
        df = pd.DataFrame({'a': [1.0], 'b': [2.0], 'c': [3.0]})
        result = calc(df, 'r = a + b * c')
        assert result['r'].iloc[0] == 7.0  # 1 + 2*3 = 7

    def test_nested_parentheses(self):
        """嵌套括号：((a + b) - c) / d"""
        df = pd.DataFrame({'a': [10.0], 'b': [5.0], 'c': [3.0], 'd': [2.0]})
        result = calc(df, 'r = ((a + b) - c) / d')
        assert result['r'].iloc[0] == 6.0  # (15 - 3) / 2 = 6.0

    # ── 数字常量 ──

    def test_numeric_constant(self):
        """数字常量作为操作数：tax = revenue * 0.13"""
        df = pd.DataFrame({'revenue': [100.0, 200.0]})
        result = calc(df, 'tax = revenue * 0.13')
        assert result['tax'].iloc[0] == 13.0
        assert result['tax'].iloc[1] == 26.0

    def test_integer_constant(self):
        """整数常量"""
        df = pd.DataFrame({'a': [10.0]})
        result = calc(df, 'r = a * 2')
        assert result['r'].iloc[0] == 20.0

    def test_constant_expression(self):
        """纯常量表达式"""
        df = pd.DataFrame({'a': [1.0]})
        result = calc(df, 'r = 1 + 2')
        assert result['r'].iloc[0] == 3.0

    # ── NaN 和零值处理 ──

    def test_nan_input_coerce(self):
        """NaN 输入 → 结果 NaN"""
        df = pd.DataFrame({'a': [10.0, np.nan], 'b': [5.0, 3.0]})
        result = calc(df, 'r = a + b')
        assert result['r'].iloc[0] == 15.0
        assert pd.isna(result['r'].iloc[1])

    def test_division_zero_denominator(self):
        """除法零值保护：分母为 0"""
        df = pd.DataFrame({'a': [10.0, 0.0, 0.0], 'b': [0.0, 0.0, 5.0]})
        result = calc(df, 'r = a / b')
        assert result['r'].iloc[0] == 0.0   # 非零/0 → 脏数据强制 0
        assert pd.isna(result['r'].iloc[1])  # 0/0 → NaN
        assert result['r'].iloc[2] == 0.0    # 0/5 = 0

    def test_nan_in_compound_expression(self):
        """混合表达式中 NaN 传播"""
        df = pd.DataFrame({'revenue': [100.0, np.nan], 'cogs': [40.0, 30.0]})
        result = calc(df, 'gm_rate = (revenue - cogs) / revenue')
        assert result['gm_rate'].iloc[0] == 0.6
        assert pd.isna(result['gm_rate'].iloc[1])

    # ── errors 参数 ──

    def test_errors_raise_nan(self):
        """errors='raise' → NaN 行抛异常"""
        df = pd.DataFrame({'a': [1.0, np.nan], 'b': [2.0, 3.0]})
        with pytest.raises(ValueError, match='NaN'):
            calc(df, 'r = a + b', errors='raise')

    def test_errors_raise_zero_denom(self):
        """errors='raise' → 零分母行抛异常"""
        df = pd.DataFrame({'a': [10.0], 'b': [0.0]})
        with pytest.raises(ValueError, match='零分母'):
            calc(df, 'r = a / b', errors='raise')

    def test_errors_raise_no_problems(self):
        """errors='raise' → 无问题时正常计算"""
        df = pd.DataFrame({'a': [10.0], 'b': [2.0]})
        result = calc(df, 'r = a / b', errors='raise')
        assert result['r'].iloc[0] == 5.0

    def test_errors_ignore(self):
        """errors='ignore' → 问题行保留 NaN"""
        df = pd.DataFrame({'a': [10.0, np.nan], 'b': [5.0, 3.0]})
        result = calc(df, 'r = a + b', errors='ignore')
        assert result['r'].iloc[0] == 15.0
        assert pd.isna(result['r'].iloc[1])

    def test_errors_invalid_value(self):
        """非法 errors 参数"""
        df = pd.DataFrame({'a': [1.0], 'b': [2.0]})
        with pytest.raises(ValueError, match="coerce"):
            calc(df, 'r = a + b', errors='bad')

    # ── 格式和 rounding ──

    def test_percentage_format(self):
        """百分比格式后缀"""
        df = pd.DataFrame({'a': [0.6]})
        result = calc(df, 'r = a >>> %')
        assert result['r'].iloc[0] == '60.00%'

    def test_rounding(self):
        """rounding 参数生效"""
        df = pd.DataFrame({'a': [1.0], 'b': [3.0]})
        result = calc(df, 'r = a / b', rounding=4)
        assert result['r'].iloc[0] == 0.3333

    # ── 安全性 ──

    def test_reject_function_call(self):
        """拒绝函数调用"""
        df = pd.DataFrame({'a': [1.0]})
        with pytest.raises(ValueError, match="不支持"):
            calc(df, 'r = __import__("os").system("ls")')

    def test_reject_attribute_access(self):
        """拒绝属性访问"""
        df = pd.DataFrame({'a': [1.0]})
        with pytest.raises(ValueError, match="不支持"):
            calc(df, 'r = a.__class__')

    def test_reject_subscript(self):
        """拒绝下标访问"""
        df = pd.DataFrame({'a': [1.0]})
        with pytest.raises(ValueError, match="不支持"):
            calc(df, 'r = a[0]')

    def test_reject_comparison(self):
        """拒绝比较运算符"""
        df = pd.DataFrame({'a': [1.0], 'b': [2.0]})
        with pytest.raises(ValueError, match="不支持"):
            calc(df, 'r = a > b')

    # ── 格式错误 ──

    def test_no_equals_sign(self):
        """无 = 号"""
        df = pd.DataFrame({'a': [1.0]})
        with pytest.raises(ValueError, match="'='"):
            calc(df, 'a + b')

    def test_empty_column_name(self):
        """空列名"""
        df = pd.DataFrame({'a': [1.0]})
        with pytest.raises(ValueError, match="新列名为空"):
            calc(df, ' = a + b')

    def test_empty_expression(self):
        """空计算表达式"""
        df = pd.DataFrame({'a': [1.0]})
        with pytest.raises(ValueError, match="计算表达式为空"):
            calc(df, 'r = ')

    def test_syntax_error_in_expression(self):
        """表达式语法错误"""
        df = pd.DataFrame({'a': [1.0]})
        with pytest.raises(ValueError, match="语法错误"):
            calc(df, 'r = a +')

    # ── 列不存在 ──

    def test_column_not_found(self):
        """列不存在 → KeyError"""
        df = pd.DataFrame({'a': [1.0]})
        with pytest.raises(KeyError, match='nonexistent'):
            calc(df, 'r = a + nonexistent')

    # ── MultiIndex ──

    def test_multiindex(self, df_multiindex):
        """MultiIndex DataFrame 正常计算"""
        result = calc(df_multiindex.copy(), 'sum = a + b')
        assert result['sum'].iloc[0] == 15.0
        assert result['sum'].iloc[3] == 42.0

    # ── 多步计算链 ──

    def test_chained_calculations(self):
        """多步计算链：先算 margin，再用 margin 算 margin_rate"""
        df = pd.DataFrame({'revenue': [100.0, 200.0], 'cost': [40.0, 80.0]})
        calc(df, 'margin = revenue - cost')
        calc(df, 'margin_rate = margin / revenue >>> %')
        assert df['margin'].iloc[0] == 60.0
        assert df['margin_rate'].iloc[0] == '60.00%'

    # ── 一元运算符 ──

    def test_unary_negative(self):
        """一元负号：-a"""
        df = pd.DataFrame({'a': [5.0, 10.0]})
        result = calc(df, 'neg = -a')
        assert result['neg'].iloc[0] == -5.0
        assert result['neg'].iloc[1] == -10.0

    def test_unary_positive(self):
        """一元正号：+a（恒等）"""
        df = pd.DataFrame({'a': [5.0]})
        result = calc(df, 'pos = +a')
        assert result['pos'].iloc[0] == 5.0

    def test_negative_constant(self):
        """负号常量：a * -1"""
        df = pd.DataFrame({'a': [5.0]})
        result = calc(df, 'r = a * -1')
        assert result['r'].iloc[0] == -5.0

    # ── 返回值 ──

    def test_returns_same_dataframe(self):
        """返回同一个 DataFrame 实例"""
        df = pd.DataFrame({'a': [1.0], 'b': [2.0]})
        result = calc(df, 'r = a + b')
        assert result is df


# ──────────────────────────────────────────────
# 拆解引擎 _calc_decompose 测试
# ──────────────────────────────────────────────

class TestCalcDecompose:
    """测试 _calc_decompose 拆解引擎。"""

    def test_simple_addition(self):
        """A = B + C → 1 步拆解。"""
        df = pd.DataFrame({"B": [10, 20], "C": [5, 8]})
        result = _calc_decompose(df, "A = B + C")
        assert isinstance(result, CalcResult)
        assert len(result.steps) == 1
        assert result.steps[0].operator == '+'
        assert "A" in result.df.columns
        assert list(result.df["A"]) == [15.0, 28.0]

    def test_complex_three_operand(self):
        """A = B + C / D → 2 步。"""
        df = pd.DataFrame({"B": [10, 20], "C": [6, 8], "D": [3, 4]})

        result = _calc_decompose(df, "A = B + C / D")
        assert len(result.steps) == 2
        # Step 1: C / D
        assert result.steps[0].operator == '/'
        # Step 2: B + __calc_tmp_1
        assert result.steps[1].operator == '+'
        # A = B + C/D = 10 + 2 = 12, 20 + 2 = 22
        assert list(result.df["A"]) == [12.0, 22.0]

    def test_parentheses(self):
        """A = (B + C) * D → 2 步。"""
        df = pd.DataFrame({"B": [1, 2], "C": [3, 4], "D": [5, 6]})

        result = _calc_decompose(df, "A = (B + C) * D")
        assert len(result.steps) == 2
        assert result.steps[0].operator == '+'
        assert result.steps[1].operator == '*'
        # (1+3)*5=20, (2+4)*6=36
        assert list(result.df["A"]) == [20.0, 36.0]

    def test_nested_parentheses(self):
        """A = (B + C) / (E - F) → 3 步。"""
        df = pd.DataFrame({"B": [10, 20], "C": [5, 10], "E": [3, 6], "F": [1, 2]})

        result = _calc_decompose(df, "A = (B + C) / (E - F)")
        assert len(result.steps) == 3
        # Step 1: B+C=15,30  Step 2: E-F=2,4  Step 3: 15/2=7.5, 30/4=7.5
        assert list(result.df["A"]) == [7.5, 7.5]

    def test_keep_tmp_true(self):
        """keep_tmp=True 保留临时列。"""
        df = pd.DataFrame({"B": [10, 20], "C": [5, 8]})

        result = _calc_decompose(df, "A = B + C", keep_tmp=True)
        tmp_cols = [c for c in result.df.columns if c.startswith("__calc_tmp_")]
        assert len(tmp_cols) > 0

    def test_keep_tmp_false_default(self):
        """默认清理临时列。"""
        df = pd.DataFrame({"B": [10, 20], "C": [5, 8]})
        result = _calc_decompose(df, "A = B + C")
        tmp_cols = [c for c in result.df.columns if c.startswith("__calc_tmp_")]
        assert len(tmp_cols) == 0

    def test_calc_api_unchanged(self):
        """calc() 公开 API 行为不变（回归测试）。"""
        df = pd.DataFrame({"revenue": [100, 200], "cogs": [40, 80]})
        result_df = calc(df, "gm = revenue - cogs")
        assert isinstance(result_df, pd.DataFrame)
        assert "gm" in result_df.columns
        assert list(result_df["gm"]) == [60.0, 120.0]

    def test_calc_result_object_structure(self):
        """CalcResult 对象结构验证。"""
        df = pd.DataFrame({"B": [10], "C": [5]})
        result = _calc_decompose(df, "A = B + C")
        assert isinstance(result, CalcResult)
        assert isinstance(result.df, pd.DataFrame)
        assert isinstance(result.steps, list)
        assert all(isinstance(s, CalcStep) for s in result.steps)
        assert isinstance(result.tmp_columns, list)


class TestCalchemyChain:
    """测试 Calchemy 链式调用类（Phase 4）。"""

    def test_basic_chain(self):
        """基础链式调用。"""
        df = pd.DataFrame({"B": [10, 20], "C": [5, 8]})
        result = Calchemy(df).calc("A = B + C").result()
        assert "A" in result.columns
        assert result["A"].tolist() == [15.0, 28.0]

    def test_does_not_modify_original_df(self):
        """原始 DataFrame 不被修改。"""
        df = pd.DataFrame({"B": [10, 20], "C": [5, 8]})
        original_cols = list(df.columns)
        Calchemy(df).calc("A = B + C").result()
        assert list(df.columns) == original_cols
        assert "A" not in df.columns

    def test_multi_step_chain(self):
        """多步链式调用。"""
        df = pd.DataFrame({"revenue": [100, 200], "cost": [40, 80]})
        result = (Calchemy(df)
                  .calc("margin = revenue - cost")
                  .calc("margin_rate = margin / revenue >>> %")
                  .result())
        assert "margin" in result.columns
        assert "margin_rate" in result.columns
        assert result["margin"].tolist() == [60.0, 120.0]

    def test_steps_accumulated(self):
        """steps 跨调用累积。"""
        df = pd.DataFrame({"B": [10, 20], "C": [5, 8]})
        chain = Calchemy(df).calc("A = B + C").calc("D = A * 2")
        assert len(chain.steps) > 0
        step_exprs = [s.expression for s in chain.steps]
        assert any("A" in e for e in step_exprs)

    def test_tmp_columns_with_keep_tmp(self):
        """keep_tmp=True 时 tmp_columns 累积。"""
        df = pd.DataFrame({"B": [10, 20], "C": [5, 8], "D": [2, 3]})
        chain = Calchemy(df).calc("A = B + C / D", keep_tmp=True)
        assert len(chain.tmp_columns) > 0
        # 验证临时列确实在 result 中
        for col in chain.tmp_columns:
            assert col in chain.result().columns

    def test_empty_chain(self):
        """不调用 calc() 直接 result()。"""
        df = pd.DataFrame({"B": [10, 20]})
        result = Calchemy(df).result()
        assert "B" in result.columns
        assert len(result.columns) == 1

    def test_lineage_single_col(self):
        """lineage 追踪单列血缘。"""
        df = pd.DataFrame({"revenue": [100, 200], "cost": [40, 80]})
        chain = (Calchemy(df)
                 .calc("margin = revenue - cost")
                 .calc("margin_rate = margin / revenue >>> %"))
        deps = chain.lineage("margin")
        assert "revenue" in deps
        assert "cost" in deps

    def test_lineage_all_cols(self):
        """lineage 返回所有列映射。"""
        df = pd.DataFrame({"revenue": [100, 200], "cost": [40, 80]})
        chain = Calchemy(df).calc("margin = revenue - cost")
        mapping = chain.lineage()
        assert "margin" in mapping
        assert "revenue" in mapping["margin"]
        assert "cost" in mapping["margin"]

    def test_chaining_with_parentheses(self):
        """括号表达式链式调用。"""
        df = pd.DataFrame({"revenue": [100, 200], "cogs": [40, 80]})
        result = (Calchemy(df)
                  .calc("gm_rate = (revenue - cogs) / revenue >>> %")
                  .result())
        assert result["gm_rate"].tolist() == ["60.00%", "60.00%"]

    def test_chaining_with_scalar(self):
        """常量参与链式调用。"""
        df = pd.DataFrame({"revenue": [100, 200]})
        result = (Calchemy(df)
                  .calc("tax = revenue * 0.13")
                  .result())
        assert result["tax"].tolist() == [13.0, 26.0]

    def test_parameter_rounding(self):
        """rounding 参数正确传递。"""
        df = pd.DataFrame({"B": [10, 20], "C": [3, 7]})
        result = Calchemy(df).calc("A = B / C", rounding=4).result()
        assert result["A"].tolist() == [round(10 / 3, 4), round(20 / 7, 4)]

    def test_parameter_errors_raise(self):
        """errors='raise' 参数正确传递。"""
        df = pd.DataFrame({"B": [10, 0], "C": [5, 0]})
        chain = Calchemy(df)
        with pytest.raises(ValueError):
            chain.calc("A = B / C", errors='raise')

    def test_result_returns_same_instance(self):
        """多次 result() 返回同一 DataFrame 实例。"""
        df = pd.DataFrame({"B": [10, 20], "C": [5, 8]})
        chain = Calchemy(df).calc("A = B + C")
        r1 = chain.result()
        r2 = chain.result()
        assert r1 is r2

    def test_chaining_returns_self(self):
        """calc() 返回 self 支持链式。"""
        df = pd.DataFrame({"B": [10, 20], "C": [5, 8]})
        chain = Calchemy(df)
        returned = chain.calc("A = B + C")
        assert returned is chain


# ──────────────────────────────────────────────
# Calchemy 链式调用类测试
# ──────────────────────────────────────────────

class TestCalchemyChain:
    """测试 Calchemy 链式调用类。"""

    def test_basic_chain(self):
        """基础链式调用。"""
        df = pd.DataFrame({"B": [10, 20], "C": [5, 8]})
        result = (Calchemy(df)
                  .calc("A = B + C")
                  .result())
        assert "A" in result.columns
        assert result["A"].tolist() == [15.0, 28.0]

    def test_does_not_modify_original_df(self):
        """原始 DataFrame 不被修改。"""
        df = pd.DataFrame({"B": [10, 20], "C": [5, 8]})
        original_cols = list(df.columns)
        (Calchemy(df)
         .calc("A = B + C")
         .result())
        assert list(df.columns) == original_cols
        assert "A" not in df.columns

    def test_multi_step_chain(self):
        """多步链式调用。"""
        df = pd.DataFrame({"revenue": [100, 200], "cost": [40, 80]})
        result = (Calchemy(df)
                  .calc("margin = revenue - cost")
                  .calc("margin_rate = margin / revenue >>> %")
                  .result())
        assert "margin" in result.columns
        assert "margin_rate" in result.columns
        assert result["margin"].tolist() == [60.0, 120.0]

    def test_steps_accumulated(self):
        """steps 跨调用累积。"""
        df = pd.DataFrame({"B": [10, 20], "C": [5, 8]})
        chain = (Calchemy(df)
                 .calc("A = B + C")
                 .calc("D = A * 2"))
        assert len(chain.steps) > 0
        # 第二步应该引用了 A
        step_exprs = [s.expression for s in chain.steps]
        assert any("A" in e for e in step_exprs)

    def test_tmp_columns_with_keep_tmp(self):
        """keep_tmp=True 时 tmp_columns 有值。"""
        df = pd.DataFrame({"B": [10, 20], "C": [5, 8], "D": [2, 3]})
        chain = (Calchemy(df)
                 .calc("A = B + C / D", keep_tmp=True))
        assert len(chain.tmp_columns) > 0

    def test_empty_chain(self):
        """不调用 calc() 直接 result()。"""
        df = pd.DataFrame({"B": [10, 20]})
        result = Calchemy(df).result()
        assert "B" in result.columns
        assert len(result.columns) == 1

    def test_lineage_single_col(self):
        """lineage 追踪单列血缘。"""
        df = pd.DataFrame({"revenue": [100, 200], "cost": [40, 80]})
        chain = (Calchemy(df)
                 .calc("margin = revenue - cost")
                 .calc("margin_rate = margin / revenue >>> %"))
        deps = chain.lineage("margin")
        assert "revenue" in deps
        assert "cost" in deps

    def test_lineage_all_cols(self):
        """lineage 返回所有列映射。"""
        df = pd.DataFrame({"revenue": [100, 200], "cost": [40, 80]})
        chain = (Calchemy(df)
                 .calc("margin = revenue - cost"))
        mapping = chain.lineage()
        assert "margin" in mapping
        assert "revenue" in mapping["margin"]
        assert "cost" in mapping["margin"]

    def test_chaining_with_parentheses(self):
        """括号表达式链式调用。"""
        df = pd.DataFrame({"revenue": [100, 200], "cogs": [40, 80]})
        result = (Calchemy(df)
                  .calc("gm_rate = (revenue - cogs) / revenue >>> %")
                  .result())
        assert result["gm_rate"].tolist() == ["60.00%", "60.00%"]

    def test_parameter_propagation(self):
        """rounding/format/errors 参数正确传递。"""
        df = pd.DataFrame({"B": [10, 20], "C": [3, 7]})
        result = (Calchemy(df)
                  .calc("A = B / C", rounding=4)
                  .result())
        assert result["A"].tolist() == [round(10/3, 4), round(20/7, 4)]


class TestExtendedOperators:
    """测试扩展运算符：**/^, abs, log, sqrt, root。"""

    def test_pow_basic(self):
        """基础指数运算 A = B ** 2。"""
        df = pd.DataFrame({"B": [2, 3, 4]})
        result = calc_pow(df, "A = B ** 2")
        assert result["A"].tolist() == [4.0, 9.0, 16.0]

    def test_pow_excel_style(self):
        """Excel 风格指数 A = B ^ 2。"""
        df = pd.DataFrame({"B": [2, 3, 4]})
        result = calc_pow(df, "A = B ^ 2")
        assert result["A"].tolist() == [4.0, 9.0, 16.0]

    def test_pow_calc_integration(self):
        """calc() 中支持 ** 运算。"""
        df = pd.DataFrame({"B": [2, 3], "C": [3, 2]})
        result = calc(df, "A = B ** C")
        assert result["A"].tolist() == [8.0, 9.0]

    def test_pow_zero_zero(self):
        """0 ** 0 → NaN。"""
        df = pd.DataFrame({"B": [0, 1], "C": [0, 0]})
        result = calc_pow(df, "A = B ** C")
        assert pd.isna(result["A"].iloc[0])
        assert result["A"].iloc[1] == 1.0

    def test_abs_basic(self):
        """基础绝对值 A = abs(B)。"""
        df = pd.DataFrame({"B": [-5, 0, 3]})
        result = calc_abs(df, "A = abs(B)")
        assert result["A"].tolist() == [5.0, 0.0, 3.0]

    def test_abs_calc_integration(self):
        """calc() 中支持 abs()。"""
        df = pd.DataFrame({"B": [-5, 3]})
        result = calc(df, "A = abs(B)")
        assert result["A"].tolist() == [5.0, 3.0]

    def test_log_natural(self):
        """自然对数 A = log(B)。"""
        df = pd.DataFrame({"B": [1, np.e, np.e ** 2]})
        result = calc_log(df, "A = log(B)")
        assert result["A"].iloc[0] == pytest.approx(0.0, abs=0.01)
        assert result["A"].iloc[1] == pytest.approx(1.0, abs=0.01)
        assert result["A"].iloc[2] == pytest.approx(2.0, abs=0.01)

    def test_log_base10(self):
        """以 10 为底的对数 A = log(B, 10)。"""
        df = pd.DataFrame({"B": [1, 10, 100]})
        result = calc_log(df, "A = log(B, 10)")
        assert result["A"].iloc[0] == pytest.approx(0.0, abs=0.01)
        assert result["A"].iloc[1] == pytest.approx(1.0, abs=0.01)
        assert result["A"].iloc[2] == pytest.approx(2.0, abs=0.01)

    def test_sqrt_basic(self):
        """基础平方根 A = sqrt(B)。"""
        df = pd.DataFrame({"B": [1, 4, 9]})
        result = calc_sqrt(df, "A = sqrt(B)")
        assert result["A"].tolist() == [1.0, 2.0, 3.0]

    def test_sqrt_negative(self):
        """负数平方根 → NaN。"""
        df = pd.DataFrame({"B": [-1, 4]})
        result = calc_sqrt(df, "A = sqrt(B)")
        assert pd.isna(result["A"].iloc[0])
        assert result["A"].iloc[1] == 2.0

    def test_root_cube(self):
        """立方根 A = root(B, 3)。"""
        df = pd.DataFrame({"B": [1, 8, 27]})
        result = calc_root(df, "A = root(B, 3)")
        assert result["A"].iloc[0] == pytest.approx(1.0, abs=0.01)
        assert result["A"].iloc[1] == pytest.approx(2.0, abs=0.01)
        assert result["A"].iloc[2] == pytest.approx(3.0, abs=0.01)

    def test_func_in_chain(self):
        """Calchemy 链式中使用函数。"""
        df = pd.DataFrame({"B": [-5, 3], "C": [2, 4]})
        result = (Calchemy(df)
                  .calc("A = abs(B)")
                  .calc("D = A + C")
                  .result())
        assert result["D"].tolist() == [7.0, 7.0]

    def test_pow_in_mixed_expr(self):
        """指数在混合表达式中 A = B ** 2 + C。"""
        df = pd.DataFrame({"B": [2, 3], "C": [1, 2]})
        result = calc(df, "A = B ** 2 + C")
        assert result["A"].tolist() == [5.0, 11.0]
