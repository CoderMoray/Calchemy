"""
Calchemy 核心模块测试

覆盖全部四个 helper 函数：calc_add / calc_sub / calc_mul / calc_div
测试维度：正常计算、NaN 输入、零值、百分比格式、errors 三种模式、列不存在、MultiIndex、格式错误
"""

import pytest
import pandas as pd
import numpy as np

from calchemy import calc_add, calc_sub, calc_mul, calc_div, calc


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
