"""
PandasForLLM 核心模块测试

覆盖全部四个 helper 函数：pandas_add/subtract/multiply/divided_helper
测试维度：正常计算、NaN 输入、零值、百分比格式、errors 三种模式、列不存在、MultiIndex、格式错误
"""

import pytest
import pandas as pd
import numpy as np

from core import (
    pandas_add_helper,
    pandas_subtract_helper,
    pandas_multiply_helper,
    pandas_divided_helper,
)


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

class TestPandasAddHelper:
    """pandas_add_helper 测试套件"""

    def test_normal_calculation(self, df_basic):
        """正常加法计算"""
        result = pandas_add_helper(df_basic.copy(), 'sum = a + b')
        assert result['sum'].iloc[0] == 15.0
        assert result['sum'].iloc[1] == 24.0
        assert result['sum'].iloc[2] == 30.0
        assert result['sum'].iloc[3] == 0.0

    def test_nan_input_coerce(self, df_basic):
        """NaN 输入 → 结果 NaN（coerce 默认行为）"""
        result = pandas_add_helper(df_basic.copy(), 'sum = a + b')
        assert pd.isna(result['sum'].iloc[4])

    def test_zero_operands(self):
        """0 + 0 = 0"""
        df = pd.DataFrame({'a': [0.0], 'b': [0.0]})
        result = pandas_add_helper(df, 'sum = a + b')
        assert result['sum'].iloc[0] == 0.0

    def test_percentage_format(self, df_basic):
        """百分比格式后缀"""
        result = pandas_add_helper(df_basic.copy(), 'sum = a + b >>> %')
        assert result['sum'].iloc[0] == '1500.00%'
        # NaN 行应保持 NaN
        assert pd.isna(result['sum'].iloc[4])

    def test_errors_raise(self, df_basic):
        """errors='raise' → NaN 输入行抛出 ValueError"""
        with pytest.raises(ValueError, match='NaN'):
            pandas_add_helper(df_basic.copy(), 'sum = a + b', errors='raise')

    def test_errors_raise_no_nan(self):
        """errors='raise' → 无 NaN 时不抛异常"""
        df = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
        result = pandas_add_helper(df, 'sum = a + b', errors='raise')
        assert result['sum'].iloc[0] == 4.0

    def test_errors_ignore(self, df_basic):
        """errors='ignore' → NaN 行保留原值（NaN），其他行正常计算"""
        result = pandas_add_helper(df_basic.copy(), 'sum = a + b', errors='ignore')
        assert result['sum'].iloc[0] == 15.0
        assert pd.isna(result['sum'].iloc[4])

    def test_column_not_found(self, df_basic):
        """列不存在 → 友好 KeyError"""
        with pytest.raises(KeyError, match='x'):
            pandas_add_helper(df_basic.copy(), 'sum = x + y')

    def test_multiindex(self, df_multiindex):
        """MultiIndex DataFrame 正常计算"""
        result = pandas_add_helper(df_multiindex.copy(), 'sum = a + b')
        assert result['sum'].iloc[0] == 15.0
        assert result['sum'].iloc[3] == 42.0

    def test_format_error_no_equals(self, df_basic):
        """格式错误：无 = 号"""
        with pytest.raises(ValueError, match="'='"):
            pandas_add_helper(df_basic.copy(), 'bad format')

    def test_format_error_multiple_plus(self, df_basic):
        """格式错误：多个 + 号"""
        with pytest.raises(ValueError, match="多个"):
            pandas_add_helper(df_basic.copy(), 'bad = a + b + c')

    def test_errors_invalid_value(self, df_basic):
        """非法 errors 参数"""
        with pytest.raises(ValueError, match="coerce"):
            pandas_add_helper(df_basic.copy(), 'sum = a + b', errors='bad')

    def test_rounding(self):
        """rounding 参数生效"""
        df = pd.DataFrame({'a': [1.111], 'b': [2.222]})
        result = pandas_add_helper(df, 'sum = a + b', rounding=1)
        assert result['sum'].iloc[0] == 3.3

    def test_returns_same_dataframe(self, df_basic):
        """返回的是同一个 DataFrame 实例"""
        df = df_basic.copy()
        result = pandas_add_helper(df, 'sum = a + b')
        assert result is df


# ──────────────────────────────────────────────
# 减法测试
# ──────────────────────────────────────────────

class TestPandasSubtractHelper:
    """pandas_subtract_helper 测试套件"""

    def test_normal_calculation(self, df_basic):
        """正常减法计算"""
        result = pandas_subtract_helper(df_basic.copy(), 'diff = a - b')
        assert result['diff'].iloc[0] == 5.0
        assert result['diff'].iloc[1] == 16.0
        assert result['diff'].iloc[2] == 30.0
        assert result['diff'].iloc[3] == 0.0

    def test_negative_result(self):
        """减法允许负数结果"""
        df = pd.DataFrame({'a': [5.0], 'b': [10.0]})
        result = pandas_subtract_helper(df, 'diff = a - b')
        assert result['diff'].iloc[0] == -5.0

    def test_nan_input_coerce(self, df_basic):
        """NaN 输入 → 结果 NaN"""
        result = pandas_subtract_helper(df_basic.copy(), 'diff = a - b')
        assert pd.isna(result['diff'].iloc[4])

    def test_zero_operands(self):
        """0 - 0 = 0"""
        df = pd.DataFrame({'a': [0.0], 'b': [0.0]})
        result = pandas_subtract_helper(df, 'diff = a - b')
        assert result['diff'].iloc[0] == 0.0

    def test_percentage_format(self, df_basic):
        """百分比格式后缀"""
        result = pandas_subtract_helper(df_basic.copy(), 'diff = a - b >>> %')
        assert result['diff'].iloc[0] == '500.00%'

    def test_errors_raise(self, df_basic):
        """errors='raise' → NaN 行抛异常"""
        with pytest.raises(ValueError, match='NaN'):
            pandas_subtract_helper(df_basic.copy(), 'diff = a - b', errors='raise')

    def test_errors_ignore(self, df_basic):
        """errors='ignore' → NaN 行保留，其他行正常"""
        result = pandas_subtract_helper(df_basic.copy(), 'diff = a - b', errors='ignore')
        assert result['diff'].iloc[0] == 5.0
        assert pd.isna(result['diff'].iloc[4])

    def test_column_not_found(self, df_basic):
        """列不存在 → KeyError"""
        with pytest.raises(KeyError, match='nonexistent'):
            pandas_subtract_helper(df_basic.copy(), 'diff = nonexistent - a')

    def test_multiindex(self, df_multiindex):
        """MultiIndex 正常计算"""
        result = pandas_subtract_helper(df_multiindex.copy(), 'diff = a - b')
        assert result['diff'].iloc[0] == 5.0
        assert result['diff'].iloc[3] == 38.0

    def test_format_error_no_equals(self, df_basic):
        """格式错误：无 = 号"""
        with pytest.raises(ValueError, match="'='"):
            pandas_subtract_helper(df_basic.copy(), 'bad format')

    def test_format_error_multiple_minus(self, df_basic):
        """格式错误：多个 - 号"""
        with pytest.raises(ValueError, match="多个"):
            pandas_subtract_helper(df_basic.copy(), 'bad = a - b - c')

    def test_returns_same_dataframe(self, df_basic):
        """返回同一个 DataFrame 实例"""
        df = df_basic.copy()
        result = pandas_subtract_helper(df, 'diff = a - b')
        assert result is df


# ──────────────────────────────────────────────
# 乘法测试
# ──────────────────────────────────────────────

class TestPandasMultiplyHelper:
    """pandas_multiply_helper 测试套件"""

    def test_normal_calculation(self, df_basic):
        """正常乘法计算"""
        result = pandas_multiply_helper(df_basic.copy(), 'product = a * b')
        assert result['product'].iloc[0] == 50.0
        assert result['product'].iloc[1] == 80.0

    def test_multiply_by_zero(self, df_basic):
        """非零 × 0 = 0"""
        result = pandas_multiply_helper(df_basic.copy(), 'product = a * b')
        assert result['product'].iloc[2] == 0.0

    def test_zero_multiply_zero(self):
        """0 × 0 = 0"""
        df = pd.DataFrame({'a': [0.0], 'b': [0.0]})
        result = pandas_multiply_helper(df, 'product = a * b')
        assert result['product'].iloc[0] == 0.0

    def test_zero_multiply_nan(self):
        """0 × NaN = NaN（不做隐式填 0）"""
        df = pd.DataFrame({'a': [0.0], 'b': [np.nan]})
        result = pandas_multiply_helper(df, 'product = a * b')
        assert pd.isna(result['product'].iloc[0])

    def test_nan_input_coerce(self, df_basic):
        """NaN 输入 → 结果 NaN"""
        result = pandas_multiply_helper(df_basic.copy(), 'product = a * b')
        assert pd.isna(result['product'].iloc[4])

    def test_percentage_format(self, df_basic):
        """百分比格式后缀"""
        result = pandas_multiply_helper(df_basic.copy(), 'product = a * b >>> %')
        assert result['product'].iloc[0] == '5000.00%'

    def test_errors_raise(self, df_basic):
        """errors='raise' → NaN 行抛异常"""
        with pytest.raises(ValueError, match='NaN'):
            pandas_multiply_helper(df_basic.copy(), 'product = a * b', errors='raise')

    def test_errors_raise_no_nan(self):
        """errors='raise' → 无 NaN 时不抛异常"""
        df = pd.DataFrame({'a': [2.0], 'b': [3.0]})
        result = pandas_multiply_helper(df, 'product = a * b', errors='raise')
        assert result['product'].iloc[0] == 6.0

    def test_errors_ignore(self, df_basic):
        """errors='ignore' → NaN 行保留，其他行正常"""
        result = pandas_multiply_helper(df_basic.copy(), 'product = a * b', errors='ignore')
        assert result['product'].iloc[0] == 50.0
        assert pd.isna(result['product'].iloc[4])

    def test_column_not_found(self, df_basic):
        """列不存在 → KeyError"""
        with pytest.raises(KeyError, match='xyz'):
            pandas_multiply_helper(df_basic.copy(), 'product = xyz * a')

    def test_multiindex(self, df_multiindex):
        """MultiIndex 正常计算"""
        result = pandas_multiply_helper(df_multiindex.copy(), 'product = a * b')
        assert result['product'].iloc[0] == 50.0
        assert result['product'].iloc[3] == 80.0

    def test_format_error_multiple_star(self, df_basic):
        """格式错误：多个 * 号"""
        with pytest.raises(ValueError, match="多个"):
            pandas_multiply_helper(df_basic.copy(), 'bad = a * b * c')

    def test_returns_same_dataframe(self, df_basic):
        """返回同一个 DataFrame 实例"""
        df = df_basic.copy()
        result = pandas_multiply_helper(df, 'product = a * b')
        assert result is df


# ──────────────────────────────────────────────
# 除法测试
# ──────────────────────────────────────────────

class TestPandasDividedHelper:
    """pandas_divided_helper 测试套件（含新增 errors 参数）"""

    def test_normal_calculation(self, df_basic):
        """正常除法计算"""
        result = pandas_divided_helper(df_basic.copy(), 'ratio = a / b')
        assert result['ratio'].iloc[0] == 2.0
        assert result['ratio'].iloc[1] == 5.0

    def test_zero_numerator_zero_denominator(self, df_basic):
        """0 / 0 → NaN"""
        result = pandas_divided_helper(df_basic.copy(), 'ratio = a / b')
        assert pd.isna(result['ratio'].iloc[3])

    def test_nonzero_numerator_zero_denominator(self, df_basic):
        """非零 / 0 → 脏数据强制 0"""
        result = pandas_divided_helper(df_basic.copy(), 'ratio = a / b')
        assert result['ratio'].iloc[2] == 0.0

    def test_nan_input_coerce(self, df_basic):
        """NaN 输入 → 结果 NaN"""
        result = pandas_divided_helper(df_basic.copy(), 'ratio = a / b')
        assert pd.isna(result['ratio'].iloc[4])

    def test_percentage_format(self, df_basic):
        """百分比格式后缀"""
        result = pandas_divided_helper(df_basic.copy(), 'pct = a / b >>> %')
        assert result['pct'].iloc[0] == '200.00%'

    def test_errors_raise_with_nan(self, df_basic):
        """errors='raise' → NaN 输入行抛异常"""
        with pytest.raises(ValueError, match='NaN'):
            pandas_divided_helper(df_basic.copy(), 'ratio = a / b', errors='raise')

    def test_errors_raise_with_zero_denom(self):
        """errors='raise' → 零分母行抛异常"""
        df = pd.DataFrame({'a': [10.0, 0.0], 'b': [2.0, 0.0]})
        with pytest.raises(ValueError, match='零分母'):
            pandas_divided_helper(df, 'ratio = a / b', errors='raise')

    def test_errors_raise_no_problems(self):
        """errors='raise' → 无问题时正常计算"""
        df = pd.DataFrame({'a': [10.0], 'b': [2.0]})
        result = pandas_divided_helper(df, 'ratio = a / b', errors='raise')
        assert result['ratio'].iloc[0] == 5.0

    def test_errors_ignore(self, df_basic):
        """errors='ignore' → 问题行保留 NaN，其他行正常"""
        result = pandas_divided_helper(df_basic.copy(), 'ratio = a / b', errors='ignore')
        assert result['ratio'].iloc[0] == 2.0
        assert pd.isna(result['ratio'].iloc[2])  # 零分母行 → NaN
        assert pd.isna(result['ratio'].iloc[3])  # 0/0 → NaN
        assert pd.isna(result['ratio'].iloc[4])  # NaN 输入 → NaN

    def test_column_not_found(self, df_basic):
        """列不存在 → KeyError"""
        with pytest.raises(KeyError, match='missing'):
            pandas_divided_helper(df_basic.copy(), 'ratio = missing / a')

    def test_multiindex(self, df_multiindex):
        """MultiIndex 正常计算"""
        result = pandas_divided_helper(df_multiindex.copy(), 'ratio = a / b')
        assert result['ratio'].iloc[0] == 2.0

    def test_format_error_multiple_slash(self, df_basic):
        """格式错误：多个 / 号"""
        with pytest.raises(ValueError, match="多个"):
            pandas_divided_helper(df_basic.copy(), 'bad = a / b / c')

    def test_errors_default_is_coerce(self, df_basic):
        """errors 默认值 = coerce（向后兼容）"""
        result_default = pandas_divided_helper(df_basic.copy(), 'ratio = a / b')
        result_coerce = pandas_divided_helper(df_basic.copy(), 'ratio = a / b', errors='coerce')
        # 对比每一行的结果
        pd.testing.assert_series_equal(result_default['ratio'], result_coerce['ratio'])

    def test_returns_same_dataframe(self, df_basic):
        """返回同一个 DataFrame 实例"""
        df = df_basic.copy()
        result = pandas_divided_helper(df, 'ratio = a / b')
        assert result is df


# ──────────────────────────────────────────────
# 通用边界测试（跨函数）
# ──────────────────────────────────────────────

class TestCrossFunctionEdgeCases:
    """跨函数通用边界测试"""

    def test_all_functions_with_pure_numbers(self):
        """四个函数在纯数值 DataFrame 上结果正确"""
        df = pd.DataFrame({'a': [6.0], 'b': [3.0]})
        assert pandas_add_helper(df.copy(), 'r = a + b')['r'].iloc[0] == 9.0
        assert pandas_subtract_helper(df.copy(), 'r = a - b')['r'].iloc[0] == 3.0
        assert pandas_multiply_helper(df.copy(), 'r = a * b')['r'].iloc[0] == 18.0
        assert pandas_divided_helper(df.copy(), 'r = a / b')['r'].iloc[0] == 2.0

    def test_all_functions_reject_invalid_errors(self):
        """四个函数统一拒绝非法 errors 参数"""
        df = pd.DataFrame({'a': [1.0], 'b': [2.0]})
        for func in [pandas_add_helper, pandas_subtract_helper,
                     pandas_multiply_helper, pandas_divided_helper]:
            with pytest.raises(ValueError, match="coerce"):
                func(df.copy(), 'r = a + b' if func != pandas_subtract_helper else 'r = a - b',
                     errors='invalid')

    def test_spaces_in_expression(self):
        """表达式中空格随意添加"""
        df = pd.DataFrame({'a': [10.0], 'b': [5.0]})
        # 紧凑写法
        r1 = pandas_add_helper(df.copy(), 'sum=a+b')
        # 松散写法
        r2 = pandas_add_helper(df.copy(), 'sum = a + b')
        # 超松散写法
        r3 = pandas_add_helper(df.copy(), '  sum   =   a   +   b  ')
        assert r1['sum'].iloc[0] == 15.0
        assert r2['sum'].iloc[0] == 15.0
        assert r3['sum'].iloc[0] == 15.0
