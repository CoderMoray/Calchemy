<p align="center">
  <a href="README_CN.md">
    <img src="https://img.shields.io/badge/语言-中文-blue?style=for-the-badge" alt="中文">
  </a>
  <a href="README.md">
    <img src="https://img.shields.io/badge/Language-English-slategray?style=for-the-badge" alt="English">
  </a>
</p>

<h1 align="center">⚗️ Calchemy</h1>

<p align="center">
  <strong>数据炼金术</strong> — 为 DataFrame 列运算设计的声明式 DSL<br>
  让业务方、LLM 和技术方说同一种语言
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/tests-124%20passing-brightgreen" alt="Tests">
  <a href="https://github.com/CoderMoray/Calchemy/actions/workflows/ci.yml">
    <img src="https://github.com/CoderMoray/Calchemy/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License">
</p>

---

## ✨ 一行表达式完成列计算

传统 pandas 代码对业务方不可读，也让 LLM 承担不必要的翻译成本。**Calchemy** 用一行自然语言风格的表达式完成列计算：

### 业务人员也能看懂的对比

Calchemy 的优势随**复杂度指数级放大**——公式越复杂，差距越夸张。

---

#### 第一层：简单加减 — pandas 还能凑合

**场景**：算毛利

```python
# pandas —— 勉强能读，但 df["..."] 已经是噪音
df["毛利"] = df["销售额"] - df["成本"]

# Calchemy —— 去掉噪音，直接是业务语言
calc(df, "毛利 = 销售额 - 成本")
```

> 差距不大，但 Calchemy 更干净。

---

#### 第二层：除法 + 格式化 — pandas 开始吃力

**场景**：算毛利率并显示为百分比

```python
# pandas —— 从 1 行变成 2 行，冒出 lambda 和 apply
df["毛利率"] = (df["销售额"] - df["成本"]) / df["销售额"]
df["毛利率"] = df["毛利率"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else x)

# Calchemy —— 一行搞定，格式是后缀，业务方直接能读
calc(df, "毛利率 = (销售额 - 成本) / 销售额 >>> %")
```

> pandas 暴露"技术细节噪音"，业务方开始困惑。**Calchemy 仍然像写公式。**

---

#### 第三层：复合指标 — pandas 变成天书

**场景**：算**复合健康度指标**（DAU × 留存率 - 获客成本）/ 收入

```python
# pandas —— 列名重复 6 次 df["..."]，业务方完全迷失
df["健康度"] = (df["DAU"] * df["留存率"] - df["获客成本"]) / df["收入"]
df["健康度"] = df["健康度"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else x)

# Calchemy —— 无论多复杂，始终一行自然语言
calc(df, "健康度 = (DAU * 留存率 - 获客成本) / 收入 >>> %")
```

> **pandas 的 `df["..."]` 随列数线性增长**，5 个字段就出现 6 次噪音；**Calchemy 保持自然语言密度不变**，无论多复杂都是一行业务公式。

---

#### 关键洞察

```
pandas 可读性 ≈ 公式复杂度 × df["..."] 重复次数
Calchemy 可读性 = 公式复杂度（常数，始终一行）
```

### 常见业务场景速查

| 业务需求 | Calchemy 表达式 |
|---------|----------------|
| 计算毛利 | `calc(df, "毛利 = 销售额 - 成本")` |
| 计算毛利率 | `calc(df, "毛利率 = (销售额 - 成本) / 销售额 >>> %")` |
| 计算增值税（13%） | `calc(df, "税额 = 销售额 * 0.13")` |
| 计算净利润 | `calc(df, "净利润 = 销售额 - 成本 - 税额")` |
| 计算同比增长率 | `calc(df, "同比增长 = (本年 - 去年) / 去年 >>> %")` |
| 计算客单价（GMV / 订单数） | `calc(df, "客单价 = GMV / 订单数")` |
| 计算人均产出 | `calc(df, "人均产出 = 总产出 / 人数")` |
| 计算标准差中的平方项 | `calc(df, "平方差 = (X - 均值) ** 2")` |
| 计算对数收益率 | `calc(df, "对数收益 = log(收盘价 / 昨收)")` |
| 计算 n 次方根 | `calc(df, "立方根 = root(X, 3)")` |

> 💡 字段名支持**中文**（`销售额`、`成本`）或**常见英文简写**（`GMV`、`COGS`、`DAU`），DSL 自动识别。

**Calchemy = Calc + Alchemy**。把原始数据"炼"成业务指标——这就是数据炼金术。

> *"Every DataFrame has gold in it. Calchemy helps you extract it."*

### 三方受益

| 角色 | 价值 |
|------|------|
| 🧑‍💼 **业务方** | 直接读懂计算逻辑，无需看 pandas 代码，可自行核对字段定义 |
| 🧑‍💻 **技术方** | 用 DSL 表达式替代重复性 pandas 模板代码，防御性处理内置 |
| 🤖 **LLM / AI** | 直接输出 DSL 表达式而非 pandas 代码，减少语法错误与调试往返 |

---

## 🚀 快速上手

### 安装

```bash
pip install calchemy
```

或直接将 `calchemy/` 目录复制到项目中，仅依赖 `pandas` 和 `numpy`。

### 30 秒示例

```python
import pandas as pd
from calchemy import calc

df = pd.DataFrame({
    "销售额": [100, 200, 0, 400],
    "成本":   [60,  150, 0, 300],
})

# 混合运算 + 百分比格式（字段名直接用中文！）
calc(df, "毛利 = 销售额 - 成本")
calc(df, "毛利率 = 毛利 / 销售额 >>> %")

print(df[["销售额", "成本", "毛利", "毛利率"]])
```

输出：

```
   销售额  成本  毛利   毛利率
0    100   60   40  40.00%
1    200  150   50  25.00%
2      0    0    0     nan
3    400  300  100  25.00%
```

---

## 📖 DSL 语法

### 基础格式

```
新列名 = 列A <运算符> 列B
新列名 = 列A <运算符> 列B >>> 格式
```

### 混合运算（支持括号）

```
gm_rate = (revenue - cogs) / revenue >>> %
tax = revenue * 0.13
```

**操作数**：列名（无引号）或数字常量
**运算符**：`+` `-` `*` `/` `**` `^`
**函数**：`abs(列名)` `log(列名)` `log(列名, 底数)` `sqrt(列名)`
**格式后缀**（`>>>` 后）：`%`/`pct`/`percent`/`百分比` → 百分比；省略 → 浮点数

### API 列表

| 函数 | 用途 | 示例 |
|------|------|------|
| `calc()` | 🌟 **混合运算引擎**（推荐） | `calc(df, "rate = (a - b) / c >>> %")` |
| `calc_add()` | 加法 | `calc_add(df, "total = a + b")` |
| `calc_sub()` | 减法 | `calc_sub(df, "margin = a - b")` |
| `calc_mul()` | 乘法 | `calc_mul(df, "gmv = qty * price")` |
| `calc_div()` | 除法（含零值保护） | `calc_div(df, "rate = a / b >>> %")` |

> 💡 简单两操作数运算可用 `calc_*` helper，混合运算请用 `calc()`。

---

## 🛡️ 错误处理（`errors` 参数）

所有函数均支持 `errors` 参数，命名对齐 `pd.to_datetime(errors=...)`：

| `errors` 值 | 行为 | 适用场景 |
|-------------|------|----------|
| `'coerce'`（默认） | NaN 和零分母 → `NaN`；脏数据 → `0` | 🏭 生产报表，容错优先 |
| `'raise'` | 遇零分母或 NaN 立刻抛 `ValueError` | 🔍 数据校验，严格模式 |
| `'ignore'` | 跳过问题行，保留原列值 | 🔄 幂等重跑，不覆盖已有结果 |

```python
# 严格模式：有问题直接报错
calc(df, "rate = a / b", errors='raise')

# 忽略模式：有问题跳过
calc(df, "rate = a / b", errors='ignore')
```

### NaN 处理语义

| 情况 | 结果 |
|------|------|
| 任一操作数为 NaN | `NaN` |
| 除法：0 / 0 | `NaN`（无意义） |
| 除法：非零 / 0 | `0`（脏数据强制归零，避免 ∞） |
| 0 × NaN | `NaN`（不做隐式填 0） |

---

## 🔒 安全性

`calc()` 使用 `ast.parse` 受限子集解析表达式：

- ✅ 允许：列名（`Name`）、数字常量（`Constant`）、四则运算（`BinOp`）、正负号（`UnaryOp`）、白名单函数（`abs`/`log`/`sqrt`/`root`）
- ❌ 禁止：任意函数调用、属性访问、下标、比较运算、`eval()`

```python
# ❌ 以下表达式会被拒绝
calc(df, 'r = __import__("os").system("ls")')  # 函数调用
calc(df, 'r = a.__class__')                     # 属性访问
calc(df, 'r = a > b')                           # 比较运算
```

---

## 📁 项目结构

```
calchemy/
├── calchemy/               # 包目录
│   ├── __init__.py         # 公共 API 入口
│   ├── types.py            # 数据结构：CalcStep, CalcResult
│   ├── utils.py            # 校验与格式化工具
│   ├── helpers.py          # 运算 helper（四则运算 + 扩展运算）
│   ├── parse.py            # AST 解析与拆解
│   ├── calc.py             # 混合运算引擎
│   └── chain.py            # 链式调用 API
├── tests/                  # 测试目录
│   ├── __init__.py
│   └── test_calchemy.py   # 测试套件（124 用例）
├── README.md               # 中文文档（本文件）
├── README_EN.md            # English documentation
└── .gitignore
```

---

## 🗺️ 开发路线

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | 除法 `calc_div` | ✅ 完成 |
| Phase 2 | 加减乘 helper + errors 参数统一 | ✅ 完成 |
| Phase 2.5 | 项目结构重组：标准 Python 包 | ✅ 完成 |
| Phase 3 | 混合运算引擎 `calc()`（括号 + 常量） | ✅ 完成 |
| Phase 3.5 | 拆解引擎 `_calc_decompose()`：逐步执行 + 血缘追踪 | ✅ 完成 |
| Phase 4 | 链式调用 `Calchemy` 类 | ✅ 完成（111 测试通过） |
| Phase 4.5 | 扩展运算符（指数 `**`/`^`、对数 `log`、绝对值 `abs`、平方根 `sqrt`、n次方根 `root`） | ✅ 完成（124 测试通过） |
| Phase 5 | LLM Function Calling schema + Skill 文档 | 🔲 规划中 |
| Phase 6 | 跨后端适配（polars / SQL） | 🔲 规划中 |
| Phase 7 | 指标注册表 + 血缘图谱 | 🔲 规划中 |

---

## 🎯 设计原则

1. **零隐式魔法**：NaN 处理语义必须通过 `errors` 参数显式声明
2. **列名即文档**：DSL 表达式本身就是业务逻辑的可读文档
3. **防御性优先**：类型检查先于运算，精确报错优于强制转换
4. **LLM 友好**：函数签名和 docstring 以 LLM 直接调用为目标
5. **跨后端可迁移**：同一套 DSL 可在不同后端执行（Phase 6）

---

## License

Apache 2.0
