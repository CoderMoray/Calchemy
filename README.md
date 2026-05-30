<p align="center">
  <a href="README.md">
    <img src="https://img.shields.io/badge/语言-中文-blue?style=for-the-badge" alt="中文">
  </a>
  <a href="README_EN.md">
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
  <img src="https://img.shields.io/badge/tests-101%20passing-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License">
</p>

---

## ✨ 一行表达式完成列计算

传统 pandas 代码对业务方不可读，也让 LLM 承担不必要的翻译成本。**Calchemy** 用一行自然语言风格的表达式完成列计算：

```python
# 传统 pandas —— 业务方看不懂，LLM 容易写错
df["gm_rate"] = (df["revenue"] - df["cogs"]) / df["revenue"]
df["gm_rate"] = df["gm_rate"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else x)

# Calchemy —— 业务方能读懂，LLM 直接输出
calc(df, "gm_rate = (revenue - cogs) / revenue >>> %")
```

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
    "revenue": [100, 200, 0, 400],
    "cogs":    [60,  150, 0, 300],
})

# 混合运算 + 百分比格式
calc(df, "margin = revenue - cogs")
calc(df, "margin_rate = margin / revenue >>> %")

print(df[["revenue", "cogs", "margin", "margin_rate"]])
```

输出：

```
   revenue  cogs  margin margin_rate
0      100    60      40       40.00%
1      200   150      50       25.00%
2        0     0       0         nan
3      400   300     100       25.00%
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

- ✅ 允许：列名（`Name`）、数字常量（`Constant`）、四则运算（`BinOp`）、正负号（`UnaryOp`）
- ❌ 禁止：函数调用、属性访问、下标、比较运算、`eval()`

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
│   └── calchemy.py         # 核心 DSL 实现
├── tests/                  # 测试目录
│   ├── __init__.py
│   └── test_calchemy.py   # 测试套件（101 用例）
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
