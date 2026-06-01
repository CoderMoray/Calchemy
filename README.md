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
  <strong>Data Alchemy</strong> — A declarative DSL for DataFrame column calculations<br>
  Bridging humans, LLMs, and data with one language
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

## ✨ One Expression, One Calculation

Traditional pandas code is unreadable to business users and error-prone for LLMs. **Calchemy** uses a single natural-language-style expression to perform column calculations:

### Side-by-Side Comparison

Calchemy's advantage grows **exponentially with complexity** — the more complex the formula, the more dramatic the gap.

---

#### Level 1: Simple addition/subtraction — pandas is still okay

**Scenario**: Calculate gross profit

```python
# pandas — barely readable, but df["..."] is already noise
df["gross_profit"] = df["revenue"] - df["cogs"]

# Calchemy — removes the noise, reads like business language
calc(df, "gross_profit = revenue - cogs")
```

> The gap is small, but Calchemy is cleaner.

---

#### Level 2: Division + formatting — pandas starts to struggle

**Scenario**: Calculate gross margin and display as percentage

```python
# pandas — 1 line becomes 2, lambda and apply appear
df["gm_rate"] = (df["revenue"] - df["cogs"]) / df["revenue"]
df["gm_rate"] = df["gm_rate"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else x)

# Calchemy — one line, format is a suffix, instantly readable
calc(df, "gm_rate = (revenue - cogs) / revenue >>> %")
```

> pandas exposes "technical detail noise"; business users start to get confused. **Calchemy still reads like a formula.**

---

#### Level 3: Composite metric — pandas becomes gibberish

**Scenario**: Calculate **composite health score** (DAU × retention - CAC) / revenue

```python
# pandas — column names repeated 6× with df["..."], business users are lost
df["health"] = (df["DAU"] * df["retention"] - df["CAC"]) / df["revenue"]
df["health"] = df["health"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else x)

# Calchemy — no matter how complex, always one line of natural language
calc(df, "health = (DAU * retention - CAC) / revenue >>> %")
```

> **pandas' `df["..."]` grows linearly with column count** — 5 fields produce 6 repetitions of noise; **Calchemy maintains constant natural-language density**, always one line of business formula.

---

#### Key insight

```
pandas readability ≈ formula complexity × df["..."] repetition count
Calchemy readability = formula complexity (constant, always one line)
```

### Common Business Scenarios

| Business Need | Calchemy Expression |
|---------------|---------------------|
| Calculate gross profit | `calc(df, "gross_profit = revenue - cogs")` |
| Calculate gross margin | `calc(df, "gm_rate = (revenue - cogs) / revenue >>> %")` |
| Calculate VAT (13%) | `calc(df, "tax = revenue * 0.13")` |
| Calculate net profit | `calc(df, "net_profit = revenue - cogs - tax")` |
| Calculate YoY growth | `calc(df, "yoy = (this_year - last_year) / last_year >>> %")` |
| Calculate ARPU / AOV | `calc(df, "aov = GMV / orders")` |
| Calculate per-capita output | `calc(df, "per_capita = total_output / headcount")` |
| Calculate squared deviation | `calc(df, "squared = (X - mean) ** 2")` |
| Calculate log return | `calc(df, "log_return = log(close / prev_close)")` |
| Calculate n-th root | `calc(df, "cuberoot = root(X, 3)")` |

> 💡 Column names support **Chinese** (`销售额`, `成本`) or **common English abbreviations** (`GMV`, `COGS`, `DAU`). The DSL handles both seamlessly.

**Calchemy = Calc + Alchemy**. Turn raw data into business metrics — that's data alchemy.

> *"Every DataFrame has gold in it. Calchemy helps you extract it."*

### Benefits for Everyone

| Role | Benefit |
|------|---------|
| 🧑‍💼 **Business Users** | Read calculation logic directly — no pandas knowledge needed |
| 🧑‍💻 **Developers** | Replace repetitive pandas boilerplate — defensive handling built in |
| 🤖 **LLMs / AI** | Output DSL expressions instead of pandas code — fewer syntax errors and debug loops |

---

## 🚀 Quick Start

### Install

```bash
pip install calchemy
```

Or copy the `calchemy/` directory into your project. Only `pandas` and `numpy` are required.

### 30-Second Example

```python
import pandas as pd
from calchemy import calc

df = pd.DataFrame({
    "revenue": [100, 200, 0, 400],
    "cogs":    [60,  150, 0, 300],
})

# Compound expression + percentage format
calc(df, "margin = revenue - cogs")
calc(df, "margin_rate = margin / revenue >>> %")

print(df[["revenue", "cogs", "margin", "margin_rate"]])
```

Output:

```
   revenue  cogs  margin margin_rate
0      100    60      40       40.00%
1      200   150      50       25.00%
2        0     0       0         nan
3      400   300     100       25.00%
```

---

## 📖 DSL Syntax

### Basic Format

```
new_col = col_A <operator> col_B
new_col = col_A <operator> col_B >>> format
```

### Compound Expressions (with parentheses)

```
gm_rate = (revenue - cogs) / revenue >>> %
tax = revenue * 0.13
```

**Operands**: column names (unquoted) or numeric constants
**Operators**: `+` `-` `*` `/` `**` `^`
**Functions**: `abs(col)` `log(col)` `log(col, base)` `sqrt(col)`
**Format suffix** (after `>>>`): `%`/`pct`/`percent` → percentage; omitted → float

### API Reference

| Function | Purpose | Example |
|----------|---------|---------|
| `calc()` | 🌟 **Compound expression engine** (recommended) | `calc(df, "rate = (a - b) / c >>> %")` |
| `calc_add()` | Addition | `calc_add(df, "total = a + b")` |
| `calc_sub()` | Subtraction | `calc_sub(df, "margin = a - b")` |
| `calc_mul()` | Multiplication | `calc_mul(df, "gmv = qty * price")` |
| `calc_div()` | Division (with zero-protection) | `calc_div(df, "rate = a / b >>> %")` |

> 💡 Use `calc_*` helpers for simple two-operand operations; use `calc()` for compound expressions.

---

## 🛡️ Error Handling (`errors` Parameter)

All functions support the `errors` parameter, named after `pd.to_datetime(errors=...)`:

| `errors` Value | Behavior | Use Case |
|----------------|----------|----------|
| `'coerce'` (default) | NaN & zero-denom → `NaN`; dirty data → `0` | 🏭 Production reports, fault-tolerant |
| `'raise'` | Raise `ValueError` on zero-denom or NaN | 🔍 Data validation, strict mode |
| `'ignore'` | Skip problem rows, preserve original values | 🔄 Idempotent re-runs |

```python
# Strict mode: fail fast on problems
calc(df, "rate = a / b", errors='raise')

# Ignore mode: skip problems
calc(df, "rate = a / b", errors='ignore')
```

### NaN Handling Semantics

| Case | Result |
|------|--------|
| Either operand is NaN | `NaN` |
| Division: 0 / 0 | `NaN` (undefined) |
| Division: non-zero / 0 | `0` (dirty data forced to zero, avoiding ∞) |
| 0 × NaN | `NaN` (no implicit fill) |

---

## 🔒 Security

`calc()` uses a restricted subset of `ast.parse` to evaluate expressions:

- ✅ Allowed: column names (`Name`), numeric constants (`Constant`), arithmetic (`BinOp`), unary +/- (`UnaryOp`), whitelisted functions (`abs`, `log`, `sqrt`, `root`)
- ❌ Rejected: arbitrary function calls, attribute access, subscripting, comparisons, `eval()`

```python
# ❌ These expressions will be rejected
calc(df, 'r = __import__("os").system("ls")')  # Function call
calc(df, 'r = a.__class__')                     # Attribute access
calc(df, 'r = a > b')                           # Comparison
```

---

## 📁 Project Structure

```
calchemy/
├── calchemy/               # Package directory
│   ├── __init__.py         # Public API entry point
│   ├── types.py            # Data structures: CalcStep, CalcResult
│   ├── utils.py            # Validation & formatting utilities
│   ├── helpers.py          # Calculation helpers (arithmetic + extended ops)
│   ├── parse.py            # AST parsing & decomposition
│   ├── calc.py             # Compound expression engine
│   └── chain.py            # Chain-style API
├── tests/                  # Test directory
│   ├── __init__.py
│   └── test_calchemy.py   # Test suite (124 cases)
├── README.md               # Chinese documentation
├── README_EN.md            # English documentation (this file)
└── .gitignore
```

---

## 🗺️ Roadmap

| Phase | Content | Status |
|-------|---------|--------|
| Phase 1 | Division `calc_div` | ✅ Done |
| Phase 2 | Add/sub/mul helpers + unified `errors` param | ✅ Done |
| Phase 2.5 | Project restructuring: standard Python package | ✅ Done |
| Phase 3 | Compound expression engine `calc()` (parentheses + constants) | ✅ Done |
| Phase 3.5 | Decompose engine `_calc_decompose()`: step-by-step execution + lineage tracking | ✅ Done |
| Phase 4 | Chain-style API: `Calchemy` class | ✅ Done (111 tests) |
| Phase 4.5 | Extended operators (exponent `**`/`^`, logarithm `log`, absolute `abs`, square root `sqrt`, n-th root `root`) | ✅ Done (124 tests) |
| Phase 5 | LLM Function Calling schema + Skill docs | 🔲 Planned |
| Phase 6 | Multi-backend support (polars / SQL) | 🔲 Planned |
| Phase 7 | Metric registry + lineage graph | 🔲 Planned |

---

## 🎯 Design Principles

1. **No Implicit Magic**: NaN handling must be explicitly declared via the `errors` parameter
2. **Column Names as Documentation**: DSL expressions ARE the readable documentation for business logic
3. **Defensive First**: Type checking before operations; precise errors over silent coercion
4. **LLM-Friendly**: Function signatures and docstrings designed for direct LLM invocation
5. **Backend-Portable**: Same DSL runs on different backends (Phase 6)

---

## License

Apache 2.0
