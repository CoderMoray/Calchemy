# PandasForLLM

> 为 DataFrame 列运算设计的声明式 DSL，让业务方、LLM 和技术方说同一种语言。

---

## 项目定位

传统 pandas 代码对业务方不可读，也让 LLM 在生成代码时承担不必要的翻译成本。`PandasForLLM` 提供一个轻量的**声明式计算 DSL（领域特定语言）**，用一行自然语言风格的表达式完成列计算：

```
gross_margin_rate = (revenue - cogs) / revenue >>> %
```

### 三方受益

| 角色 | 价值 |
|------|------|
| **业务方** | 直接读懂计算逻辑，无需看 pandas 代码，可自行核对字段定义 |
| **技术方** | 用 DSL 表达式替代重复性 pandas 模板代码，防御性处理内置 |
| **LLM / AI** | 直接输出 DSL 表达式而非 pandas 代码，减少语法错误与调试往返 |

---

## DSL 语法

### 基础语法

```
新列名 = 列A <运算符> 列B
```

### 带格式后缀

```
新列名 = 列A <运算符> 列B >>> 格式
```

### 支持的运算符

| 运算符 | 含义 | 示例 |
|--------|------|------|
| `/` | 除法（含零值保护） | `rate = profit / revenue` |
| `+` | 加法 | `total = base + bonus` |
| `-` | 减法 | `margin = revenue - cost` |
| `*` | 乘法 | `gmv = qty * unit_price` |

### 支持的格式后缀（`>>>` 后）

| 格式标识 | 含义 | 输出示例 |
|----------|------|----------|
| `%` / `pct` / `percent` / `percentage` / `百分比` | 百分比字符串 | `"65.43%"` |
| 省略 / `float` | 浮点数 | `0.6543` |

---

## 安装与依赖

```bash
pip install pandas numpy
```

直接复制 `core.py` 到项目即可使用，无额外依赖。

---

## 快速上手

```python
import pandas as pd
from core import pandas_divided_helper

df = pd.DataFrame({
    "revenue": [100, 200, 0, 400],
    "cost":    [60,  150, 0, 300],
})

# 除法 + 百分比格式
df = pandas_divided_helper(df, "margin_rate = revenue / cost >>> %")
print(df["margin_rate"])
# 0    166.67%
# 1    133.33%
# 2       None
# 3    133.33%
```

---

## 错误处理（`errors` 参数）

所有 helper 函数均支持 `errors` 参数，控制异常数据的处理方式：

| `errors` 值 | 行为 | 适用场景 |
|-------------|------|----------|
| `'coerce'`（默认） | NaN 和零分母 → `NaN`；脏数据（分子非0但分母0）→ `0` | 生产报表，容错优先 |
| `'raise'` | 遇到零分母或 NaN 立刻抛出 `ValueError`，附带行索引信息 | 数据校验，严格模式 |
| `'ignore'` | 跳过问题行，保留原列值 | 幂等重跑，不覆盖已有结果 |

> 命名与 `pd.to_datetime(errors=...)` 保持一致。

---

## 项目结构

```
PandasForLLM/
├── core.py               # 核心 DSL 实现（四则运算 helper + calc() 引擎）
├── README.md             # 本文件
└── .workbuddy/
    └── memory/           # 项目知识库（架构设计、开发标准、进度记录）
        ├── MEMORY.md     # 长期项目笔记与知识沉淀
        └── YYYY-MM-DD.md # 每日开发日志（append-only）
```

---

## 开发路线

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | 除法 `pandas_divided_helper` | ✅ 完成 |
| Phase 2 | 加减乘三个 helper 补全 | 🔲 待开发 |
| Phase 3 | 混合运算引擎 `calc()`，支持括号和链式计算 | 🔲 待开发 |
| Phase 4 | LLM Function Calling schema + Skill 文档 | 🔲 待开发 |
| Phase 5 | 跨后端适配（polars / SQL） | 🔲 规划中 |
| Phase 6 | 指标注册表 + 血缘图谱 | 🔲 规划中 |

详见 [`.workbuddy/memory/`](.workbuddy/memory/) 目录下的项目知识库

---

## 设计原则

1. **零隐式魔法**：NaN 处理语义必须通过 `errors` 参数显式声明，不做隐式填充
2. **列名即文档**：DSL 表达式本身就是业务逻辑的文档
3. **防御性优先**：类型检查先于运算，精确报错优于强制转换
4. **LLM 友好**：函数签名和 docstring 设计以 LLM 直接调用为目标
