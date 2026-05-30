#!/usr/bin/env python3
"""
Calchemy Code-Map Auto-Updater
===============================

自动扫描 calchemy/ 目录，用 ast 模块提取结构信息，
更新 .workbuddy/code-map.md 的「模块清单」和「依赖关系图」部分。

保留 AI 写的语义部分（快速定位表、新增模块时）不变。

用法：
    python scripts/update_code_map.py [--project-dir /path/to/Calchemy]
"""

import ast
import os
import re
import sys
from pathlib import Path


# ── 配置 ──────────────────────────────────────────────────────────────

PKG_DIR = "calchemy"
CODE_MAP_PATH = ".workbuddy/code-map.md"
TEST_DIR = "tests"

# 模块职责描述（需手动维护，脚本无法自动推断语义）
MODULE_DESCRIPTIONS = {
    "types.py": "数据结构定义",
    "utils.py": "校验与格式化工具",
    "helpers.py": "四则运算 helper",
    "parse.py": "AST 解析与拆解",
    "calc.py": "混合运算引擎",
    "chain.py": "链式调用 API",
    "__init__.py": "统一 re-export",
}


# ── AST 扫描 ─────────────────────────────────────────────────────────

def scan_module(filepath: str) -> dict:
    """扫描单个 Python 文件，提取 exports 和内部 imports。"""
    with open(filepath, encoding="utf-8") as f:
        source = f.read()
    lines = source.count("\n") + 1
    tree = ast.parse(source)

    exports = []
    internal_imports = []

    for node in ast.walk(tree):
        # 提取顶层函数和类
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("__"):
                exports.append(node.name)
        elif isinstance(node, ast.ClassDef):
            exports.append(node.name)

        # 提取内部 import（from calchemy.xxx import ...）
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("calchemy"):
                internal_imports.append(node.module)

    # 去重并排序
    internal_imports = sorted(set(internal_imports))

    # 提取模块级别的常量赋值（如 _BINOP_MAP = ...）
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("__"):
                    exports.append(target.id)

    # 去重排序，函数/类在前，常量在后
    func_exports = [e for e in exports if not e.isupper() and not e.startswith("_")]
    const_exports = [e for e in exports if e.isupper() or (e.startswith("_") and not e.startswith("__"))]
    all_exports = func_exports + const_exports

    return {
        "lines": lines,
        "exports": all_exports,
        "internal_imports": internal_imports,
    }


def scan_package(pkg_path: str) -> dict:
    """扫描整个包目录。"""
    modules = {}
    for fname in sorted(os.listdir(pkg_path)):
        if not fname.endswith(".py"):
            continue
        fpath = os.path.join(pkg_path, fname)
        info = scan_module(fpath)
        info["description"] = MODULE_DESCRIPTIONS.get(fname, "")
        modules[fname] = info
    return modules


def scan_tests(test_dir: str) -> dict:
    """扫描测试目录，统计行数和测试类。"""
    total_lines = 0
    test_classes = []
    for fname in sorted(os.listdir(test_dir)):
        if not fname.endswith(".py") or fname == "__init__.py":
            continue
        fpath = os.path.join(test_dir, fname)
        with open(fpath, encoding="utf-8") as f:
            source = f.read()
        total_lines += source.count("\n") + 1
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                # 统计测试方法数
                methods = [
                    n.name for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and n.name.startswith("test_")
                ]
                test_classes.append({"name": node.name, "methods": len(methods)})
    return {"lines": total_lines, "classes": test_classes}


# ── Markdown 生成 ────────────────────────────────────────────────────

def format_exports(exports: list[str], max_shown: int = 5) -> str:
    """格式化导出列表，超过 max_shown 的截断。"""
    if not exports:
        return "—"
    formatted = ", ".join(f"`{e}`" for e in exports[:max_shown])
    if len(exports) > max_shown:
        formatted += f", ...（共 {len(exports)} 个）"
    return formatted


def format_deps(internal_imports: list[str]) -> str:
    """从内部 import 提取依赖模块名。"""
    deps = []
    for imp in internal_imports:
        # calchemy.helpers → helpers
        parts = imp.split(".")
        if len(parts) >= 2:
            dep_name = parts[1] + ".py"
            if dep_name not in deps:
                deps.append(dep_name)
    return ", ".join(f"`{d.replace('.py', '')}`" for d in deps) if deps else "—"


def generate_module_table(modules: dict) -> str:
    """生成模块清单 Markdown 表格。"""
    rows = []
    for fname, info in modules.items():
        if fname == "__init__.py":
            desc = MODULE_DESCRIPTIONS.get("__init__.py", "统一 re-export")
            rows.append(
                f"| `calchemy/__init__.py` | {desc} | ~{info['lines']} | 所有公共 API | 所有模块 |"
            )
        else:
            rows.append(
                f"| `calchemy/{fname}` | {info['description']} | ~{info['lines']} "
                f"| {format_exports(info['exports'])} | {format_deps(info['internal_imports'])} |"
            )
    return "\n".join(rows)


def generate_dependency_graph(modules: dict) -> str:
    """生成依赖关系图（ASCII + 文字描述）。"""
    # 构建依赖映射
    dep_map = {}
    for fname, info in modules.items():
        if fname == "__init__.py":
            continue
        deps = []
        for imp in info["internal_imports"]:
            parts = imp.split(".")
            if len(parts) >= 2:
                dep = parts[1] + ".py"
                if dep != fname and dep not in deps:
                    deps.append(dep)
        if deps:
            dep_map[fname] = deps

    # 文字描述
    lines = []
    for fname, deps in sorted(dep_map.items()):
        dep_strs = [f"`{d}`" for d in deps]
        lines.append(f"- `{fname}` → {', '.join(dep_strs)}")

    return "\n".join(lines)


# ── Code-Map 更新 ────────────────────────────────────────────────────

def update_code_map(project_dir: str, modules: dict, test_info: dict) -> bool:
    """
    更新 .workbuddy/code-map.md，替换「模块清单」和「依赖关系图」部分，
    保留其他部分（快速定位表、新增模块时）不变。
    """
    code_map_path = os.path.join(project_dir, CODE_MAP_PATH)

    if not os.path.exists(code_map_path):
        print(f"⚠️  {code_map_path} 不存在，将创建新文件")
        return create_code_map(code_map_path, modules, test_info)

    with open(code_map_path, encoding="utf-8") as f:
        content = f.read()

    # 生成新内容
    new_module_table = generate_module_table(modules)
    new_dep_graph = generate_dependency_graph(modules)

    # 替换模块清单表格（## 模块清单 到 ## 依赖关系图 之间的表格）
    content = replace_section_content(
        content,
        "模块清单",
        lambda: (
            f"\n\n| 文件 | 职责 | 行数 | 导出函数/类 | 依赖 |\n"
            f"|------|------|------|------------|------|\n"
            f"{new_module_table}\n"
            f"| `tests/test_calchemy.py` | 测试 | ~{test_info['lines']} | — | `calchemy` |\n"
        ),
    )

    # 替换依赖关系图
    content = replace_section_content(
        content,
        "依赖关系图",
        lambda: f"\n\n完整依赖链：\n{new_dep_graph}\n",
    )

    with open(code_map_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"✅ {code_map_path} 已更新")
    return True


def replace_section_content(
    content: str, section_title: str, content_generator
) -> str:
    """
    替换 Markdown 中某个 ## section 的内容，
    保留标题行，替换标题到下一个 ## 之间的内容。
    """
    pattern = rf"(## {re.escape(section_title)}\n)"
    match = re.search(pattern, content)
    if not match:
        print(f"⚠️  未找到 section「{section_title}」，跳过")
        return content

    start = match.end()

    # 找到下一个 ## 标题或文件末尾
    next_section = re.search(r"\n## ", content[start:])
    if next_section:
        end = start + next_section.start()
    else:
        end = len(content)

    new_content = content_generator()
    return content[:start] + new_content + content[end:]


def create_code_map(path: str, modules: dict, test_info: dict) -> bool:
    """创建全新的 code-map.md。"""
    # ... (完整创建逻辑，此处省略，因为已有文件)
    print("请先手动创建 code-map.md 或运行完整版脚本")
    return False


# ── 主流程 ────────────────────────────────────────────────────────────

def main():
    project_dir = os.environ.get("CALCHEMY_DIR", ".")
    if len(sys.argv) > 1 and sys.argv[1] == "--project-dir":
        project_dir = sys.argv[2]

    pkg_path = os.path.join(project_dir, PKG_DIR)
    test_path = os.path.join(project_dir, TEST_DIR)

    if not os.path.isdir(pkg_path):
        print(f"❌ 包目录不存在：{pkg_path}")
        sys.exit(1)

    print(f"📂 扫描 {pkg_path} ...")
    modules = scan_package(pkg_path)

    print(f"📂 扫描 {test_path} ...")
    test_info = scan_tests(test_path)

    # 打印摘要
    print("\n📊 扫描结果：")
    total_lines = 0
    for fname, info in modules.items():
        total_lines += info["lines"]
        exports_str = ", ".join(info["exports"][:5])
        if len(info["exports"]) > 5:
            exports_str += f" ...（共 {len(info['exports'])} 个）"
        print(f"  {fname}: {info['lines']} 行 | exports: {exports_str}")
    print(f"\n  总计: {total_lines} 行 | 测试: ~{test_info['lines']} 行")

    # 更新 code-map
    print(f"\n🔄 更新 {CODE_MAP_PATH} ...")
    success = update_code_map(project_dir, modules, test_info)

    if success:
        print("\n✨ 完成！结构部分已自动更新，语义部分（快速定位表）保持不变。")
        print("   如需更新语义部分，运行 WorkBuddy Automation 或手动编辑。")


if __name__ == "__main__":
    main()
