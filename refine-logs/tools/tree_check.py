#!/usr/bin/env python3
"""tree_check —— 实验记录文档树完整性闸门（refine-logs/README.md 扇出规则的执行器）

仿 mini Biosphere 的 tools/doc_tree_check.mjs，检查三项（任一失败退出码 1）：
  1. 挂树：refine-logs/** 下每篇 .md 必须出现在 refine-logs/README.md 的扇出树（```text 块）里
  2. 实存：树里每个 .md 节点必须真实存在（行内带 ⏳ 的规划节点跳过）
  3. 链接：refine-logs/**/*.md 与 AGENTS.md 里的相对 .md 链接必须能解析到真实文件

用法：python refine-logs/tools/tree_check.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # 仓库根
BASE = Path(__file__).resolve().parents[1]   # refine-logs/（树内路径相对它解析）
README = BASE / "README.md"

problems = {"not_on_tree": [], "ghost": [], "broken_links": []}


def rel(p: Path) -> str:
    return p.resolve().relative_to(ROOT).as_posix()


disk_files = sorted(BASE.rglob("*.md"))

# --- 1/2. 树 vs 磁盘（树内路径相对 refine-logs/ 解析） ---
readme = README.read_text(encoding="utf-8").replace("\r\n", "\n")
tree_paths = set()
for block in re.findall(r"```text\n(.*?)```", readme, re.S):
    for line in block.split("\n"):
        planned = "⏳" in line
        for m in re.finditer(r"[^\s`|├└│…]+\.md", line):
            t = m.group(0)
            if t.startswith("refine-logs/"):   # 兼容根节点行带 refine-logs/ 前缀
                t = t[len("refine-logs/"):]
            p = (BASE / t).resolve()
            tree_paths.add(rel(p))
            if not planned and not p.exists():
                problems["ghost"].append(rel(p))

for f in disk_files:
    if rel(f) not in tree_paths:
        problems["not_on_tree"].append(rel(f))

# --- 3. 相对 .md 链接 ---
link_re = re.compile(r"\]\(([^)#\s]+\.md)\)")
link_scope = disk_files + [ROOT / "AGENTS.md"]
n_links = 0
for file in link_scope:
    if not file.exists():
        continue
    text = file.read_text(encoding="utf-8").replace("\r\n", "\n")
    for m in link_re.finditer(text):
        target = m.group(1)
        if target.startswith(("http://", "https://")):
            continue
        n_links += 1
        if not (file.parent / target).resolve().exists():
            problems["broken_links"].append(f"{rel(file)} -> {target}")

# --- 报告 ---
fail = problems["not_on_tree"] or problems["ghost"] or problems["broken_links"]
print(f"tree_check：磁盘 {len(disk_files)} 篇 / 树内 {len(tree_paths)} 节点 / 相对链接 {n_links} 条")
if problems["not_on_tree"]:
    print("✗ 未挂树（磁盘有、树里没有）:")
    for p in problems["not_on_tree"]:
        print(f"    {p}")
if problems["ghost"]:
    print("✗ 幽灵节点（树里有、磁盘没有）:")
    for p in problems["ghost"]:
        print(f"    {p}")
if problems["broken_links"]:
    print("✗ 断链:")
    for p in problems["broken_links"]:
        print(f"    {p}")
if fail:
    sys.exit(1)
print("✓ 三项全绿：挂树 / 实存 / 链接")
