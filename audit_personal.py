"""TO39 数据迁出审计：列出 /personal/flux_runs 全部产物并 cat 关键 JSON。

startScript: gm-run audit_personal.py
产物（清单+JSON 内容）全部 print 进任务日志；结束把整套打包回 /personal。
"""

import json
import os
from pathlib import Path

P = Path("/personal/flux_runs")
print("=== AUDIT /personal/flux_runs ===", flush=True)
if not P.is_dir():
    print("MISSING /personal/flux_runs", flush=True)
    raise SystemExit(1)

total = 0
for f in sorted(P.rglob("*")):
    if f.is_file():
        kb = f.stat().st_size / 1024
        total += kb
        print(f"{kb:9.1f} KB  {f}", flush=True)
print(f"=== total {total/1024:.1f} MB ===", flush=True)

for j in sorted(P.rglob("*summary*.json")) + sorted(P.glob("to39c/train_log.json")):
    print(f"=== CAT {j} ===", flush=True)
    try:
        print(j.read_text(), flush=True)
    except Exception as exc:
        print("READ FAIL:", exc, flush=True)
print("=== AUDIT DONE ===", flush=True)
