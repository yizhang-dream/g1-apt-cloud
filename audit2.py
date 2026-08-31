import os
from pathlib import Path
P = Path("/personal/flux_runs")
print("listdir flux_runs:", sorted(os.listdir(P)) if P.is_dir() else "NO DIR", flush=True)
t = P / "to39c"
print("to39c exists:", t.is_dir(), flush=True)
if t.is_dir():
    fs = sorted(os.listdir(t))
    print("to39c files:", len(fs), flush=True)
    print("first/last:", fs[:3], fs[-3:], flush=True)
print("it150 exists:", (t / "policy_it_150.pt").is_file(), flush=True)
print("train_log exists:", (t / "train_log.json").is_file(), flush=True)
