"""D055 max-vx 臂数值单测：前向速度奖励 +max_vx_scale·clamp(vx_body,0,vx_cap)/vx_cap（纯 torch/numpy，无 isaaclab 依赖）。

被测对象（DS_CONTINUOUS_EXECUTION_PLAN §5o 预注册；apt_flat_env.py 的
max_vx_scale 奖励路径 + train_apt_isaac.py 的 --max-vx-scale 接线）。本文件
**不 import apt_flat_env**（其模块级 import isaaclab 本机不可用），改用 AST
从源文件提取真身代码/断言结构做同源对拍（计算行逐字 exec，仅改写输入变量
名）——方法与 test_yaw_rew.py / test_obs_heading.py 同款：

  1. max_vx 数值：vx=0 → 0；vx=1 → scale/2；vx≥cap → scale；负 vx → 0
     （clamp 下界归零不反向罚）；±2cap 网格逐点 = scale·clamp(vx,0,cap)/cap；
     cap 内线性、单调不减
  2. 公式与守卫结构：奖励行/分项行 token 逐字钉（§5o 预注册式）；
     `max_vx_term = None` 初始化在守卫前；计算块全在 `max_vx_scale > 0.0`
     守卫内且位于 anti_stop 块之后；cfg 默认 max_vx_scale=0.0 / vx_cap=2.0
     （逐字）；守卫块零 token 引用（R4 载体 token_mode 零改动约束的静态面）
  3. 零默认零变化 + 分项记录守卫内：_last_rew_terms 基础键=旧 6 键原样；
     max_vx 分项仅 `max_vx_term is not None` 时补记（快照行在守卫体内、
     位于基础 dict 之后）；DIAG_KEYS 不含 max_vx（diag 键集合不变）；
     [CFG] 回显条件式（>0 才显示，默认逐字不变）
  4. train 接线在位：`--max-vx-scale` 默认 0.0；cfg.max_vx_scale 接线行且
     位于 env 构造之前；回显定义在 print 之前、尾巴 {max_vx_cfg} 在位

边界说明：env `_get_rewards` 的完整拼装深度耦合 Isaac（robot.data 张量），
本机不可实例化——以「真身块 exec + mock 张量」覆盖奖励项的数值行为，
Isaac 侧集成行为由 D053/D054 同款边界说明豁免（零默认路径由 case2/case3
静态保证零改动）。

用法（本机 CPU torch / 服务器 .venv_isaac 均可）：
    PYTHONPATH=. python apt_g1/isaac/test_max_vx.py
全部用例 PASS 时 exit 0。
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ENV_PATH = HERE / "apt_flat_env.py"
TRAIN_PATH = HERE / "train_apt_isaac.py"

GUARD_MAX_VX = "self.cfg.max_vx_scale > 0.0"
GUARD_ANTI_STOP = "self.cfg.anti_stop_scale > 0.0"
GUARD_SNAPSHOT = "max_vx_term is not None"

SCALE = 2.0  # §5o 预注册臂值（地板对账推算：≥2×站立地板 1.641 的差值需求）
CAP = 2.0  # §5o 预注册封顶速度（m/s，cfg 写死不暴露 CLI）


def _report(name: str, ok: bool, detail: str = "") -> bool:
    print(
        f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""),
        flush=True,
    )
    return ok


# --------------------------------------------------------------------------- AST helpers
def _parse(path: Path) -> tuple[ast.Module, str]:
    src = path.read_text(encoding="utf-8")
    return ast.parse(src), src


def _find_func(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function def {name!r} not found")


def _find_guarded_if(func: ast.FunctionDef, guard: str) -> ast.If:
    hits = [
        n for n in ast.walk(func)
        if isinstance(n, ast.If) and ast.unparse(n.test) == guard
    ]
    if not hits:
        raise AssertionError(f"guard {guard!r} not found in {func.name}")
    return hits[0]


def _exec_lines(lines: list[str], rename: dict[str, str], name: str, ret: str):
    """把源码行原样 exec 成可调用体（仅改写输入变量名，计算行逐字不动）。"""
    for old, new in rename.items():
        lines = [ln.replace(old, new) for ln in lines]
    src = f"def {name}({', '.join(rename.values())}):\n" + "\n".join(
        f"    {ln}" for ln in lines
    ) + f"\n    return {ret}"
    ns: dict = {"torch": torch}
    exec(compile(src, f"<{name}>", "exec"), ns)
    return ns[name]


def _exec_guard_body(if_node: ast.If, rename: dict[str, str], name: str, ret: str):
    return _exec_lines([ast.unparse(s) for s in if_node.body], rename, name, ret)


def _env_max_vx_block():
    """apt_flat_env._get_rewards 的 max_vx 守卫块原样 exec。

    签名 (vx, scale, cap, reward)——把 base_lin_vel / self.cfg.max_vx_scale /
    self.cfg.vx_cap / reward 显式化为入参，计算行逐字保留。
    返回 (reward, max_vx_term)。
    """
    tree, _ = _parse(ENV_PATH)
    rew_fn = _find_func(tree, "_get_rewards")
    return _exec_guard_body(
        _find_guarded_if(rew_fn, GUARD_MAX_VX),
        {
            "base_lin_vel": "vx",
            "self.cfg.max_vx_scale": "scale",
            "self.cfg.vx_cap": "cap",
            "reward": "reward",
        },
        "_env_max_vx_block",
        ret="reward, max_vx_term",
    )


# --------------------------------------------------------------------------- cases
def case1_max_vx_numeric_pins() -> bool:
    """max_vx 数值：0→0、vx=1→scale/2、vx≥cap→scale、负 vx→0（clamp 下界）+网格逐点。"""
    blk = _env_max_vx_block()
    ok = True

    # 预注册钉（scale=cap=2.0）：vx=0→0、vx=1→scale·0.5、vx=cap→scale、
    # vx>cap→scale、负 vx→0（clamp 下界归零不反向罚）
    for vx, want_delta, want_term in [
        (0.0, 0.0, 0.0),
        (1.0, SCALE * 0.5, 0.5),
        (2.0, SCALE, 1.0),  # vx = cap
        (3.0, SCALE, 1.0),  # vx > cap
        (-0.7, 0.0, 0.0),
        (-2.0, 0.0, 0.0),
    ]:
        reward, term = blk(
            torch.tensor([[vx]], dtype=torch.float64), SCALE, CAP,
            torch.zeros(1, dtype=torch.float64),
        )
        ok &= abs(term[0].item() - want_term) < 1e-12
        ok &= abs(reward[0].item() - want_delta) < 1e-12

    # ±2cap 网格逐点：delta == scale·clamp(vx,0,cap)/cap
    vxs = [round(-2.0 * CAP + 0.25 * CAP * i, 4) for i in range(17)]
    vx_t = torch.tensor(vxs, dtype=torch.float64).unsqueeze(1)
    reward, term = blk(
        vx_t, SCALE, CAP, torch.zeros(len(vxs), dtype=torch.float64)
    )
    for i, v in enumerate(vxs):
        want = min(max(v, 0.0), CAP) / CAP
        ok &= abs(term[i].item() - want) < 1e-12
        ok &= abs(reward[i].item() - SCALE * want) < 1e-12

    # cap 内线性（斜率 1/cap）且全程单调不减
    ok &= bool((term[1:] >= term[:-1] - 1e-12).all())
    vxs_lin = [0.0, 0.5, 1.0, 1.5, 2.0]
    _, term_lin = blk(torch.tensor(vxs_lin, dtype=torch.float64).unsqueeze(1),
                      SCALE, CAP, torch.zeros(len(vxs_lin), dtype=torch.float64))
    ok &= abs((term_lin[-1] - term_lin[0]).item() - (vxs_lin[-1] - vxs_lin[0]) / CAP) < 1e-12
    # 通用 scale：scale=0.5 时 delta 减半（权重与分项解耦）
    r2, t2 = blk(torch.tensor([[1.0]], dtype=torch.float64), 0.5, CAP,
                 torch.zeros(1, dtype=torch.float64))
    ok &= abs(t2[0].item() - 0.5) < 1e-12 and abs(r2[0].item() - 0.25) < 1e-12
    return _report(
        "case1 max_vx 数值（0→0、1→scale/2、≥cap→scale、负→0、±2cap 网格+线性单调）", ok,
        f"scale={SCALE} cap={CAP}  grid=±{2 * CAP}/0.5·cap",
    )


def case2_formula_and_guard_structure() -> bool:
    """公式/守卫结构：两行 token 逐字钉、None 初始化在守卫前、anti_stop 之后、cfg 默认逐字、零 token 引用。"""
    tree, src_env = _parse(ENV_PATH)
    ok = True
    rew_fn = _find_func(tree, "_get_rewards")
    guard_if = _find_guarded_if(rew_fn, GUARD_MAX_VX)

    # ① 奖励行/分项行 token 逐字（§5o 预注册式，house 模式=先算分项再加权）
    body_toks = [ast.unparse(s).replace(" ", "") for s in guard_if.body]
    ok &= "max_vx_term=torch.clamp(base_lin_vel[:,0],0.0,self.cfg.vx_cap)/self.cfg.vx_cap" in body_toks
    ok &= "reward=reward+self.cfg.max_vx_scale*max_vx_term" in body_toks
    # ② 守卫块零 token 引用（R4 载体 token_mode 零改动的静态面；raw 源段含注释）
    seg = ast.get_source_segment(src_env, guard_if) or ""
    ok &= "token" not in seg
    # ③ max_vx_term=None 初始化存在且在守卫之前
    init_ln = [
        n.lineno for n in ast.walk(rew_fn)
        if isinstance(n, ast.Assign)
        and any(ast.unparse(t) == "max_vx_term" for t in n.targets)
        and ast.unparse(n.value) == "None"
    ]
    ok &= len(init_ln) == 1 and init_ln[0] < guard_if.lineno
    # ④ 守卫块位于 anti_stop 块之后（§5o「照 anti_stop 分支口径写」）
    anti_if = _find_guarded_if(rew_fn, GUARD_ANTI_STOP)
    ok &= guard_if.lineno > anti_if.lineno
    # ⑤ cfg 默认逐字：max_vx_scale=0.0（零变化）/ vx_cap=2.0（预注册值）
    cfg_cls = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "AptFlatG1EnvCfg"
    )
    defaults = {}
    for name in ("max_vx_scale", "vx_cap"):
        field = next(
            n for n in ast.walk(cfg_cls)
            if isinstance(n, ast.AnnAssign) and ast.unparse(n.target) == name
        )
        defaults[name] = ast.unparse(field.value) if field.value else ""
    ok &= defaults["max_vx_scale"] == "0.0"
    ok &= defaults["vx_cap"] == "2.0"
    return _report(
        "case2 公式/守卫结构（两行 token 钉 + None 前置 + anti_stop 后 + cfg 默认 0.0/2.0 + 零 token 引用）", ok,
        f"defaults={defaults}",
    )


def case3_zero_default_and_snapshot_guard() -> bool:
    """零默认零变化 + 分项记录守卫内：基础 6 键原样、快照 None 守卫、DIAG_KEYS 不扩、回显条件式。"""
    tree_env, _ = _parse(ENV_PATH)
    tree_train, _ = _parse(TRAIN_PATH)
    ok = True
    rew_fn = _find_func(tree_env, "_get_rewards")
    # ① _last_rew_terms 基础键 = 旧 6 键原样（顺序含；max_vx 不进基础键）
    dict_assign = next(
        n for n in ast.walk(rew_fn)
        if isinstance(n, ast.Assign)
        and any(ast.unparse(t) == "self._last_rew_terms" for t in n.targets)
        and isinstance(n.value, ast.Dict)
    )
    base_keys = [k.value for k in dict_assign.value.keys]
    ok &= base_keys == [
        "track_xy", "track_yaw", "upright", "height", "stillness", "vx_err",
    ]
    # ② max_vx 分项仅 None 守卫时补记，快照行在守卫体内、位于基础 dict 之后
    snap_if = _find_guarded_if(rew_fn, GUARD_SNAPSHOT)
    ok &= 'self._last_rew_terms["max_vx"] = max_vx_term' in ast.unparse(snap_if).replace(
        "'", '"'
    )
    ok &= len(snap_if.body) == 1  # 守卫体内恰一条（快照行）
    ok &= snap_if.lineno > dict_assign.lineno
    # ③ train DIAG_KEYS 不含 max_vx（diag 键集合不变）
    train_main = _find_func(tree_train, "main")
    diag_assign = next(
        n for n in ast.walk(train_main)
        if isinstance(n, ast.Assign)
        and any(ast.unparse(t) == "DIAG_KEYS" for t in n.targets)
    )
    diag_keys = [e.value for e in diag_assign.value.elts]
    ok &= "max_vx" not in diag_keys
    ok &= diag_keys == ["track_xy", "track_yaw", "upright", "height", "stillness"]
    # ④ [CFG] 回显条件式（>0 才显示）且定义在 print 之前、尾巴在位
    echo_def = next(
        n for n in ast.walk(train_main)
        if isinstance(n, ast.Assign)
        and any(ast.unparse(t) == "max_vx_cfg" for t in n.targets)
    )
    u_echo = ast.unparse(echo_def).replace("'", '"')
    ok &= "if cli.max_vx_scale > 0.0 else" in u_echo
    ok &= u_echo.endswith('else ""')
    print_call = next(
        n for n in ast.walk(train_main)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name) and n.func.id == "print"
        and "max_vx_cfg" in ast.unparse(n)
    )
    ok &= "{heading_cfg}{yaw_rew_cfg}{vx_min_cfg}{max_vx_cfg}" in ast.unparse(print_call)
    ok &= echo_def.lineno < print_call.lineno
    return _report(
        "case3 零默认零变化 + 分项守卫（6 键原样 + None 守卫快照 + DIAG_KEYS 不扩 + 回显条件式）", ok,
    )


def case4_train_wiring() -> bool:
    """train 接线：--max-vx-scale 默认 0.0、cfg 接线行先于 env 构造。"""
    _, src_train = _parse(TRAIN_PATH)
    tree_train, _ = _parse(TRAIN_PATH)
    ok = True
    train_main = _find_func(tree_train, "main")
    # ① argparse 默认 0.0（零变化）
    ok &= '"--max-vx-scale", type=float, default=0.0' in src_train
    # ② cfg 接线行（裸 Assign，D054 yaw_rew 接线点同款）
    wire = next(
        n for n in ast.walk(train_main)
        if isinstance(n, ast.Assign)
        and any(ast.unparse(t) == "cfg.max_vx_scale" for t in n.targets)
    )
    ok &= ast.unparse(wire) == "cfg.max_vx_scale = cli.max_vx_scale"
    # ③ 位于 env 构造之前（wiring 落在 env cfg 上，必须先于 AptFlatG1Env(...)）
    env_ctors = [
        n for n in ast.walk(train_main)
        if isinstance(n, ast.Assign)
        and any(ast.unparse(t) == "env" for t in n.targets)
        and "AptFlatG1Env(" in ast.unparse(n)
    ]
    ok &= len(env_ctors) >= 1
    ok &= all(wire.lineno < c.lineno for c in env_ctors)
    # ④ vx_cap 不暴露 CLI（cfg 写死 2.0，§5o 预注册）：无 --vx-cap 旗标、
    # main 内无 cfg.vx_cap 接线行（help/注释里的 vx_cap=2.0 字样不算暴露）
    ok &= "--vx-cap" not in src_train
    ok &= not [
        n for n in ast.walk(train_main)
        if isinstance(n, ast.Assign)
        and any(ast.unparse(t) == "cfg.vx_cap" for t in n.targets)
    ]
    return _report(
        "case4 train 接线（argparse 默认 0 + cfg 行先于 env 构造 + vx_cap 不暴露 CLI）", ok,
    )


def main() -> None:
    cases = [
        case1_max_vx_numeric_pins,
        case2_formula_and_guard_structure,
        case3_zero_default_and_snapshot_guard,
        case4_train_wiring,
    ]
    results = []
    for c in cases:
        try:
            results.append(c())
        except Exception as e:  # noqa: BLE001 —— 用例内异常按 FAIL 记账不中断
            print(f"[FAIL] {c.__name__}  (exception: {e!r})", flush=True)
            results.append(False)
    n_fail = results.count(False)
    print(f"\n{len(results) - n_fail}/{len(results)} cases PASS", flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
