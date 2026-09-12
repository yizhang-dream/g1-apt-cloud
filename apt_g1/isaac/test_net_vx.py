"""D056 netfwd 臂数值单测：净前向奖励 +net_vx_scale·clamp(net_vx,0,vx_cap)/vx_cap（纯 torch/numpy，无 isaaclab 依赖）。

被测对象（DS_CONTINUOUS_EXECUTION_PLAN §5p 预注册；apt_flat_env.py 的
net_vx_scale 奖励路径 + train_apt_isaac.py 的 --net-vx-scale 接线）。本文件
**不 import apt_flat_env**（其模块级 import isaaclab 本机不可用），改用 AST
从源文件提取真身代码/断言结构做同源对拍（计算行逐字 exec，仅改写输入变量
名）——方法与 test_yaw_rew.py / test_max_vx.py 同款：

  1. net_vx 数值：yaw_rel=0 → net_vx=vx_body；yaw_rel=90° → −vy_body
     （clamp 后 0）；yaw_rel=180° → −vx_body（clamp 后 0）；组合点
     (vx=1, vy=0.5, yaw_rel=30°) 手算对拍；±180°/15° 网格逐点 =clamp
     口径；cap 饱和（vx=4 → term=1）；负 net_vx → 0（clamp 下界不反向罚）；
     通用 scale 权重解耦
  2. 与 yaw_rew 块 yaw_rel 同源（D048b 双旋转前科防复发）：同 quat 同
     _init_yaw 下，net_vx 块 yaw_rel 与 yaw_rew 块 yaw_rel 数值逐点相等；
     且两块的 quat 别名行 / atan2 提取式 / wrap 式 **token 逐字同构**
  3. 守卫方案验证（第三条 elif 同体分支，方案 B）：test_yaw_rew.py case3
     对前两条支做 AST 精确匹配（`ast.unparse(test) == GUARD_YAW_REW`），
     扩条件会破冻结断言——故 __init__ 分配与 _reset_idx 回填链扩**第三条
     elif `net_vx_scale > 0.0`**：支在位、orelse 尽头、分支体与 obs_heading
     支 token 逐字相同；mock exec 证明 net_vx 单开（obs_heading=0,
     yaw_rew=0, scale>0）分配 (N,) float32 张量并按同式回填（30° quat → 30°）
  4. 奖励块结构 + 零默认零变化：三行公式 token 逐字钉（§5p 预注册式）；
     `net_vx_term = None` 初始化在守卫前、守卫位于 max_vx 块后；分项仅
     `net_vx_term is not None` 时补记（快照行在守卫体内、位于基础 dict
     之后）；_last_rew_terms 基础键=旧 6 键原样；cfg 默认 net_vx_scale=0.0
     / vx_cap=2.0（逐字）；守卫块零 token 引用（token_mode 零改动静态面）
  5. train 接线在位：`--net-vx-scale` 默认 0.0；cfg.net_vx_scale 接线行且
     位于 env 构造之前；回显定义在 print 之前、尾巴 {net_vx_cfg} 在位；
     DIAG_KEYS 不含 net_vx；main 无 net_vx If 分支（回显走三元式）；
     observation_space 既有 +=2 bump 恰 3 条原样（net_vx 不 bump obs）

边界说明：env `_get_rewards` 的完整拼装深度耦合 Isaac（robot.data 张量），
本机不可实例化——以「真身块 exec + mock 张量」覆盖奖励项与回填段的数值
行为，Isaac 侧集成行为由 D053/D054/D055 同款边界说明豁免（零默认路径由
case4/case5 静态保证零改动）。

用法（本机 CPU torch / 服务器 .venv_isaac 均可）：
    PYTHONPATH=. python apt_g1/isaac/test_net_vx.py
全部用例 PASS 时 exit 0。
"""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
ENV_PATH = HERE / "apt_flat_env.py"
TRAIN_PATH = HERE / "train_apt_isaac.py"

GUARD_OBS = "self.cfg.obs_heading"
GUARD_YAW_REW = "self.cfg.yaw_rew_scale > 0.0"
GUARD_NET_VX = "self.cfg.net_vx_scale > 0.0"
GUARD_MAX_VX = "self.cfg.max_vx_scale > 0.0"
GUARD_SNAPSHOT = "net_vx_term is not None"

SCALE = 2.0  # §5p 预注册臂值（与 D055 max-vx 同档）
CAP = 2.0  # §5p 封顶复用 §5o vx_cap（cfg 写死不暴露 CLI）


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
    ns: dict = {"torch": torch, "math": math}
    exec(compile(src, f"<{name}>", "exec"), ns)
    return ns[name]


def _exec_guard_body(if_node: ast.If, rename: dict[str, str], name: str, ret: str):
    return _exec_lines([ast.unparse(s) for s in if_node.body], rename, name, ret)


def _strip_outer_parens(text: str) -> str:
    """剥掉语句开头的整组外层括号——Python 3.10 ast.unparse 渲染元组赋值带括号
    `(w, x, y, z) = (...)`，3.12+ 不带；归一后跨版本前缀匹配一致（CVGL py3.10 实证）。"""
    t = text.lstrip()
    while t.startswith("("):
        # 找与首括号配对的那个右括号，其后必须跟 " =" 或 ","（整组包裹才剥）
        depth, end = 0, -1
        for i, ch in enumerate(t):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end == -1 or not t[end + 1 :].lstrip().startswith("="):
            break
        t = t[1:end] + t[end + 1 :]
        t = t.lstrip()
    return t


def _stmt_tokens(if_node: ast.If, prefixes: tuple[str, ...]) -> dict[str, str]:
    """守卫块体内以 prefixes 开头的语句，归一空白后的 token 串（字面同构对比用）。"""
    out: dict[str, str] = {}
    for s in if_node.body:
        u = _strip_outer_parens(ast.unparse(s))
        for p in prefixes:
            if u.startswith(p):
                out[p] = u.replace(" ", "")
    return out


# --------------------------------------------------------------------------- 真身块提取
def _env_net_vx_block():
    """apt_flat_env._get_rewards 的 net_vx 守卫块原样 exec。

    签名 (root_quat, init_yaw, vel, reward, scale, cap)——把
    self.robot.data.root_quat_w / self._init_yaw / base_lin_vel / reward /
    self.cfg.net_vx_scale / self.cfg.vx_cap 显式化为入参，计算行逐字保留。
    返回 (reward, net_vx_term, net_vx, yaw_rel)。
    """
    tree, _ = _parse(ENV_PATH)
    rew_fn = _find_func(tree, "_get_rewards")
    return _exec_guard_body(
        _find_guarded_if(rew_fn, GUARD_NET_VX),
        {
            "self.robot.data.root_quat_w": "root_quat",
            "self._init_yaw": "init_yaw",
            "base_lin_vel": "vel",
            "reward": "reward",
            "self.cfg.net_vx_scale": "scale",
            "self.cfg.vx_cap": "cap",
        },
        "_env_net_vx_block",
        ret="reward, net_vx_term, net_vx, yaw_rel",
    )


def _env_yaw_rew_block():
    """apt_flat_env._get_rewards 的 yaw_rew 守卫块原样 exec（同源对拍基准）。

    返回 (reward, yaw_rew_term, yaw_rel)。
    """
    tree, _ = _parse(ENV_PATH)
    rew_fn = _find_func(tree, "_get_rewards")
    return _exec_guard_body(
        _find_guarded_if(rew_fn, GUARD_YAW_REW),
        {
            "self.robot.data.root_quat_w": "root_quat",
            "self._init_yaw": "init_yaw",
            "reward": "reward",
            "self.cfg.yaw_rew_scale": "scale",
        },
        "_env_yaw_rew_block",
        ret="reward, yaw_rew_term, yaw_rel",
    )


# --------------------------------------------------------------------------- 数学小件
def quat_from_zyx_euler(psi: float, theta: float, phi: float) -> torch.Tensor:
    """内在 ZYX（yaw-pitch-roll）欧拉 → w-first quat（D044 i:zyx 实测轴序）。"""
    hpsi, hth, hphi = psi / 2.0, theta / 2.0, phi / 2.0
    cp, sp = math.cos(hpsi), math.sin(hpsi)
    ct, st = math.cos(hth), math.sin(hth)
    cr, sr = math.cos(hphi), math.sin(hphi)
    w = cp * ct * cr + sp * st * sr
    x = cp * ct * sr - sp * st * cr
    y = cp * st * cr + sp * ct * sr
    z = sp * ct * cr - cp * st * sr
    return torch.tensor([w, x, y, z], dtype=torch.float64)


def _yaw_of(quat_row: torch.Tensor) -> float:
    """与 env 块同式的标量 yaw（w-first quat，手算 ground truth 用）。"""
    w, x, y, z = (float(v) for v in quat_row)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap_pi(d: float) -> float:
    # 与 env 块同式：torch.remainder=floor 模（除数正）↔ Python % 同语义；
    # 不用 math.remainder（IEEE 最近整舍入，量程/符号不同）
    return (d + math.pi) % (2.0 * math.pi) - math.pi


# --------------------------------------------------------------------------- cases
def case1_net_vx_numeric_pins() -> bool:
    """net_vx 数值：预注册点手算对拍 + ±180°/15° 网格逐点 + cap 饱和 + 负值下界 + 权重解耦。"""
    blk = _env_net_vx_block()
    ok = True

    def _run(vx: float, vy: float, yaw_deg: float, scale: float = SCALE):
        """yaw_deg 单开（init_yaw=0，identity 由纯 yaw quat 承载）跑真身块。"""
        q = quat_from_zyx_euler(math.radians(yaw_deg), 0.0, 0.0).unsqueeze(0)
        return blk(
            q,
            torch.zeros(1, dtype=torch.float64),
            torch.tensor([[vx, vy, 0.0]], dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
            scale, CAP,
        )

    # 预注册钉（scale=cap=2.0）：yaw_rel=0 → vx_body；90° → −vy_body（clamp
    # 后 0）；180° → −vx_body（clamp 后 0）；组合点 (1,0.5,30°) 手算；
    # §5p cos 因子（0.5,0,60° → 贡献减半）；cap 饱和；负 net_vx 下界归零
    c30, s30 = math.cos(math.radians(30.0)), math.sin(math.radians(30.0))
    for vx, vy, yaw_deg in [
        (1.0, 0.0, 0.0),      # yaw_rel=0 → vx_body
        (0.0, 0.5, 90.0),     # 90° → −vy_body
        (1.0, 0.0, 180.0),    # 180° → −vx_body
        (1.0, 0.5, 30.0),     # 组合点手算 0.6160…
        (1.0, 0.5, -30.0),    # 对称点手算 1.1160…
        (0.5, 0.0, 60.0),     # §5p：偏 60° 贡献减半
        (4.0, 0.0, 0.0),      # 超帽（term 饱和到 1）
        (-1.0, 0.0, 0.0),     # 后退（clamp 下界归零）
        (0.0, -0.5, 90.0),    # 反侧漂+转 90° → 正前向
    ]:
        rdelta, term, net_vx, yaw_rel = _run(vx, vy, yaw_deg)
        yaw_rad = math.radians(yaw_deg)
        want_netvx = vx * math.cos(yaw_rad) - vy * math.sin(yaw_rad)
        ok &= abs(yaw_rel[0].item() - _wrap_pi(yaw_rad)) < 1e-12
        ok &= abs(net_vx[0].item() - want_netvx) < 1e-12
        want_term = min(max(want_netvx, 0.0), CAP) / CAP
        ok &= abs(term[0].item() - want_term) < 1e-12
        ok &= abs(rdelta[0].item() - SCALE * want_term) < 1e-12  # 增量=scale·term

    # ±180°/15° 网格逐点：term == clamp(vx·cos−vy·sin, 0, cap)/cap
    vxs, vys = 1.2, 0.5
    degs = [round(-180.0 + 15.0 * i, 6) for i in range(25)]
    qs = torch.stack(
        [quat_from_zyx_euler(math.radians(d), 0.0, 0.0) for d in degs]
    )
    _, terms, _, yaws_rel = blk(
        qs,
        torch.zeros(len(degs), dtype=torch.float64),
        torch.tensor([[vxs, vys, 0.0]] * len(degs), dtype=torch.float64),
        torch.zeros(len(degs), dtype=torch.float64),
        SCALE, CAP,
    )
    for i, d in enumerate(degs):
        r = math.radians(d)
        ok &= abs(yaws_rel[i].item() - _wrap_pi(r)) < 1e-12
        want = min(max(vxs * math.cos(r) - vys * math.sin(r), 0.0), CAP) / CAP
        ok &= abs(terms[i].item() - want) < 1e-12

    # cap 饱和：vx=cap → term=1、vx>cap 仍=1；cap 内线性（yaw_rel=0 斜率 1/cap）
    _, term_cap, _, _ = _run(2.0, 0.0, 0.0)
    ok &= abs(term_cap[0].item() - 1.0) < 1e-12
    _, term_over, _, _ = _run(3.0, 0.0, 0.0)
    ok &= abs(term_over[0].item() - 1.0) < 1e-12
    _, term_lin, _, _ = _run(1.0, 0.0, 0.0)
    ok &= abs(term_lin[0].item() - 1.0 / CAP) < 1e-12

    # 通用 scale：scale=0.5 时增量减半（权重与分项解耦）
    _, t_half, _, _ = _run(1.0, 0.0, 0.0, scale=0.5)
    ok &= abs(t_half[0].item() - 0.5) < 1e-12
    return _report(
        "case1 net_vx 数值（yaw_rel=0/90°/180° 钉 + 组合点手算 + 网格逐点 + cap 饱和/下界 + 权重解耦）", ok,
        f"scale={SCALE} cap={CAP}  combo(vx=1,vy=0.5,30°)={c30 - 0.5 * s30:.6f}",
    )


def case2_same_source_yaw_rel() -> bool:
    """与 yaw_rew 块 yaw_rel 同源：数值逐点相等 + 别名/atan2/wrap 三式 token 逐字同构。"""
    tree, _ = _parse(ENV_PATH)
    rew_fn = _find_func(tree, "_get_rewards")
    ok = True

    # 数值：9 quat（纯 yaw × 含 pitch/roll）× 4 init 逐点，两块 yaw_rel 相等
    # 且 == wrap(yaw−init) 手算
    quats = []
    for psi in (0.0, 30.0, -30.0, 90.0, -90.0, 180.0):
        quats.append(quat_from_zyx_euler(math.radians(psi), 0.0, 0.0))
    for psi, th, phi in [(20.0, 10.0, -5.0), (-40.0, 5.0, 15.0), (75.0, -8.0, 3.0)]:
        quats.append(quat_from_zyx_euler(math.radians(psi), math.radians(th), math.radians(phi)))
    inits = [0.0, math.radians(45.0), math.radians(-45.0), math.radians(135.0)]
    n = len(quats) * len(inits)
    q_rows, init_rows = [], []
    for q in quats:
        for y0 in inits:
            q_rows.append(q)
            init_rows.append(y0)
    root_quat = torch.stack(q_rows)
    init_yaw = torch.tensor(init_rows, dtype=torch.float64)
    n_pts = 0
    for i in range(n):
        yaw = _yaw_of(root_quat[i])
        want = _wrap_pi(yaw - float(init_yaw[i]))
        _, _, _, rj = _env_net_vx_block()(
            root_quat[i : i + 1], init_yaw[i : i + 1],
            torch.zeros(1, 3, dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64), SCALE, CAP,
        )
        _, _, ry = _env_yaw_rew_block()(
            root_quat[i : i + 1], init_yaw[i : i + 1],
            torch.zeros(1, dtype=torch.float64), SCALE,
        )
        ok &= abs(rj[0].item() - want) < 1e-12
        ok &= abs(rj[0].item() - ry[0].item()) < 1e-12
        n_pts += 1

    # token：net_vx 块与 yaw_rew 块的 quat 别名行 / atan2 提取式 / wrap 式
    # 逐字同构（D048b 双旋转前科——错符号=教反方向，效度核心）
    prefixes = ("yaw =", "yaw_rel =", "w, x, y, z =")
    toks_net = _stmt_tokens(_find_guarded_if(rew_fn, GUARD_NET_VX), prefixes)
    toks_yaw = _stmt_tokens(_find_guarded_if(rew_fn, GUARD_YAW_REW), prefixes)
    for p in prefixes:
        ok &= p in toks_net and p in toks_yaw and toks_net[p] == toks_yaw[p]
    ok &= toks_net.get("yaw =") == (
        "yaw=torch.atan2(2.0*(w*z+x*y),1.0-2.0*(y*y+z*z))"
    )
    return _report(
        "case2 与 yaw_rew 块 yaw_rel 同源（9×4 quat×init 数值逐点 + atan2/wrap/别名 token 同构）", ok,
        f"{n_pts} 点对拍  block-token 逐字相等",
    )


def _chain_of(first_if: ast.If) -> list[ast.If]:
    """if/elif 链按序展开（AST 里 elif 嵌在 orelse[0]）。"""
    chain, node = [], first_if
    while node is not None:
        chain.append(node)
        nxt = node.orelse
        node = nxt[0] if len(nxt) == 1 and isinstance(nxt[0], ast.If) else None
    return chain


def case3_shared_init_yaw_guard_option_b() -> bool:
    """守卫方案 B：__init__/_reset_idx 链扩第三条 elif net_vx（同体分支）——
    前两条支的 AST 精确匹配（test_yaw_rew case3 冻结断言）不被破坏。"""
    tree, src_env = _parse(ENV_PATH)
    ok = True

    # __init__：先 None；链 = [obs_heading, yaw_rew, net_vx]，net_vx 支在位
    # 且是链尾；net_vx 支体与 obs 支体 token 逐字相同
    init_fn = _find_func(tree, "__init__")
    ok &= "self._init_yaw = None" in (ast.get_source_segment(src_env, init_fn) or "")
    alloc_if = _find_guarded_if(init_fn, GUARD_OBS)
    chain = _chain_of(alloc_if)
    ok &= len(chain) == 3
    ok &= ast.unparse(chain[1].test) == GUARD_YAW_REW  # 前两支原样（冻结断言面）
    ok &= ast.unparse(chain[2].test) == GUARD_NET_VX
    ok &= not chain[2].orelse  # net_vx 是链尾（不吞后续分支）
    ok &= [ast.unparse(s) for s in chain[0].body] == [
        ast.unparse(s) for s in chain[2].body
    ]

    # mock exec（net_vx 单开语义）：第三支分配 (N,) float32 张量
    alloc_lines = [ast.unparse(s) for s in chain[2].body]
    alloc_mock = _exec_lines(
        alloc_lines, {"self": "self"}, "_alloc_mock", ret="self._init_yaw"
    )
    yaw = alloc_mock(SimpleNamespace(num_envs=4, device="cpu"))
    ok &= tuple(yaw.shape) == (4,) and yaw.dtype == torch.float32

    # _reset_idx：回填链同构；net_vx 支与 obs 支回填同输入数值相等（30° quat）
    reset_fn = _find_func(tree, "_reset_idx")
    reset_if = _find_guarded_if(reset_fn, GUARD_OBS)
    rchain = _chain_of(reset_if)
    ok &= len(rchain) == 3
    ok &= ast.unparse(rchain[1].test) == GUARD_YAW_REW
    ok &= ast.unparse(rchain[2].test) == GUARD_NET_VX
    ok &= not rchain[2].orelse
    ok &= [ast.unparse(s) for s in rchain[0].body] == [
        ast.unparse(s) for s in rchain[2].body
    ]
    ok &= "self._init_yaw[env_ids]" in ast.unparse(reset_if)
    write_ln = next(
        n.lineno for n in ast.walk(reset_fn)
        if isinstance(n, ast.Call) and "write_root_state_to_sim" in ast.unparse(n)
    )
    ok &= reset_if.lineno > write_ln
    rename = {"self._init_yaw": "init_yaw", "default_root": "default_root",
              "env_ids": "env_ids"}
    backfill_if = _exec_guard_body(rchain[0], rename, "_backfill_if", ret="init_yaw")
    backfill_net = _exec_guard_body(rchain[2], rename, "_backfill_net", ret="init_yaw")
    root30 = torch.zeros(2, 7, dtype=torch.float64)
    root30[:, 3:7] = quat_from_zyx_euler(math.radians(30.0), 0.0, 0.0)
    env_ids = torch.tensor([0, 1], dtype=torch.long)
    y_if = backfill_if(torch.zeros(2, dtype=torch.float64), root30.clone(), env_ids)
    y_net = backfill_net(torch.zeros(2, dtype=torch.float64), root30.clone(), env_ids)
    ok &= torch.allclose(y_if, y_net)
    ok &= bool((y_if - math.radians(30.0)).abs().max() < 1e-12)
    return _report(
        "case3 守卫方案 B（第三条 elif net_vx 同体在位 + 前两支冻结断言面原样 + mock 分配/回填）", ok,
        "obs_heading/yaw_rew 单开路径不进第三支，逐位不变",
    )


def case4_reward_block_structure_zero_default() -> bool:
    """奖励块结构 + 零默认零变化：公式 token 钉、None 前置、max_vx 后、快照守卫内、cfg 默认逐字、零 token 引用。"""
    tree, src_env = _parse(ENV_PATH)
    ok = True
    rew_fn = _find_func(tree, "_get_rewards")
    guard_if = _find_guarded_if(rew_fn, GUARD_NET_VX)

    # ① 三行公式 token 逐字钉（§5p 预注册式，house 模式=先算分项再加权）
    body_toks = [ast.unparse(s).replace(" ", "") for s in guard_if.body]
    ok &= (
        "net_vx=base_lin_vel[:,0]*torch.cos(yaw_rel)-base_lin_vel[:,1]*torch.sin(yaw_rel)"
        in body_toks
    )
    ok &= "net_vx_term=torch.clamp(net_vx,0.0,self.cfg.vx_cap)/self.cfg.vx_cap" in body_toks
    ok &= "reward=reward+self.cfg.net_vx_scale*net_vx_term" in body_toks
    # ② 守卫块零 token 引用（token_mode 零改动约束的静态面；raw 源段含注释）
    seg = ast.get_source_segment(src_env, guard_if) or ""
    ok &= "token" not in seg
    # ③ net_vx_term=None 初始化存在且在守卫之前（守卫关时快照守卫可判 None）
    init_ln = [
        n.lineno for n in ast.walk(rew_fn)
        if isinstance(n, ast.Assign)
        and any(ast.unparse(t) == "net_vx_term" for t in n.targets)
        and ast.unparse(n.value) == "None"
    ]
    ok &= len(init_ln) == 1 and init_ln[0] < guard_if.lineno
    # ④ 守卫块位于 max_vx 块之后（照 D055 分支口径接续）
    ok &= guard_if.lineno > _find_guarded_if(rew_fn, GUARD_MAX_VX).lineno
    # ⑤ net_vx 分项仅 None 守卫时补记，快照行在守卫体内、位于基础 dict 之后
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
    snap_if = _find_guarded_if(rew_fn, GUARD_SNAPSHOT)
    ok &= 'self._last_rew_terms["net_vx"] = net_vx_term' in ast.unparse(snap_if).replace(
        "'", '"'
    )
    ok &= len(snap_if.body) == 1  # 守卫体内恰一条（快照行）
    ok &= snap_if.lineno > dict_assign.lineno
    # ⑥ cfg 默认逐字：net_vx_scale=0.0（零变化）/ vx_cap=2.0（§5o 值复用）
    cfg_cls = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "AptFlatG1EnvCfg"
    )
    defaults = {}
    for name in ("net_vx_scale", "vx_cap"):
        field = next(
            n for n in ast.walk(cfg_cls)
            if isinstance(n, ast.AnnAssign) and ast.unparse(n.target) == name
        )
        defaults[name] = ast.unparse(field.value) if field.value else ""
    ok &= defaults["net_vx_scale"] == "0.0"
    ok &= defaults["vx_cap"] == "2.0"
    return _report(
        "case4 奖励块结构+零默认零变化（三行 token 钉 + None 前置 + max_vx 后 + 快照守卫 + 6 键/cfg 默认逐字）", ok,
        f"defaults={defaults}",
    )


def case5_train_wiring_and_obs_untouched() -> bool:
    """train 接线：--net-vx-scale 默认 0.0、cfg 行先于 env 构造、回显尾巴在位、DIAG_KEYS/obs 零波及。"""
    _, src_train = _parse(TRAIN_PATH)
    tree_train, _ = _parse(TRAIN_PATH)
    ok = True
    train_main = _find_func(tree_train, "main")
    # ① argparse 默认 0.0（零变化）
    ok &= '"--net-vx-scale", type=float, default=0.0' in src_train
    # ② cfg 接线行（裸 Assign，D055 max_vx 接线点同款）且位于 env 构造之前
    wire = next(
        n for n in ast.walk(train_main)
        if isinstance(n, ast.Assign)
        and any(ast.unparse(t) == "cfg.net_vx_scale" for t in n.targets)
    )
    ok &= ast.unparse(wire) == "cfg.net_vx_scale = cli.net_vx_scale"
    env_ctors = [
        n for n in ast.walk(train_main)
        if isinstance(n, ast.Assign)
        and any(ast.unparse(t) == "env" for t in n.targets)
        and "AptFlatG1Env(" in ast.unparse(n)
    ]
    ok &= len(env_ctors) >= 1
    ok &= all(wire.lineno < c.lineno for c in env_ctors)
    # ③ [CFG] 回显条件式（>0 才显示）且定义在 print 之前、尾巴在位
    echo_def = next(
        n for n in ast.walk(train_main)
        if isinstance(n, ast.Assign)
        and any(ast.unparse(t) == "net_vx_cfg" for t in n.targets)
    )
    u_echo = ast.unparse(echo_def).replace("'", '"')
    ok &= "if cli.net_vx_scale > 0.0 else" in u_echo
    ok &= u_echo.endswith('else ""')
    print_call = next(
        n for n in ast.walk(train_main)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name) and n.func.id == "print"
        and "net_vx_cfg" in ast.unparse(n)
    )
    ok &= "{heading_cfg}{yaw_rew_cfg}{vx_min_cfg}{max_vx_cfg}{net_vx_cfg}" in ast.unparse(
        print_call
    )
    ok &= echo_def.lineno < print_call.lineno
    # ④ main 内无 net_vx 关联 If 分支（接线全裸 Assign + 回显走三元式）
    ok &= not [
        n for n in ast.walk(train_main)
        if isinstance(n, ast.If) and "net_vx_scale" in ast.unparse(n.test)
    ]
    # ⑤ DIAG_KEYS 不含 net_vx（diag 键集合不变，5 键原样）
    diag_assign = next(
        n for n in ast.walk(train_main)
        if isinstance(n, ast.Assign)
        and any(ast.unparse(t) == "DIAG_KEYS" for t in n.targets)
    )
    diag_keys = [e.value for e in diag_assign.value.elts]
    ok &= "net_vx" not in diag_keys
    ok &= diag_keys == ["track_xy", "track_yaw", "upright", "height", "stillness"]
    # ⑥ obs 维度不随 net_vx 走：既有 observation_space += 2 恰 3 条原样
    augs = [
        n for n in ast.walk(train_main)
        if isinstance(n, ast.AugAssign)
        and ast.unparse(n.target) == "cfg.observation_space"
        and ast.unparse(n.value) == "2"
    ]
    ok &= len(augs) == 3
    return _report(
        "case5 train 接线（argparse 默认 0 + cfg 行先于 env 构造 + 回显尾巴 + DIAG_KEYS/obs 零波及）", ok,
    )


def main() -> None:
    cases = [
        case1_net_vx_numeric_pins,
        case2_same_source_yaw_rel,
        case3_shared_init_yaw_guard_option_b,
        case4_reward_block_structure_zero_default,
        case5_train_wiring_and_obs_untouched,
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
