"""D053 obs-head 臂数值单测：航向可观测性 2 维 [sin(yaw_rel), cos(yaw_rel)]（纯 torch/numpy，无 isaaclab 依赖）。

被测对象（DS_CONTINUOUS_EXECUTION_PLAN §5m 预注册；apt_flat_env.py 的
obs_heading 旗标路径 + train_apt_isaac.py 的 --obs-heading 接线）。本文件
**不 import apt_flat_env**（其模块级 import isaaclab 本机不可用），改用
AST 从源文件提取真身代码/断言结构做同源对拍（计算行逐字 exec，仅改写
输入变量名）：

  1. yaw 提取与 eval 同源对拍：AST 抽取 eval_apt_isaac.py 嵌套函数
     ``_yaw_of``（w-first root quat 标准 yaw 公式）原样 exec，与
     apt_flat_env.py `_get_observations` obs_heading 块（向量版公式 + wrap
     + sin/cos 拼装）在 quat 电池上数值逐点相等；并静态断言两处 atan2
     表达式字面同构（D048b 双旋转 bug 前科——错符号=教反方向，效度核心）
  2. 纯 yaw quat（绕 z 轴）：yaw = 0/45°/90°/−90°/180° → 提取正确（180°
     提取为 +π，wrap 后 = −π，与 eval yaw_err wrap 口径一致）
  3. D044 i:zyx 欧拉序样例（官方 G1 数据实测内在轴序，D044 判读：
     i:zyx 旋转误差 0.00°）：(ψ, θ, φ) 组合构造 quat → 提取 yaw 恢复 ψ；
     另用旋转矩阵投影（x̂ 世界系航向）做独立几何 ground truth 交叉验证
     （θ=0 时投影航向 == ψ，任意 roll）
  4. wrap_pi 边界：env 源码提取的向量式（torch.remainder）与 eval 标量式
     (d+π)%(2π)−π（L598 原式）在 [−4π, 4π] 网格 + 边界钉（±π→−π、
     ±3π/2→∓π/2）逐点相等
  5. reset→step 合成序列（env _reset_idx 块 + obs_heading 块 AST 原样
     exec 的 mock 拼装）：identity quat 复位后首步 (sin,cos)=(0,1)；机体
     yaw 偏移 45°/−90°/180° 后 heading 块 = (0.7071,0.7071)/(−1,0)/
     (≈0,−1)；yaw0≠0 的通用 wrap(yaw−yaw0) 口径同验
  6. D051 b 臂 obs 维度拼装：base 91（3+3+3+29+29+3+2+12+5+1+1）→
     latent 105 → +29 res = 134 → +3 vb = 137 → +2 heading = 139（与
     train 侧身份断言字面 139 对拍）；无旗标 = 137（+2 块不进 cat）
  7. 无旗标路径零变化（AST 静态）：cfg 默认 False；_init_yaw 分配/
     _reset_idx 回填/_get_observations 追加三处全部由 obs_heading 守卫
     （追加块单条 append 2 维、位于 vb_w 反馈之后、reset 回填位于 root
     写入之后）；train bump 守卫 `if cli.obs_heading:` 且位于 decft 覆写
     之后；旗标关时不进任何新分支 → obs 维度表达式逐字节不变
  8. 训练日志钩子静态在位：hist["yaw_rel_mean"] 仅旗标开时初始化、
     obs[:, -2:] 累积、it 行 yaw_sin/yaw_cos 尾巴——三处同守卫；
     argparse --obs-heading 默认 0；[CFG] 回显旗标开才追加

边界说明：env `_get_observations` 的完整拼装深度耦合 Isaac（robot.data
张量），本机不可实例化——case5 以「真身块 exec + mock parts」覆盖 obs
组装的 heading 段与 reset 回填段，case6 覆盖 cat 宽度契约；既有 11+4 个
parts 块的 Isaac 侧行为本文件不重复（无旗标路径由 case7 静态保证零改动）。

用法（本机 CPU torch / 服务器 .venv_isaac 均可）：
    PYTHONPATH=. python apt_g1/isaac/test_obs_heading.py
全部用例 PASS 时 exit 0。
"""

from __future__ import annotations

import ast
import math
import sys
import textwrap
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ENV_PATH = HERE / "apt_flat_env.py"
EVAL_PATH = HERE / "eval_apt_isaac.py"
TRAIN_PATH = HERE / "train_apt_isaac.py"

GUARD_ENV = "self.cfg.obs_heading"
GUARD_TRAIN = "cli.obs_heading"


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
    """把源码行原样 exec 成可调用体（仅改写输入变量名，计算行逐字不动）。

    rename 的 value 序列 = 形参顺序；同名映射（value==key）用于把隐式
    外部输入显式化进签名；ret = 追加在函数体末尾的 return 表达式源码。
    """
    for old, new in rename.items():
        lines = [ln.replace(old, new) for ln in lines]
    src = f"def {name}({', '.join(rename.values())}):\n" + textwrap.indent(
        "\n".join(lines) + f"\nreturn {ret}", "    "
    )
    ns: dict = {"torch": torch, "math": math}
    exec(compile(src, f"<{name}>", "exec"), ns)
    return ns[name]


def _exec_guard_body(if_node: ast.If, rename: dict[str, str], name: str, ret: str):
    return _exec_lines([ast.unparse(s) for s in if_node.body], rename, name, ret)


def _env_heading_block():
    """apt_flat_env._get_observations 的 obs_heading 块原样 exec。

    签名 (root_quat, init_yaw, parts)——把 self.robot.data.root_quat_w /
    self._init_yaw / parts 显式化为入参，计算行逐字保留。
    """
    tree, _ = _parse(ENV_PATH)
    obs_fn = _find_func(tree, "_get_observations")
    return _exec_guard_body(
        _find_guarded_if(obs_fn, GUARD_ENV),
        {
            "self.robot.data.root_quat_w": "root_quat",
            "self._init_yaw": "init_yaw",
            "parts": "parts",
        },
        "_env_heading_block",
        ret="parts[-1]",
    )


def _env_reset_block():
    """apt_flat_env._reset_idx 的 obs_heading 块（init yaw 回填）原样 exec。"""
    tree, _ = _parse(ENV_PATH)
    reset_fn = _find_func(tree, "_reset_idx")
    return _exec_guard_body(
        _find_guarded_if(reset_fn, GUARD_ENV),
        {"default_root": "default_root", "env_ids": "env_ids", "self._init_yaw": "init_yaw"},
        "_env_reset_init_yaw",
        ret="init_yaw",
    )


def _env_wrap():
    """obs_heading 块内的 yaw_rel wrap 行原样 exec（case4 同源校验用）。"""
    tree, _ = _parse(ENV_PATH)
    obs_fn = _find_func(tree, "_get_observations")
    if_node = _find_guarded_if(obs_fn, GUARD_ENV)
    line = next(
        ast.unparse(s) for s in if_node.body
        if isinstance(s, ast.Assign) and ast.unparse(s).startswith("yaw_rel")
    )
    return _exec_lines(
        [line], {"yaw": "yaw", "self._init_yaw": "init_yaw"}, "_env_wrap", ret="yaw_rel"
    )


def _eval_yaw_of():
    """AST 抽取 eval_apt_isaac.py 嵌套函数 _yaw_of 原样 exec（同源对拍基准）。"""
    tree, src = _parse(EVAL_PATH)
    seg = ast.get_source_segment(src, _find_func(tree, "_yaw_of"))
    assert seg is not None, "_yaw_of source segment missing"
    ns: dict = {"torch": torch, "math": math}
    exec(compile(textwrap.dedent(seg), "<eval._yaw_of>", "exec"), ns)
    return ns["_yaw_of"]


def _is_enclosed_by_guard(fn: ast.FunctionDef, node: ast.AST, guard: str) -> bool:
    """node（按行号）是否位于 fn 内 test==guard 的 If 块行号范围内。"""
    for ifn in ast.walk(fn):
        if isinstance(ifn, ast.If) and ast.unparse(ifn.test) == guard:
            if ifn.lineno <= node.lineno <= (ifn.end_lineno or ifn.lineno):
                return True
    return False


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


def zyx_rotmat(psi: float, theta: float, phi: float) -> torch.Tensor:
    """R = Rz(psi) Ry(theta) Rx(phi)（内在 ZYX 同一旋转的矩阵形，投影 ground truth 用）。"""
    def rz(a):
        c, s = math.cos(a), math.sin(a)
        return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)

    def ry(a):
        c, s = math.cos(a), math.sin(a)
        return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float64)

    def rx(a):
        c, s = math.cos(a), math.sin(a)
        return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=torch.float64)

    return rz(psi) @ ry(theta) @ rx(phi)


def wrap_eval_scalar(d: float) -> float:
    """eval_apt_isaac yaw_err 的标量式（eval L598 原式镜像；case4 与 env 式对拍）。"""
    return (d + math.pi) % (2.0 * math.pi) - math.pi


# --------------------------------------------------------------------------- cases
def case1_eval_same_source_extraction() -> bool:
    """env 真身块（向量 yaw 提取+wrap+拼装）== eval _yaw_of 数值逐点 + 字面同构。"""
    heading_block = _env_heading_block()
    yaw_of = _eval_yaw_of()
    _, src_eval = _parse(EVAL_PATH)
    tree_eval, _ = _parse(EVAL_PATH)
    eval_seg = ast.get_source_segment(src_eval, _find_func(tree_eval, "_yaw_of")) or ""

    degs = [0.0, 45.0, 90.0, -90.0, 180.0, 30.0, -170.0, 123.456]
    quats = torch.stack([quat_from_zyx_euler(math.radians(d), 0.0, 0.0) for d in degs])
    init_yaw = torch.zeros(len(degs), dtype=torch.float64)
    parts: list = []
    sin_cos = heading_block(quats, init_yaw, parts)

    ok = True
    for i, d in enumerate(degs):
        y_eval = yaw_of(quats[i])  # eval 标量公式（原始提取，∈(−π, π]）
        want = wrap_eval_scalar(math.radians(d))
        ok &= abs(sin_cos[i, 0].item() - math.sin(want)) < 1e-9
        ok &= abs(sin_cos[i, 1].item() - math.cos(want)) < 1e-9
        # eval 原始提取与 eval 口径 wrap 后的差 = 整数个 2π（wrap_pi 一致性）
        ok &= abs(wrap_eval_scalar(y_eval - want)) < 1e-9
    # 块结构：单条 append、输出 2 维（=obs 尾块）
    ok &= len(parts) == 1 and tuple(parts[0].shape) == (len(degs), 2)
    # 字面同构：env 块与 eval _yaw_of 的 atan2 提取式在归一空白后逐字符相等
    expr = "torch.atan2(2.0*(w*z+x*y),1.0-2.0*(y*y+z*z))"
    ok &= expr in ast.unparse(_find_guarded_if(_find_func(ast.parse(
        ENV_PATH.read_text(encoding="utf-8")), "_get_observations"), GUARD_ENV
    )).replace(" ", "")
    ok &= expr in eval_seg.replace(" ", "")
    return _report(
        "case1 eval 同源对拍（数值逐点+atan2 字面同构）", ok,
        f"quats={len(degs)}  eval=_yaw_of AST-exec  env=obs_heading 块 AST-exec",
    )


def case2_pure_yaw_quats() -> bool:
    """绕 z 轴纯 yaw quat：env 真身块提取=角度；180° 钉 +π（wrap 后 −π）。"""
    heading_block = _env_heading_block()
    env_wrap = _env_wrap()
    yaw_of = _eval_yaw_of()
    ok = True
    for d in (0.0, 45.0, 90.0, -90.0, 180.0, -180.0, 270.0):
        half = math.radians(d) / 2.0
        q = torch.tensor([[math.cos(half), 0.0, 0.0, math.sin(half)]], dtype=torch.float64)
        sc = heading_block(q, torch.zeros(1, dtype=torch.float64), [])
        want = wrap_eval_scalar(math.radians(d))
        ok &= abs(sc[0, 0].item() - math.sin(want)) < 1e-9
        ok &= abs(sc[0, 1].item() - math.cos(want)) < 1e-9
        # eval 原始提取同验
        y = yaw_of(q[0])
        ok &= abs(wrap_eval_scalar(y - want)) < 1e-12
    # 180° 钉：q=(0,0,0,1) → atan2(0,−1)=+π（正支），env wrap 后 = −π
    q180 = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
    ok &= abs(yaw_of(q180) - math.pi) < 1e-12
    ok &= abs(env_wrap(torch.tensor([math.pi], dtype=torch.float64),
                       torch.zeros(1, dtype=torch.float64))[0].item()
              - (-math.pi)) < 1e-9
    return _report("case2 纯 yaw quat 提取（0/45/90/−90/180/−180/270°）", ok)


def case3_d044_zyx_euler() -> bool:
    """D044 i:zyx 欧拉序样例：通用 (ψ,θ,φ) quat 提取恢复 ψ + 矩阵投影交叉验证。"""
    yaw_of = _eval_yaw_of()
    ok = True
    combos = [
        (0.0, 0.0, 0.0),
        (45.0, -15.0, 10.0),
        (90.0, 5.0, -8.0),
        (-90.0, -12.0, 3.0),
        (180.0, 9.0, -6.0),
        (30.0, 0.0, 25.0),   # θ=0：投影航向 == ψ（任意 roll），几何 ground truth
        (-135.0, 0.0, -20.0),
    ]
    for dpsi, dth, dphi in combos:
        psi, th, phi = map(math.radians, (dpsi, dth, dphi))
        q = quat_from_zyx_euler(psi, th, phi)
        y = yaw_of(q)
        ok &= abs(wrap_eval_scalar(y - psi)) < 1e-12
        if dth == 0.0:
            xw = zyx_rotmat(psi, th, phi) @ torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
            proj = math.atan2(xw[1].item(), xw[0].item())
            ok &= abs(wrap_eval_scalar(proj - psi)) < 1e-12
            ok &= abs(wrap_eval_scalar(y - proj)) < 1e-12
    return _report("case3 D044 i:zyx 欧拉序 quat（含投影 ground truth）", ok)


def case4_wrap_pi_boundaries() -> bool:
    """wrap_pi 边界：env 真身向量式 == eval 标量式（L598）；±π→−π、±3π/2→∓π/2。"""
    env_wrap = _env_wrap()
    grid = torch.arange(-4.0 * math.pi, 4.0 * math.pi + 1e-9, 0.001, dtype=torch.float64)
    zero = torch.zeros_like(grid)
    env_vals = env_wrap(grid, zero)
    ok = env_vals.shape == grid.shape
    for g, v in zip(grid.tolist(), env_vals.tolist()):
        ok &= abs(v - wrap_eval_scalar(g)) < 1e-9
        ok &= -math.pi - 1e-9 <= v < math.pi + 1e-9
    pins = {
        math.pi: -math.pi,
        -math.pi: -math.pi,
        1.5 * math.pi: -0.5 * math.pi,
        -1.5 * math.pi: 0.5 * math.pi,
        0.0: 0.0,
    }
    for d, want in pins.items():
        got = env_wrap(torch.tensor([d], dtype=torch.float64), zero[:1])[0].item()
        ok &= abs(got - want) < 1e-9
    return _report("case4 wrap_pi 边界（env 真身式 == eval L598 标量式，±π 钉）", ok)


def case5_reset_then_yaw_offset_sequence() -> bool:
    """reset→step 合成序列：复位首步 (0,1)；yaw 偏移与通用 yaw0 口径。"""
    reset_block = _env_reset_block()
    heading_block = _env_heading_block()
    env_ids = torch.tensor([0], dtype=torch.long)
    init_yaw = torch.zeros(1, dtype=torch.float64)

    # 复位：identity quat（与 env reset 写入值 [1,0,0,0] 同）
    root0 = torch.zeros(1, 7, dtype=torch.float64)
    root0[0, 3] = 1.0
    reset_block(root0, env_ids, init_yaw)
    ok = abs(init_yaw[0].item()) < 1e-12

    # 首步 obs：机体 yaw=identity → (sin,cos)=(0,1)（§5m spot check 钉）
    parts: list = []
    sc = heading_block(root0[:, 3:7], init_yaw, parts)
    ok &= abs(sc[0, 0].item()) < 1e-9 and abs(sc[0, 1].item() - 1.0) < 1e-9

    # 机体 yaw 偏移序列（yaw0=0）：45° / −90° / 180°
    for d, want_s, want_c in [
        (45.0, math.sin(math.radians(45.0)), math.cos(math.radians(45.0))),
        (-90.0, -1.0, 0.0),
        (180.0, 0.0, -1.0),
    ]:
        q = quat_from_zyx_euler(math.radians(d), 0.0, 0.0).unsqueeze(0)
        sc = heading_block(q, init_yaw, [])
        ok &= abs(sc[0, 0].item() - want_s) < 1e-9
        ok &= abs(sc[0, 1].item() - want_c) < 1e-9

    # 通用口径：yaw0=30° 后机体 yaw=75° → yaw_rel=45°
    root30 = torch.zeros(1, 7, dtype=torch.float64)
    root30[0, 3:7] = quat_from_zyx_euler(math.radians(30.0), 0.0, 0.0)
    reset_block(root30, env_ids, init_yaw)
    ok &= abs(init_yaw[0].item() - math.radians(30.0)) < 1e-12
    q75 = quat_from_zyx_euler(math.radians(75.0), 0.0, 0.0).unsqueeze(0)
    sc = heading_block(q75, init_yaw, [])
    ok &= abs(sc[0, 0].item() - math.sin(math.radians(45.0))) < 1e-9
    ok &= abs(sc[0, 1].item() - math.cos(math.radians(45.0))) < 1e-9
    return _report("case5 reset→step 合成序列（首步 (0,1)，偏移与通用 yaw0）", ok)


def case6_b_arm_assembly_dims() -> bool:
    """D051 b 臂 obs 拼装宽度：91→105→134→137→139；无旗标 = 137。"""
    # base 块宽（_get_observations parts 固定前 11 块；mode_oh=5 档）
    base_widths = [3, 3, 3, 29, 29, 3, 2, 12, 5, 1, 1]
    ok = sum(base_widths) == 91  # cfg.observation_space 默认（全旗标关，env assert 在位）
    latent_widths = list(base_widths)
    latent_widths[6] = 16                       # latent _last_phase 2→16（train +14）
    b_arm = latent_widths + [29, 3]             # +29 res 反馈 +3 vb 反馈 → 137
    ok &= sum(latent_widths) == 105 and sum(b_arm) == 137
    ok &= sum(b_arm + [2]) == 139               # 旗标开：+2 heading 块
    # cat 层面数值对拍
    parts = [torch.zeros(4, w) for w in b_arm]
    ok &= torch.cat(parts, dim=1).shape[1] == 137          # 无旗标路径
    ok &= torch.cat(parts + [torch.zeros(4, 2)], dim=1).shape[1] == 139
    # train 侧 bump 块：+2 与身份断言字面 139 在 cli.obs_heading 守卫内
    tree_train, _ = _parse(TRAIN_PATH)
    main_fn = _find_func(tree_train, "main")
    bump_if = next(
        (n for n in ast.walk(main_fn)
         if isinstance(n, ast.If) and ast.unparse(n.test) == GUARD_TRAIN
         and "cfg.observation_space += 2" in ast.unparse(n)),
        None,
    )
    ok &= bump_if is not None
    if bump_if is not None:
        body = ast.unparse(bump_if)
        ok &= "cfg.observation_space += 2" in body
        ok &= "cfg.observation_space == _obs_dim_pre_heading + 2" in body
        ok &= "cfg.observation_space == 139" in body
    return _report("case6 b 臂拼装维度 91/105/134/137/139（静态式+cat 数值）", ok)


def case7_no_flag_zero_change_static() -> bool:
    """无旗标路径零变化（AST 静态）：全部新代码路径都在 obs_heading 守卫内。"""
    tree_env, src_env = _parse(ENV_PATH)
    ok = True
    # ① cfg 字段默认 False
    cfg_cls = next(
        n for n in ast.walk(tree_env)
        if isinstance(n, ast.ClassDef) and n.name == "AptFlatG1EnvCfg"
    )
    field = next(
        n for n in ast.walk(cfg_cls)
        if isinstance(n, ast.AnnAssign) and ast.unparse(n.target) == "obs_heading"
    )
    ok &= field.value is not None and ast.unparse(field.value) == "False"
    # ② __init__：_init_yaw 先置 None（旗标关不分配张量），zeros 分配在守卫内
    init_fn = _find_func(tree_env, "__init__")
    init_src = ast.get_source_segment(src_env, init_fn) or ""
    ok &= "self._init_yaw = None" in init_src
    alloc_if = _find_guarded_if(init_fn, GUARD_ENV)
    ok &= "self._init_yaw = torch.zeros" in ast.unparse(alloc_if)
    # ③ _reset_idx：init yaw 回填在守卫内、位于 write_root_state_to_sim 之后
    reset_fn = _find_func(tree_env, "_reset_idx")
    reset_if = _find_guarded_if(reset_fn, GUARD_ENV)
    ok &= "self._init_yaw[env_ids]" in ast.unparse(reset_if)
    write_ln = next(
        n.lineno for n in ast.walk(reset_fn)
        if isinstance(n, ast.Call) and "write_root_state_to_sim" in ast.unparse(n)
    )
    ok &= reset_if.lineno > write_ln
    # ④ _get_observations：追加块在守卫内、恰一条 append（2 维）、位于 vb_w 之后
    obs_fn = _find_func(tree_env, "_get_observations")
    obs_if = _find_guarded_if(obs_fn, GUARD_ENV)
    appends = [s for s in obs_if.body if ast.unparse(s).startswith("parts.append(")]
    ok &= len(appends) == 1
    ok &= "torch.stack([torch.sin(yaw_rel), torch.cos(yaw_rel)], dim=1)" in ast.unparse(obs_if)
    vb_ln = next(
        n.lineno for n in ast.walk(obs_fn)
        if isinstance(n, ast.Call) and ast.unparse(n) == "parts.append(self._last_vb_w)"
    )
    ok &= obs_if.lineno > vb_ln
    # ⑤ cat 的 parts 列表原式零改动（obs = torch.cat([...parts], dim=-1) 仍在）
    ok &= "obs = torch.cat(" in (ast.get_source_segment(src_env, obs_fn) or "")
    # ⑥ train：bump 在 decft 整体覆写之后（组合序稳健）、守卫内
    tree_train, _ = _parse(TRAIN_PATH)
    main_fn = _find_func(tree_train, "main")
    bump_aug = next(
        n for n in ast.walk(main_fn)
        if isinstance(n, ast.AugAssign) and ast.unparse(n.target) == "cfg.observation_space"
        and isinstance(n.op, ast.Add) and ast.unparse(n.value) == "2"
        and _is_enclosed_by_guard(main_fn, n, GUARD_TRAIN)
    )
    ok &= _is_enclosed_by_guard(main_fn, bump_aug, GUARD_TRAIN)
    decft_assign = next(
        n for n in ast.walk(main_fn)
        if isinstance(n, ast.Assign)
        and any(ast.unparse(t) == "cfg.observation_space" for t in n.targets)
        and "DECFT_OBS_DIM" in ast.unparse(n)
    )
    ok &= bump_aug.lineno > decft_assign.lineno
    # ⑦ 全部 cfg.observation_space 既有 bump 块原样在位（未动既有行）
    # （历史 +2 共 3 条：token_phase_obs/to42_sel 各 1 + 本臂 1）
    aug_txts = sorted(
        ast.unparse(n) for n in ast.walk(main_fn)
        if isinstance(n, ast.AugAssign) and ast.unparse(n.target) == "cfg.observation_space"
    )
    ok &= aug_txts.count("cfg.observation_space += 2") == 3
    for keep in ("cfg.observation_space += cfg.elev_grid * cfg.elev_grid",
                 "cfg.observation_space += 12",
                 "cfg.observation_space += 14",
                 "cfg.observation_space += 29",
                 "cfg.observation_space += 3",
                 "cfg.observation_space += 62"):
        ok &= keep in aug_txts
    return _report(
        "case7 无旗标零变化静态断言（env 3 处守卫 + train bump 守卫/位置/既有式原样）", ok,
        "obs_heading=0 时 observation_space 表达式与改动前逐字节一致",
    )


def case8_train_logging_hooks() -> bool:
    """训练日志钩子静态在位：hist 初始化/obs 尾 2 维累积/it 行尾巴，同守卫。"""
    tree_train, src_train = _parse(TRAIN_PATH)
    main_fn = _find_func(tree_train, "main")
    guarded = "\n".join(
        ast.unparse(n) for n in ast.walk(main_fn)
        if isinstance(n, ast.If) and ast.unparse(n.test) == GUARD_TRAIN
    )
    ok = True
    # ast.unparse 把字符串字面量归一为单引号，两种引号形态都认
    ok &= ("hist['yaw_rel_mean'] = []" in guarded
           or 'hist["yaw_rel_mean"] = []' in guarded)
    ok &= "yaw_rel_sum += obs[:, -2:].sum(0)" in guarded
    ok &= ("hist['yaw_rel_mean'].append(" in guarded
           or 'hist["yaw_rel_mean"].append(' in guarded)
    ok &= "yaw_sin=" in guarded and "yaw_cos=" in guarded
    # [CFG] 回显：条件表达式旗标开才追加 obs_heading=1（回显逐字不变纪律）
    ok &= ('" obs_heading=1" if cli.obs_heading else ""' in src_train
           and "heading_cfg" in src_train)
    # argparse：--obs-heading 存在、默认 0
    flag_pos = src_train.find('"--obs-heading"')
    ok &= flag_pos > 0 and "default=0" in src_train[flag_pos:flag_pos + 80]
    return _report("case8 训练日志 yaw_rel 钩子（hist/累积/it 行尾巴同守卫）", ok)


def main() -> None:
    cases = [
        case1_eval_same_source_extraction,
        case2_pure_yaw_quats,
        case3_d044_zyx_euler,
        case4_wrap_pi_boundaries,
        case5_reset_then_yaw_offset_sequence,
        case6_b_arm_assembly_dims,
        case7_no_flag_zero_change_static,
        case8_train_logging_hooks,
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
