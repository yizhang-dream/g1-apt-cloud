"""D054 yaw-rew 臂数值单测：航向误差奖励 +yaw_rew_scale·(1−|yaw_rel|/π)（纯 torch/numpy，无 isaaclab 依赖）。

被测对象（DS_CONTINUOUS_EXECUTION_PLAN §5n 预注册；apt_flat_env.py 的
yaw_rew_scale 奖励路径 + train_apt_isaac.py 的 --yaw-rew-scale / --vx-min
接线）。本文件**不 import apt_flat_env**（其模块级 import isaaclab 本机不可
用），改用 AST 从源文件提取真身代码/断言结构做同源对拍（计算行逐字 exec，
仅改写输入变量名）——方法与 test_obs_heading.py 同款：

  1. yaw_rew 数值：yaw_rel=0 → 奖励增量=scale；|yaw_rel|=π → 0；45° →
     scale·(1−0.25)=0.75·scale；±180° 网格逐点 = scale·(1−|wrap(d)|/π)；
     yaw0≠0 通用口径（init 30°/yaw 75° → yaw_rel=45°）与 ±对称、单调性
  2. 与 obs 块同源性：同 quat 同 _init_yaw 下，奖励块 yaw_rel 与
     obs_heading 块 [sin,cos] 反解的 yaw_rel 数值逐点相等；且两块的
     atan2 提取式 / wrap 式 / 四元数分量别名行 **token 逐字同构**（D048b
     双旋转前科——错符号=教反方向，效度核心）
  3. 共享 _init_yaw 守卫：__init__ 分配与 _reset_idx 回填的 elif
     yaw_rew 支在位、分支体与 obs_heading 支 **token 逐字相同**（防漂移）；
     mock exec 证明 yaw_rew 单开（obs_heading=0, scale>0）分配 (N,) float32
     张量并按同式回填（30° quat → 30°）
  4. obs 维度不随 yaw_rew 走：_get_observations 全函数无 yaw_rew 引用、
     heading 追加块唯一守卫仍= obs_heading；train main 无任何 yaw_rew/vx_min
     关联的 If 分支，observation_space 既有 bump 式原样（+2 计数仍 3 且全在
     cli.obs_heading 守卫内）
  5. 双零默认零变化静态断言：env cfg yaw_rew_scale 默认 0.0；train 两旗标
     默认 0.0；_get_rewards 新块全在 `yaw_rew_scale > 0.0` 守卫内、分解快照
     仅 `yaw_rew_term is not None` 时补记；_last_rew_terms 基础键=旧 6 键
     原样；DIAG_KEYS 不含 yaw_rew（diag 键集合不变）；[CFG] 回显条件式
     （>0 才显示，默认逐字不变）
  6. train 接线在位：cfg.yaw_rew_scale / cfg.vx_min 两行 Assign 且位于 env
     构造之前；env cfg.vx_min 默认 0.0 + uniform(vx_min, vx_max) 采样行既有；
     DecFtPolicy 构造仍只传 vx_max（decft 路径无 vx_min 消费，D054 curr 臂
     走 latent 路径——范围声明断言）

边界说明：env `_get_rewards`/`_get_observations` 的完整拼装深度耦合 Isaac
（robot.data 张量），本机不可实例化——以「真身块 exec + mock 张量」覆盖
奖励项与回填段的数值行为，Isaac 侧集成行为由 D053 同款边界说明豁免
（双零默认路径由 case4/case5 静态保证零改动）。

用法（本机 CPU torch / 服务器 .venv_isaac 均可）：
    PYTHONPATH=. python apt_g1/isaac/test_yaw_rew.py
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
GUARD_SNAPSHOT = "yaw_rew_term is not None"
GUARD_TRAIN = "cli.obs_heading"

SCALE = 0.4  # §5n 预注册臂值（量级对齐 heading_scale）


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


def _env_yaw_rew_block():
    """apt_flat_env._get_rewards 的 yaw_rew 守卫块原样 exec。

    签名 (root_quat, init_yaw, reward, scale)——把 self.robot.data.root_quat_w /
    self._init_yaw / self.cfg.yaw_rew_scale 显式化为入参，计算行逐字保留。
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


def _env_heading_block():
    """apt_flat_env._get_observations 的 obs_heading 块原样 exec（同源对拍基准）。"""
    tree, _ = _parse(ENV_PATH)
    obs_fn = _find_func(tree, "_get_observations")
    return _exec_guard_body(
        _find_guarded_if(obs_fn, GUARD_OBS),
        {
            "self.robot.data.root_quat_w": "root_quat",
            "self._init_yaw": "init_yaw",
            "parts": "parts",
        },
        "_env_heading_block",
        ret="parts[-1]",
    )


def _stmt_tokens(if_node: ast.If, prefixes: tuple[str, ...]) -> dict[str, str]:
    """守卫块体内以 prefixes 开头的语句，归一空白后的 token 串（字面同构对比用）。"""
    out: dict[str, str] = {}
    for s in if_node.body:
        u = ast.unparse(s)
        for p in prefixes:
            if u.startswith(p):
                out[p] = u.replace(" ", "")
    return out


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


def quat_pure_yaw(deg: float) -> torch.Tensor:
    """绕 z 轴纯 yaw quat（w-first）。"""
    half = math.radians(deg) / 2.0
    return torch.tensor(
        [[math.cos(half), 0.0, 0.0, math.sin(half)]], dtype=torch.float64
    )


def wrap_eval_scalar(d: float) -> float:
    """eval_apt_isaac yaw_err 的标量式（wrap_pi 口径）。"""
    return (d + math.pi) % (2.0 * math.pi) - math.pi


# --------------------------------------------------------------------------- cases
def case1_yaw_rew_numeric_pins() -> bool:
    """yaw_rew 数值：0→scale、|yaw_rel|=π→0、45°→0.75·scale、网格逐点+yaw0 通用口径。"""
    blk = _env_yaw_rew_block()
    ok = True

    # ±180° 每 15° 网格：delta == scale·(1−|wrap(d)|/π)
    degs = list(range(-180, 181, 15))
    quats = torch.cat([quat_pure_yaw(d) for d in degs])
    zero = torch.zeros(len(degs), dtype=torch.float64)
    base = torch.zeros(len(degs), dtype=torch.float64)
    reward, term, _ = blk(quats, zero, base, SCALE)
    for i, d in enumerate(degs):
        want = 1.0 - abs(wrap_eval_scalar(math.radians(d))) / math.pi
        ok &= abs(term[i].item() - want) < 1e-9
        ok &= abs(reward[i].item() - SCALE * want) < 1e-9

    # 预注册钉：yaw_rel=0 → scale；45° → scale·0.75；90° → 0.5·scale；180° → 0
    for d, frac in [(0.0, 1.0), (45.0, 0.75), (90.0, 0.5), (-90.0, 0.5), (180.0, 0.0)]:
        r, t, _ = blk(quat_pure_yaw(d), torch.zeros(1, dtype=torch.float64),
                      torch.zeros(1, dtype=torch.float64), SCALE)
        ok &= abs(r[0].item() - SCALE * frac) < 1e-9
        ok &= abs(t[0].item() - frac) < 1e-9

    # 通用 yaw0 口径：init=30°，yaw=75° → yaw_rel=45° → 0.75·scale；
    # yaw=−105° → yaw_rel=−135° → 0.25·scale
    q30 = quat_from_zyx_euler(math.radians(30.0), 0.0, 0.0).unsqueeze(0)
    init30 = torch.tensor([math.radians(30.0)], dtype=torch.float64)
    for dyaw, frac in [(75.0, 0.75), (-105.0, 0.25)]:
        q = quat_from_zyx_euler(math.radians(dyaw), 0.0, 0.0).unsqueeze(0)
        r, t, yrel = blk(q, init30, torch.zeros(1, dtype=torch.float64), SCALE)
        ok &= abs(t[0].item() - frac) < 1e-9
        ok &= abs(r[0].item() - SCALE * frac) < 1e-9
        ok &= abs(yrel[0].item() - wrap_eval_scalar(math.radians(dyaw - 30.0))) < 1e-9

    # ±对称（abs 口径）与单调（越正对越高分）
    _, t45, _ = blk(quat_pure_yaw(45.0), torch.zeros(1, dtype=torch.float64),
                    torch.zeros(1, dtype=torch.float64), SCALE)
    _, tm45, _ = blk(quat_pure_yaw(-45.0), torch.zeros(1, dtype=torch.float64),
                     torch.zeros(1, dtype=torch.float64), SCALE)
    _, t30, _ = blk(quat_pure_yaw(30.0), torch.zeros(1, dtype=torch.float64),
                    torch.zeros(1, dtype=torch.float64), SCALE)
    _, t60, _ = blk(quat_pure_yaw(60.0), torch.zeros(1, dtype=torch.float64),
                    torch.zeros(1, dtype=torch.float64), SCALE)
    ok &= abs(t45[0].item() - tm45[0].item()) < 1e-12
    ok &= t30[0].item() > t45[0].item() > t60[0].item()  # 越正对越高分
    return _report(
        "case1 yaw_rew 数值（0→scale、π→0、45°→0.75·scale、±180° 网格+yaw0 通用）", ok,
        f"scale={SCALE}  grid=±180°/15°",
    )


def case2_same_source_with_obs_block() -> bool:
    """同源性：同 quat 同 _init_yaw 下奖励块 yaw_rel == obs 块反解 yaw_rel + 公式 token 同构。"""
    rew_blk = _env_yaw_rew_block()
    obs_blk = _env_heading_block()
    combos = [
        (0.0, 0.0, 0.0), (45.0, 0.0, 0.0), (-90.0, 0.0, 0.0), (180.0, 0.0, 0.0),
        (75.0, -12.0, 5.0), (-135.0, 0.0, -20.0), (30.0, 8.0, 3.0),
        (123.456, 0.0, 0.0), (-170.0, 4.0, -2.0),
    ]
    inits = [0.0, 30.0, -45.0, 170.0]
    ok = True
    n = 0
    for dpsi, dth, dphi in combos:
        q = quat_from_zyx_euler(math.radians(dpsi), math.radians(dth),
                                math.radians(dphi)).unsqueeze(0)
        for init in inits:
            init_yaw = torch.tensor([math.radians(init)], dtype=torch.float64)
            parts: list = []
            sc = obs_blk(q, init_yaw, parts)
            obs_yaw_rel = torch.atan2(sc[0, 0], sc[0, 1])
            _, _, rew_yaw_rel = rew_blk(
                q, init_yaw, torch.zeros(1, dtype=torch.float64), SCALE
            )
            ok &= abs(obs_yaw_rel.item() - rew_yaw_rel[0].item()) < 1e-9
            n += 1

    # 字面同构：两块的 atan2 提取式 / wrap 式 / quat 别名行 token 逐字相同
    tree, _ = _parse(ENV_PATH)
    obs_if = _find_guarded_if(_find_func(tree, "_get_observations"), GUARD_OBS)
    rew_if = _find_guarded_if(_find_func(tree, "_get_rewards"), GUARD_YAW_REW)
    prefixes = ("yaw =", "yaw_rel =", "w, x, y, z =")
    toks_obs = _stmt_tokens(obs_if, prefixes)
    toks_rew = _stmt_tokens(rew_if, prefixes)
    for p in prefixes:
        ok &= p in toks_obs and p in toks_rew and toks_obs[p] == toks_rew[p]
    # atan2 提取式显式钉（eval_apt_isaac._yaw_of / elevation 分支三方同源式）
    ok &= toks_rew.get("yaw =") == (
        "yaw=torch.atan2(2.0*(w*z+x*y),1.0-2.0*(y*y+z*z))"
    )
    return _report(
        "case2 与 obs 块同源（9×4 quat×init 数值逐点 + atan2/wrap/别名 token 同构）", ok,
        f"{n} 点对拍  block-token 逐字相等",
    )


def case3_shared_init_yaw_guards() -> bool:
    """共享 _init_yaw：elif 支在位且与 obs 支 token 逐字相同；mock exec 分配/回填同验。"""
    tree, src_env = _parse(ENV_PATH)
    ok = True

    # __init__：先 None 再 if obs_heading 分配 + elif yaw_rew 同体分支
    init_fn = _find_func(tree, "__init__")
    ok &= "self._init_yaw = None" in (ast.get_source_segment(src_env, init_fn) or "")
    alloc_if = _find_guarded_if(init_fn, GUARD_OBS)
    ok &= len(alloc_if.orelse) == 1 and isinstance(alloc_if.orelse[0], ast.If)
    alloc_elif = alloc_if.orelse[0]
    ok &= ast.unparse(alloc_elif.test) == GUARD_YAW_REW
    ok &= [ast.unparse(s) for s in alloc_if.body] == [
        ast.unparse(s) for s in alloc_elif.body
    ]

    # _reset_idx：回填 if/elif 同体分支、elif 位于 write_root_state_to_sim 之后
    reset_fn = _find_func(tree, "_reset_idx")
    reset_if = _find_guarded_if(reset_fn, GUARD_OBS)
    ok &= len(reset_if.orelse) == 1 and isinstance(reset_if.orelse[0], ast.If)
    reset_elif = reset_if.orelse[0]
    ok &= ast.unparse(reset_elif.test) == GUARD_YAW_REW
    ok &= [ast.unparse(s) for s in reset_if.body] == [
        ast.unparse(s) for s in reset_elif.body
    ]
    ok &= "self._init_yaw[env_ids]" in ast.unparse(reset_if)
    write_ln = next(
        n.lineno for n in ast.walk(reset_fn)
        if isinstance(n, ast.Call) and "write_root_state_to_sim" in ast.unparse(n)
    )
    ok &= reset_if.lineno > write_ln

    # mock exec（yaw_rew 单开语义）：elif 支分配 (N,) float32 张量
    alloc_lines = [ast.unparse(s) for s in alloc_elif.body]
    alloc_mock = _exec_lines(
        alloc_lines, {"self": "self"}, "_alloc_mock", ret="self._init_yaw"
    )
    yaw = alloc_mock(SimpleNamespace(num_envs=4, device="cpu"))
    ok &= tuple(yaw.shape) == (4,) and yaw.dtype == torch.float32

    # mock exec：if/elif 两支回填同输入数值相等（30° quat → ±30°）
    rename = {"self._init_yaw": "init_yaw", "default_root": "default_root",
              "env_ids": "env_ids"}
    backfill_if = _exec_guard_body(reset_if, rename, "_backfill_if", ret="init_yaw")
    backfill_elif = _exec_guard_body(reset_elif, rename, "_backfill_elif", ret="init_yaw")
    root30 = torch.zeros(2, 7, dtype=torch.float64)
    root30[:, 3:7] = quat_from_zyx_euler(math.radians(30.0), 0.0, 0.0)
    env_ids = torch.tensor([0, 1], dtype=torch.long)
    y_if = backfill_if(torch.zeros(2, dtype=torch.float64), root30.clone(), env_ids)
    y_elif = backfill_elif(torch.zeros(2, dtype=torch.float64), root30.clone(), env_ids)
    ok &= torch.allclose(y_if, y_elif)
    ok &= bool((y_if - math.radians(30.0)).abs().max() < 1e-12)
    return _report(
        "case3 共享 _init_yaw 守卫（init/reset elif 同体 token 同构 + mock 分配/回填）", ok,
        "obs_heading 单开路径不进 elif 支，逐位不变",
    )


def case4_obs_dim_unchanged() -> bool:
    """obs 维度只跟 obs_heading 走：obs 函数无 yaw_rew 引用；train 无 yaw_rew/vx_min 分支。"""
    tree_env, src_env = _parse(ENV_PATH)
    ok = True
    obs_fn = _find_func(tree_env, "_get_observations")
    obs_src = ast.get_source_segment(src_env, obs_fn) or ""
    ok &= "yaw_rew" not in obs_src  # yaw_rew 单开不加 obs 维度（全函数零引用）
    obs_if = _find_guarded_if(obs_fn, GUARD_OBS)
    appends = [s for s in obs_if.body if ast.unparse(s).startswith("parts.append(")]
    ok &= len(appends) == 1  # 唯一追加块仍只属 obs_heading 守卫
    ok &= "obs.shape[1] == self.cfg.observation_space" in obs_src  # 维度断言原样

    tree_train, _ = _parse(TRAIN_PATH)
    main_fn = _find_func(tree_train, "main")
    # main 内不得出现 yaw_rew/vx_min 关联的 If 分支（wiring 全是裸 Assign，
    # observation_space bump 只跟 cli.obs_heading 走）
    ok &= not [
        n for n in ast.walk(main_fn)
        if isinstance(n, ast.If)
        and ("yaw_rew_scale" in ast.unparse(n.test) or "vx_min" in ast.unparse(n.test))
    ]
    # 既有 bump 式原样：+= 2 恰 3 条，其中恰 1 条（D053 bump）在 cli.obs_heading
    # 守卫内；其余历史 bump 的守卫归属不被本臂改动
    augs = [
        n for n in ast.walk(main_fn)
        if isinstance(n, ast.AugAssign)
        and ast.unparse(n.target) == "cfg.observation_space"
    ]
    ok &= sum(ast.unparse(n.value) == "2" for n in augs) == 3
    ok &= sum(
        ast.unparse(n.value) == "2"
        and _is_enclosed_by_guard(main_fn, n, GUARD_TRAIN)
        for n in augs
    ) == 1
    return _report(
        "case4 obs 维度不随 yaw_rew 走（env 零引用 + train 无新分支 + 既有 bump 原样）", ok,
    )


def case5_double_zero_defaults() -> bool:
    """双零默认零变化静态断言：cfg/CLI 默认 0、守卫在位、键集合与回显逐字不变。"""
    tree_env, _ = _parse(ENV_PATH)
    ok = True
    # ① env cfg 默认 0.0
    cfg_cls = next(
        n for n in ast.walk(tree_env)
        if isinstance(n, ast.ClassDef) and n.name == "AptFlatG1EnvCfg"
    )
    field = next(
        n for n in ast.walk(cfg_cls)
        if isinstance(n, ast.AnnAssign) and ast.unparse(n.target) == "yaw_rew_scale"
    )
    ok &= field.value is not None and ast.unparse(field.value) == "0.0"
    # ② 奖励新块全在 `yaw_rew_scale > 0.0` 守卫内；快照仅 None 守卫补记
    rew_fn = _find_func(tree_env, "_get_rewards")
    rew_if = _find_guarded_if(rew_fn, GUARD_YAW_REW)
    ok &= "reward = reward + self.cfg.yaw_rew_scale * yaw_rew_term" in ast.unparse(rew_if)
    snap_if = next(
        n for n in ast.walk(rew_fn)
        if isinstance(n, ast.If) and ast.unparse(n.test) == GUARD_SNAPSHOT
    )
    ok &= 'self._last_rew_terms["yaw_rew"] = yaw_rew_term' in ast.unparse(snap_if).replace(
        "'", '"'
    )
    # ③ _last_rew_terms 基础键 = 旧 6 键原样（顺序含）
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
    # ④ train：两旗标默认 0.0、DIAG_KEYS 不含 yaw_rew
    _, src_train = _parse(TRAIN_PATH)
    ok &= '"--yaw-rew-scale", type=float, default=0.0' in src_train
    ok &= '"--vx-min", type=float, default=0.0' in src_train
    tree_train, _ = _parse(TRAIN_PATH)
    train_main = _find_func(tree_train, "main")
    diag_assign = next(
        n for n in ast.walk(train_main)
        if isinstance(n, ast.Assign)
        and any(ast.unparse(t) == "DIAG_KEYS" for t in n.targets)
    )
    diag_keys = [e.value for e in diag_assign.value.elts]
    ok &= "yaw_rew" not in diag_keys
    ok &= diag_keys == ["track_xy", "track_yaw", "upright", "height", "stillness"]
    # ⑤ [CFG] 回显条件式（>0 才显示）且拼在 print 尾部、定义在 print 之前
    echo_defs = {}
    for name in ("yaw_rew_cfg", "vx_min_cfg"):
        a = next(
            n for n in ast.walk(train_main)
            if isinstance(n, ast.Assign)
            and any(ast.unparse(t) == name for t in n.targets)
        )
        echo_defs[name] = a
    u_yaw = ast.unparse(echo_defs["yaw_rew_cfg"]).replace("'", '"')
    u_vx = ast.unparse(echo_defs["vx_min_cfg"]).replace("'", '"')
    ok &= "if cli.yaw_rew_scale > 0.0 else" in u_yaw
    ok &= "if cli.vx_min > 0.0 else" in u_vx
    ok &= u_yaw.endswith('else ""') and u_vx.endswith('else ""')
    print_call = next(
        n for n in ast.walk(train_main)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name) and n.func.id == "print"
        and "yaw_rew_cfg" in ast.unparse(n)
    )
    # f-string 体含新尾巴（ast.unparse 引号风格无关，只对字段序列）
    ok &= "{heading_cfg}{yaw_rew_cfg}{vx_min_cfg}" in ast.unparse(print_call)
    ok &= echo_defs["yaw_rew_cfg"].lineno < print_call.lineno
    ok &= echo_defs["vx_min_cfg"].lineno < print_call.lineno
    return _report(
        "case5 双零默认零变化（cfg/CLI 默认 0 + 守卫在位 + 键集合/DIAG_KEYS/回显不变）", ok,
    )


def case6_train_wiring_and_env_vx() -> bool:
    """train 接线在位（cfg 两行、位于 env 构造前）+ env vx_min 既有消费 + decft 范围声明。"""
    tree_env, src_env = _parse(ENV_PATH)
    tree_train, _ = _parse(TRAIN_PATH)
    ok = True
    train_main = _find_func(tree_train, "main")
    # ① 两行 cfg 接线（裸 Assign，D053 obs_heading 接线点同款）
    wires = {}
    for name in ("cfg.yaw_rew_scale", "cfg.vx_min"):
        a = next(
            n for n in ast.walk(train_main)
            if isinstance(n, ast.Assign)
            and any(ast.unparse(t) == name for t in n.targets)
        )
        wires[name] = a
    ok &= ast.unparse(wires["cfg.yaw_rew_scale"]) == "cfg.yaw_rew_scale = cli.yaw_rew_scale"
    ok &= ast.unparse(wires["cfg.vx_min"]) == "cfg.vx_min = cli.vx_min"
    # ② 位于 env 构造之前（wiring 落在 env cfg 上，必须先于 AptFlatG1Env(...)）
    env_ctor = next(
        n for n in ast.walk(train_main)
        if isinstance(n, ast.Assign)
        and any(ast.unparse(t) == "env" for t in n.targets)
        and "AptFlatG1Env(" in ast.unparse(n)
    )
    ok &= wires["cfg.yaw_rew_scale"].lineno < env_ctor.lineno
    ok &= wires["cfg.vx_min"].lineno < env_ctor.lineno
    # ③ env cfg.vx_min 默认 0.0 + uniform(vx_min, vx_max) 采样既有
    cfg_cls = next(
        n for n in ast.walk(tree_env)
        if isinstance(n, ast.ClassDef) and n.name == "AptFlatG1EnvCfg"
    )
    vx_min_field = next(
        n for n in ast.walk(cfg_cls)
        if isinstance(n, ast.AnnAssign) and ast.unparse(n.target) == "vx_min"
    )
    ok &= vx_min_field.value is not None and ast.unparse(vx_min_field.value) == "0.0"
    ok &= "uniform_(self.cfg.vx_min, self.cfg.vx_max)" in src_env
    # ④ 范围声明：DecFtPolicy 构造仍只传 vx_max（decft 无 vx_min 消费）
    decft_call = next(
        n for n in ast.walk(train_main)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name) and n.func.id == "DecFtPolicy"
    )
    kws = {k.arg for k in decft_call.keywords}
    ok &= "vx_max" in kws and "vx_min" not in kws
    return _report(
        "case6 train 接线（cfg 两行先于 env 构造）+ env vx_min 既有采样 + decft 范围声明", ok,
    )


def main() -> None:
    cases = [
        case1_yaw_rew_numeric_pins,
        case2_same_source_with_obs_block,
        case3_shared_init_yaw_guards,
        case4_obs_dim_unchanged,
        case5_double_zero_defaults,
        case6_train_wiring_and_env_vx,
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
