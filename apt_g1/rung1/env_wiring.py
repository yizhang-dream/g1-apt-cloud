"""TO41 Rung 1 launch sanity：Mode A runtime → 真实 apt_flat_env 接线 shim。

角色（SCRIPT_MAP 登记）：**state-changing execution code**（module，被
launch_sanity.py 驱动；Rung 1 compute 的 eval 侧接线届时复用同一实现，
conformance 随之继承）。

三十七轮 owner 裁定的执行 gate：D1/D2/D3 只证明了 decode plumbing
（`T_runtime = T_mapping`、`τ_runtime = τ_frozen`），未触碰真实 env 的
τ 注入/控制路径。本模块把冻结 Mode A 契约接入**未改动的**真实
`AptFlatG1Env`：

    τ 轴    : cfg.to_ref_npz = 冻结 τ(v) 材料（env __init__ 自行 np.load，
              冻结路径 apt_flat_env.py:318-328 不变）；消费点探针 =
              实例级 wrapper 拦截 _to_ref_lookup()（:768 的唯一消费者）。
    condition 轴: 实例级 wrapper 拦截 env._vae.decode（:589 冻结调用点）。
              冻结 bucketize(:583-586) 每步照跑，wrapper 记录 natural
              vb/db 后替换为 mapping_lookup 的 speed_bin —— override
              位于 decode 输入边界，天然不可被后续逻辑覆盖，且每次
              decode 调用都留 natural/overridden 双记录（L2 证据）。

接线纪律（与 D 协议同构）：

1. **零 env 文件改动**：env 的 sha256 = mapping `preprocessing_hash` 冻结
   锚；本模块只做实例属性 shadowing（`env._vae.decode = wrapped` /
   `env._to_ref_lookup = wrapped`），冻结类与冻结代码路径逐字不变。
2. **record only**：所有 handle 只产记录（per-call 记录 / 计数 / 哈希），
   不得产 verdict / PASS 字段（协议 §9 同款禁自证）。
3. **buffer 身份**：τ 的"实际消费身份"= env._to_tau 缓冲哈希（冻结代码
   读的唯一来源）+ per-call tau 序列 digest；npz 文件哈希只是输入身份。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

# SCRIPT_MAP/文档引用的消费点行号（apt_flat_env.py@187f2fb 冻结版位）
TAU_CONSUMER_IDENTITY = (
    "apt_flat_env.py:765-773 _apply_action (to_tau branch) "
    "<- _to_ref_lookup():776-787 <- self._to_tau (np.load cfg.to_ref_npz:319-328)"
)
CONDITION_ENTRY_IDENTITY = (
    "apt_flat_env.py:568-612 _compute_q_des latent_mode+latent_dir_bins branch "
    "<- self._vae.decode(phase, sc, vb, db):589 <- bucketize(cmd_v, edges):583-586"
)


def canonical_array_sha256(a) -> str:
    """冻结数组身份的规范哈希（wiring 侧实现；l_checker 另有独立实现，
    selftest 交叉验证一致性）。

    规范化 = f"shape={shape};dtype={dtype.str};data=" + 原始小端字节。
    τ buffer 语义：np.load(tau_ref6) -> float32（env 冻结路径
    `torch.from_numpy(...).float()` 的等价 numpy 形式），C-contiguous。
    """
    import numpy as np

    arr = np.ascontiguousarray(a, dtype=np.float32)
    h = hashlib.sha256()
    h.update(f"shape={arr.shape};dtype={arr.dtype.str};data=".encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def _scalar_int(x) -> int:
    """bin 张量 → int（冻结路径传 (num_envs,) 张量；selftest mock 传 python
    int）。非单元素或不可转换 = 接线契约违反，raise fail-fast。"""
    if hasattr(x, "detach"):
        if x.numel() != 1:
            raise RuntimeError(
                f"wiring contract violation: bin 张量非单元素 shape={tuple(x.shape)}")
        return int(float(x.detach().reshape(-1)[0].item()))
    if isinstance(x, (int, float)):
        return int(x)
    raise RuntimeError(f"wiring contract violation: bin 参数类型 {type(x)!r}")


def _same_bins_like(template, value: int):
    """以 template 的 dtype/device/shape 重建 bin 张量（冻结调用点语义不变；
    template 为 python int 时返回 int —— selftest mock 路径）。"""
    if hasattr(template, "detach"):
        import torch

        return torch.full_like(template, int(value))
    return int(value)


def _to_float32_np(x):
    """torch 张量 / numpy 数组 → float32 C-contiguous numpy（其余类型 =
    接线契约违反，fail-fast）。"""
    import numpy as np

    if hasattr(x, "detach"):
        return np.ascontiguousarray(x.detach().cpu().numpy(), dtype=np.float32)
    if isinstance(x, np.ndarray):
        return np.ascontiguousarray(x, dtype=np.float32)
    raise RuntimeError(f"wiring contract violation: 数值类型 {type(x)!r}")


def tau_buffer_snapshot(env) -> dict:
    """env._to_tau 缓冲身份快照（冻结代码读的唯一 τ 来源）。

    返回 shape/dtype/规范哈希；float32 化与 env 冻结加载路径
    （torch.from_numpy(...).float()）等价。
    """
    import numpy as np

    t = env._to_tau
    arr = _to_float32_np(t)
    return {
        "shape": list(arr.shape),
        "dtype": arr.dtype.str,
        "buffer_sha256": canonical_array_sha256(arr),
    }


class ConditionOverrideHandle:
    """condition 轴接线：拦截 env._vae.decode（实例属性 shadowing）。

    每 cell 一个 handle；env 每 cell 重建，无跨 cell 状态。per-call 记录
    natural_vb / applied_vb / natural_db / applied_db —— L2（override
    persistence）与 receipt "actual condition" 字段的事实源。只记录，
    不判定。
    """

    def __init__(self, env, speed_bin: int, dir_bin: int):
        self.speed_bin = int(speed_bin)
        self.dir_bin = int(dir_bin)
        self.records: list[dict] = []
        self._orig_decode = env._vae.decode
        handle = self

        def _wrapped_decode(phase, sc, vb, db=None, **kw):
            if db is None:
                raise RuntimeError(
                    "wiring contract violation: env 以无 db 形式调用 decode"
                    "（cfg 应为 latent_dir_bins，见 " + CONDITION_ENTRY_IDENTITY + "）")
            natural_vb = _scalar_int(vb)
            natural_db = _scalar_int(db)
            vb_new = _same_bins_like(vb, handle.speed_bin)
            db_new = _same_bins_like(db, handle.dir_bin)
            out = handle._orig_decode(phase, sc, vb_new, db_new, **kw)
            handle.records.append({
                "i": len(handle.records),
                "natural_vb": natural_vb,
                "applied_vb": handle.speed_bin,
                "natural_db": natural_db,
                "applied_db": handle.dir_bin,
            })
            return out

        env._vae.decode = _wrapped_decode  # 实例属性 shadowing；冻结类零改动

    @property
    def n_calls(self) -> int:
        return len(self.records)

    @property
    def n_override_changed(self) -> int:
        return sum(1 for r in self.records if r["natural_vb"] != r["applied_vb"])

    def natural_vb_distribution(self) -> dict[int, int]:
        dist: dict[int, int] = {}
        for r in self.records:
            dist[r["natural_vb"]] = dist.get(r["natural_vb"], 0) + 1
        return dist

    def calls_after(self, n_calls_at_event: int) -> int:
        return max(0, len(self.records) - n_calls_at_event)

    def record_block(self) -> dict:
        return {
            "mechanism": "instance-level shadowing of env._vae.decode "
                         "(frozen call site apt_flat_env.py:589)",
            "frozen_path_untouched": True,
            "condition_entry_identity": CONDITION_ENTRY_IDENTITY,
            "mapped_speed_bin": self.speed_bin,
            "mapped_dir_bin": self.dir_bin,
            "n_decode_calls": self.n_calls,
            "n_override_changed": self.n_override_changed,
            "natural_vb_distribution": {
                str(k): v for k, v in sorted(self.natural_vb_distribution().items())},
            "per_call": self.records,
        }


class TauConsumptionProbe:
    """τ 轴消费点探针：拦截 env._to_ref_lookup（实例属性 shadowing）。

    _apply_action 的 to_tau 分支（:768）是冻结代码消费 τ 的唯一调用点；
    本探针记录每次调用的 tau 张量（序列 digest + 首末哈希 + 非有限计数）。
    OFF 臂（to_tau=False）冻结代码不调用本函数 —— n_calls==0 即预注册
    的"不注入"记录。只记录，不判定。
    """

    def __init__(self, env):
        self.records_meta: list[dict] = []
        self._digest = hashlib.sha256()
        self._first_sha16: str | None = None
        self._last_sha16: str | None = None
        self.n_nonfinite = 0
        self._orig_lookup = env._to_ref_lookup
        probe = self

        def _wrapped_lookup():
            q, s, tau = probe._orig_lookup()
            if tau is not None:
                arr = _to_float32_np(tau)
                d = canonical_array_sha256(arr)
                probe._digest.update(d.encode())
                sha16 = d[:16]
                if probe._first_sha16 is None:
                    probe._first_sha16 = sha16
                probe._last_sha16 = sha16
                if not bool(np.isfinite(arr).all()):
                    probe.n_nonfinite += 1
                probe.records_meta.append({
                    "i": len(probe.records_meta),
                    "tau_shape": list(arr.shape),
                    "tau_sha256_16": sha16,
                })
            return q, s, tau

        env._to_ref_lookup = _wrapped_lookup  # 实例属性 shadowing；冻结类零改动

    @property
    def n_calls(self) -> int:
        return len(self.records_meta)

    def calls_after(self, n_calls_at_event: int) -> int:
        return max(0, len(self.records_meta) - n_calls_at_event)

    def record_block(self) -> dict:
        return {
            "mechanism": "instance-level shadowing of env._to_ref_lookup "
                         "(frozen consumer apt_flat_env.py:768)",
            "consumer_identity": TAU_CONSUMER_IDENTITY,
            "n_tau_calls": self.n_calls,
            "calls_tau_digest_sha256": self._digest.hexdigest(),
            "first_tau_sha256_16": self._first_sha16,
            "last_tau_sha256_16": self._last_sha16,
            "n_nonfinite_tau_calls": self.n_nonfinite,
        }
