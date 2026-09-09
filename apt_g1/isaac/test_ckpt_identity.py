"""D048h 确定性测试：ckpt 配置身份信封（ckpt_identity，无 isaaclab 依赖）。

纯 torch + tempfile + stdlib。覆盖面：
  1. 带身份块存取往返：save_ckpt -> load_ckpt 正确解包（state_dict 逐张量
     一致、identity 逐键一致）
  2. verify_ckpt_identity 全匹配 -> 空失配清单
  3. 失配检出：res_scale / res_clip / latent_residual / obs_dim /
     action_space / vae_md5 单键扰动 -> 清单恰含对应键一条、无其他误报键
  4. legacy 兼容：纯 state_dict（旧格式）-> load 返回 (sd, None)；信封缺
     ckpt_identity 键 -> 同样 (sd, None)
  5. 损坏文件（垃圾字节）-> torch.load 异常向上抛（eval 侧 try 捕获后
     exit 3 的路径）
  6. legacy 缺键跳过：format 非 1（含无 format 键 / identity=None）时
     expect/identity 侧缺键仍静默跳过（只报 format 条目，无逐键误报）
  7. file_md5：内容 md5 稳定 32 hex；缺失文件 -> None（train/eval 资产
     md5 容错口径）
  8. format=1 必填硬化（2026-09-09 owner 裁定）：裸 {"format":1} 信封对
     错误 res_scale 的完整 expect 仍被拒且条目点名缺失；ckpt 侧缺
     vae_md5 -> 失配条目点名缺失；expect 侧缺一个 VERIFY_KEY -> 同样
     失配（双侧必填，不再单侧静默跳过）
  9. 回归：双侧完整且值一致 -> 空清单；双侧完整但 res_scale 不同 -> 失配；
     float 容差 1e-9 不变

用法（服务器 .venv_isaac；本机无 torch 仅 py_compile）：
    PYTHONPATH=. python apt_g1/isaac/test_ckpt_identity.py
全部用例 PASS 时 exit 0。
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch

from apt_g1.isaac.ckpt_identity import (
    FORMAT,
    VERIFY_KEYS,
    build_identity,
    file_md5,
    load_ckpt,
    save_ckpt,
    verify_ckpt_identity,
)


def _report(name: str, ok: bool, detail: str = "") -> bool:
    print(
        f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""),
        flush=True,
    )
    return ok


def _ident(**over) -> dict:
    """A fully-populated identity resembling a D048g-style latent-residual run."""
    base = dict(
        entry="isaac_d048g_prog1_ent0_s0",
        it=50,
        res_scale=0.15,
        res_clip=1.0,
        res_l2=2e-2,
        res_freeze_steps=0,
        latent_mode=True,
        latent_residual=True,
        action_space=45,
        obs_dim=134,
        rew_contract=2,
        env_sha256="d" * 64,
        vae_md5="a" * 32,
        decoder_md5="b" * 32,
        git_head="0" * 40,
    )
    base.update(over)
    return build_identity(**base)


def _expect(**over) -> dict:
    """Matching eval-side expect (all VERIFY_KEYS derivable)."""
    base = dict(
        res_scale=0.15,
        res_clip=1.0,
        latent_residual=True,
        obs_dim=134,
        action_space=45,
        vae_md5="a" * 32,
        decoder_md5="b" * 32,
    )
    base.update(over)
    return base


def _sd() -> dict:
    return {"w": torch.arange(6, dtype=torch.float32).reshape(2, 3)}


# ---------------------------------------------------------------------------
def case1_roundtrip(tmp: str) -> bool:
    p = os.path.join(tmp, "envelope.pt")
    ident = _ident(it=150)
    save_ckpt(p, _sd(), ident)
    sd, got = load_ckpt(p)
    ok = isinstance(sd, dict) and "w" in sd and torch.equal(sd["w"], _sd()["w"])
    ok = ok and isinstance(got, dict) and got.get("format") == FORMAT
    for k in ("entry", "it", "res_scale", "res_clip", "res_l2",
              "latent_mode", "latent_residual", "action_space", "obs_dim",
              "rew_contract", "env_sha256", "vae_md5", "decoder_md5",
              "git_head"):
        ok = ok and got.get(k) == ident[k]
    ok = ok and "timestamp" in got
    return _report("1 roundtrip: save_ckpt/load_ckpt unwraps identity", ok)


def case2_verify_match(tmp: str) -> bool:
    msgs = verify_ckpt_identity(_ident(), _expect())
    return _report("2 verify all-match -> empty list", msgs == [], str(msgs))


def case3_mismatches(tmp: str) -> bool:
    bad = {
        "res_scale": 0.4,        # the D048d-class accident
        "res_clip": 0.5,
        "latent_residual": False,
        "obs_dim": 91,
        "action_space": 16,
        "vae_md5": "c" * 32,
        "decoder_md5": "e" * 32,
    }
    ok = True
    detail = []
    for key, val in bad.items():
        msgs = verify_ckpt_identity(_ident(), _expect(**{key: val}))
        hit = [m for m in msgs if m.startswith(key + ":")]
        if len(msgs) != 1 or len(hit) != 1 or str(val) not in msgs[0]:
            ok = False
            detail.append(f"{key}->{msgs}")
    # message format check: "res_scale: ckpt=0.4 eval=0.15"-style
    msgs = verify_ckpt_identity(_ident(), _expect(res_scale=0.4))
    ok = ok and msgs == ["res_scale: ckpt=0.15 eval=0.4"]
    return _report("3 single-key mismatch detected, no false keys", ok,
                   "; ".join(detail))


def case4_legacy(tmp: str) -> bool:
    # old-format bare state_dict
    p1 = os.path.join(tmp, "legacy.pt")
    torch.save(_sd(), p1)
    sd, ident = load_ckpt(p1)
    ok = ident is None and torch.equal(sd["w"], _sd()["w"])
    # envelope written without the identity key -> same legacy treatment
    p2 = os.path.join(tmp, "envelope_no_ident.pt")
    torch.save({"state_dict": _sd()}, p2)
    sd2, ident2 = load_ckpt(p2)
    ok = ok and ident2 is None and torch.equal(sd2["w"], _sd()["w"])
    return _report("4 legacy bare state_dict -> (sd, None)", ok)


def case5_corrupt(tmp: str) -> bool:
    p = os.path.join(tmp, "corrupt.pt")
    with open(p, "wb") as f:
        f.write(b"\x00garbage-not-a-torch-archive" * 32)
    raised = False
    try:
        load_ckpt(p)
    except Exception:
        raised = True  # eval catches this path and os._exit(3)
    return _report("5 corrupt file raises (eval -> exit 3 path)", raised)


def case6_expect_missing_keys(tmp: str) -> bool:
    # legacy semantics (2026-09-09 硬化后仅限非 format=1 身份)：eval 只喂
    # 自身旗标能推导的键，缺键 ≠ 失配；format 条目本身仍要报告
    expect = _expect()
    del expect["vae_md5"], expect["decoder_md5"], expect["action_space"]
    legacy = _ident()
    del legacy["format"]  # 无 format 键 -> 按 legacy 处理，逐键跳过
    msgs = verify_ckpt_identity(legacy, expect)
    ok = msgs == ["format: ckpt=None expected=1"]
    # legacy 身份块自身的 None 值同样跳过（format 改写为 0 = 格式不符）
    legacy2 = _ident(vae_md5=None, action_space=None)
    legacy2["format"] = 0
    msgs2 = verify_ckpt_identity(legacy2, expect)
    ok = ok and msgs2 == ["format: ckpt=0 expected=1"]
    # identity=None（load 返回 legacy 裸 state_dict）：只有 format 条目
    msgs3 = verify_ckpt_identity(None, _expect())
    ok = ok and msgs3 == ["format: ckpt=None expected=1"]
    return _report("6 legacy identity: missing keys still skipped (no false report)",
                   ok, str(msgs + msgs2 + msgs3))


def case7_file_md5(tmp: str) -> bool:
    p = os.path.join(tmp, "blob.bin")
    with open(p, "wb") as f:
        f.write(b"hello-md5")
    d1 = file_md5(p)
    ok = d1 is not None and len(d1) == 32 and file_md5(p) == d1
    ok = ok and file_md5(os.path.join(tmp, "missing.bin")) is None
    return _report("7 file_md5 stable / missing -> None", ok)


def case8_bare_format1_rejected(tmp: str) -> bool:
    # owner 复现用例（2026-09-09）：裸 {"format":1} 信封对错误 res_scale 的
    # 完整 expect 曾返回无失配；硬化后必须被拒且条目点名缺失
    msgs = verify_ckpt_identity({"format": FORMAT}, _expect(res_scale=0.4))
    ok = len(msgs) == len(VERIFY_KEYS)  # 7 键全部 ckpt 侧缺失，各一条
    ok = ok and (
        "res_scale: missing on ckpt side (required for format=1)" in msgs
    )
    return _report("8 bare format=1 envelope vs wrong expect -> rejected",
                   ok, str(msgs))


def case9_ckpt_side_missing_key(tmp: str) -> bool:
    # format=1 但 ckpt 侧缺 vae_md5（None 等价缺失）-> 失配条目点名缺失
    msgs = verify_ckpt_identity(_ident(vae_md5=None), _expect())
    ok = msgs == ["vae_md5: missing on ckpt side (required for format=1)"]
    return _report("9 format=1 ckpt-side missing key -> entry names it",
                   ok, str(msgs))


def case10_expect_side_missing_key(tmp: str) -> bool:
    # 双侧必填：完整 format=1 ckpt vs 缺一个 VERIFY_KEY 的 expect -> 失配
    expect = _expect()
    del expect["obs_dim"]
    msgs = verify_ckpt_identity(_ident(), expect)
    ok = msgs == ["obs_dim: missing on eval side (required for format=1)"]
    return _report("10 format=1 expect-side missing key -> mismatch",
                   ok, str(msgs))


def case11_complete_regression(tmp: str) -> bool:
    # 回归：双侧完整时行为与硬化前一致
    ok = verify_ckpt_identity(_ident(), _expect()) == []
    # res_scale 不同 -> 恰一条，消息格式不变
    ok = ok and verify_ckpt_identity(_ident(), _expect(res_scale=0.4)) == [
        "res_scale: ckpt=0.15 eval=0.4"
    ]
    # float 容差 1e-9 不变：5e-10 漂移仍算匹配
    ok = ok and verify_ckpt_identity(
        _ident(res_scale=0.15 + 5e-10), _expect(res_scale=0.15)
    ) == []
    return _report(
        "11 regression: complete envelopes compare as before (1e-9 tol)", ok
    )


def main() -> None:
    cases = [
        case1_roundtrip,
        case2_verify_match,
        case3_mismatches,
        case4_legacy,
        case5_corrupt,
        case6_expect_missing_keys,
        case7_file_md5,
        case8_bare_format1_rejected,
        case9_ckpt_side_missing_key,
        case10_expect_side_missing_key,
        case11_complete_regression,
    ]
    with tempfile.TemporaryDirectory() as tmp:
        results = [c(tmp) for c in cases]
    n_fail = results.count(False)
    print(f"\n{len(results) - n_fail}/{len(results)} cases PASS", flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
