"""Checkpoint identity envelope (D048h).

Every new training checkpoint is saved as
``{"state_dict": ..., "ckpt_identity": {...}}`` instead of the bare
``policy.state_dict()``. The identity block pins the exact training config
(res_scale / res_clip / latent_residual / obs-action dims / rew_contract /
env_sha256 / asset md5s / git_head / iteration), so the eval side can reject
a train/eval misconfiguration (the 0.4-vs-0.15 res_scale class of accidents)
BEFORE running a single rollout.

Compatibility: ``load_ckpt`` accepts both formats -- legacy bare state_dicts
return ``(sd, None)`` and the caller skips verification.

Only torch + stdlib deps (no isaaclab) so the unit test can run anywhere.
"""

from __future__ import annotations

import hashlib
import subprocess
import time

import torch

FORMAT = 1  # envelope format version (identity["format"])

# Keys compared by verify_ckpt_identity. Semantics split by envelope format
# (2026-09-09 owner ruling): a format=1 identity block must be self-
# certifying -- every key here has to exist (non-None) on BOTH the ckpt and
# the expect side, a missing side is itself a mismatch. Legacy ckpts (no
# identity block) keep the old behavior: a key is SKIPPED when absent (or
# None) on either side -- the eval side only feeds keys it can derive
# unambiguously from its own flags, so "missing" must never mean "mismatch"
# there.
VERIFY_KEYS = (
    "res_scale",
    "res_clip",
    "latent_residual",
    "obs_dim",
    "action_space",
    "vae_md5",
    "decoder_md5",
)


def file_md5(path) -> str | None:
    """md5 hex digest of a file; None on IO error (missing asset etc.)."""
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _git_head(cwd=None) -> str:
    """git rev-parse HEAD; "" outside a repo / on any failure (train_log 同款)."""
    try:
        g = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return g.stdout.strip() if g.returncode == 0 else ""
    except Exception:
        return ""


def build_identity(
    *,
    entry: str,
    it: int,
    res_scale: float,
    res_clip: float,
    res_l2: float,
    res_freeze_steps: int,
    latent_mode: bool,
    latent_residual: bool,
    action_space,
    obs_dim,
    rew_contract,
    env_sha256: str,
    vae_md5,
    decoder_md5,
    git_head: str | None = None,
) -> dict:
    """Construct the identity dict embedded in every checkpoint.

    ``git_head=None`` -> computed here (subprocess); pass the train_log value
    to keep a single source. ``vae_md5`` / ``decoder_md5`` are computed once
    at the asset-load site by the caller (avoid re-reading large files).
    """
    return {
        "format": FORMAT,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_head": _git_head() if git_head is None else git_head,
        "entry": str(entry),
        "it": int(it),
        "res_scale": float(res_scale),
        "res_clip": float(res_clip),
        "res_l2": float(res_l2),
        "res_freeze_steps": int(res_freeze_steps),
        "latent_mode": bool(latent_mode),
        "latent_residual": bool(latent_residual),
        "action_space": None if action_space is None else int(action_space),
        "obs_dim": None if obs_dim is None else int(obs_dim),
        "rew_contract": rew_contract,
        "env_sha256": env_sha256,
        "vae_md5": vae_md5,
        "decoder_md5": decoder_md5,
    }


def save_ckpt(path, state_dict, identity: dict) -> None:
    """Save the D048h envelope {"state_dict", "ckpt_identity"}."""
    torch.save({"state_dict": state_dict, "ckpt_identity": identity}, path)


def load_ckpt(path, map_location=None):
    """Load a checkpoint -> (state_dict, identity | None).

    New format (dict with a "state_dict" key): unwrap and return the identity
    block. Legacy bare state_dict (or an envelope without the identity key):
    return (sd, None) so the caller can warn-and-continue.
    """
    obj = torch.load(path, map_location=map_location)
    if isinstance(obj, dict) and "state_dict" in obj:
        ident = obj.get("ckpt_identity")
        return obj["state_dict"], (dict(ident) if isinstance(ident, dict) else None)
    return obj, None


def verify_ckpt_identity(identity, expect: dict) -> list:
    """Pure comparison: return a list of mismatch messages (empty = match).

    Message format: ``"res_scale: ckpt=0.15 eval=0.4"``. For a format=1
    identity block every VERIFY_KEY is REQUIRED on both sides; a missing (or
    None) key reports as ``"res_scale: missing on ckpt side (required for
    format=1)"`` instead of being skipped. Legacy blocks (no identity /
    format mismatch) keep the skip-if-absent semantics; a wrong/missing
    "format" is itself reported.
    """
    msgs = []
    fmt = identity.get("format") if isinstance(identity, dict) else None
    if fmt != FORMAT:
        msgs.append(f"format: ckpt={fmt!r} expected={FORMAT}")
    strict = isinstance(identity, dict) and fmt == FORMAT
    for key in VERIFY_KEYS:
        ck = identity.get(key) if isinstance(identity, dict) else None
        ev = expect.get(key)
        if strict:
            # format=1 envelope must be self-certifying: a missing/None key on
            # either side is a mismatch, never a silent skip.
            if ck is None:
                msgs.append(
                    f"{key}: missing on ckpt side (required for format={FORMAT})"
                )
            if ev is None:
                msgs.append(
                    f"{key}: missing on eval side (required for format={FORMAT})"
                )
            if ck is None or ev is None:
                continue
        else:
            # legacy: skipped when absent (or None) on either side -- the eval
            # side only feeds keys it can derive unambiguously from its own
            # flags, so "missing" must not fire here.
            if ck is None or ev is None:
                continue
        if isinstance(ck, float) or isinstance(ev, float):
            ok = abs(float(ck) - float(ev)) <= 1e-9
        else:
            ok = ck == ev
        if not ok:
            msgs.append(f"{key}: ckpt={ck} eval={ev}")
    return msgs
