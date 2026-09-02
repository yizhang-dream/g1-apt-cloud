# TO41 D dry-run audit report（machine-generated）

- artifact: `rung1-d-audit/v1`  selftest=False
- generated: 2026-09-02T13:47:53Z  env_tag: local
- platform: Windows-11-10.0.26200-SP0 / python 3.13.5
- receipts: 0 @ None
- static coverage: 28/28

## verdict

| check | verdict |
|---|---|
| schema_check | **NOT_RUN** |
| D1 | **NOT_RUN** |
| D2 | **NOT_RUN** |
| D3A | **NOT_RUN** |
| D3B | **NOT_RUN** |
| D3 | **NOT_RUN** |
| overall | **NOT_RUN** |

## material baseline（G_DOWN_SPEC §9 照录）

| target | artifact | v_realized | abs_err | source | determinism | note |
|---:|---|---|---|---|---|---|
| 0.200 | gdown_v0200_k0.npz | 0.2 | 6.7e-09 | G↓ ↓-k0 | ✅ |  |
| 0.225 | gdown_v0225_k0.npz | 0.225 | 1.7e-06 | G↓ ↓-k0 | ✅ |  |
| 0.250 | gdown_v0250_k0.npz | 0.25 | 7.0e-09 | G↓ ↓-k0 | ✅ |  |
| 0.275 | gdown_smoke_v275_k0.npz | 0.275 | 8.0e-09 | G↓ ↓-k0 | ✅（run A/B） |  |
| 0.277 | to36_hybrid_gait_F11b_flat.npz | 0.2768 | 2.0e-04 | registry 既有（F 线审计验收） | registry 冻结 |  |
| 0.300 | to36_hybrid_gait_F11_slope2.npz | 0.2925 | 7.5e-03 | registry 既有 | registry 冻结 | << accepted under the pre-registered ±0.02 m/s realization tolerance（禁触发重解，协议 §4） |
| 0.325 | to36_hybrid_gait_F9.npz | 0.3179 | 7.1e-03 | registry 既有 | registry 冻结 | << accepted under the pre-registered ±0.02 m/s realization tolerance（禁触发重解，协议 §4） |

## scope notes / observations

- D 只回答 plumbing 是否忠实执行 specification（协议 §4）；PASS 解释上限 = implementation conforms to treatment specification
- 本 report 不含任何 performance / locomotion 字段（协议 §4 禁收）
- mapping YAML freeze_status = 'generated-not-frozen'（generated-not-frozen 字样；owner 冻结动作记录于 tracker/TO.md 三十三轮——implementation 不改冻结工件，仅如实记录）
- eval 侧 decoder 类源文件 apt_g1/isaac/token_window_vae.py 在 mapping artifact 中无冻结 hash；D1 架构身份以 state_dict 签名 + 调用点（apt_flat_env.py，preprocessing_hash 冻结）+ 训练源（decoder_architecture_hash 冻结）覆盖，类源 sha 记录于各 receipt 供追溯
- decode-only dry-run：τ 注入 env 的 exercise 属 Rung 1 launch sanity（IMPL §6），不在 D 范围
