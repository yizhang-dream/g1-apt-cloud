# TO41 D dry-run audit report（machine-generated）

- artifact: `rung1-d-audit/v1`  selftest=False
- generated: 2026-09-02T14:00:25Z  env_tag: lab-ts
- platform: Linux-6.8.0-136-generic-x86_64-with-glibc2.35 / python 3.10.20
- receipts: 28 @ apt_g1/outputs/sync/to41_d/receipts
- static coverage: 28/28

## verdict

| check | verdict |
|---|---|
| schema_check | **PASS** |
| D1 | **PASS** |
| D2 | **PASS** |
| D3A | **PASS** |
| D3B | **PASS** |
| D3 | **PASS** |
| overall | **PASS** |

## D2 Mode A fingerprint（same v → same τ identity）

| v | C1 τ hash(16) | C2 τ hash(16) | equal | lineage uniform | verdict |
|---:|---|---|---|---|---|
| 0.200 | 0038afb8a22cee26 | 0038afb8a22cee26 | True | True | PASS |
| 0.225 | eb87ad8491adb447 | eb87ad8491adb447 | True | True | PASS |
| 0.250 | 626b340737e9d953 | 626b340737e9d953 | True | True | PASS |
| 0.275 | 09c2915c5c713afb | 09c2915c5c713afb | True | True | PASS |
| 0.277 | 3f239cbf991b382b | 3f239cbf991b382b | True | True | PASS |
| 0.300 | da6cb3a6d72c33b5 | da6cb3a6d72c33b5 | True | True | PASS |
| 0.325 | e7b4cdf455ab0864 | e7b4cdf455ab0864 | True | True | PASS |

## D3A assignment conformance（逐 cell）

| cell | yaml | runtime | verdict |
|---|---|---|---|
| v0200_C1_on | vb0_db4 | vb0_db4 | PASS |
| v0200_C1_off | vb0_db4 | vb0_db4 | PASS |
| v0200_C2_on | vb1_db4 | vb1_db4 | PASS |
| v0200_C2_off | vb1_db4 | vb1_db4 | PASS |
| v0225_C1_on | vb0_db4 | vb0_db4 | PASS |
| v0225_C1_off | vb0_db4 | vb0_db4 | PASS |
| v0225_C2_on | vb1_db4 | vb1_db4 | PASS |
| v0225_C2_off | vb1_db4 | vb1_db4 | PASS |
| v0250_C1_on | vb0_db4 | vb0_db4 | PASS |
| v0250_C1_off | vb0_db4 | vb0_db4 | PASS |
| v0250_C2_on | vb1_db4 | vb1_db4 | PASS |
| v0250_C2_off | vb1_db4 | vb1_db4 | PASS |
| v0275_C1_on | vb0_db4 | vb0_db4 | PASS |
| v0275_C1_off | vb0_db4 | vb0_db4 | PASS |
| v0275_C2_on | vb1_db4 | vb1_db4 | PASS |
| v0275_C2_off | vb1_db4 | vb1_db4 | PASS |
| v0277_C1_on | vb0_db4 | vb0_db4 | PASS |
| v0277_C1_off | vb0_db4 | vb0_db4 | PASS |
| v0277_C2_on | vb1_db4 | vb1_db4 | PASS |
| v0277_C2_off | vb1_db4 | vb1_db4 | PASS |
| v0300_C1_on | vb0_db4 | vb0_db4 | PASS |
| v0300_C1_off | vb0_db4 | vb0_db4 | PASS |
| v0300_C2_on | vb1_db4 | vb1_db4 | PASS |
| v0300_C2_off | vb1_db4 | vb1_db4 | PASS |
| v0325_C1_on | vb0_db4 | vb0_db4 | PASS |
| v0325_C1_off | vb0_db4 | vb0_db4 | PASS |
| v0325_C2_on | vb1_db4 | vb1_db4 | PASS |
| v0325_C2_off | vb1_db4 | vb1_db4 | PASS |

## D3B material conformance（逐 cell）

| cell | artifact | τ hash(16) | lineage | verdict |
|---|---|---|---|---|
| v0200_C1_on | gdown_v0200_k0.npz | 0038afb8a22cee26 | G↓ ↓-k0 | PASS |
| v0200_C1_off | gdown_v0200_k0.npz | 0038afb8a22cee26 | G↓ ↓-k0 | PASS |
| v0200_C2_on | gdown_v0200_k0.npz | 0038afb8a22cee26 | G↓ ↓-k0 | PASS |
| v0200_C2_off | gdown_v0200_k0.npz | 0038afb8a22cee26 | G↓ ↓-k0 | PASS |
| v0225_C1_on | gdown_v0225_k0.npz | eb87ad8491adb447 | G↓ ↓-k0 | PASS |
| v0225_C1_off | gdown_v0225_k0.npz | eb87ad8491adb447 | G↓ ↓-k0 | PASS |
| v0225_C2_on | gdown_v0225_k0.npz | eb87ad8491adb447 | G↓ ↓-k0 | PASS |
| v0225_C2_off | gdown_v0225_k0.npz | eb87ad8491adb447 | G↓ ↓-k0 | PASS |
| v0250_C1_on | gdown_v0250_k0.npz | 626b340737e9d953 | G↓ ↓-k0 | PASS |
| v0250_C1_off | gdown_v0250_k0.npz | 626b340737e9d953 | G↓ ↓-k0 | PASS |
| v0250_C2_on | gdown_v0250_k0.npz | 626b340737e9d953 | G↓ ↓-k0 | PASS |
| v0250_C2_off | gdown_v0250_k0.npz | 626b340737e9d953 | G↓ ↓-k0 | PASS |
| v0275_C1_on | gdown_smoke_v275_k0.npz | 09c2915c5c713afb | G↓ ↓-k0 | PASS |
| v0275_C1_off | gdown_smoke_v275_k0.npz | 09c2915c5c713afb | G↓ ↓-k0 | PASS |
| v0275_C2_on | gdown_smoke_v275_k0.npz | 09c2915c5c713afb | G↓ ↓-k0 | PASS |
| v0275_C2_off | gdown_smoke_v275_k0.npz | 09c2915c5c713afb | G↓ ↓-k0 | PASS |
| v0277_C1_on | to36_hybrid_gait_F11b_flat.npz | 3f239cbf991b382b | registry 既有（F 线审计验收） | PASS |
| v0277_C1_off | to36_hybrid_gait_F11b_flat.npz | 3f239cbf991b382b | registry 既有（F 线审计验收） | PASS |
| v0277_C2_on | to36_hybrid_gait_F11b_flat.npz | 3f239cbf991b382b | registry 既有（F 线审计验收） | PASS |
| v0277_C2_off | to36_hybrid_gait_F11b_flat.npz | 3f239cbf991b382b | registry 既有（F 线审计验收） | PASS |
| v0300_C1_on | to36_hybrid_gait_F11_slope2.npz | da6cb3a6d72c33b5 | registry 既有 | PASS |
| v0300_C1_off | to36_hybrid_gait_F11_slope2.npz | da6cb3a6d72c33b5 | registry 既有 | PASS |
| v0300_C2_on | to36_hybrid_gait_F11_slope2.npz | da6cb3a6d72c33b5 | registry 既有 | PASS |
| v0300_C2_off | to36_hybrid_gait_F11_slope2.npz | da6cb3a6d72c33b5 | registry 既有 | PASS |
| v0325_C1_on | to36_hybrid_gait_F9.npz | e7b4cdf455ab0864 | registry 既有 | PASS |
| v0325_C1_off | to36_hybrid_gait_F9.npz | e7b4cdf455ab0864 | registry 既有 | PASS |
| v0325_C2_on | to36_hybrid_gait_F9.npz | e7b4cdf455ab0864 | registry 既有 | PASS |
| v0325_C2_off | to36_hybrid_gait_F9.npz | e7b4cdf455ab0864 | registry 既有 | PASS |

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
