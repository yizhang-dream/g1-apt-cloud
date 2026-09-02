# TO41 Rung 1 launch sanity report（machine-generated）

- environment_tag: `lab-ts`  
- generated_utc: `2026-09-02T17:06:28Z`  
- receipts: `28/28`  

## verdict
| check | verdict |
|---|---|
| schema_check | PASS |
| L1 | PASS |
| L2 | PASS |
| L3 | PASS |
| L4 | PASS |
| **overall** | **PASS** |

## L2 override persistence（per cell）
| cell | natural | mapped | calls | changed | boundaries | verdict |
|---|---:|---:|---:|---:|---:|---|
| v0200_C1_on | vb0 | vb0 | 300 | 0 | 3 | PASS |
| v0200_C1_off | vb0 | vb0 | 300 | 0 | 3 | PASS |
| v0200_C2_on | vb0 | vb1 | 300 | 300 | 3 | PASS |
| v0200_C2_off | vb0 | vb1 | 300 | 300 | 3 | PASS |
| v0225_C1_on | vb0 | vb0 | 300 | 0 | 3 | PASS |
| v0225_C1_off | vb0 | vb0 | 300 | 0 | 3 | PASS |
| v0225_C2_on | vb0 | vb1 | 300 | 300 | 3 | PASS |
| v0225_C2_off | vb0 | vb1 | 300 | 300 | 3 | PASS |
| v0250_C1_on | vb0 | vb0 | 300 | 0 | 3 | PASS |
| v0250_C1_off | vb0 | vb0 | 300 | 0 | 3 | PASS |
| v0250_C2_on | vb0 | vb1 | 300 | 300 | 3 | PASS |
| v0250_C2_off | vb0 | vb1 | 300 | 300 | 3 | PASS |
| v0275_C1_on | vb1 | vb0 | 300 | 300 | 3 | PASS |
| v0275_C1_off | vb1 | vb0 | 300 | 300 | 3 | PASS |
| v0275_C2_on | vb1 | vb1 | 300 | 0 | 3 | PASS |
| v0275_C2_off | vb1 | vb1 | 300 | 0 | 3 | PASS |
| v0277_C1_on | vb1 | vb0 | 300 | 300 | 3 | PASS |
| v0277_C1_off | vb1 | vb0 | 300 | 300 | 3 | PASS |
| v0277_C2_on | vb1 | vb1 | 300 | 0 | 3 | PASS |
| v0277_C2_off | vb1 | vb1 | 300 | 0 | 3 | PASS |
| v0300_C1_on | vb1 | vb0 | 300 | 300 | 3 | PASS |
| v0300_C1_off | vb1 | vb0 | 300 | 300 | 3 | PASS |
| v0300_C2_on | vb1 | vb1 | 300 | 0 | 3 | PASS |
| v0300_C2_off | vb1 | vb1 | 300 | 0 | 3 | PASS |
| v0325_C1_on | vb1 | vb0 | 300 | 300 | 3 | PASS |
| v0325_C1_off | vb1 | vb0 | 300 | 300 | 3 | PASS |
| v0325_C2_on | vb1 | vb1 | 300 | 0 | 3 | PASS |
| v0325_C2_off | vb1 | vb1 | 300 | 0 | 3 | PASS |

## L1 Mode A env-level fingerprint（per v）
| target_speed | buffer sha(16) | verdict |
|---:|---|---|
| 0.2 | 3f3e3fed5c8800f7 | PASS |
| 0.225 | b2b42c1661a107a4 | PASS |
| 0.25 | 0dd9ce2c205bdf85 | PASS |
| 0.275 | 9a40d021670106e4 | PASS |
| 0.277 | c711e7397e1e266e | PASS |
| 0.3 | ace5753faece466a | PASS |
| 0.325 | e0ff887270f39ed2 | PASS |

## material baseline（checker 独立重算）
| target | artifact | v_realized | abs_err | sha(16) | note |
|---:|---|---:|---:|---|---|
| 0.2 | gdown_v0200_k0.npz | 0.2 | 6.7e-09 | 0038afb8a22cee26 |  |
| 0.225 | gdown_v0225_k0.npz | 0.225 | 1.7e-06 | eb87ad8491adb447 |  |
| 0.25 | gdown_v0250_k0.npz | 0.25 | 7e-09 | 626b340737e9d953 |  |
| 0.275 | gdown_smoke_v275_k0.npz | 0.275 | 8e-09 | 09c2915c5c713afb |  |
| 0.277 | to36_hybrid_gait_F11b_flat.npz | 0.2768 | 0.0002 | 3f239cbf991b382b |  |
| 0.3 | to36_hybrid_gait_F11_slope2.npz | 0.2925 | 0.0075 | da6cb3a6d72c33b5 | accepted under the pre-registered ±0.02 m/s realization tolerance（禁触发重解） |
| 0.325 | to36_hybrid_gait_F9.npz | 0.3179 | 0.0071 | e7b4cdf455ab0864 | accepted under the pre-registered ±0.02 m/s realization tolerance（禁触发重解） |

