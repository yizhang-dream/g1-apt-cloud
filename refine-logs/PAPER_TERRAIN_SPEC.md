# 论文地形定义对照（APT-RL, Sci. Robot. 11, eadz7397, 2026）

> 来源：arXiv:2607.13579（与 Science Robotics 接受版同内容，含 Supplementary）。
> 本地缓存：`tmp/pdfs/arxiv_2607.13579.html`、`tmp/pdfs/arxiv_plain.txt`。
> 生成日期：2026-08 复查。回答的问题：论文的 "rough" 是否等于我们的 rough 0.06/0.08。

## 1. 论文侧：训练地形（Supplementary, "Detailed terrain types and levels for training"）

七种地形：stair / stepping stones / rough / hurdle / discrete / high step / gap，
curriculum 共 10 级（级 1–10 线性缩放参数），之前再加 2 级平地。参数上限：

| 地形 | 论文参数 |
|---|---|
| stair | 步宽固定 0.3 m；步高 0.05–0.315 m |
| stepping stones | 石头尺寸 0.4–0.48 m；石高 0–0.18 m；非石地面沿程下沉 0–0.36 m |
| hurdle | 栏厚随机 0.2/0.3 m；栏高 0–0.792 m |
| high step | 厚 1.5–3 m；高 0–0.9 m |
| gap | 缝宽 0.1–1.5 m |
| **rough** | **每个地形实例：最低点高度 U(−0.06, −0.02) m，最高点 U(0.02, 0.06) m；网格 cell 用 downsampled scale 0.2（粗 0.2 m 生成后升采样到 0.1 m 物理栅格）；同一组 rough 参数全局叠加到所有地形** |
| discrete | 块高 0–0.16 m；块尺寸 0.2–0.8 m；300 个随机矩形块 |

关键原文：
> "In the rough terrain, the minimum and maximum heights were randomly sampled from
> –0.06 to –0.02 m and 0.02 to 0.06 m, respectively, and a downsampled scale of 0.2
> was used."
> "We applied the same roughness parameters as the rough terrain globally to all terrains."

## 2. 我们侧：Isaac Lab HfRandomUniformTerrainCfg（`apt_g1/isaac/terrain_cfg.py`）

`make_terrain_importer_cfg("rough", noise, seed)`：

- `noise_range=(0.0, noise)`：每个 0.1 m cell 高度 U(0, noise)（只有凸起，无坑；
  均值 = noise/2，即地形整体被抬升 3–4 cm）。
- `horizontal_scale=0.1`（独立 cell 0.1 m，是论文 0.2 m 的 2 倍频率）；
  `vertical_scale=0.005`；`noise_step=0.01`（量化步）；`slope_threshold=0.75`。
- 8×8 m 瓷砖，10×20 阵列，fixed seed 0/1。
- 评测档位：noise = 0.06 / 0.08。

## 3. 对照结论

| 维度 | 论文 rough | 我们 rough 0.06 | 我们 rough 0.08 |
|---|---|---|---|
| 振幅上限 | ±0.06 m | 0–0.06 m | 0–0.08 m（**超出论文 33%**） |
| 形状 | 对称（有凸有坑，均值≈0） | 只凸不坑（均值 +0.03 m） | 只凸不坑（均值 +0.04 m） |
| 独立 feature 尺寸 | 0.2 m | 0.1 m（2× 频率） | 0.1 m |
| 同振幅下局部坡度 | ~0.3–0.6 | ~0.6–0.75（slope_threshold 截断） | 常达 0.75+ |

一句话：
- **振幅上，我们的 0.06 ≈ 论文 rough 的上限；0.08 在论文定义之外。**
- 形状上，我们的地形（更细、只凸）在同等振幅下**更难**（局部坡度更大），
  而论文 0.2 m cell 的对称噪声更"钝"。
- 论文其余地形（台阶 0.315 m、栏 0.792 m、高台 0.9 m、缝 1.5 m）远高于我们
  测过的离散物/楼梯档位（0.05–0.14 m）；rough 是论文七种地形里最温和的一种。

## 4. 对本项目结论的影响

> **〔2026-08-15 实证修订，见 EXPERIMENT_TRACKER G0 节〕** 本节原先按"振幅对照"推断
> "论文意义上的 rough 泛化我们并未落后"。**该推断已被论文形状实测推翻**：用论文
> 形状（对称 ±0.06、0.2m 粗格；`terrain_cfg.py` 的 `rough_paper`）评测，E47 与 E39
> 双双 **0/12 全倒**（fall 5–31s）；对称 ±0.06、0.1m 格（`rough_sym` 对照）同样
> 0/12 → **坑（负障碍）是唯一必要难点变量，格子大小无关**。"0.06 通过"仅对
> 只凸形状成立。下文条目 1–3 保留为当时的分析记录，结论以本段为准。

1. 冻结先验管道在 rough 0.06 已是 6/6（E41/E42），而 0.06 已达到论文 rough 的
   振幅上限 → **论文意义上的 "rough 泛化" 我们并未落后**。〔已被上段修订〕
2. **0.08 悬崖是我们自设的压力测试**（同早期 MuJoCo 本地扫描 ±0.02–0.10），
   不是复现论文的缺口；论文从未声称 0.08。
3. 若要做"论文级严格对照"，缺的是一次**论文形状** rough 评测（对称 ±0.06、
   0.2 m cell、8×8 m），这是纯配置改动、零训练成本，可作任何微调实验前的
   gate 0。〔已于 2026-08-15 完成，结果见 G0：有坑 0.06 即全倒〕
4. 若继续 path a（RL 微调冻结 SONIC 解码器冲 0.08），其定位应明确为
   **超出论文的能力边界压力测试**（回答"解码器流形是否可被 RL 顶开"），
   而非"补复现缺口"。
