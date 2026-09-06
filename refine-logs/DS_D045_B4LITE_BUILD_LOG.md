# DS D045 执行日志：B4-lite 首版构建（四路并行执行计划 + S1–S6 进度区）

> 【层位 L3 侧轴｜执行日志（2026-09-07 D045 开户落盘）】↑ `refine-logs/README.md`
> （扇出树根）｜父协议：`DS_B4LITE_ACTION_SPLIT_PLAN.md`（本文 = 其 §6 执行工序
> S1–S6 的执行日志，owner 三点默认裁定与 9h 四路并行执行计划的事实记录）｜
> 上游：纲领 `DS_TERRAIN_ADAPTER_CHARTER.md`（§4 阶段一）、
> `DS_OFFICIAL_DATA_PLAN.md`（§3 流水线 B3' 门判据）｜事实源：
> `tracker/D.md` D045 行（Run 数据唯一事实源；本文只放执行计划与进度判读）｜
> 状态：**已冻结（2026-09-07 数据侧结算完成；S1–S6 与 §5 收账均已回填终态，Run 行见 `tracker/D.md` D045 行 DONE）**

**这篇讲什么**：D045 = B4-lite 首版构建的开户执行日志。2026-09-07 主会话定
9 小时四路并行执行计划、owner 批准三点默认裁定后 T0 发令。本文落盘：开户
事实与三点裁定（§0）、四路分工（§1）、时间轴与验收点（§2）、风险预案五条
（§3）、S1–S6 六节进度区（§4，逐节回填状态/产物路径/判读）、收账节（§5）。
Run 行纪律：行只进 `tracker/D.md`（协议 §6 同款），本文不存 Run 数据。

## 0. 开户事实与三点默认裁定（2026-09-07 owner 批准）

- **D045 定位**：B4-lite 首版构建（协议 `DS_B4LITE_ACTION_SPLIT_PLAN.md` §6
  执行工序 S1–S6 全工序）。D044 B3' 门四核心类（dance/jump/run 100% +
  walk 96.0%）全过后，数据侧正式协议的首次执行。
- **裁定①（语义族清单）= 10 族**：forward_walk / slow_walk / fast_walk_run /
  forward_jump / dance_rhythm / turn_walk / lateral / start_stop_transition /
  asym_upper / posture_change（S2 覆盖缺口表出来后允许微调）。
- **裁定②（三集指定 + 演员留出）**：10 族中 6 训练 / 2 开发验证 / 2 最终测试
  （T-fam 留出）；演员留出独立：522 演员 ID 哈希 80/10/10；留出族内部演员
  不与训练族共享；演员×族交叉空格时调族不调规则（协议 §7.3 同款）。
- **裁定③（边界集口径）= 双轨**：manifest 内 `quality_status=L3` +
  `boundary_set=true` 字段 + 独立 `boundary_set.json` 清单（A005 类段两处
  可见，协议 §2 能力边界集落法）。

## 1. 四路并行分工（9h，T0 发令）

| 路 | 载体/位置 | 任务链 | 交付 |
|---|---|---|---|
| L1 数据主线 | lab-ts | S1 metadata 落位 → S2 描述名→语义族映射层+五维标签 → S3 三层清洗 → S4 划分冻结 → S5 批量转换+B3' 扩面回放+≥10% 渲染抽看 | 映射表 / 五维标签 / 清洗台账 / 划分清单 / npz+manifest |
| L2 算力线 | CVGL → node01 3090 池（32c64t_256_3090） | 端代码同步 → Isaac 卡死复验 → B3' 扩面回放蹲守 → stretch：VAE 训练管线冒烟（CVGL 4090） | 复验结论 / 回放 JSON / 冒烟记录 |
| L3 文档台账线 | 本机 | D045 行 + 本叶 + 挂树 + tree_check 全绿 + commit | 本文 + tracker D045 行 |
| L4 只读调研 | 本机 | D044 产物格式盘点 + A005 能力边界先例 | 盘点笔记（不入 Run 行） |

- L1 的 B3' 扩面回放在 lab-ts 同硬件执行，与 D044 判读基准可比。

## 2. 时间轴与验收点

T0 发令 → T0:30–2:30 S1+映射层 → **T2:30 验收点①**（族清单 + 三裁定落定）→
T3–5 S2 后半 + S3 + S4 → **T5 验收点②**（S5 发射；双查目录 + 进程防双发射）→
T5:30–7:30 S5 跑量 → **T7:30 验收点③**（B3' 逐族判定 + S6 冻结）→ T8–9
stretch 冒烟 + 收账。

## 3. 风险预案（五条）

1. **temporal_labels 下载失败** → 走整段粗标签降级，标注置信度降级记录
   （协议 §7.2 同款）；
2. **node01 3090 Isaac 再卡死** → 上 CVGL 4090；
3. **扩面族存活 <95%** → 先离线回环排除转换器伪影（D044 A005 同法：离线
   回环健康 + Isaac 闭环系统性摔倒 = 闭环空间漂移型，非转换器伪影），再定性
   L3 能力边界，不急删；
4. **S4 泄漏违例** → 调族不调规则（演员留出优先级高于族覆盖，协议 §7.3
   同款）；
5. **B4-lite 衍生 npz 触 BONES-SEED「不公开转发」纪律** → 只存服务器，不进
   公开仓（D043 license 三纪律落账同款）。

## 4. S1–S6 进度区（滚动更新）

### S1 metadata 落位

- 状态：DONE
- 产物路径：seed_metadata_v004.parquet 上服务器（lab-ts `~/ros2_data/apt_g1/data/ds_bones/`），**142,220 行×51 列核验**；temporal_labels 复验无有效 token（D043 revoke 后未恢复）
- 判读：全量可查零缺失，判据达成；temporal_labels 缺失按风险预案①直落协议 §7.2 整段粗标签 fallback，标签置信度降级随映射层记录，非阻断

### S2 候选池与标签

- 状态：DONE（v1 缺陷修复后定稿，commit 8a2e665）
- 产物路径：`desc_family_map.json`（lab-ts g1_b4lite/，本地镜像 tmp/g1_b4lite/）；候选池+五维标签（51 列官方标注驱动：移动/上肢/姿态/接触/时间）
- 判读：归一 **4,067 描述类→owner 10 族，覆盖率 87.6%**（置信度 high/med/low 三档随映射落盘）；is_neutral 标定实为标准动作语料总旗标（**91.2%**）非 box 族专属 → v1 一刀切排除曾致 dance/slow_walk/lateral 三族零入选，改**描述名定向排除 box_climb** 后 10 族全覆盖；映射判据（协议 §7 R1）与裁定①②落定达成

### S3 三层清洗

- 状态：DONE
- 产物路径：清洗台账 ledger（逐段排除原因，lab-ts g1_b4lite/ledger/）
- 判读：**L1 转换失败 0**；L2 排除合计 79,404 段——props 16,945 / duration_short 11,443 / box_climb_desc 5,342 / duration_long 498 / mirror_dedup 45,171 / L2_label_mismatch 5（Step_Rotate_Reaction_Idle_0135_001__A020、orange_justice_slow_001__A465、crawl_ff_start_180_R_001__A125/A126/A127，渲染复看+轨迹证据逐段裁定）；L3 全部保留入能力边界口径（执行失败≠脏数据，协议 §2 同款）

### S4 划分冻结

- 状态：DONE（v2 冻结）
- 产物路径：split/ 划分清单 + leak/ 泄漏检查报告（lab-ts g1_b4lite/）
- 判读：152 段入选 → 5 段 L2 移出 → **147 段冻结**：train 84 / T-seg 18 / dev 20 / T-fam 22（lateral 12+slow_walk 10）/ boundary_ref 3；演员 ID 哈希三桶（80/10/10）；**泄漏三项全 pass**（镜像同集 0 违例、T-fam 演员零交集 68 vs 14 演员）；判据「零违例、调族不调规则未触发」达成

### S5 编码与回放扩面

- 状态：DONE
- 产物路径：npz/×147 + gate_summary_d045.json + renders/×16 mp4（lab-ts g1_b4lite/；渲染抽看 16/152=10.5%，MuJoCo offscreen）
- 判读：152→147 段**转换 0 失败**（lattice 违例全 0；roundtrip MAE mean 0.1468，A005=0.1136 与 D044 该段 0.114 复现一致）；B3' 双门 seed0：A 门 101/108=93.5%、B 门 43/44=97.7%；**3-seed 复现**——arc_jog_left_loop_002__A029、burning_start_R_001__A470、A005 各 3/3 系统性（A005=D044 边界段跨实验复现），Jump_002__A019 与 Jump_Left_001__A018 各 1/3=**seed-variance**（seed0 重跑通过，Isaac 非确定性实证，不判系统性）；渲染抽看 11 一致/4 存疑/1 轻矛盾（均已裁定处置）；lateral 族 12 段 locomotion 标签修正（前进→侧移，带前后 diff）入已知噪声

### S6 manifest 冻结

- 状态：DONE（冻结）
- 产物路径：`manifest_v2.json` + `boundary_set.json`（A029/A470/A005 三段）+ `gate_summary_d045.json` + `publication_table_d045.json`（lab-ts g1_b4lite/）
- 判读：逐族存活 **7 族全 PASS**——forward_walk 17/17、dance 18/18、turn 18/18、asym_upper 10/10、posture 10/10、lateral 12/12、slow_walk 10/10 全 100%；**3 FAIL 族如实保留**（系统性 fall 按协议保留族内计数）——fast_walk_run 17/18=94.4%、start_stop 14/15=93.3%、forward_jump 16/18=88.9%；**整体 142/147=96.6%**；T-seg 18/18、dev 20/20、T-fam 22/22 全存活；三表示参与声明 147 段统一；四问逐段可答判据达成（协议 §5）

## 5. 收账（2026-09-07 数据侧结算）

- **终态**：D045 数据侧 DONE。B4-lite 首版 **147 段冻结**（train 84 / T-seg 18 / dev 20 / T-fam 22 / boundary_ref 3），B3' **整体 142/147=96.6%**（7 族 PASS + 3 族 FAIL 如实保留），**边界集 3 段**（A029/A470/A005，均 3-seed 系统性复现），**T-seg 18/18、dev 20/20、T-fam 22/22 全存活**；seed-variance 2 段（Jump_002__A019、Jump_Left_001__A018，各 1/3，seed0 重跑通过=Isaac 非确定性实证）。
- **产物索引**：服务器 lab-ts `~/ros2_data/apt_g1/data/ds_bones/g1_b4lite/`（npz/×147、manifest_v2.json、boundary_set.json、gate_summary_d045.json、publication_table_d045.json、desc_family_map.json、candidates/、split/、leak/、ledger/、renders/×16 mp4）；本地镜像 `tmp/g1_b4lite/`。
- **协议偏差与坑位**：①temporal_labels 无有效 token（D043 revoke）→ 预案①启用，§7.2 粗标签 fallback+置信度降级；②is_neutral 一刀切缺陷（91.2% 总旗标误当 box 族）→ 描述名定向排除 box_climb 修复（commit 8a2e665），修复前 dance/slow_walk/lateral 三族零入选；③lateral 族 12 段 locomotion 标签修正（前进→侧移）带前后 diff 入已知噪声；④渲染复看 16/152=10.5%：11 一致 / 4 存疑 / 1 轻矛盾，均已裁定处置。
- **提交链（8 笔，均未 push）**：9366bfa → c980528 → 6994963 → 7ab931c → 12433a5 → 8a2e665 → f352fbb → 37f05d3。
- **未决事项**：①CVGL 3090 Isaac 复验 det 10167 仍排队（外部 8 卡任务占池，非本实验失败，另号跟踪）；②HANDOFF 同步留 owner 审后另步；③owner mp4 终审清单 3 段待看（终审 mp4 在 lab-ts `~/ros2_data/apt_g1/data/ds_bones/g1_b4lite/renders/`）：dance_basic_chaines_180_R_002__A310、Jump_002__A017、Loop_Forward_Walk_001__A017（弱）。

下游：B4-lite VAE 训练（n_vbins 坑待验证）另号开户。
