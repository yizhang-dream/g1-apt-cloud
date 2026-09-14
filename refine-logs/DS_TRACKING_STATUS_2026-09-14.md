# WBT 语料接入 × 语料效应第三点：D057 进行中交接（2026-09-14 交接快照）

【层位】L1 阶段状态/交接 | 上级：[refine-logs 扇出树](README.md) · 预注册/判读框架：[DS_CONTINUOUS_EXECUTION_PLAN](DS_CONTINUOUS_EXECUTION_PLAN.md) §5q · 上一快照：[DS_TRACKING_STATUS_2026-09-12](DS_TRACKING_STATUS_2026-09-12.md) · 结论口径：[HANDOFF/README](../HANDOFF/README.md) §3

> 时点：2026-09-14 上午（d4f88a0 部署、9 进程并行转换发射后）。面向新会话接手，一页读全。
> 主线变化：09-12 快照的「平地命令跟踪三瓶颈」线收束在 D056（Pareto 三角）；09-14 owner 指令转向
> **UnifoLM-WBT 语料训练**（「那就开始训练这份语料吧」）= D057，本快照只覆盖这条线。

## 0. 一句话现状

D057（WBT 语料接入 + speedA 式 VAE 重训 = 语料效应第三点）**进行中**：G0 选料（12/22 子集
~429 分钟行走材料）→ G2 冒烟门（两轮，门重锚后过）→ G3→G4→训练全链路真数据验证（挖掘器/
builder/训练器沙箱全通）已毕；**G3 全量转换 9 进程并行在 lab-ts 后台跑**（长杆 ~7.5h 隔夜，
发射时点 09:30 服务器时间左右），完成后照 §4 命令序列接：挖掘 → vb 标签 → 30ep 训练 →
init 档位图（Isaac 电池走 CVGL）。判读框架预注册在 §5q（甲=单调∧全档低于 B4-lite
{0.29,0.47,1.31}）。

## 1. 科学问题（为什么做这条线）

- **语料效应第三点**：D048n 已证 speedA(B4-lite) init 档位图 {0.29,0.47,1.31} 严格单调
  （速度标签→冻结解码器因果链）+ ctrlB 弱单调（语料效应并行发现）。换一份速度分布截然不同
  （低速带 0.1-0.5、真机载荷/重心转移）的语料重训同架构 speedA——若 init 档位图随语料分布
  平移 ⇒「语料分布→档位-速度映射」方向性复证。
- 兼 D056 丙分支「材料层」归因的首个可检材料（方向多样语料；db 轴仅观察项不设门）。
- 摸底结论（09-13，探针 lab-ts /tmp/wbt/）：UnifoLM-WBT **无 >1.694 高速材料**（最富行走
  子集 200 局 v_p90 全 ≤0.51）——语料量程上限=遥操采集方式的结构性质，与 D048r 本体路径
  1.71 构成论文边界叙事对仗的两半。

## 2. 已完成（commits 617e2f0→d4f88a0，全部已推送）

- **G0**：22 子集 data-only 下载 3.8GB（hf-mirror，UA=curl/8.0 必带；3 个 MainCamOnly 嵌套
  目录 tree 列表在 mirror 异常未取得，~1-2h 低行走材料，如实放弃）；扫描选料
  `scan_wbt_g0.json`：**12/22 入选**（walk_min≥10min），合计 ~429min 纯行走/~5.6k 窗；
  WalkToTable 63.5min 居首，Fridge 类站立操作落选。
- **G2 冒烟门（两轮）**：首轮 6/6 被膝签名断言误杀（站立 episode 膝 median 自然 -0.08，
  深屈曲 max 0.9-1.2 在=关节序没错）→ 454ab84 口径修正（per-run 只拦 median<-0.25/max>2.2，
  站立位降级 warning+聚合签名 stdout），reviewer 复核 pass；二轮 6/6 过（lattice 全 0、
  roundtrip mean 0.1493/max 0.1731、default 基线 0.25-0.28）。**门重锚**（§5q append-only）：
  原 0.16 门锚 D038 单配对样本非接收带；B4-lite 现役 147 段实测带 mean 0.1468/p90 0.2074/
  max 0.2539 ⇒ 重锚「WBT ≤ B4-lite 带」，冒烟整段落于带内。
- **链路真数据验证**：挖掘器（6 npz→12 窗/48s、edges=[0.153,0.389]、vb 三桶均衡）→
  builder（11 件产物）→ 训练器 2ep 沙箱（train_token_vae_e39 --vb-npy 正常收敛落盘七件套）。
- **四脚本入仓+SCRIPT_MAP 登记**：scan_wbt_datasets / convert_wbt_g1_parquet /
  mine_wbt_walk_segments / build_wbt_vae_inputs；reviewer 三轮（交付 pass-with-notes →
  签名修正 pass → batch-encode pass）。
- **性能攻坚三连败记录**（重要教训，勿重蹈）：①串行逐帧 ~33 帧/s（ETA 47h）；②--batch-encode
  批推理 **encoder/decoder ONNX 输入批维双双静态 [1,·]**（813+78 全 FAIL INVALID_ARGUMENT，
  「encoder 可批/护栏已过」曾是我的错误推断——护栏从未执行到）；③动态批维重导出被图内
  `Reshape_3` 硬编码 {1,1,10,6} 阻断（副产品：encoder 图内 ScatterND 按 obs[0] 模式头选
  分支）。**终案=零代码 9 进程分 repo 并行**。--batch-encode/decode_actions_threaded 保留
  在代码里并注明限制（换可批导出模型时即生效）。

## 3. 正在运行的服务器作业（lab-ts，与本对话解耦，nohup 存活）

- **9 个转换进程**（`convert_wbt_g1_parquet.py --roundtrip`，默认串行路径，OMP_NUM_THREADS=1），
  分组：G1=Laundry_Basket_Washer（长杆 ~888k 帧@50 ≈7.5h）/G2=Inspire_Washing_Machine/
  G3=Brainco_Collect_Plates/G4=WalkToTable/G5=Inspire_Pick_Up_Drinks/G6=Brainco_Empty_Folder/
  G7=Inspire_Empty_Folder+Inspire_Hang_Up/G8=Brainco_MakeTheBed+Pickup_Pillow+Hang_Up/
  G9=Brainco_Pick_Cloth_On_Bed。
- 日志 `/tmp/wbt/conv_mp1.log`~`conv_mp9.log`；产物 `~/ros2_data/apt_g1/data/ds_wbt/g1wbt_conv/`
  （npz/ + manifest.json，发射时点已 181 npz）。发射器 `/tmp/wbt/launch_mp_convert.sh`。
- **~2h 检查点实测（watcher）**：全局 ~4 run/min、已转 410；mp1（Laundry 长杆）争用下
  16.4 帧/s，**完成窗口=今夜 00:00-06:00**（小组退出释放 CPU 后提速；rt_mae 全程
  mean 0.141/max 0.193 n=381，低于冒烟水平——质量面持续健康）。
- **~3h 检查点（重构落地）**：四脚本 + 新共享模块 `wbt_common.py` 结构重构入仓
  81a9048（SCRIPT_MAP 已登记；零漂移三重证据=四脚本 --selftest 检查名与 --help
  逐字节一致 + AST 级数值路径对照 + 服务器 17 项产物 md5 对拍全过〔smoke 挖掘/
  构建 + Fridge 子集转换默认路含 roundtrip，覆写重跑逐位一致〕）；§4 命令不受
  影响（脚本名/旗标全不变）。两笔勘误/再证：①下文「发射时点已 181 npz」的
  manifest.json 实际不存在——manifest 只在各进程收尾落盘，中途仅 npz/ 有内容；
  ②--batch-encode 基线复测 6/6 INVALID_ARGUMENT（encoder 批维同为静态）=§2
  勘误再证。
- 完成判据：9 进程全退 + 各日志出现 `SUMMARY: converted` 与 `[signature]` 聚合行。
- 进程查法必须用方括号防自匹配：`pgrep -f "convert_wbt_g1_parque[t]"`（**pkill 自杀坑已踩一次**：
  ssh 命令行含匹配串会把远端 shell 一起杀）。

## 4. 接手指令（新会话直接执行，全部在 lab-ts）

```bash
# 1) 查转换是否完成（9 组都要有 SUMMARY；未完则等/隔间查）
ssh lab-ts 'pgrep -fc "convert_wbt_g1_parque[t]"; for i in 1 2 3 4 5 6 7 8 9; do tail -3 /tmp/wbt/conv_mp$i.log | grep -E "SUMMARY|signature"; done'

# 2) manifest 完整性核验（9 进程尾端合并竞态防护，D044 坑①先例：以 npz 目录为准重建）
#    npz 数 == manifest 条目数（error 条目除外）；不符则以 npz 内 meta 重建 manifest
ssh lab-ts 'ls ~/ros2_data/apt_g1/data/ds_wbt/g1wbt_conv/npz | wc -l; ~/ros2_data/.venv_isaac/bin/python -c "import json; m=json.load(open(\"/home/cvgluser/ros2_data/apt_g1/data/ds_wbt/g1wbt_conv/manifest.json\")); ok=[e for e in m if \"error\" not in e]; print(len(m), len(ok))"'

# 3) 挖掘行走窗 + 构建 vae_inputs + vb 三分位标签（冒烟已验证过的同款命令，换正式目录）
ssh lab-ts 'cd ~/ros2_data/g1-apt-cloud-sync && export PYTHONPATH=$PWD/apt_g1:$PWD && V=~/ros2_data/.venv_isaac/bin/python && D=~/ros2_data/apt_g1/data/ds_wbt/g1wbt_conv && $V apt_g1/mine_wbt_walk_segments.py --conv-dir $D && $V apt_g1/build_wbt_vae_inputs.py --segments-json $D/walk_segments.json'

# 4) 30ep 训练（3060 CUDA 可用——NVML 坏只影响 nvidia-smi 监控；预计 ~10-20min 量级）
ssh lab-ts 'cd ~/ros2_data/g1-apt-cloud-sync && export PYTHONPATH=$PWD/apt_g1:$PWD && D=~/ros2_data/apt_g1/data/ds_wbt/g1wbt_conv && nohup ~/ros2_data/.venv_isaac/bin/python apt_g1/train_token_vae_e39.py --data-dir $D/vae_inputs_wbt --vb-npy $D/vae_inputs_wbt/vb_speed.npy --epochs 30 --out-dir ~/ros2_data/apt_g1/outputs/token_vae_d057_wbtA > /tmp/wbt/train_d057.log 2>&1 < /dev/null & disown'

# 5) G5 init 档位图：D048n G2 recipe（tracker D.md D048n 行），Isaac 电池走 CVGL node06
#    （lab-ts GPU 驱动错配未修：内核 NVRM 595.84 vs 用户态 595.91，D052 先例直接 CVGL）
```

判读（§5q 预注册，勿改）：**甲**=三档严格单调 ∧ 全档低于 B4-lite {0.29,0.47,1.31}；
**乙**=单调但交叉/持平（edges 数值入账）；**丙**=不单调或材料不可用。观察项：db 方向轴档位分布。

## 5. 坑位与已入账事实（接手必读）

- **膝签名拒绝**：3/813 run max 2.227-2.256 微超 2.2 限（真机软限外微超），维持拒绝不放宽，
  manifest error 条目可查；**ep78/ep130 完全同值**=Washing_Machine 疑似重复上传 take，不剔除。
- **sts_ppo 并存**：owner 并行项目 mdiv2 gate（14 worker）在 lab-ts 上，勿杀；我方 OMP=1 已限流。
- **venv 依赖拆分**：转换器（mujoco+onnxruntime+pyarrow）跑 `.venv_mjlab`（已补装 pyarrow
  25.0.1）；挖掘/builder/训练器跑 `.venv_isaac`。脚本一律从同步克隆跑
  （`cd ~/ros2_data/g1-apt-cloud-sync && PYTHONPATH=$PWD/apt_g1:$PWD`）——Mimosa 钩子禁 bash
  部署 cp .py，克隆直跑是合规替代。
- **hf-mirror**：Python urllib 默认 UA 被 403，带 `User-Agent: curl/8.0`。
- **WBT 数据契约**（双 schema）：q36=`robot_q_current[36]`（根位姿 7 wxyz+29 关节）/grouped=
  `state_base_pose[7]`+`lower_body[15]+left_arm[7]+right_arm[7]`；30fps；关节序=MJ 序（值域
  签名+roundtrip 兜底，名字级 URDF 对拍待补=§5q 风险①未清偿项）。

## 6. 待 owner 裁量（无阻塞项）

- G5 评测平台：CVGL node06（默认，D052 先例）vs 等 lab-ts 驱动修复（需管理员重载内核模块）。
- 闭账后可选续刀：语料混合臂（B4-lite+WBT）、db 方向轴标签重训（材料层检验的正对面）。
