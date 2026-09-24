"""D067 G2 输出端并行残差臂训练入口（三臂对照 + max-forward 奖励）。

预注册 = refine-logs/ds/DS_CONTINUOUS_EXECUTION_PLAN.md §5y（:496-506）。
立项口径（owner 2026-09-22 拍板第 4 项 + 目标改写）：
    a_t = decoder(token_t) + r(s_t)
  - token_t = author v0 ckpt（D061a）推理 token（--author-ckpt）；**推理期 ctx 命令槽
    钉最大速度档**（--intent-pin-max 默认开，假设 A2）。
  - decoder = 冻结 SONIC ONNX decoder（同 D064/D065）；其输入 = 994 维
    `[token(64), 本体历史(930)]`（sonic_action_term.py:24-34 通道契约）。
  - r = 轻量 MLP [256,128]+SiLU 输出 29 维关节目标残差，**末层零初始化** ⇒ 构造期
    与 decoder 臂输出逐位一致（残差恒等 0，不破坏 token 先验）。
  - PPO 只训 r（decoder/token 源全程冻结，梯度只达 r；critic 属值函数必要可训项，
    同 D065 adapter-only 口径，假设 A4）。
奖励（2026-09-22 owner 目标改写）：residual 臂默认 `--reward-mode max_forward`
  = 机体坐标前进速度（root_lin_vel_b[:, 0]）正向奖励 + 官方配方防摔/姿态/动作平滑正则，
  **无人工速度上限**，移除速度命令跟踪项（track_lin_vel_xy_exp / track_ang_vel_z_exp）。
  direct/decoder 臂默认保持 D064 原命令跟踪口径不动（保 D064 可比性）。

三臂（--arm direct|decoder|residual）：
  direct   : env 动作 = 官方 JointPositionAction（29 维），reward = 命令跟踪（D064 逐字）。
  decoder  : env 动作 = 冻结 SONIC decoder ActionTerm（64 维 token），reward = 命令跟踪（D064 逐字）。
  residual : env 动作 = direct 动作通路逐字（29 维，经 sibling `action="lora_policy"` 变体
             = direct 通路 + policy 组末位 930 维 decoder 本体历史观测）；policy obs 另追加
             64 维 `author_token`（env 侧冻结 author adapter 产出，观测常量，无梯度）；
             policy μ = 冻结 decoder(cat[token, proprio]) + r(官方 policy obs)，
             reward 默认 max_forward。

为什么 decoder 必须 in-graph（不在 env 侧 ActionTerm）：D065 v1 已实测——env 侧
  ActionTerm 的输出对 PPO 的 log_prob/熵无梯度（全冻结 policy 时 loss.requires_grad=False，
  backward 直接 RuntimeError），残差会退化成「逐位等于 decoder 臂」的假阴性。故残差臂与
  D065 C 臂同形态：decoder 在 policy 图内，μ 经分布头，梯度经 log N(a; μ, σ) 直达 r
  （见 train_g1_decoder_lora.py 模块 docstring 的完整论证）。

与 D064/D065 的关系（最小侵入，不改既有默认行为）：
  - 本文件是**新增入口**，不修改 train_g1_decoder.py（D064）任何默认值/行为。
  - env cfg 一律经 sibling `g1_velocity_decoder_env.make_env_cfg` 工厂获取（残差臂用既有
    `action="lora_policy"` 变体，即 direct 动作通路 + 930 维本体观测）；奖励/obs 需求全部
    在**本文件内**做 cfg 覆写（参考 eval_author_v0_decoder.py 的 cmd_cfg 覆写先例）；
    不改 g1_velocity_decoder_env.py，不动 sonic_action_term.py。
  - PPO 配方逐字段继承 D064 官方 G1 rough（保三臂可比性）。

分层（同 sonic_action_term.py / eval_author_v0_decoder.py 惯例）：
  Layer 1（纯 torch/numpy，零 isaaclab/onnx/rsl_rl 依赖，本机 Windows 可 import 可测）
    - 残差 MLP `ResidualActionMLP`（末层零初始化）
    - μ 路径 `ResidualMuPath`（冻结 decoder(decoder_obs) + r(r_obs)）
    - 奖励纯数学 `forward_vel_reward` / `max_forward_total_reward` + 配方表 `MAX_FORWARD_RECIPE`
    - 三臂配置表 `ARM_TABLE`、身份信封 `build_identity_envelope`
    - 契约件：`CTX_TERRAIN_MAP`/`ctx_terrain`（地形映射）、`forward_token_affine`（token
      前向仿射）、`load_token_stats`、`official_obs_slice`（r 输入切片守卫）、
      `assemble_decoder_obs`（decoder 输入组装）
    - `run_selftest()`：零初始化恒等 / 梯度隔离 / 奖励数学 / 三臂表 / 身份信封 /
      **前向仿射往返** / **residual vs decoder 臂 decoder 输入逐位一致** / 地形契约 /
      切片守卫 / **执行-反馈闭环增益恒等** / **旧标量 0.5 反例** / **29 维 scale 形状与来源** /
      **adapter env 维分块 ≡ 全 batch（逐位，含整块部分 reset 与尾块）** / **分块 CLI 默认值** /
      **adapter bf16 autocast 前向（合成 logits + 非近平局 mock argmax 一致）** /
      **adapter tf32 标志进/出恢复** / **dtype 与 chunk CLI 默认值（bf16/1024）** /
      **rollout 期 obs 消毒器（脏 obs 净化+计数行+首例坐标 / 健康逐位零写 / 透传性）**
  Layer 2（isaac，import 集中在函数内；本机不 import）
    - `_import_heavy`（复用 train_g1_decoder._import_heavy，单份组装防漂移）
    - env cfg 构建 + max_forward 奖励覆写 + author_token 观测项注入
    - `ResidualDecoderPolicy` + rsl_rl 类名注册
    - runner 训练回路

用法（服务器 cvgl，cwd=仓根）：
    # 本机 CPU 自测（不 import isaaclab）
    python apt_g1/isaac/train_g1_decoder_residual.py --selftest     # -> D067_SELFTEST_PASS

    # 残差臂训练（author v0 为 token 源）
    python apt_g1/isaac/train_g1_decoder_residual.py --arm residual \\
        --terrain rough --num-envs 4096 --max-iterations 3000 \\
        --author-ckpt <d061a/ckpt_final.pt> --token-stats <g1_token_stats.npz>

    # direct / decoder 对照臂（reward 保持 D064 命令跟踪）
    python apt_g1/isaac/train_g1_decoder_residual.py --arm decoder \\
        --terrain rough --token-stats <g1_token_stats.npz>

契约映射（reviewer 验收 blocker 修复，2026-09-22）：
  M1 terrain 契约：CLI 地形名（plane/rough/stairs/stones/discrete，本文件 TERRAINS）
     **不得直传** `ctx_from_command`（后者只接受 eval_author_v0_decoder.TERRAIN_TYPES =
     plane/rough_paper/climbing_box，:193）。显式映射表 `CTX_TERRAIN_MAP`：
       plane -> plane；rough/stairs/stones/discrete -> rough_paper
     （本仓无 climbing_box env 地形，故不映射到 climbing_box；如需 climbing_box 语境，
     要另建对应 env 地形，不是本表的事。）映射在 `_token_obs_params`/`residual_token_obs`
     两处接入（`ctx_terrain()`，幂等：已是 ctx 地形则原样返回），并进身份信封
     `ctx_terrain`/`ctx_terrain_map` 字段。
  M2 token 前向仿射：`residual_token_obs` 里 `adapter.step` 返回的是**归一化动作 a**
     （= (tok-mean)/(alpha*std)，eval_author_v0_decoder.AuthorPolicyAdapter.step:638-642），
     不是 token 坐标；喂 decoder 分支前必须补**前向仿射**
       token = mean + alpha*std*tanh(a)（token_bound=="tanh"）/ mean + alpha*std*a（"none"）
     （权威式：sonic_action_term.py:205-215 map_token、sonic_lora_policy.py:157-161
     token_from_raw；与 adapter 的逆仿射互逆）。mean/std 与 decoder 分支**同一份
     `--token-stats` npz**（同一 std 下限 TOKEN_STD_FLOOR，与 author_token_stats /
     sonic_action_term.token_std_floor 同值）显式传入 `residual_token_obs` 并在运行期与
     adapter 内部统计做逐位一致性断言；alpha/bound 与 `_token_obs_params` 现值一致。
     该仿射身份进信封 `token_affine` 字段。
  M3 特权 critic（SF-1，**有意设计，不改行为**）：`author_token` 同时追加进 policy 与
     critic 两组，即 critic 可见的观测比 actor 多（特权 critic 不对称）；信封
     `privileged_critic` 字段声明。
  M4 plane 档奖励口径（SF-4）：max_forward 配方对**所有地形统一**施加官方 G1 rough
     正则（含 plane），与官方 flat 语义 `_apply_flat_semantics` 的正则**不同源** ⇒
     plane 档与官方 flat 的对照**非严格单变量**；信封 `reward_recipe_scope` 字段声明。

显式假设（指令含糊处的解释，均落成常量/打印，便于事后改判）：
  A1 token 经 policy obs 传递：author adapter 是**有状态自回归**件，若在 policy.forward
     里运行，PPO 的 evaluate 阶段会重跑并污染滚动窗 ⇒ 必须 env 侧每控制步算一次、把
     64 维 token 作为观测常量喂给 policy（`author_token` obs 项，本文件内新增）。
     decoder 是无状态 MLP，重跑安全。
  A2 intent-pin 值：`--intent-pin-value` 缺省 1.0（= 官方 G1 rough 命令 lin_vel_x 上界，
     rough_env_cfg.py:146-148 的 (0,1) 上界）。若语料速度带另有最大档，用
     `--intent-pin-value` 显式覆盖。
  A3 残差臂动作语义：env 侧 = direct 臂 JointPositionAction（use_default_offset=True），
     **scale 覆写为逐关节 sonic_scale 向量**（不是 direct 臂的标量 0.5；见下方「执行/反馈
     仿射一致性」小节）；policy μ（29 维）= 冻结 decoder 归一化输出 + r。
     **修订记录（2026-09-22，CVGL 探针坐实根因）**：原 A3 口径「scale=0.5、与 D065 C 臂
     同口径」存在闭环增益缺陷——反馈通路 `sonic_proprio_hist` 的 last_act 通道按**逐关节
     sonic_scale** 反归一化（`last_act=(q_des-default)/sonic_scale`），而执行端 scale=0.5 ⇒
     `last_act = μ·0.5/sonic_scale ≠ μ`，闭环增益 0.91–6.71×（踝≈0.0745 ⇒ 增益 6.71），
     12 步几何发散至 2.49e7（与训练 iter0-2 NaN/action_rate 1e15 签名吻合）。修法 = 执行端
     scale 改逐关节 sonic_scale，使 (q_des-default)/sonic_scale ≡ μ 严格互逆、闭环增益恒 1。
  A4 critic 可训：PPO 值函数必须有梯度路径（同 D065 adapter-only），故 critic 与 r 可训，
     decoder/actor 占位/σ 冻结。
  A5 r 的输入 = 官方 policy obs 切片（`total - 930(proprio) - 64(token)`），即原 D064
     policy 八项（含 height_scan）——保持残差看到与 direct/decoder 臂相同的状态口径。
     **SF-2**：该「前缀切片」不再靠隐式假设——`_build_residual_policy_kwargs` 经
     `official_obs_slice` 按运行期 term 序推出宽度，并断言 `sonic_proprio`/`author_token`
     两 term 全部落在 official 段之后（起始=0 前缀契约），对不上即 fail loud。

执行/反馈仿射一致性（2026-09-22 修复，CVGL 探针坐实根因）：
  残差臂闭环里同一条关节目标要过两处仿射，二者必须严格互逆，否则每步乘一个 ≠1 的增益：
    - **执行端**（env 侧动作项）：JointPositionAction
      `q_des = default + raw * scale + 0`（joint_actions.py:130-139）；本臂 raw = μ。
    - **反馈端**（decoder 本体历史观测项）：`sonic_proprio_hist` -> `SonicActionCore.
      last_act_from_q_des`：`last_act = (q_des - default) / sonic_scale`
      （sonic_lora_policy.py:517 附近取 processed_actions；sonic_action_term.py:244）。
  取执行端 `scale = sonic_scale`（**逐关节 29 维向量**）⇒
    `last_act = (default + μ*sonic_scale - default)/sonic_scale ≡ μ`（逐位），闭环增益恒 1。
  原实现执行端 scale=0.5（标量）⇒ `last_act = μ*0.5/sonic_scale`，增益 0.5/sonic_scale
  逐关节 0.91–6.71×（踝 sonic_scale≈0.0745 ⇒ 6.71），12 步几何发散至 2.49e7。
  本文件的覆写落点 = `_override_residual_action_scale`（**只改本臂 cfg 实例**，不动
  g1_velocity_decoder_env.py 的 DirectActionsCfg 默认值，保 direct 臂 D064 逐字可比）。
  `sonic_scale` 与 `sonic_proprio_hist` **同源**：两者都经 `sonic_action_term._sonic_scale_isaac()`
  取（单一事实源），本文件不再复制第二份常量（防漂移）。

Isaac Sim 关闭路径死锁修复 + 打断事件落盘仪器（2026-09-22，faulthandler 栈实证）：
  实证栈（本地副本 `tmp/_freeze_labts_iter950_PYSTACK.txt`，lab-ts iter 950 ≈ 23.35M 步）：
    main -> `simulation_app.close()`（原 :2450 finally）-> isaaclab `simulation_context.py:743
    _app_control_on_stop_handle_fn` -> `while not omni.timeline.get_timeline_interface().
    is_playing(): self.render()` **无限忙等**（timeline 永不恢复播放；其余线程全 idle）。
  ⇒ 主循环被未知事件打断后，finally 的裸 close() 把进程卡死在关闭渲染循环里，异常被吞。
  两层修复（**不改训练/奖励/策略任何路径**）：
    1. `_safe_close(simulation_app, timeout_s=120, exit_code=0)` 看门狗替代裸 close()：
       ① 先尝试 `omni.timeline...play()` 把 timeline 设回 playing（失败忽略，延迟 import）；
       ② `close()` 放子线程执行，主线程 `join(timeout)`；超时则打印警告 + 线程栈
          （`faulthandler.dump_traceback` 可用时）后 `os._exit(exit_code)` 强退
          （ckpt 在循环内已周期落盘，强退可接受）；正常关闭成功则由调用方正常退出。
    2. `_log_main_loop_crash`：主训练循环外套 `try/except BaseException`，把完整 traceback
       写入 `outputs/<out>/main_loop_crash.log`（时间戳/iter/异常类型）再 raise——
       下次复现时「谁在 23M 步打断主循环」直接见尸（当前该异常被 finally 的死锁吞了）。
  退出码语义不变：正常完跑 rc=0；异常时 `_safe_close` 不调用 `os._exit`（exit_code=0），
  由调用方 `raise` 沿原异常冒泡。

D064 同款裸 close 警示（**只登记不修**，2026-09-22 口径）：
  `train_g1_decoder.py:1035` 的 finally 仍是裸 `simulation_app.close()`，与本次修复前的
  本文件同款关闭死锁风险（CVGL 两臂挂死同根因）；本任务只动本文件，该修复随后续批次推广
  （改 D064 会牵动其历史可比性/复现命令，故不在本批次）。

D065 潜伏缺陷警示（**只登记不修**，owner 2026-09-22 口径）：
  同一缺陷潜伏在 D065 C 臂：`train_g1_decoder_lora.py` 的 C 臂 env 侧走
  `G1SonicLoRAPolicyEnvCfg`，其动作项 = `DirectActionsCfg`（`g1_velocity_decoder_env.py:247-249`
  的 `JointPositionActionCfg(scale=0.5, use_default_offset=True)`），而 policy 组末位同样挂了
  `sonic_proprio_hist`（按逐关节 sonic_scale 反归一化）⇒ C 臂存在与残差臂**同款**的执行/反馈
  仿射不一致（增益 0.5/sonic_scale）。本文件**不修** C 臂：D065 已产出的结论与 ckpt 身份绑定
  于旧口径，就地改动会让历史可比性失真；如需修，应在 `train_g1_decoder_lora.py` 侧以新实验号
  单独立项（改动范围、重训与结论修订均超出本次任务）。

σ 病态区夹紧 + σ 漂移仪器（2026-09-23，第二层修复，crash 日志实证）：
  实证（`tmp/_fixverify_main_loop_crash.log`；1024env×1250it direct max_forward seed0 确定性
  崩于 iter 950 ≈ 23.35M 步）：rsl_rl `ppo.py:260` `self.policy.act(...)` ->
  `actor_critic.py:122` `self.distribution.sample()` -> torch `normal.py:73`
  `torch.normal(loc, scale)` -> `RuntimeError: normal expects all elements of std >= 0.0`。
  机制 = rsl_rl ActorCritic 的可训 σ（`self.std`，init_noise_std 路径）是**裸 nn.Parameter、
  无任何下界约束**，PPO 梯度更新可把它推成负值；负 σ 构造出的 Normal 在 sample 时抛异常
  （构造期因 `Normal.set_default_validate_args(False)` 不校验，只在采样/log_prob 处炸）。
  track 奖励下 D064 曾完整跑完未触发，max_forward 下确定性触发（训练动力学不同）。
  两层修复（**不改奖励/网络结构/超参，不动 rsl_rl 库本体**，只在本文件内 subclass）：
    1. σ 采样夹紧 `ClampedStdPolicyMixin`：覆写 `update_distribution`，在**构造 `Normal` 前**
       对 σ 做 `std.clamp(min=STD_CLAMP_MIN=1e-4)`（**只下夹不上夹**）。三臂统一经该 mixin：
       direct/decoder 臂用 `ClampedStdActorCritic`（继承 rsl_rl ActorCritic + mixin，`__init__`
       逐字继承 ⇒ 与 D064 ActorCritic 同初始化）；residual 臂的 `ResidualDecoderPolicy` 同样
       继承 mixin（μ 路径不变）。**夹紧仅在 σ≤1e-4 的病态区生效：健康轨迹（σ≥1e-4）下 clamp
       是恒等映射，Normal 的 loc/scale 与反传梯度逐位不变 ⇒ D064 可比性与健康训练零改动。**
       夹紧只作用于**采样/求 log_prob 用的分布**，不改写 `self.std` 参数本体（漂移仪器仍能观测
       原始 σ 滑向负值的全过程）。
    2. σ 漂移仪器 `install_std_trajectory_instrument`：包住 `alg.update`（rsl_rl 主循环每 iter
       调一次），每次（默认每 iter）把 `policy.std` 的 min/mean/max/负值个数追加到
       `outputs/<out>/std_trajectory.log`（pre/post/crash 三段；crash 段记录 update 抛异常
       瞬间的 σ，正好是本次崩溃的负值）。下次长跑即可看到 σ 从健康滑向负值的时间与速率。
  退出码语义/身份信封/复现命令均不变。

σ NaN 净化 + update 内首 NaN 溯源探针（2026-09-23，第三层修复，crash 日志实证）：
  实证（lab-ts `~/ros2_data/d067_verify/freeze_sigmafix_iter950_crash.txt`）：σ 轨迹 1902 行
  **全程健康**（n_negative=0、min=0.51 从未贴地板），iter 951 pre 正常、post=NaN ——
  **一次 update 内部把 σ 参数打成 NaN**；`torch.normal` 对 NaN 同样抛 `std >= 0.0`，而第二层
  的 `clamp(min=1e-4)` 对 NaN 恒等（`NaN.clamp = NaN`）⇒ 保命带对 NaN 失效。三次崩点全部
  确定性落在 ~23.35M 环境步。首要嫌疑（**未实证**）：max_forward 奖励无上限 -> 某 env 物理
  速度爆炸 -> obs 出 inf/NaN -> 前向/梯度回传污染 σ。
  两层修复（**不改奖励/网络/超参，不动 rsl_rl 本体**，延迟 import）：
    1. **σ NaN 净化（保命带）**：`_clamp_std_param` 改为
       `where(isfinite, clamp(min=STD_CLAMP_MIN), STD_CLAMP_MIN)` 等价式（NaN/±inf -> 1e-4；
       有限值路径与旧 clamp **逐位一致**）。docstring 注明「NaN 溯源探针在记录真相，净化只保
       进程不崩」——**参数本体不被改写**，NaN 仍留在 `self.std`，故仪器/探针能看到污染真相。
    2. **update 内首 NaN 溯源探针（治本仪器）**：`install_std_trajectory_instrument` 的
       `_wrapped_update` 在 orig_update 前后与内部可及点探测：① obs_batch（`alg.storage.
       observations` 逐 key；policy 段按 `official/proprio/token` 切片**能分就分**）② 返回的
       loss_dict 逐 key ③ update 后 `policy.std` 与策略输出（`mu`/`actor`，可及才跑）。
       首个非有限量以 `tag=nan_probe` 行落 `std_trajectory.log`（字段 name/n_nonfinite/
       min/max/first/first_index/shape/dtype + stage/update_calls），**只记录不阻断**
       （保命带在，训练继续，让探针在后续崩点自然复现时留下完整现场）。梯度注册钩子未做
       （指令标可选；现有可及点已足够定位「污染在 obs 还是 loss」）。

update 内分步 NaN 溯源探针（2026-09-22，第四层，纯记录仪器）：
  已定案证据链：iter 950 单次 `alg.update()` 内 σ 参数 29 维中 27-28 维变 NaN（entropy
  由 +45.66 算术反解坐实）；lr 假说否定（全程 1e-5~3.8e-4 被代码硬夹）；幅值假说否定
  （崩前速度/奖励/loss 全 <2 正常）；update 用的 obs 缓冲有限（pre 探针）。首 NaN 必在
  update 内部计算序中：advantage→ratio→surrogate/value loss→backward 梯度→σ 参数更新；
  最可能单点 = 某中间量 NaN → loss NaN → 反向染全部梯度 → `clip_grad_norm_(1.0)` 对 NaN
  无效 → 参数全灭。本层**只扩仪器、不改任何训练数值路径**（不改奖励/网络/超参/rsl_rl 本体，
  延迟 import，best-effort 全兜），把 update 内部按计算序分步落盘：
    ① `pre_mid` stage（advantage/returns 探针）：orig_update 前从 `alg.storage` 取
       `advantages`/`returns`/`values`（rsl_rl RolloutStorage 属性名，按服务器 .venv_isaac
       源码核对；取不到静默跳过）——isfinite + 幅值。
    ② `post` stage loss 逐 key 幅值升级：不只 isfinite，逐 key 记有限元素的
       min/max/abs_max/mean；并记 `kl_mean`（见下「kl 口径」）与 `learning_rate`
       （adaptive 分支唯一可后验观测的量）。
    ③ 参数快照对比：orig_update 前后快照 `policy.std` 全量 + `actor`/`critic` 各首末
       `nn.Linear` weight 的 norm/isfinite；post 对比若 actor 权重与 std 同刻变 NaN 则
       印证「loss NaN 染全部梯度」单点假说。**不做 backward 钩子**（侵入风险）。
    ④ obs 探针逐 key 幅值：pre 探针在原有「首个非有限」之外，追加每个切分段
       （official/proprio/token）/整块的 min/max 幅值——量级趋势从此可见。
  新增行统一 tag=`nan_probe_mag`（幅值趋势行；**与首 NaN 行 `nan_probe` 分 tag**，使
  `nan_probe` 的「有 NaN 才有一行」语义与既有 20 项断言零回归），字段向后兼容。

  kl 口径（重要，勿误读）：rsl_rl 2.3.3 `PPO.update` 的 `kl`/`kl_mean` 是**函数内局部变量**
  （仅 adaptive 分支用），**不落 `self`**，update 返回后不可读 ⇒ `kl_mean` 字段 best-effort
  读 `alg.kl_mean`/`alg.kl`（当前版本恒为 null，`kl_source="absent"`）。adaptive 分支对
  `self.learning_rate` 的改写**是**可后验的：记 `learning_rate_pre`/`learning_rate` 两值，
  lr 下调即 kl_mean>2·desired_kl、上调即 kl_mean<desired_kl/2，可反推 KL 是否爆炸。

advantage/returns 消毒器 + critic 前向探针（2026-09-22，第五层修复/溯源，probe4 逐环日志实证）：
  已定案根因链（probe4 逐环日志）：iter 950 rollout 中 critic `evaluate` 返回的 values 出 **1 个
  NaN**（obs/权重均有限）→ GAE 把该 NaN 扩散成 **19 个 returns** → advantage 归一化
  `(A-mean)/(std+1e-8)` 的分母含 NaN ⇒ **放大成全 24576 个 NaN** → 单次 update 梯度全染
  （参数快照 5/5 翻转）→ 全灭。**放大器 = 归一化对单个 NaN 无免疫**（一个 NaN 拉低/污染
  mean/std，除法把 NaN 摊到全量）。本层两件（**不改奖励/网络/超参，不动 rsl_rl 本体**）：
    1. **消毒器（发射解锁件）**：`_wrapped_update` 在 `orig_update` **之前**对
       `storage.advantages`/`storage.returns`/`storage.values`/**`storage.privileged_observations`**/
       **`storage.observations`** 五路做非有限检测（检测范围随第六/七层扩到五路，见第七层段）；若有：
         ① 用 `_first_nonfinite_probe` 落 `tag=nan_probe` 行（`stage=pre_mid`，含
            n_nonfinite/first_coord，与第四层 pre_mid 语义一致）；
         ② **写回消毒**：非有限元素置 0（重赋值 `storage.advantages/returns/values`，见第六层），
            并落 `tag=nan_probe_sanitize` 行（n_nonfinite/n_total/n_zeroed/purged_full/abs_max_before）；
         ③ update 照常进行。
       **口径（声明）**：24576 中 1 个离群置 0，对归一化统计量的偏置可忽略。
       **口径修正（reviewer 2026-09-22）**：归一化先于消毒器 → 实际触发 = `purged_full` 单点
       场景仅在归一化后（`mini_batch_generator` 的 `(A-mean)/(std+1e-8)` 会把单个 NaN 摊成全量，
       消毒器在 `orig_update` 前拦下原始单点；storage 层 `purged_full` 只在原始 advantages 已
       全 NaN 时才出现）。
       **口径修正（reviewer 2026-09-22）**：全量置 0 = actor 梯度 0 + critic 被拉向 0 + loss 仍报
       NaN = 权重存活**非空**学习步（critic 仍有梯度路径）；依赖 torch clamp 对 NaN 的反向掩码，
       2.5.1/2.12 实测一致。
       **只在检测到非有限时才写**：健康路径（advantages/returns/values 全有限）逐位零改动、零写回。

第六层：消毒器 inference-tensor 写回修复（2026-09-22，第六跑实证）：
  实证：`storage.advantages` 是 **inference tensor**（rollout 在 `torch.inference_mode()` 下跑，
  on_policy_runner.py:201）——第五层的**就地** `masked_fill_` 对 inference tensor 抛
  `RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed`
  ⇒ 300 次消毒**全失败**（returns 恰好非 inference tensor 故成功）⇒ update 照吃全 NaN
  advantages ⇒ 权重全毁。另 critic 探针已抓真源头 = env 507 critic_obs 单点 NaN（1/292864）
  → critic values 出 NaN。修法 = **重赋值路径**（见 `_sanitize_storage_tensor`）：对
  `storage.advantages/returns/**values**` 逐个检测非有限，命中则
  `setattr(storage, name, torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0))`——`nan_to_num`
  返回**普通 tensor**（非 inference），重赋值绕开 inference-tensor 的就地限制；服务器源码已核
  `mini_batch_generator` 每次调用**现读** `self.advantages` ⇒ 重赋值对 update 可见。旧就地路径
  已删（仅留注释说明为何不可用）。`values` 一并消毒 = GAE 输入/源头同层净化。健康路径仍**零写**。
  obs 不动（rollout 已发生，消毒器同层再动 obs 已晚）；奖励/网络/超参不变。
    2. **critic 前向探针（溯源件）**：`install_std_trajectory_instrument` 仪器安装时对
       `alg.policy.evaluate` 做**只读包装**——每次调用后检查返回 values 的 `isfinite`；发现非有限
       时落 `tag=nan_probe` 行（`stage=critic_forward`，含 values 的 first_coord）+ 该次
       `critic_obs` **逐 key min/max/abs_max**（critic 组含**无噪** height_scan 187 维——重点看；
       大张量按同一阈值跳过，copy 前置判断）。**只记录不阻断、不改返回值、不改 critic_obs。**
  另收上一轮 reviewer 的 should-fix：`_wrapped_update` 中两处 `_nan_probe_param_snapshot` 调用
  包 `try/except`（探针绝不打断训练原则）。

第七层：消毒器上移到观测端（critic 重算路径的污染源，2026-09-22，第七跑实证）：
  实证（第七跑）：第六层三张量（advantages/returns/values）消毒 **0 失败**，但 `ppo.update()`
  内部（`ppo.py:263`）用 `storage.privileged_observations` **现算 critic**——env 507 的单个 NaN
  观测（critic_obs n_nonf=1/292864，u950，两跑位置**逐位复现** = 确定性 Isaac artifact）就在
  这条重算路径上，消毒 values 管不到「重算」，故 `value_loss` 再 NaN → 参数全毁。
  **坑**：obs 张量（privileged 24×1024×~1500≈37M / policy ≈32M 元素）**超过**消毒器大张量
  跳过阈值 `NAN_PROBE_MAX_TENSOR_ELEMS=2^24`，直接加 attr 会被跳过（= 消毒器形同未加）。
  修法两件（**不改奖励/网络/超参，不动 rsl_rl 本体**，延迟 import，健康路径零写零同步）：
    1. **消毒器加观测端**：`_SANITIZE_STORAGE_ATTRS` 扩为
       `(advantages, returns, values, privileged_observations, observations)`；
       `observations` 在 rsl_rl 里是 **dict（policy/critic 两组）**、`privileged_observations`
       是**张量**——`_sanitize_storage_tensor` 按 `getattr` 后类型分派：dict 则逐 key 走同一
       消毒路径（写回时 `setattr` 整 dict），张量直接走。
    2. **大张量检测改 GPU 侧（关键）**：`_sanitize_storage_tensor` 对超阈值张量**不再跳过**
       ——检测改为 **GPU 侧 `isfinite` 归约**（`bad_n = int((~torch.isfinite(t)).sum())`，单标量
       同步，**零全量拷贝**）；`bad_n==0` 提前返回零写；`bad_n>0` 才做 **GPU 侧 `nan_to_num`
       重赋值**（同样零拷贝）+ 统计（first/first_coord 在 GPU 找首个非有限索引后**只拷小片**）。
       即「obs 张量超阈值但消毒**不**跳过 = GPU 侧归约零拷贝」。
       **探针类候选收集（`_collect_*`，只读记录用）的大张量跳过逻辑保持不变**（那是日志性能
       保护，与消毒器职责不同；探针仍跳过超阈值张量以免反复 GPU→CPU 全量 copy 拖慢/OOM）。
    3. 健康路径开销：**一次 GPU 侧归约同步**（`bad_n` 标量 `.item()`）——这是「不跳过超大张量」
       的必要代价（无法在不读数据的前提下判断有没有 NaN）；健康路径仍**零写回**（对象引用不变）。

residual 臂 author token 源 env 维分块前向（2026-09-24，residual 臂 iter0 CUDA OOM 修复）：
  实证：residual 臂 iter0 即崩——`residual_token_obs` -> `adapter.step` 是**全 batch 完整
    前向**，author transformer 的码嵌入按 (N·t·64, d_model) 一次性分配
    （train_author_v0.py:421 F.embedding；1024env×200帧窗×64维×768d×4B ≈ 3.75GiB 量级），
    叠加 Isaac 驻留必爆 3060 12G；碎片缓解/降 env 数均不可取（后者破坏三臂同预算对照）。
  修法（**只动本文件**）：新 CLI `--adapter-chunk-envs`（默认 128，0=不分块）——
    `residual_token_obs` 懒建期把 N 个 env 不重叠地拆成 ceil(N/C) 个**独立
    AuthorPolicyAdapter 实例**（`_ChunkedAuthorAdapter` 包装：逐块 step 后 cat），前向
    峰值 ~1/M（128 块 ≈470MB/次前向）。adapter 本体（eval_author_v0_decoder.py =
    轴 B 生产件）零改动。
  正确性（adapter 状态 env 维独立性的代码证据 + 唯一 batch 维耦合的等价改法）：
    adapter 自回归缓冲 codes/states/ctx/n_valid/_last_idx **全部按 env 维独立**
    （eval_author_v0_decoder.py:579-588；`_push` :608-614 逐 env 滚动；`reset` :650-658
    只动指定 env；step 的模型前向对 batch 逐行独立）⇒ 不重叠拆到多个实例 ≡ 全 batch。
    **唯一 batch 维耦合** = `window_len()`（:603-606）：前向帧数 t = n_valid 对整个
    adapter batch 取 max，且 `step` 是**先 `_push`（n_valid +1）再读** ⇒ 单 adapter 下每
    env 的前向窗都吃「推进后」的全局 max。naive 分块各块只看块内 max，第 2 步起就会少
    1 帧、部分 reset 后更偏离。等价改法：每步取 t_global = **推进后**的全局 max
    （`_step_window_len`：min(推进前全局 max + 1, win)，同一 env 集合的同一值 = 单
    adapter 的 window_len()），经 `_pinned_window_len` 实例级钉桩各块的 window_len
    （**不改 adapter 本体文件**，仅实例属性覆写、步内即还原）⇒ codes_w 切片/前向/
    argmax/逆仿射与全 batch 逐位一致。已知边界：不同 batch 形状理论上可能选到不同 GEMM 分块策略产生末位
    舍入差（argmax 仅近平局时可能翻转）；CPU 与常见 CUDA 路径实测逐位一致，selftest
    断言 ①③ 以真实 AuthorPolicyAdapter + mock transformer 验证（含整块部分 reset 的
    非均匀 n_valid 场景与尾块）。

adapter 前向精度档 + 分块默认拉大（2026-09-24，性能补丁：精度换速度）：
  实测+算账：adapter 每控制步对全 envs×200 帧窗完整前向，计算量结构性巨大
    （1024env ≈ 790 TFLOPs/iter；fp32 理论下限 ~80s/it、chunk128 实测 179s/it，
    5000it ≈ 10 天不可行）。精确 KV cache 不可能（v0 位置编码窗口相对 + 滑动驱逐
    ⇒ 缓存每步失效，架构性结论已裁定）⇒ 唯一可行杠杆 = 精度换速度。
  修法（**只动本文件**，不改奖励/网络/超参/adapter 本体）：
    1. 新 CLI `--adapter-dtype {fp32,bf16,tf32}`（默认 bf16，`ADAPTER_DTYPES`）——
       bf16 = autocast 包 adapter.step 的 model 前向（`_AutocastModelForward` 替身 +
       `_adapter_dtype_ctx` 实例级钉桩，同 `_pinned_window_len` 先例）：autocast 下
       matmul bf16 / embedding fp32 **由 autocast 决定，不手动控制具体算子**；logits
       在 autocast 边界升回 fp32 ⇒ step 内 argmax 与逆仿射保持 fp32 口径（bf16
       logits 与其 fp32 提升严格序同构，argmax 逐位一致——selftest 断言 ④ 直证）。
       tf32 = `torch.backends.cuda.matmul.allow_tf32=True` 上下文（仅 wrapper 范围，
       进/出 finally 恢复，不全局泄漏——selftest 断言 ⑤）。fp32 = 旧行为逐字。
       **口径声明（信封 adapter_perf 字段登记）**：冻结 80M transformer 纯推理
       argmax 生成 token，bf16 数值近似可接受、token 流偶发近平局翻转不影响 RL 适配；
       与轴 A/B（D061a author 训练 / D064 decoder 臂）fp32 口径的偏差显式声明于此。
    2. `--adapter-chunk-envs` 默认 128 -> 1024：bf16 下嵌入/激活内存减半，1024env
       全批量单块峰值预期 ~7-8GB 可行（估计非承诺）；仍 OOM 时 CLI 可降回 512/128，
       分块语义不变（selftest 断言 ② 更新为新默认值）。
    3. 首步仪器：`residual_token_obs` 装载后首个 adapter step 打一行
       `[adapter-perf] dtype/chunk/首次 step 耗时`（CUDA 侧同步后计时；首步含预热，
       供首跑判读量级，非稳态基准）。

rollout 期观测源消毒器（2026-09-24，第八层 = 第七层 update 侧消毒的上游补全，r4 里程碑实证）：
  实证（r4 里程碑）：update 侧消毒器 1634 次/1254it（1.30 次/it），其中 `purged_full=411`
  （33% update 为空学习步）——观测 NaN 在 rollout 期进入 buffer 后，update 侧消毒只能救权重、
  救不了该次 update（归一化早已放大成全量）；且 rollout 期策略直接消费含 NaN 的 obs → 该 env
  动作 NaN → 物理更炸的正反馈。事件链 = obs 先 NaN，即 env 传感器侧（激进动作 → 物理压力 →
  偶发 inf；residual 特有高频暴烈动作风格放大频率，direct 0.055/it 同族）。
  修法（治本 = obs 进入训练前净化；**不改奖励/网络/超参，update 侧五路消毒器保留为防御纵深**）：
    1. 新 CLI `--obs-sanitize`（默认开，BooleanOptionalAction）：`_run` 里 env 构造后、runner
       构造前调 `install_obs_sanitize_wrapper` 给 **env 本体**打「只读+净化」包装——runner 内部
       rollout 不可侵入，故覆写 `step`/`reset`，返回前对 obs dict 逐 key GPU 侧 `isfinite`
       归约：健康路径零写回、原 dict/张量原样（每 key 一次标量同步 = 唯一开销）；命中则
       `nan_to_num(nan=0, posinf=+1e4, neginf=-1e4)`（**±inf 保号映射 ±1e4 有限大值**，供
       学习信号；与 update 侧第五~七层「置 0」口径不同，语义进信封 `obs_sanitize` 字段登记）。
    2. 包装手法 = **动态子类 + `__class__` 重赋值**（同一对象、identity 不变）：
       `type("ObsSanitized_<原类名>", (type(env),), {...})` 只覆写 step/reset 再重分类——
       isinstance/属性/方法**原生**兼容（比 `__getattr__` 透传更强的兼容性：`isinstance(env, 原类)`
       恒真，`assert_residual_action_scale` 等既有持有 env 引用的路径零感知，未知属性无需转发）。
    3. 计数落同一 `std_trajectory.log`（`tag=obs_sanitize`，逐 key 逐命中行：call/step/
       n_nonfinite/n_nan/n_posinf/n_neginf/n_total/cum/first/first_index/first_coord）；
       stdout WARN 逐 key 限频一次（命中可能高频，防刷屏）。
  selftest 断言：`obs_sanitize_dirty`（①脏 obs 净化+计数行+首例坐标）/ `obs_sanitize_
  healthy_bitwise`（②健康逐位不变+零计数）/ `obs_sanitize_passthrough`（③透传性+幂等）。
  性能声明：健康路径每 env.step/reset 每 float key 一次 GPU 侧 isfinite 归约 + 单标量同步
  （obs 通常 policy/critic 两 key，每控制步 ≈2 次同步，相对 rollout 计算与 adapter 前向可忽略）；
  命中路径另有 nonzero 首索引小拷贝（罕见）。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import sys
import threading
import time
import traceback
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    # 以 `python apt_g1/isaac/train_g1_decoder_residual.py` 直跑时 sys.path[0] 是脚本
    # 目录而非仓根（train_g1_decoder.py:531-534 同款处理）。
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Layer 1: 契约常量 + 纯 torch 件（零 isaaclab / onnx / rsl_rl 依赖）
# ---------------------------------------------------------------------------
ARMS = ("direct", "decoder", "residual")
TERRAINS = ("plane", "rough", "stairs", "stones", "discrete")
SMOKE_NUM_ENVS = 8
SMOKE_ITERS = 2

TOKEN_DIM = 64
ACTION_DIM = 29              # 关节目标维（G1 29 DOF，SONIC/IsaacLab 序）
HISTORY_LEN = 10
PROPRIO_DIM = HISTORY_LEN * 3 + 3 * HISTORY_LEN * ACTION_DIM + HISTORY_LEN * 3  # 930
DECODER_OBS_DIM = TOKEN_DIM + PROPRIO_DIM  # 994（sonic_action_term.py:69-74）
RESIDUAL_HIDDEN = (256, 128)
RESIDUAL_ACTIVATION = "silu"

REWARD_MODES = ("auto", "track", "max_forward")
DEFAULT_REWARD_MODE = "auto"   # auto: residual -> max_forward；direct/decoder -> track
DEFAULT_FORWARD_WEIGHT = 1.0
DEFAULT_INTENT_PIN_VALUE = 1.0  # 官方 G1 rough 命令 lin_vel_x 上界（假设 A2）
# residual 臂 author adapter env 维分块前向的默认块大小（0=不分块）。
# 依据（2026-09-24 性能补丁，128 -> 1024）：历史默认 128 是 fp32 OOM 修复的保守值
# （1024env 全 batch 前向的码嵌入一次分配 ~3.75GiB，train_author_v0.py:421
# F.embedding，3060 12G 叠加 Isaac 驻留必爆）；引入 --adapter-dtype bf16 后嵌入/激活
# 内存减半，1024env 全批量单块峰值预期 ~7-8GB 可行（估计非承诺），大块减小批开销；
# 若仍 OOM，CLI 可降回 512/128（分块语义不变）。
DEFAULT_ADAPTER_CHUNK_ENVS = 1024
# residual 臂 author adapter 前向精度档（性能补丁：精度换速度，默认 bf16）。
# 口径声明见模块 docstring「adapter 前向精度档」与身份信封 adapter_perf 字段：冻结
# 80M transformer 纯推理 argmax 生成 token，bf16 数值近似可接受、token 流偶发近平局
# 翻转不影响 RL 适配；与轴 A/B 的 fp32 口径偏差显式声明。
ADAPTER_DTYPES = ("fp32", "bf16", "tf32")
DEFAULT_ADAPTER_DTYPE = "bf16"

# env 地形名 -> author ctx 地形名（ctx_from_command 只认 plane/rough_paper/climbing_box，
# 见 eval_author_v0_decoder.TERRAIN_TYPES:193）。本仓无 climbing_box env 地形，故不映射到
# 该档；rough/stairs/stones/discrete 全部归到语料里唯一的非平坦档 rough_paper。
# 未列出的地形名 -> `ctx_terrain` 显式报错（不静默回落，防新地形悄悄用错语境）。
CTX_TERRAIN_MAP = {
    "plane": "plane",
    "rough": "rough_paper",
    "stairs": "rough_paper",
    "stones": "rough_paper",
    "discrete": "rough_paper",
}
# token 统计 std 下限（与 eval_author_v0_decoder.author_token_stats floor 与
# sonic_action_term.SonicTokenCfg.token_std_floor 同值，保证两份统计逐位一致）。
TOKEN_STD_FLOOR = 1.0e-3

# max_forward 奖励配方（唯一事实源：Layer 2 覆写与 selftest 同读此表）。
# 正则项权重逐字取自官方 G1 rough 配方（tmp/isaac_ref/rough_env_cfg.py:20-100 +
# velocity_env_cfg.py:222-254 的 G1 post_init 覆写，= D064 继承的同一批正则）。
# 注意：本配方**对所有地形统一施加**（含 plane）——官方 flat 语义
# （_apply_flat_semantics）会改若干正则权重，但 max_forward 下三臂/跨地形用同一套
# 正则更利于归因（§5y「官方配方防摔/姿态/动作平滑正则」未按地形分档）。
MAX_FORWARD_RECIPE = {
    # 前进速度项（唯一正向任务项；无人工上限，见 forward_vel_reward）
    "forward_vel": {"weight": DEFAULT_FORWARD_WEIGHT, "unit": "m/s", "kind": "task"},
    # 官方防摔/姿态/动作平滑正则（sign: -1=罚，+1=塑形，0=关闭）
    "termination_penalty": {"weight": -200.0, "sign": -1, "kind": "safety"},
    "flat_orientation_l2": {"weight": -1.0, "sign": -1, "kind": "posture"},
    "lin_vel_z_l2": {"weight": 0.0, "sign": 0, "kind": "posture"},
    "ang_vel_xy_l2": {"weight": -0.05, "sign": -1, "kind": "posture"},
    "dof_torques_l2": {"weight": -1.5e-7, "sign": -1, "kind": "smooth"},
    "dof_acc_l2": {"weight": -1.25e-7, "sign": -1, "kind": "smooth"},
    "action_rate_l2": {"weight": -0.005, "sign": -1, "kind": "smooth"},
    "feet_air_time": {"weight": 0.25, "sign": 1, "kind": "gait"},
    "feet_slide": {"weight": -0.1, "sign": -1, "kind": "smooth"},
    "dof_pos_limits": {"weight": -1.0, "sign": -1, "kind": "safety"},
    "joint_deviation_hip": {"weight": -0.1, "sign": -1, "kind": "posture"},
    "joint_deviation_arms": {"weight": -0.1, "sign": -1, "kind": "posture"},
    "joint_deviation_torso": {"weight": -0.1, "sign": -1, "kind": "posture"},
}
# 移除的命令跟踪项（§5y：命令跟踪项移除）
MAX_FORWARD_REMOVED_TERMS = ("track_lin_vel_xy_exp", "track_ang_vel_z_exp")

# 三臂配置表（唯一事实源：CLI 校验 / 训练编排 / 身份信封 / selftest 同读）。
# env_action="lora_policy" 的 env 侧动作通路 = direct 臂逐字（JointPositionAction，
# g1_velocity_decoder_env.py:378-391），仅 policy obs 组末位多 930 维本体历史。
ARM_TABLE = {
    "direct": {
        "env_action": "direct",
        "action_dim": ACTION_DIM,
        "reward_mode": "track",
        "policy": "rsl_rl ActorCritic（D064 逐字）",
        "frozen_branch": False,
        "token_source": None,
    },
    "decoder": {
        "env_action": "decoder",
        "action_dim": TOKEN_DIM,
        "reward_mode": "track",
        "policy": "rsl_rl ActorCritic（D064 逐字）",
        "frozen_branch": False,
        "token_source": None,
    },
    "residual": {
        "env_action": "lora_policy",   # = direct 动作通路逐字 + 930 维本体观测
        "action_dim": ACTION_DIM,
        "reward_mode": "max_forward",
        "policy": "ResidualDecoderPolicy（decoder 分支冻结 + r 可训）",
        "frozen_branch": True,
        "token_source": "author_v0_ckpt",
    },
}


def resolve_reward_mode(arm: str, requested: str) -> str:
    """--reward-mode 解析：auto 时 residual->max_forward、direct/decoder->track。"""
    if requested not in REWARD_MODES:
        raise ValueError(f"reward_mode 须 ∈ {REWARD_MODES}，收到 {requested!r}")
    if requested != "auto":
        return requested
    return ARM_TABLE[arm]["reward_mode"]


# ------------------------------------------------- token 前向仿射 / terrain 契约
def forward_token_affine(
    raw: torch.Tensor,
    token_mean: torch.Tensor,
    token_std: torch.Tensor,
    token_alpha: float = 1.0,
    token_bound: str = "tanh",
) -> torch.Tensor:
    """归一化动作 a -> token 坐标（与 author adapter 的逆仿射互逆）。

    逐式同 `sonic_action_term.SonicActionCore.map_token`（:205-215）与
    `sonic_lora_policy.SonicLoRAMuPath.token_from_raw`（:157-161）：
        token_bound == "tanh": token = mean + alpha*std*tanh(a)
        否则                 : token = mean + alpha*std*a

    **为什么必须补这一步**：`AuthorPolicyAdapter.step` 返回的是归一化动作
    a = (tok-mean)/(alpha*std)（eval_author_v0_decoder.py:638-642），不是 token 坐标。
    把 a 直喂 decoder 分支的 FSQ 量化会破坏冻结 decoder 先验（reviewer 实测
    max|a-tok|=3.51）。
    """
    m = torch.as_tensor(token_mean, dtype=torch.float32, device=raw.device).reshape(-1)
    s = torch.as_tensor(token_std, dtype=torch.float32, device=raw.device).reshape(-1)
    if int(m.numel()) != TOKEN_DIM or int(s.numel()) != TOKEN_DIM:
        raise ValueError(f"token_mean/std 须为 ({TOKEN_DIM},)，收到 {int(m.numel())}/{int(s.numel())}")
    bound = str(token_bound)
    if bound not in ("none", "tanh"):
        raise ValueError(f"token_bound 须 ∈ ('none','tanh')，收到 {bound!r}")
    a = raw if bound != "tanh" else torch.tanh(raw)
    return m + float(token_alpha) * s * a


# ------------------------------------------------- 执行/反馈仿射（闭环增益契约）
# 残差臂闭环里同一条关节目标过两处仿射，二者必须严格互逆（否则每步乘 ≠1 的增益）：
#   执行端 JointPositionAction：q_des = default + raw * scale（joint_actions.py:130-139），
#                              本臂 raw = μ；use_default_offset=True ⇒ offset = default。
#   反馈端 SonicActionCore.last_act_from_q_des：last_act = (q_des-default)/sonic_scale
#                              （sonic_action_term.py:244；sonic_proprio_hist 取
#                              action_manager.get_term("joint_pos").processed_actions）。
# 取执行端 scale == sonic_scale（逐关节 29 维）⇒ last_act ≡ μ（逐位），闭环增益恒 1。
# 旧实现（direct 臂逐字）执行端 scale = 标量 0.5 ⇒ 增益 0.5/sonic_scale（0.91–6.71×）。
RESIDUAL_LEGACY_ACTION_SCALE = 0.5   # direct 臂逐字标量；仅作反例/信封记录，不用于本臂
RESIDUAL_ACTION_SCALE_FIELD = "residual_action_scale"
# sonic_scale 的单一事实源（与 sonic_proprio_hist 同源）：sonic_action_term._sonic_scale_isaac
SONIC_SCALE_SOURCE = "sonic_action_term._sonic_scale_isaac()（= sonic_proprio_hist 反归一化同一份）"


def sonic_joint_names() -> tuple[str, ...]:
    """SONIC 29 序关节名（= G1_ISAACLab_ORDER，与 sonic_scale/sonic_proprio_hist 同序）。

    从 `gear_sonic.envs.env_utils.joint_utils.G1_ISAACLab_ORDER` 取——**与
    `sonic_action_term.py:50` 同一 import 源**（该模块 SONIC_JOINT_NAMES = 此序），
    故本文件不再复制第二份关节名/scale 常量（防漂移）。本机（无 isaaclab）亦可 import。
    """
    try:  # 服务器执行根平铺 import / 仓库根包 import 双兼容
        from gear_sonic.envs.env_utils.joint_utils import G1_ISAACLab_ORDER
    except ImportError as exc:  # pragma: no cover（gear_sonic 是 SONIC 契约来源，缺则无法定序）
        raise RuntimeError(
            "gear_sonic.envs.env_utils.joint_utils 不可导入——无法确定 SONIC 29 序关节名"
            "（与 sonic_scale 同源契约）"
        ) from exc
    names = tuple(str(n) for n in G1_ISAACLab_ORDER)
    if len(names) != ACTION_DIM:
        raise RuntimeError(f"SONIC 关节名应为 {ACTION_DIM} 个，收到 {len(names)}")
    if len(set(names)) != ACTION_DIM:
        raise RuntimeError(f"SONIC 关节名存在重复：{names}")
    return names


def build_sonic_scale_map(
    sonic_scale, joint_names: Sequence[str] | None = None
) -> dict[str, float]:
    """逐关节 sonic_scale 向量 -> `JointPositionActionCfg.scale` 接受的 {关节名: 标量}。

    **为什么是 dict 而不是张量**：isaaclab 2.1.0（服务器 installed）的 `JointAction`
    scale 只支持 `float | dict[str,float]`（joint_actions.py:80-88 实测；**不接受
    裸 tensor/list**，传入即 `ValueError: Unsupported scale type`）。dict 经
    `string_utils.resolve_matching_names_values`（`re.fullmatch`）解析到 term
    `find_joints` 得到的关节名序（joint_actions.py:61-63, 83-86），命中后按 term 序
    写进 `_scale[:, index_list]`。用**精确关节名**（无正则元字符）作 key ⇒ 每关节唯一
    命中 ⇒ `_scale` 逐关节等于 sonic_scale 向量。含正则元字符的关节名直接报错
    （否则可能一 key 命中多关节 -> isaaclab 抛 "Multiple matches"，或误匹配）。
    """
    s = np.asarray(sonic_scale, dtype=np.float64).reshape(-1)
    if s.shape != (ACTION_DIM,):
        raise ValueError(f"sonic_scale 须为 ({ACTION_DIM},)，收到 {tuple(s.shape)}")
    names = tuple(sonic_joint_names() if joint_names is None else (str(n) for n in joint_names))
    if len(names) != ACTION_DIM:
        raise ValueError(f"关节名数应为 {ACTION_DIM}，收到 {len(names)}")
    if len(set(names)) != ACTION_DIM:
        raise ValueError("关节名存在重复（scale dict 会互相覆盖）")
    meta = re.compile(r"[.^$*+?{}\[\]\\|()]")
    for n in names:
        if meta.search(n):
            raise ValueError(f"关节名 {n!r} 含正则元字符（会被 isaaclab re.fullmatch 误解析）")
    return {n: float(v) for n, v in zip(names, s)}


def exec_feedback_gain(action_scale, sonic_scale) -> np.ndarray:
    """闭环增益 g = action_scale / sonic_scale（逐关节，纯函数，可本机单测）。

    推导：执行端 `q_des = default + μ*action_scale`（use_default_offset=True ⇒ offset=default），
    反馈端 `last_act = (q_des-default)/sonic_scale = μ * action_scale/sonic_scale`。
    故 `last_act = μ * g`，g≡1 ⇔ 执行/反馈仿射严格互逆（闭环无附加增益）。
    `action_scale` 可为标量（广播，如旧 0.5）或 29 维向量（本臂的 sonic_scale）。
    """
    a = np.asarray(action_scale, dtype=np.float64).reshape(-1)
    s = np.asarray(sonic_scale, dtype=np.float64).reshape(-1)
    if s.shape != (ACTION_DIM,):
        raise ValueError(f"sonic_scale 须为 ({ACTION_DIM},)，收到 {tuple(s.shape)}")
    if a.size == 1:
        a = np.full(ACTION_DIM, float(a[0]))
    if a.shape != (ACTION_DIM,):
        raise ValueError(f"action_scale 须为标量或 ({ACTION_DIM},)，收到 {tuple(a.shape)}")
    if np.any(s == 0.0):
        raise ValueError("sonic_scale 含 0（反馈端除零）")
    return a / s


def residual_action_scale_envelope(
    sonic_scale, *, joint_names: Sequence[str] | None = None, source: str = ""
) -> dict:
    """`residual_action_scale` 信封字段（值/来源/逐关节 min-max；JSON 可序列化）。

    纯本机可构建（不 import isaaclab）。
    """
    s = np.asarray(sonic_scale, dtype=np.float64).reshape(-1)
    if s.shape != (ACTION_DIM,):
        raise ValueError(f"sonic_scale 须为 ({ACTION_DIM},)，收到 {tuple(s.shape)}")
    names = tuple(sonic_joint_names() if joint_names is None else (str(n) for n in joint_names))
    if len(names) != ACTION_DIM:
        raise ValueError(f"关节名数应为 {ACTION_DIM}，收到 {len(names)}")
    return {
        "field": RESIDUAL_ACTION_SCALE_FIELD,
        "value_kind": "per_joint_vector",
        "dim": ACTION_DIM,
        "source": str(source) or SONIC_SCALE_SOURCE,
        "joint_order": "SONIC/G1_ISAACLab_ORDER",
        "per_joint": {n: float(v) for n, v in zip(names, s)},
        "min": float(s.min()),
        "max": float(s.max()),
        "argmin_joint": names[int(s.argmin())],
        "argmax_joint": names[int(s.argmax())],
        "legacy_scalar": RESIDUAL_LEGACY_ACTION_SCALE,
        "gain_identity": True,
        "note": (
            "执行端 JointPositionAction scale=逐关节 sonic_scale ⇒ "
            "(q_des-default)/sonic_scale ≡ μ，闭环增益恒 1；旧标量 0.5 ⇒ 增益 0.5/sonic_scale"
        ),
    }


def ctx_terrain(terrain: str) -> str:
    """env 地形名 -> `ctx_from_command` 接受的地形名（显式映射，幂等）。

    映射表 = CTX_TERRAIN_MAP。已是 ctx 契约内地形名（eval_author_v0_decoder.
    TERRAIN_TYPES）时原样返回（幂等，便于在 `_token_obs_params` 与 `residual_token_obs`
    两处无脑接入）。未列出且不在 ctx 契约内的地形名 -> 显式 ValueError（不静默回落）。
    """
    name = str(terrain)
    if name in CTX_TERRAIN_MAP:
        return CTX_TERRAIN_MAP[name]
    if name in ("plane", "rough_paper", "climbing_box"):
        return name
    raise ValueError(
        f"地形 {name!r} 无 ctx 映射（CLI 允许 {TERRAINS}；ctx 契约允许 "
        f"plane/rough_paper/climbing_box，见 CTX_TERRAIN_MAP）"
    )


def official_obs_slice(
    policy_terms,
    term_dims,
    excluded: tuple[str, ...] = ("sonic_proprio", "author_token"),
) -> dict:
    """r 输入（official 段）的切片守卫（SF-2，纯函数，可本机单测）。

    契约：r 的输入 = policy obs 的**前缀** `flat[:, :official_obs_dim]`（假设 A5），
    其中 `official_obs_dim` = 运行期 policy 组总宽 − 被排除 term（默认
    `sonic_proprio`(930) + `author_token`(64)）的宽度之和。

    校验（任一条不满足即 RuntimeError，fail loud）：
      1. `policy_terms` 与 `term_dims` 一一对应；
      2. 被排除的 term 全部存在（缺位 = 接线失败，不能靠"宽度恰好凑上"蒙过去）；
      3. 被排除 term 的区间**全部落在 official 段之后**（即 official 段是前缀，
         起始 = 0）——否则 `flat[:, :official_obs_dim]` 会切进 proprio/token，
         与 `decoder_obs` 的 token/proprio 切片重叠，静默错喂。
    返回 {"start": 0, "width": official_width, "excluded": {term: width}, "policy_terms": [...]}.
    """
    names = [str(n) for n in policy_terms]
    if len(names) != len(term_dims):
        raise RuntimeError(f"policy 组 names/dims 错位：{len(names)} vs {len(term_dims)}")
    dims = [int(math.prod(int(x) for x in d)) for d in term_dims]
    total = int(sum(dims))
    spans: dict[str, tuple[int, int]] = {}
    pos = 0
    for n, w in zip(names, dims):
        spans[n] = (pos, pos + w)
        pos += w
    excl: dict[str, int] = {}
    for term in excluded:
        if term not in spans:
            raise RuntimeError(f"policy 组缺被排除 term {term!r}（现有 {names}）——r 输入切片契约不成立")
        excl[term] = int(spans[term][1] - spans[term][0])
    official_width = int(total - sum(excl.values()))
    if official_width <= 0:
        raise RuntimeError(f"official 段宽度推得 {official_width}（policy 总宽 {total}，排除 {excl}）")
    for term in excluded:
        start, _end = spans[term]
        if start < official_width:
            raise RuntimeError(
                f"{term!r} 区间起点 {start} < official 段宽度 {official_width}——"
                f"official 段不是 policy obs 前缀，r 输入切片会错喂（现有序 {names}）"
            )
    return {"start": 0, "width": official_width, "excluded": excl, "policy_terms": names}


def assemble_decoder_obs(
    flat: torch.Tensor,
    token_slice: tuple[int, int],
    proprio_slice: tuple[int, int],
) -> torch.Tensor:
    """policy obs -> (N,994) = cat[token(64), proprio(930)]（纯函数，供本机跨口径对照）。

    与 `ResidualDecoderPolicy.decoder_obs` 同式；抽成模块级纯函数是为了让本机 selftest
    能在无 rsl_rl 的环境下做「residual 臂 decoder 输入 vs decoder 臂 decoder 输入」的
    逐位对照（reviewer 要求的新增断言 ②）。
    """
    ts, tw = int(token_slice[0]), int(token_slice[1])
    ps, pw = int(proprio_slice[0]), int(proprio_slice[1])
    if tw != TOKEN_DIM or pw != PROPRIO_DIM:
        raise ValueError(f"token/proprio 宽度须为 {TOKEN_DIM}/{PROPRIO_DIM}，收到 {tw}/{pw}")
    if ts + tw > int(flat.shape[-1]) or ps + pw > int(flat.shape[-1]):
        raise ValueError(f"切片越界：obs 宽 {int(flat.shape[-1])}，token=({ts},{tw}) proprio=({ps},{pw})")
    return torch.cat([flat[:, ts : ts + tw], flat[:, ps : ps + pw]], dim=-1)


# ------------------------------------------------------------- 残差 MLP
def _make_activation(name: str) -> nn.Module:
    """激活名 -> 模块（本文件只用 silu；其余为显式可选项）。"""
    key = str(name).lower()
    if key in ("silu", "swish"):
        return nn.SiLU()
    if key == "relu":
        return nn.ReLU()
    if key == "elu":
        return nn.ELU()
    raise ValueError(f"未知激活 {name!r}（支持 silu/relu/elu）")


class ResidualActionMLP(nn.Module):
    """r(s_t)：轻量 MLP [256,128]+SiLU -> 29 维关节目标残差，**末层零初始化**。

    末层（weight 与 bias）全零 ⇒ r(s) ≡ 0（任意 s），故 a_t = decoder(...) + r(s_t)
    在构造期与 decoder 臂输出**逐位一致**（零初始化恒等，selftest 断言）。PPO 训练中
    末层从零开始学习，r 的初始贡献为 0，不破坏冻结 token 先验（§5y 设计）。
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = ACTION_DIM,
        hidden: tuple[int, ...] = RESIDUAL_HIDDEN,
        activation: str = RESIDUAL_ACTIVATION,
    ) -> None:
        super().__init__()
        if int(obs_dim) <= 0:
            raise ValueError(f"obs_dim 须 > 0，收到 {obs_dim}")
        if int(action_dim) != ACTION_DIM:
            raise ValueError(f"残差输出维冻结为 {ACTION_DIM}，收到 {action_dim}")
        layers: list[nn.Module] = []
        last = int(obs_dim)
        for h in hidden:
            layers += [nn.Linear(last, int(h)), _make_activation(activation)]
            last = int(h)
        head = nn.Linear(last, int(action_dim))
        layers.append(head)
        self.net = nn.Sequential(*layers)
        self.hidden = tuple(int(h) for h in hidden)
        self.activation = str(activation)
        # 末层零初始化（恒等起点）；其余层默认初始化由 nn.Linear 提供
        with torch.no_grad():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def is_zero(self, obs: torch.Tensor, atol: float = 0.0) -> bool:
        """r(obs) 是否恒为 0（零初始化自检用；atol=0 = 逐位）。"""
        with torch.no_grad():
            out = self.forward(obs)
        return bool((out.abs() <= atol).all())


def freeze_module(module: nn.Module) -> dict:
    """把模块整体冻结（requires_grad=False + eval()），返回参数计数报告。

    这是「decoder/token 源全冻结，梯度只达 r」的机制：冻结参数仍在模块里（可 save/
    load/诊断），但不进优化器（train 侧按 requires_grad 摘除，同 D065 口径）。
    """
    n_tensors = n_elements = 0
    for p in module.parameters():
        n_tensors += 1
        n_elements += int(p.numel())
        p.requires_grad_(False)
    module.eval()
    return {"n_tensors": n_tensors, "n_frozen": n_tensors, "n_elements": n_elements}


class ResidualMuPath(nn.Module):
    """μ = decoder(decoder_obs) + r(r_obs)（decoder 冻结分支 + 可训残差）。

    - `frozen_branch(decoder_obs)`：`torch.no_grad()` 下跑 decoder（冻结先验不建图）。
    - `forward(decoder_obs, r_obs)`：冻结分支输出 + 残差 MLP 输出（29 维）。

    梯度隔离（selftest 断言）：冻结后只有 `residual` 参数 requires_grad=True，
    μ 反传只在 residual 上产生 grad，decoder 参数 grad 恒 None。
    """

    def __init__(
        self,
        decoder: nn.Module,
        residual: ResidualActionMLP,
        *,
        freeze_decoder: bool = True,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.residual = residual
        self.decoder_frozen = False
        if freeze_decoder:
            freeze_module(self.decoder)
            self.decoder_frozen = True

    def frozen_branch(self, decoder_obs: torch.Tensor) -> torch.Tensor:
        """冻结 decoder(decoder_obs) -> (N, 29)（no_grad，常量分支）。"""
        with torch.no_grad():
            return self.decoder(decoder_obs)

    def forward(self, decoder_obs: torch.Tensor, r_obs: torch.Tensor) -> torch.Tensor:
        return self.frozen_branch(decoder_obs) + self.residual(r_obs)

    def trainable_report(self) -> dict:
        def _count(mod: nn.Module) -> dict:
            ps = list(mod.parameters())
            return {
                "n_tensors": len(ps),
                "n_trainable": sum(1 for p in ps if p.requires_grad),
                "n_elements": int(sum(p.numel() for p in ps)),
                "n_trainable_elements": int(sum(p.numel() for p in ps if p.requires_grad)),
            }

        return {
            "decoder_frozen": bool(self.decoder_frozen),
            "decoder": _count(self.decoder),
            "residual": _count(self.residual),
        }

    def only_residual_trainable(self) -> bool:
        """除 residual 外无任何 requires_grad 参数（梯度隔离判据）。"""
        if any(p.requires_grad for p in self.decoder.parameters()):
            return False
        return any(p.requires_grad for p in self.residual.parameters())


# ------------------------------------------------------------- 奖励纯数学
def forward_vel_reward(
    vx_b: torch.Tensor, weight: float = DEFAULT_FORWARD_WEIGHT, positive_only: bool = True
) -> torch.Tensor:
    """机体坐标前进速度奖励（root_lin_vel_b[:, 0]）。

    - `positive_only=True`（默认）：`clamp(vx_b, min=0)` ⇒ 只奖励前进、不倒奖后退；
    - **无人工速度上限**：与 D048j `progress_bonus` 的 `clamp(vx, 0, 1)`（封顶 1 m/s）
      不同，本项不设上界（owner 2026-09-22 目标改写：无人工限速）。
    - `weight` 由调用侧传入（isaaclab RewardManager 亦会再乘 RewTerm.weight；此处
      纯数学版自含 weight 便于单测与直接求和）。
    """
    v = torch.as_tensor(vx_b, dtype=torch.float32)
    if positive_only:
        v = torch.clamp(v, min=0.0)
    return float(weight) * v


def max_forward_total_reward(
    vx_b: torch.Tensor,
    penalties: list[tuple[float, torch.Tensor]] | None = None,
    *,
    weight_forward: float = DEFAULT_FORWARD_WEIGHT,
    positive_only: bool = True,
) -> torch.Tensor:
    """前进项 + 正则项求和（`penalties` = [(weight, value), ...]，符号已含在 weight 内）。

    纯数学组合式，供 selftest 校验量纲/符号；Layer 2 的实际奖励由 isaaclab
    RewardManager 按 MAX_FORWARD_RECIPE 逐项加权求和（本函数与其同口径）。
    """
    out = forward_vel_reward(vx_b, weight_forward, positive_only)
    for w, v in (penalties or []):
        out = out + float(w) * torch.as_tensor(v, dtype=torch.float32)
    return out


# ------------------------------------------------------------- 身份信封
def _file_md5(path: str | os.PathLike) -> str | None:
    """文件 md5（复用 eval_g1_decoder._file_md5，避免双份实现漂移）。"""
    try:
        from isaac import eval_g1_decoder as egd  # noqa: PLC0415
    except ImportError:  # pragma: no cover（本机走此支）
        from apt_g1.isaac import eval_g1_decoder as egd
    return egd._file_md5(str(path))


def _git_head() -> str:
    try:
        from isaac import eval_g1_decoder as egd  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        from apt_g1.isaac import eval_g1_decoder as egd
    return egd._git_head()


def _env_module_md5() -> dict:
    """冻结件/兄弟模块的实存文件 md5（身份信封；改动可追溯）。"""
    base = REPO_ROOT / "apt_g1" / "isaac"
    return {
        "train_g1_decoder_residual.py": _file_md5(Path(__file__).resolve()),
        "g1_velocity_decoder_env.py": _file_md5(base / "g1_velocity_decoder_env.py"),
        "sonic_action_term.py": _file_md5(base / "sonic_action_term.py"),
        "eval_author_v0_decoder.py": _file_md5(base / "eval_author_v0_decoder.py"),
        "train_g1_decoder.py": _file_md5(base / "train_g1_decoder.py"),
    }


def build_identity_envelope(
    arm: str,
    reward_mode: str,
    *,
    author_ckpt: str = "",
    onnx_path: str = "",
    token_stats: str = "",
    intent_pin: dict | None = None,
    terrain: str = "",
    ctx_terrain_name: str = "",
    extra: dict | None = None,
) -> dict:
    """身份信封：script/git + 冻结件 md5 + 三臂表项 + 奖励配方 + intent-pin + 契约声明。

    纯本机可构建（只做文件 md5 / git 调用，不 import isaaclab）。

    契约字段（reviewer blocker/should-fix 修复）：
      - `ctx_terrain`/`ctx_terrain_map`：env 地形名 -> author ctx 地形名（M1）；
      - `token_affine`：前向仿射式与 bound/alpha（M2）；
      - `privileged_critic`：critic 多看 token 的不对称声明（M3/SF-1）；
      - `reward_recipe_scope`：max_forward 对 plane 统一施加与官方 flat 不同源（M4/SF-4）。
    """
    if arm not in ARM_TABLE:
        raise ValueError(f"未知 arm {arm!r}（允许 {ARMS}）")
    env = {
        "format": "d067.residual.v2",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "entry": "apt_g1/isaac/train_g1_decoder_residual.py",
        "script": str(Path(__file__).resolve()),
        "script_md5": _file_md5(Path(__file__).resolve()),
        "git_head": _git_head(),
        "arm": arm,
        "arm_table": dict(ARM_TABLE[arm]),
        "reward_mode": reward_mode,
        "reward_recipe": {k: dict(v) for k, v in MAX_FORWARD_RECIPE.items()}
        if reward_mode == "max_forward"
        else {"mode": "track", "note": "D064 命令跟踪口径逐字（direct/decoder 臂）"},
        "reward_removed_terms": list(MAX_FORWARD_REMOVED_TERMS) if reward_mode == "max_forward" else [],
        # SF-4：max_forward 正则对所有地形（含 plane）统一施加官方 G1 rough 配方，
        # 与官方 flat 语义 _apply_flat_semantics 的正则**不同源** ⇒ plane 档对照非严格单变量。
        "reward_recipe_scope": {
            "applied_to": "all_terrains",
            "regularizer_source": "official_g1_rough",
            "note": (
                "plane 档对照非严格单变量：官方 flat 语义(_apply_flat_semantics)会改若干正则权重，"
                "max_forward 对 plane 仍施加 rough 配方（§5y 未按地形分档）"
            ),
        },
        "residual": {
            "hidden": list(RESIDUAL_HIDDEN),
            "activation": RESIDUAL_ACTIVATION,
            "last_layer_zero_init": True,
            "obs_dim_note": "r 的输入 = 官方 policy obs 切片（total - 930 proprio - 64 token）",
        },
        "decoder_input": {"token_dim": TOKEN_DIM, "proprio_dim": PROPRIO_DIM, "obs_dim": DECODER_OBS_DIM},
        "token_source": {
            "author_ckpt": str(author_ckpt) if author_ckpt else "",
            "ckpt_md5": _file_md5(author_ckpt) if author_ckpt and os.path.isfile(str(author_ckpt)) else None,
        },
        # M1：地形契约映射（env 地形名 -> ctx_from_command 接受名）
        "ctx_terrain": str(ctx_terrain_name or (ctx_terrain(terrain) if terrain else "")),
        "env_terrain": str(terrain),
        "ctx_terrain_map": dict(CTX_TERRAIN_MAP),
        # M2：前向仿射声明（token = mean + alpha*std*(tanh(a)|a)）
        "token_affine": {
            "formula": "token = mean + alpha*std*(tanh(a) if token_bound=='tanh' else a)",
            "token_bound": "cli --token-bound（缺省 tanh）",
            "token_alpha": "cli --token-alpha（缺省 1.0）",
            "token_stats": str(token_stats),
            "stats_source": "与 decoder 分支同一份 --token-stats npz（load_token_stats, floor=1e-3）",
            "authoritative": ["sonic_action_term.py:205-215", "sonic_lora_policy.py:157-161"],
        },
        # M3/SF-1：特权 critic 不对称声明
        "privileged_critic": {
            "author_token_in_critic": True,
            "note": "author_token 同时进 policy 与 critic 两组：critic 可比 actor 多看该 token（有意设计，不改行为）",
        },
        "decoder": {
            "onnx_path": str(onnx_path) if onnx_path else "factory-default(make_env_cfg, SONIC_DECODER_ONNX)",
            "onnx_md5": _file_md5(onnx_path) if onnx_path and os.path.isfile(str(onnx_path)) else None,
        },
        "token_stats": {
            "path": str(token_stats) if token_stats else "",
            "md5": _file_md5(token_stats) if token_stats and os.path.isfile(str(token_stats)) else None,
        },
        "intent_pin": dict(intent_pin) if intent_pin else None,
        "env_module_md5": _env_module_md5(),
    }
    if extra:
        env.update(extra)
    return env


# ------------------------------------------------- 关闭看门狗 + 打断事件落盘仪器
# faulthandler 栈实证（tmp/_freeze_labts_iter950_PYSTACK.txt）：主循环被未知事件打断后，
# finally 的裸 simulation_app.close() 卡在 isaaclab simulation_context.py:743 的
# `while not timeline.is_playing(): self.render()` 忙等，异常被吞、进程永不退出。
# 下方两件只做「关闭路径」与「异常落盘」，不触碰训练/奖励/策略任何路径。
SAFE_CLOSE_TIMEOUT_S = 120.0   # close() 子线程 join 超时（秒）
MAIN_LOOP_CRASH_LOG = "main_loop_crash.log"


def _dump_thread_stacks() -> bool:
    """尽力 dump 全线程 Python 栈（faulthandler 可用时）；不可用/失败返回 False，不抛。"""
    try:
        import faulthandler  # noqa: PLC0415
    except Exception:  # pragma: no cover（标准库恒有；防御性）
        return False
    try:
        faulthandler.dump_traceback()   # 默认写 stderr：强退前留尸
        return True
    except Exception:  # pragma: no cover
        return False


def _try_resume_timeline() -> bool:
    """尽力把 omni timeline 设回 playing（关闭忙等的触发条件）。

    关闭忙等的根因是 `while not is_playing(): render()`——先把 timeline 拉回 playing，
    多数情况下 close() 即可正常返回。**延迟 import**（本机无 omni）；任何失败都忽略
    （返回 False），绝不能因为「拉 timeline」失败而让关闭路径更糟。
    """
    try:
        import omni.timeline  # noqa: PLC0415
    except Exception:
        return False
    try:
        tl = omni.timeline.get_timeline_interface()
        if not tl.is_playing():
            tl.play()
        return bool(tl.is_playing())
    except Exception as exc:  # noqa: BLE001
        print(f"[d067] _safe_close: timeline.play() 失败（忽略，继续关闭）：{type(exc).__name__}: {exc}", flush=True)
        return False


def _safe_close(simulation_app, timeout_s: float = SAFE_CLOSE_TIMEOUT_S, exit_code: int = 0) -> dict:
    """带看门狗的 `simulation_app.close()`（替代裸 close，见模块 docstring）。

    流程：
      1. 先 `_try_resume_timeline()`（尽力，失败忽略）；
      2. `close()` 放**子线程**执行（daemon），主线程 `join(timeout_s)`；
      3. 超时 = Isaac Sim 关闭忙等：打印警告 + dump 线程栈后 `os._exit(exit_code)`
         强退（ckpt 在循环内已周期落盘，强退可接受）；
      4. 正常返回：`close()` 成功或抛异常都在此返回报告，由调用方继续正常退出。
         退出码语义不变——正常完跑 rc=0；异常时调用方 `raise` 沿原异常冒泡。

    返回 {"closed", "timed_out", "elapsed_s", "error"}（`os._exit` 真退时不可达，
    仅供超时路径被 monkeypatch（单测）时观测）。
    """
    _try_resume_timeline()
    done = threading.Event()
    box: dict = {}

    def _worker() -> None:
        try:
            simulation_app.close()
        except BaseException as exc:  # noqa: BLE001（close 内部异常也要让主线程解脱）
            box["error"] = exc
        finally:
            done.set()

    worker = threading.Thread(target=_worker, name="d067-safe-close", daemon=True)
    t0 = time.monotonic()
    worker.start()
    finished = done.wait(timeout=float(timeout_s))
    elapsed = time.monotonic() - t0

    if finished:
        err = box.get("error")
        if err is None:
            print(f"[d067] _safe_close: simulation_app.close() 正常返回（{elapsed:.1f}s）", flush=True)
        else:
            print(f"[d067] _safe_close: close() 抛异常但已返回（{type(err).__name__}: {err}）", flush=True)
        return {
            "closed": True,
            "timed_out": False,
            "elapsed_s": elapsed,
            "error": None if err is None else repr(err),
        }

    print(
        f"[WARN] _safe_close: simulation_app.close() 超时 {timeout_s}s 未返回"
        f"（Isaac Sim 关闭忙等，栈见 tmp/_freeze_labts_iter950_PYSTACK.txt）——"
        f"强退 os._exit({exit_code})；ckpt 已在循环内周期落盘",
        flush=True,
    )
    _dump_thread_stacks()
    os._exit(int(exit_code))
    # os._exit 真退后不可达；以下仅在单测 monkeypatch 记录器时用于观测超时路径。
    return {"closed": False, "timed_out": True, "elapsed_s": elapsed, "error": None}


def _crash_log_path(out_dir) -> Path:
    """主循环异常落盘路径：`outputs/<out>/main_loop_crash.log`。"""
    return Path(out_dir) / MAIN_LOOP_CRASH_LOG


def _log_main_loop_crash(out_dir, exc: BaseException, *, iteration=None) -> Path:
    """把主循环打断异常（完整 traceback + 时间戳/iter/类型）写盘，再由调用方 raise。

    为什么需要：修复前主循环被打断的异常会被 finally 的 close 死锁吞掉，现场只剩「卡在
    关闭渲染」的栈，看不到「谁在 23M 步打断主循环」。落盘后下次复现直接见尸。
    """
    path = _crash_log_path(out_dir)
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    header = (
        "# D067 main loop crash (BaseException)\n"
        f"timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n"
        f"exception: {type(exc).__module__}.{type(exc).__name__}\n"
        f"message: {exc}\n"
        f"iteration: {iteration if iteration is not None else 'n/a'}\n"
        "note: 主循环被该异常打断（修复前被 finally 的 close 死锁吞掉）；完整 traceback 见下\n"
        "\n"
    )
    path.write_text(header + tb, encoding="utf-8")
    print(f"[d067] 主循环异常已落盘 -> {path}", flush=True)
    return path


# ------------------------------------------------- σ 病态区夹紧（三臂统一）
# 实证：rsl_rl ActorCritic 的可训 σ（self.std，裸 nn.Parameter 无下界）被 PPO 推负后，
# Normal.sample 抛 `normal expects all elements of std >= 0.0`（tmp/_fixverify_main_loop_crash.log）。
# 夹紧只下夹不上夹，且只在构造 Normal 前生效 ⇒ 健康区（σ≥STD_CLAMP_MIN）逐位恒等。
STD_CLAMP_MIN = 1.0e-4        # σ 下界（= token std floor 同量级；病态区才生效）
STD_TRAJECTORY_LOG = "std_trajectory.log"   # σ 漂移仪器落点（outputs/<out>/ 下）


def _clamp_std_param(std: torch.Tensor) -> torch.Tensor:
    """σ 病态区净化：NaN/±inf -> STD_CLAMP_MIN，其余 min(std, STD_CLAMP_MIN)（只下夹不上夹）。

    为什么只下夹：上界夹紧会改写健康轨迹的探索强度（σ>1 是正常早训状态），破坏 D064 可比性；
    下夹在健康区（σ≥STD_CLAMP_MIN）是**恒等映射**，逐位不改 loc/scale 与反传梯度。

    为什么还要管 NaN/±inf（第三层修复）：`NaN.clamp(min=1e-4) = NaN` —— 旧 `clamp(min=...)`
    对 NaN 是恒等映射，故 σ 参数被 update 打成 NaN 后，第二层保命带形同虚设，`torch.normal`
    照样抛 `std >= 0.0`（实证：`tmp/_d067_verify/freeze_sigmafix_iter950_crash.txt`，σ 轨迹
    全程健康却在 iter 951 的 update 内部变 NaN）。`torch.where(isfinite, clamp, STD_CLAMP_MIN)`
    把非有限量一律映射到 STD_CLAMP_MIN（保命带），**有限值路径与旧 clamp 逐位一致**。

    **NaN 溯源探针在记录真相，净化只保进程不崩**：本函数只作用于「构造 Normal 用的 σ 张量」，
    不改写 `self.std` 参数本体（见 ClampedStdPolicyMixin）——NaN 仍留在参数里，漂移仪器与
    nan_probe 探针才能观测到污染真相。
    """
    finite = torch.isfinite(std)
    return torch.where(finite, std.clamp(min=STD_CLAMP_MIN), torch.full_like(std, STD_CLAMP_MIN))


class ClampedStdPolicyMixin:
    """在构造 `Normal` 前把 σ 夹到 ≥STD_CLAMP_MIN 的混入（三臂统一经此）。

    用法（MRO 必须混入在前，见 `ClampedStdActorCritic` / `ResidualDecoderPolicy`）：
        class ClampedStdActorCritic(ClampedStdPolicyMixin, _ActorCriticBase): ...
    覆写点 = `update_distribution`（rsl_rl `act` / `evaluate` 都经它填充 `self.distribution`）。

    实现手法：**临时把 `self.std` 从 `_parameters` 挪到实例 `__dict__` 并换成 clamp 后的普通
    tensor**（nn.Module `__setattr__` 不允许把 Parameter 属性直接赋成 Tensor，故走底层字典），
    调用 `super().update_distribution` 后用 `finally` 逐位还原 Parameter——保证：
      - 采样/log_prob 用夹紧后的 σ（病态区不抛）；
      - `self.std` 参数本体、`state_dict`、ckpt 结构、optimizer 引用、反传梯度在健康区
        **完全不变**（夹紧 tensor 的梯度经 clamp 恒等回到原 Parameter）；
      - σ 漂移仪器仍能读到**未夹紧**的原始 σ（滑向负值的全过程不被掩盖）。

    **声明：夹紧仅在 σ≤STD_CLAMP_MIN 的病态区生效，健康轨迹零改动（逐位）；NaN/±inf 亦
    归入病态区净化到 STD_CLAMP_MIN（见 `_clamp_std_param` 第三层修复）。**
    """

    @contextlib.contextmanager
    def _std_clamped(self):
        std = self._parameters.get("std", None)
        if std is None:   # 非 "scalar" σ 变体（如 log_std）不走此路径，保持原样
            yield
            return
        del self._parameters["std"]
        self.__dict__["std"] = _clamp_std_param(std)
        try:
            yield
        finally:
            del self.__dict__["std"]
            self._parameters["std"] = std

    def update_distribution(self, observations) -> None:
        with self._std_clamped():
            super().update_distribution(observations)


# ------------------------------------------------------------- selftest
def _check(cond: bool, msg: str) -> None:
    """自测断言：不满足即 RuntimeError（带原因，避免裸 assert 被 -O 剥掉）。"""
    if not cond:
        raise RuntimeError(f"selftest check failed: {msg}")


class _MockDecoder(nn.Module):
    """本机替身 decoder（(N,994)->(N,29)）：确定性线性映射，模拟冻结 ONNX MLP。"""

    def __init__(self, in_dim: int = DECODER_OBS_DIM, action_dim: int = ACTION_DIM) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, action_dim)
        with torch.no_grad():
            # 非平凡权重，保证输出非零（恒等性断言需要 decoder 输出 != 0）
            self.lin.weight.uniform_(-0.5, 0.5)
            self.lin.bias.uniform_(-0.1, 0.1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.lin(obs)


class _MockActorCriticStd(nn.Module):
    """本机替身 rsl_rl ActorCritic（**只复刻 σ 路径**：`self.std` + `update_distribution`）。

    生产环境（服务器）用真实 `rsl_rl.modules.ActorCritic`；本机（无训练 venv，`_ActorCriticBase
    = nn.Module`）用它验证 `ClampedStdPolicyMixin` 的夹紧行为：σ 采样、负 σ 抛异常、健康区逐位。
    复刻口径与 rsl_rl actor_critic.py 一致（`std = self.std.expand_as(mean)` -> `Normal(mean, std)`）。
    """

    def __init__(self, init_noise_std: float = 1.0, num_actions: int = ACTION_DIM) -> None:
        super().__init__()
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None

    def update_distribution(self, observations) -> None:
        std = self.std.expand_as(observations)
        self.distribution = torch.distributions.Normal(observations, std)

    def act(self, observations, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)


class _MockClampedActorCritic(ClampedStdPolicyMixin, _MockActorCriticStd):
    """替身 base + σ 夹紧混入（生产对应 `ClampedStdActorCritic`）。"""


def selftest_zero_init_identity(seed: int = 0) -> dict:
    """零初始化恒等性：mock decoder 输出 + 零 r -> μ 与 decoder 输出逐位一致。"""
    g = torch.Generator().manual_seed(seed)
    n, r_obs_dim = 8, 321
    decoder = _MockDecoder()
    residual = ResidualActionMLP(r_obs_dim)
    mu_path = ResidualMuPath(decoder, residual, freeze_decoder=True)

    decoder_obs = torch.randn(n, DECODER_OBS_DIM, generator=g)
    r_obs = torch.randn(n, r_obs_dim, generator=g)

    # 1) 残差恒为零（逐位，任意 r_obs）
    _check(residual.is_zero(r_obs, atol=0.0), "末层零初始化后 r(r_obs) 非零")
    _check(bool((residual(r_obs) == 0).all()), "r(r_obs) 非逐位零")

    # 2) μ == decoder(decoder_obs) 逐位
    with torch.no_grad():
        q = decoder(decoder_obs)
        mu = mu_path(decoder_obs, r_obs)
    _check(torch.equal(mu, q), f"μ 与 decoder 输出非逐位一致（maxdiff={float((mu - q).abs().max())}）")
    _check(bool(torch.isfinite(mu).all()), "μ 含非有限值")

    # 3) 反向验证：把 r 末层设为非零后 μ 应偏离（防「恒等」被静默成因其他原因）
    with torch.no_grad():
        head = residual.net[-1]
        head.weight.fill_(0.01)
    mu2 = mu_path(decoder_obs, r_obs)
    _check(not torch.equal(mu2, q), "非零 r 下 μ 仍与 decoder 逐位一致（恒等断言失效）")

    return {"residual_is_zero": True, "mu_bitwise_equal_decoder": True, "shape": list(mu.shape)}


def selftest_token_affine_roundtrip() -> dict:
    """新增断言 ①（M2）：前向仿射 a -> tok -> a' 往返（逐位）。

    构造：mean=0、std=2 的整数次幂 ⇒ `mean + std*a` 与 `(tok-mean)/std` 都是精确
    float32 运算（二进制可表示），故逐位往返成立。这直接验证 `forward_token_affine`
    与 author adapter 的逆仿射（eval_author_v0_decoder.py:638-642）**互逆**——即
    `residual_token_obs` 补的前向仿射确实把 a 还原回 token 坐标（reviewer BLOCKER-2）。
    同时做反向验证：不做前向仿射（直用 a）时与 token 坐标显著不同（复现
    reviewer 的 max|a-tok| 现象，防「往返」因平凡原因通过）。
    """
    g = torch.Generator().manual_seed(11)
    n = 4096
    token_mean = torch.zeros(TOKEN_DIM, dtype=torch.float32)
    token_std = torch.pow(torch.tensor(2.0), (torch.arange(TOKEN_DIM) % 7 - 3).float())
    alpha = 1.0

    for bound in ("none", "tanh"):
        if bound == "none":
            # a 任意 -> tok = std*a -> a' = tok/std（精确往返）
            a = torch.randn(n, TOKEN_DIM, generator=g, dtype=torch.float32) * 2.0
        else:
            # tanh 分支：tok = std*tanh(a) 落在 (-std, std)；a' = atanh(tok/std) 还原。
            # tanh/atanh 非精确互逆（float32 饱和区损失有效位），故取中等幅值 + 容差。
            a = torch.randn(n, TOKEN_DIM, generator=g, dtype=torch.float32) * 0.5
        tok = forward_token_affine(a, token_mean, token_std, alpha, bound)
        # 逆仿射（逐式同 AuthorPolicyAdapter.step:638-642）
        z = (tok - token_mean) / (alpha * token_std)
        if bound == "tanh":
            z = torch.atanh(z.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6))
        a_back = z
        if bound == "none":
            _check(torch.equal(a, a_back), f"none 分支 a->tok->a' 非逐位往返（maxdiff={float((a - a_back).abs().max())}）")
        else:
            _check(
                float((a - a_back).abs().max()) <= 1.0e-5,
                f"tanh 分支 a->tok->a' 往返偏差过大（maxdiff={float((a - a_back).abs().max())}）",
            )
        # 反向验证：漏前向仿射（tok 误用 a）时偏离显著
        _check(
            float((tok - a).abs().max()) > 0.1,
            f"{bound} 分支 tok 与 a 过于接近（前向仿射可能是恒等，断言失去区分度）",
        )
    return {
        "affine_roundtrip_bitwise_none": True,
        "affine_roundtrip_tanh_ok": True,
        "tanh_tolerance": 1.0e-5,
        "token_dim": TOKEN_DIM,
    }


def selftest_residual_vs_decoder_arm_decoder_input(seed: int = 0) -> dict:
    """新增断言 ②（M2 跨口径对照）：零残差时 residual 臂 μ 的 decoder 输入 ==
    decoder 臂构造的 decoder 输入，逐位一致。

    现有 `μ == decoder(dec_obs)` 自洽检查抓不到「decoder 输入本身口径错」——它只证明
    μ 等于用同一个错 dec_obs 喂 decoder 的结果。本断言**跨口径对照**：residual 臂
    `assemble_decoder_obs(policy_obs, token_slot, proprio_slot)` 的输出，必须与
    「decoder 臂语义」构造的输入（token 来自 author adapter 的 a 经**前向仿射**、
    proprio 取自 policy obs 的 930 段）逐位相等。若 residual_token_obs 漏了前向仿射
    （用 a 当 tok），本断言即失败。

    用 mock token_stats（零 mean + 2 的幂 std）保证仿射精确可逆；mock adapter.step
    返回一个已知 a，再手工走「前向仿射 -> assemble」重建 decoder 输入。
    """
    g = torch.Generator().manual_seed(seed)
    n = 8
    token_mean = torch.zeros(TOKEN_DIM, dtype=torch.float32)
    token_std = torch.pow(torch.tensor(2.0), (torch.arange(TOKEN_DIM) % 7 - 3).float())
    alpha, bound = 1.0, "none"

    official_dim = 321
    total = official_dim + PROPRIO_DIM + TOKEN_DIM  # official 段在前的真实 term 序
    token_start = official_dim + PROPRIO_DIM
    proprio_start = official_dim
    policy_obs = torch.randn(n, total, generator=g, dtype=torch.float32)

    # mock adapter.step 的返回 = 归一化动作 a（故意取远离 0 的值，放大小口径差异）
    a = torch.randn(n, TOKEN_DIM, generator=g, dtype=torch.float32) * 3.0

    # --- decoder 臂语义构造的 decoder 输入：token 坐标经前向仿射 + proprio 段 ---
    tok = forward_token_affine(a, token_mean, token_std, alpha, bound)
    dec_obs_expected = torch.cat(
        [tok, policy_obs[:, proprio_start : proprio_start + PROPRIO_DIM]], dim=-1
    )

    # --- residual 臂路径：把 token 写回 policy obs 的 token 槽，再走 assemble ---
    policy_obs_r = policy_obs.clone()
    policy_obs_r[:, token_start : token_start + TOKEN_DIM] = tok
    dec_obs_residual = assemble_decoder_obs(
        policy_obs_r, (token_start, TOKEN_DIM), (proprio_start, PROPRIO_DIM)
    )
    _check(
        torch.equal(dec_obs_residual, dec_obs_expected),
        f"residual 臂 decoder 输入 != decoder 臂语义输入（maxdiff={float((dec_obs_residual - dec_obs_expected).abs().max())}）",
    )

    # 反例：漏前向仿射（token 槽写 a 而非 tok）时两者必须不等（证明本断言有区分度）
    policy_obs_bad = policy_obs.clone()
    policy_obs_bad[:, token_start : token_start + TOKEN_DIM] = a
    dec_obs_bad = assemble_decoder_obs(
        policy_obs_bad, (token_start, TOKEN_DIM), (proprio_start, PROPRIO_DIM)
    )
    _check(
        not torch.equal(dec_obs_bad, dec_obs_expected),
        "漏前向仿射的 decoder 输入与正确输入逐位相同（断言失去区分度）",
    )

    # --- 端到端：residual 臂 μ（零残差）== decoder 臂 decoder(正确 decoder 输入) ---
    decoder = _MockDecoder()
    mu_path = ResidualMuPath(decoder, ResidualActionMLP(official_dim), freeze_decoder=True)
    r_obs = policy_obs_r[:, :official_dim]
    with torch.no_grad():
        mu = mu_path(dec_obs_residual, r_obs)
        q = decoder(dec_obs_expected)
    _check(torch.equal(mu, q), f"零残差 μ != decoder(decoder 臂输入)（maxdiff={float((mu - q).abs().max())}）")
    return {
        "decoder_input_bitwise_equal": True,
        "affine_missing_arm_differs": True,
        "official_dim": official_dim,
    }


def selftest_gradient_isolation(seed: int = 0) -> dict:
    """梯度隔离：只 r 参数 requires_grad；μ 反传只在 r 上产生 grad。"""
    g = torch.Generator().manual_seed(seed)
    n, r_obs_dim = 6, 321
    decoder = _MockDecoder()
    residual = ResidualActionMLP(r_obs_dim)
    mu_path = ResidualMuPath(decoder, residual, freeze_decoder=True)

    # 1) 参数 requires_grad 状态
    _check(not any(p.requires_grad for p in mu_path.decoder.parameters()), "decoder 参数未冻结")
    _check(all(p.requires_grad for p in mu_path.residual.parameters()), "residual 参数未全部可训")
    _check(mu_path.only_residual_trainable(), "only_residual_trainable 判据不通过")

    # 2) 反传：只有 residual 拿到 grad，decoder grad 恒 None
    decoder_obs = torch.randn(n, DECODER_OBS_DIM, generator=g)
    r_obs = torch.randn(n, r_obs_dim, generator=g)
    mu_path.zero_grad(set_to_none=True)
    mu = mu_path(decoder_obs, r_obs)
    _check(mu.requires_grad, "μ 不带梯度（残差未接入计算图）")
    loss = (mu ** 2).mean()
    loss.backward()
    dec_grads = [p.grad for p in mu_path.decoder.parameters()]
    res_grads = [p.grad for p in mu_path.residual.parameters()]
    _check(all(g_ is None for g_ in dec_grads), f"decoder 参数拿到梯度（{dec_grads}）")
    _check(all(g_ is not None for g_ in res_grads), "residual 参数未拿到梯度")
    _check(any(float(g_.abs().max()) > 0 for g_ in res_grads), "residual 梯度恒为 0")

    return {
        "decoder_frozen": True,
        "only_residual_trainable": True,
        "decoder_grad_none": True,
        "residual_grad_nonzero": True,
        "trainable_report": mu_path.trainable_report(),
    }


def selftest_reward_math() -> dict:
    """奖励纯数学单测：前进项（符号/无上限/只前进）+ 正则项（符号/量纲）。"""
    # --- 前进项符号 ---
    vx = torch.tensor([-0.5, 0.0, 0.3, 1.0, 5.0])
    r = forward_vel_reward(vx, weight=1.0, positive_only=True)
    _check(bool((r >= 0).all()), "positive_only 下出现负奖励")
    _check(float(r[0]) == 0.0, "后退速度未置零（positive_only）")
    _check(abs(float(r[2]) - 0.3) < 1e-6 and abs(float(r[3]) - 1.0) < 1e-6, "前进项量纲错（应为 1:1 m/s）")
    # 无人工上限：vx=5 的奖励 > vx=1 的奖励（对照 D048j clamp(0,1) 封顶）
    _check(float(r[4]) > float(r[3]), "前进项存在人工上限（vx=5 未高于 vx=1）")
    _check(abs(float(r[4]) - 5.0) < 1e-6, "前进项被限幅（应无上限）")
    # weight 线性缩放
    r2 = forward_vel_reward(vx, weight=2.0, positive_only=True)
    _check(torch.allclose(r2, 2.0 * r), "weight 未线性缩放")
    # positive_only=False 时保留后退负值（对照口径）
    r3 = forward_vel_reward(vx, weight=1.0, positive_only=False)
    _check(float(r3[0]) < 0, "positive_only=False 时后退应保留负值")

    # --- 正则项符号/量纲（配方表） ---
    for name, spec in MAX_FORWARD_RECIPE.items():
        w = float(spec["weight"])
        if spec["kind"] == "task":
            _check(w > 0, f"任务项 {name} 权重应 > 0，实为 {w}")
        elif spec.get("sign", 0) == 0:
            _check(w == 0.0, f"关闭项 {name} 权重应为 0，实为 {w}")
        elif spec["sign"] < 0:
            _check(w < 0, f"罚项 {name} 权重应 < 0，实为 {w}")
        else:
            _check(w > 0, f"塑形项 {name} 权重应 > 0，实为 {w}")
    _check(all(t not in MAX_FORWARD_RECIPE for t in MAX_FORWARD_REMOVED_TERMS),
           "被移除的命令跟踪项仍在配方表内")

    # --- 组合式：罚项使总奖励下降；总奖励形状/有限性 ---
    base = max_forward_total_reward(vx, weight_forward=1.0)
    penalized = max_forward_total_reward(
        vx,
        penalties=[(float(MAX_FORWARD_RECIPE["action_rate_l2"]["weight"]), torch.full_like(vx, 1.0))],
        weight_forward=1.0,
    )
    _check(bool((penalized <= base + 1e-6).all()), "罚项未使总奖励下降")
    _check(penalized.shape == vx.shape and bool(torch.isfinite(penalized).all()), "总奖励形状/有限性异常")

    return {
        "forward_positive_only": True,
        "forward_no_upper_cap": True,
        "forward_unit_mps": True,
        "recipe_signs_ok": True,
        "removed_terms_ok": True,
    }


def selftest_arm_table() -> dict:
    """三臂配置表自检（动作维/奖励默认/冻结分支/token 源）。"""
    _check(tuple(ARM_TABLE) == ARMS, f"ARM_TABLE 键 {tuple(ARM_TABLE)} != {ARMS}")
    for arm, spec in ARM_TABLE.items():
        _check(spec["action_dim"] in (ACTION_DIM, TOKEN_DIM), f"{arm} action_dim 非法")
        _check(spec["env_action"] in ("direct", "decoder", "lora_policy"), f"{arm} env_action 非法")
    _check(ARM_TABLE["direct"]["action_dim"] == ACTION_DIM, "direct 臂应 29 维")
    _check(ARM_TABLE["decoder"]["action_dim"] == TOKEN_DIM, "decoder 臂应 64 维")
    _check(ARM_TABLE["residual"]["action_dim"] == ACTION_DIM, "residual 臂应 29 维")
    _check(ARM_TABLE["residual"]["frozen_branch"] is True, "residual 臂应标 frozen_branch")
    _check(ARM_TABLE["residual"]["token_source"] == "author_v0_ckpt", "residual 臂 token 源应为 author v0")
    _check(ARM_TABLE["residual"]["env_action"] == "lora_policy", "residual 臂 env 侧应为 direct 通路变体")
    # reward auto 解析
    _check(resolve_reward_mode("residual", "auto") == "max_forward", "residual auto 应 -> max_forward")
    _check(resolve_reward_mode("direct", "auto") == "track", "direct auto 应 -> track")
    _check(resolve_reward_mode("decoder", "auto") == "track", "decoder auto 应 -> track")
    _check(resolve_reward_mode("direct", "max_forward") == "max_forward", "显式 max_forward 未被采纳")
    return {"arms": list(ARMS), "auto_reward_resolved": {a: resolve_reward_mode(a, "auto") for a in ARMS}}


def selftest_identity_envelope() -> dict:
    """身份信封自检（本机可构建，含三臂表项、奖励配方与新增契约字段）。"""
    env = build_identity_envelope("residual", "max_forward", intent_pin={"intent_pin_max": True, "value": 1.0})
    _check(env["arm"] == "residual", "信封 arm 字段错")
    _check(env["arm_table"]["frozen_branch"] is True, "信封 arm_table 未带冻结标记")
    _check(env["reward_mode"] == "max_forward", "信封 reward_mode 错")
    _check("forward_vel" in env["reward_recipe"], "信封奖励配方缺前进项")
    _check(set(MAX_FORWARD_REMOVED_TERMS) <= set(env["reward_removed_terms"]), "信封未记录被移除项")
    _check(env["script_md5"] and len(env["script_md5"]) == 32, "信封 script_md5 非法")
    _check(isinstance(env["env_module_md5"], dict) and env["env_module_md5"], "信封 env_module_md5 为空")
    # 新增契约字段（M1/M2/M3/M4）
    _check("ctx_terrain_map" in env and env["ctx_terrain_map"] == CTX_TERRAIN_MAP, "信封缺 ctx_terrain_map")
    _check("ctx_terrain" in env, "信封缺 ctx_terrain 字段")
    _check("token_affine" in env and "formula" in env["token_affine"], "信封缺 token_affine 字段")
    _check(env["privileged_critic"]["author_token_in_critic"] is True, "信封缺 privileged_critic 声明")
    _check(env["reward_recipe_scope"]["applied_to"] == "all_terrains", "信封缺 reward_recipe_scope 声明")
    # JSON 可序列化（写盘前提）
    json.dumps(env, ensure_ascii=False)
    return {"envelope_keys": sorted(env)}


def selftest_ctx_terrain_contract() -> dict:
    """M1 地形契约自检：CLI 地形名 -> ctx 契约名映射全覆盖且无非法值。"""
    allowed = ("plane", "rough_paper", "climbing_box")
    for t in TERRAINS:
        got = ctx_terrain(t)
        _check(got in allowed, f"{t} -> {got} 不在 ctx 契约 {allowed}")
    _check(ctx_terrain("plane") == "plane", "plane 应映射到 plane")
    for t in ("rough", "stairs", "stones", "discrete"):
        _check(ctx_terrain(t) == "rough_paper", f"{t} 应映射到 rough_paper，实为 {ctx_terrain(t)}")
    # 幂等：ctx 契约名原样返回
    _check(ctx_terrain("rough_paper") == "rough_paper", "ctx_terrain 非幂等（rough_paper）")
    # 未知名 fail loud
    try:
        ctx_terrain("mars")
        raise RuntimeError("未知地形未 fail loud")
    except ValueError:
        pass
    return {"terrains": list(TERRAINS), "mapped": {t: ctx_terrain(t) for t in TERRAINS}}


def selftest_official_obs_slice_guard() -> dict:
    """SF-2 切片守卫自检：正确 term 序通过；前缀假设被破坏时 fail loud。"""
    # 正确序：official 段在前，proprio/token 在后
    terms = ("base_lin_vel", "base_ang_vel", "joint_pos", "sonic_proprio", "author_token")
    dims = ((3,), (3,), (29,), (930,), (64,))
    off = official_obs_slice(terms, dims)
    _check(off["start"] == 0, "official 段起点应为 0")
    _check(off["width"] == 3 + 3 + 29, f"official 段宽度错：{off['width']}")
    _check(off["excluded"] == {"sonic_proprio": 930, "author_token": 64}, f"排除宽度错：{off['excluded']}")
    # 前缀假设被破坏（proprio 排在 official 段之前）-> fail loud
    bad_terms = ("sonic_proprio", "base_lin_vel", "author_token")
    bad_dims = ((930,), (3,), (64,))
    try:
        official_obs_slice(bad_terms, bad_dims)
        raise RuntimeError("前缀假设被破坏时未 fail loud")
    except RuntimeError:
        pass
    # 缺被排除 term -> fail loud
    try:
        official_obs_slice(("base_lin_vel",), ((3,),))
        raise RuntimeError("缺 author_token 时未 fail loud")
    except RuntimeError:
        pass
    return {"official_width": off["width"], "excluded": off["excluded"], "fail_loud_ok": True}


# ------------------------------------------------- 执行/反馈仿射 selftest（修复验证）
def _fixture_sonic_scale(seed: int = 7) -> np.ndarray:
    """selftest 用的 29 维正 scale 夹具（**非生产源**；生产源 = `hv._sonic_scale_isaac()`）。

    刻意含踝部量级 0.0745（SONIC 序 13/14/17/18 = 踝 pitch/roll），其余取 [0.35,0.55]
    量级——踝部小 scale 把旧标量 0.5 的增益缺陷放大到 6.71×，使反例有区分度。
    只验证**关系**（增益/互逆/形状），不复制生产常量（防漂移）。
    """
    g = np.random.default_rng(seed)
    s = 0.35 + 0.2 * g.random(ACTION_DIM)
    for i in (13, 14, 17, 18):  # 踝 pitch/roll（SONIC/G1_ISAACLab_ORDER）
        s[i] = 0.0745008703
    return s.astype(np.float64)


def selftest_residual_action_scale_gain_identity(seed: int = 3) -> dict:
    """新增断言 ①：闭环增益恒等（执行端 scale=sonic_scale ⇒ 反馈反归一化恢复 μ，gain≡1）。

    执行端：q_des = default + μ*scale（use_default_offset=True，joint_actions.py:130-139）。
    反馈端：last_act = (q_des-default)/sonic_scale（sonic_action_term.py:244）。
    取 scale == sonic_scale ⇒ last_act ≡ μ，闭环增益 g = scale/sonic_scale ≡ 1（逐关节）。

    三层校验：
      (a) **结构性恒等（逐位）**：`exec_feedback_gain(sonic_scale, sonic_scale)` 逐元素
          == 1.0（IEEE x/x==1 精确）——这是互逆关系的硬证据；
      (b) 数值往返（精确子例）：default=0、scale=2 的幂向量 ⇒ 乘除均为精确运算 ⇒
          q_des→(q_des-default)/scale 与 μ **逐位**相等；
      (c) 数值往返（真实量级子例）：default≠0、scale=夹具向量 ⇒ 恢复 μ 的偏差仅 float
          舍入（≤1e-6；相对缺陷增益 ~6.7 小 6 个量级），证明 offset 严格消项。
    """
    n = 4096
    sonic_scale = _fixture_sonic_scale()
    _check(sonic_scale.shape == (ACTION_DIM,), "夹具 sonic_scale 形状非 29")
    _check(bool((sonic_scale > 0).all()), "夹具 sonic_scale 含非正值")

    # (a) 结构性恒等：执行端 scale 就是 sonic_scale ⇒ 增益逐位 == 1
    gain = exec_feedback_gain(sonic_scale, sonic_scale)
    _check(np.array_equal(gain, np.ones(ACTION_DIM)), f"结构性增益非逐位 1（{gain.min()}..{gain.max()}）")
    # 通过 cfg 覆写路径构造的 scale dict，其值必须与 sonic_scale 逐位一致（同源无重排）
    scale_map = build_sonic_scale_map(sonic_scale)
    vec_from_map = np.array([scale_map[j] for j in sonic_joint_names()], dtype=np.float64)
    _check(np.array_equal(vec_from_map, sonic_scale), "scale_map 重排/改变了 sonic_scale 向量")
    gain_map = exec_feedback_gain(vec_from_map, sonic_scale)
    _check(np.array_equal(gain_map, np.ones(ACTION_DIM)), "scale_map 路径增益非逐位 1")

    # (b) 精确子例：default=0、scale=2 的幂 ⇒ 乘除精确 ⇒ 逐位往返
    g = np.random.default_rng(seed)
    mu = g.standard_normal((n, ACTION_DIM))
    pow2_scale = np.power(2.0, g.integers(-3, 1, size=ACTION_DIM)).astype(np.float64)
    q_des = mu * pow2_scale                     # default = 0
    mu_back = q_des / pow2_scale
    _check(np.array_equal(mu_back, mu), f"精确子例 μ 往返非逐位（maxdiff={np.abs(mu_back - mu).max()}）")

    # (c) 真实量级子例：default≠0 ⇒ offset 消项（偏差仅 float 舍入）
    default = 0.2 * np.sign(g.standard_normal(ACTION_DIM))
    mu2 = g.standard_normal((n, ACTION_DIM))
    q_des2 = default + mu2 * sonic_scale
    mu2_back = (q_des2 - default) / sonic_scale
    dev = float(np.abs(mu2_back - mu2).max())
    _check(dev <= 1.0e-6, f"真实量级 μ 往返偏差过大（{dev}；应仅 float 舍入）")
    return {
        "gain_bitwise_one": True,
        "gain_min": float(gain.min()),
        "gain_max": float(gain.max()),
        "exact_roundtrip_bitwise": True,
        "offset_roundtrip_maxdev": dev,
    }


def selftest_residual_action_scale_legacy_counterexample(seed: int = 5) -> dict:
    """新增断言 ②：旧配置（scale=0.5 标量）增益 = 0.5/sonic_scale > 1 的反例。

    证明断言 ① 能抓原缺陷：标量 0.5 时反馈端 last_act = μ*0.5/sonic_scale，增益逐关节
    ≠1（踝 0.0745 ⇒ 6.71；大关节 0.5475 ⇒ 0.913），数值往返 μ 偏差量级 ~0.5|μ|（远大于
    1e-6）。断言 ② 要求：增益非恒 1、且 5 个量级上可区分。
    """
    sonic_scale = _fixture_sonic_scale()
    gain_old = exec_feedback_gain(RESIDUAL_LEGACY_ACTION_SCALE, sonic_scale)  # 0.5/scale
    _check(not np.array_equal(gain_old, np.ones(ACTION_DIM)), "旧标量增益竟为 1（反例失效）")
    _check(float(gain_old.max()) > 1.0, f"旧标量增益上界应 > 1，实 {gain_old.max()}")
    _check(float(gain_old.min()) < 1.0, f"旧标量增益下界应 < 1，实 {gain_old.min()}")
    # 踝部（scale≈0.0745）增益≈6.71，与探针量化的 0.91–6.71× 区间一致
    _check(abs(float(gain_old.max()) - 0.5 / 0.0745008703) < 1.0e-6,
           f"旧标量踝部增益应≈{0.5 / 0.0745008703:.4f}，实 {gain_old.max():.4f}")

    # 数值往返：旧标量下恢复的 μ 与真 μ 显著偏离（可区分断言 ① 的 1e-6 容差）
    g = np.random.default_rng(seed)
    n = 2048
    default = 0.2 * np.sign(g.standard_normal(ACTION_DIM))
    mu = g.standard_normal((n, ACTION_DIM))
    q_des = default + mu * RESIDUAL_LEGACY_ACTION_SCALE          # 旧执行端
    mu_back = (q_des - default) / sonic_scale                     # 反馈端（逐关节 sonic_scale）
    dev = float(np.abs(mu_back - mu).max())
    _check(dev > 0.1, f"旧标量往返偏差应显著（>0.1），实 {dev}")
    return {
        "legacy_gain_min": float(gain_old.min()),
        "legacy_gain_max": float(gain_old.max()),
        "legacy_roundtrip_maxdev": dev,
        "discriminates_identity": True,
    }


def selftest_residual_action_scale_shape_source() -> dict:
    """新增断言 ③：29 维 scale 向量形状/来源一致性（单一事实源，无第二份常量）。

    校验：
      - `sonic_joint_names()` = 29 个唯一名，且 == `gear_sonic...G1_ISAACLab_ORDER`
        （与 `sonic_action_term.SONIC_JOINT_NAMES` / `_sonic_scale_isaac()` 同序同源）；
      - `build_sonic_scale_map` 键 == 上述名序，值 == 输入向量逐位（无重排/丢失）；
      - `SONIC_SCALE_SOURCE` 指向 `sonic_action_term._sonic_scale_isaac`（与
        `sonic_proprio_hist` 反归一化同一份），杜绝本文件复制第二份 scale 常量；
      - 信封 `residual_action_scale` 字段：dim=29、per_joint 29 项、min/max 与向量一致、
        JSON 可序列化。
    """
    from gear_sonic.envs.env_utils.joint_utils import G1_ISAACLab_ORDER  # noqa: PLC0415

    names = sonic_joint_names()
    _check(len(names) == ACTION_DIM, f"关节名应 {ACTION_DIM} 个，实 {len(names)}")
    _check(len(set(names)) == ACTION_DIM, "关节名有重复")
    _check(names == tuple(G1_ISAACLab_ORDER), "sonic_joint_names != G1_ISAACLab_ORDER（来源不一致）")

    sonic_scale = _fixture_sonic_scale()
    scale_map = build_sonic_scale_map(sonic_scale)
    _check(tuple(scale_map) == names, "scale_map 键序 != SONIC 关节名序")
    _check(np.array_equal(np.array(list(scale_map.values())), sonic_scale),
           "scale_map 值 != 输入 sonic_scale（逐位）")

    # 来源一致性（字符串契约）：单一事实源函数名 + 与 sonic_proprio_hist 同源声明
    _check("_sonic_scale_isaac" in SONIC_SCALE_SOURCE, "SONIC_SCALE_SOURCE 未指向 _sonic_scale_isaac")
    _check("sonic_proprio_hist" in SONIC_SCALE_SOURCE, "SONIC_SCALE_SOURCE 未声明与 sonic_proprio_hist 同源")

    # 形状守卫：非 29 维必须 fail loud（防静默广播）
    for bad in (np.zeros(ACTION_DIM - 1), np.zeros(ACTION_DIM + 1)):
        try:
            build_sonic_scale_map(bad)
            raise RuntimeError("非 29 维 scale 未 fail loud")
        except ValueError:
            pass
    try:
        exec_feedback_gain(0.5, np.zeros(ACTION_DIM - 1))
        raise RuntimeError("非 29 维 sonic_scale 未 fail loud（exec_feedback_gain）")
    except ValueError:
        pass

    # 信封字段结构
    env_field = residual_action_scale_envelope(sonic_scale)
    _check(env_field["dim"] == ACTION_DIM, "信封 dim 非 29")
    _check(len(env_field["per_joint"]) == ACTION_DIM, "信封 per_joint 非 29 项")
    _check(abs(env_field["min"] - float(sonic_scale.min())) < 1e-12, "信封 min 与向量不符")
    _check(abs(env_field["max"] - float(sonic_scale.max())) < 1e-12, "信封 max 与向量不符")
    _check(env_field["gain_identity"] is True, "信封未声明 gain_identity")
    _check(env_field["legacy_scalar"] == RESIDUAL_LEGACY_ACTION_SCALE, "信封 legacy_scalar 错")
    json.dumps(env_field, ensure_ascii=False)
    return {
        "n_joints": len(names),
        "source": SONIC_SCALE_SOURCE,
        "envelope_dim": env_field["dim"],
        "scale_min": env_field["min"],
        "scale_max": env_field["max"],
    }


def selftest_safe_close_timeout(monkeypatch_exit=None) -> dict:
    """新增断言 ①：`_safe_close` 超时路径（mock 永不返回的 close() -> 走 os._exit 分支）。

    mock 一个 `close()` 永不返回的 app（Event 同步阻塞，绝不释放），把 timeout 压到 0.05s；
    再用 `monkeypatch_exit`（缺省即 monkeypatch 本模块 `os._exit` 为记录器）防真退。
    断言：超时被判定、os._exit 被以 exit_code 调用、返回报告 timed_out=True。
    另测正常路径：close() 立即返回 -> 不调 os._exit、closed=True。
    """
    recorded: list = []

    def _fake_exit(code):
        recorded.append(int(code))

    # 延迟把 os._exit 换成记录器（防真退）；若调用方传入自己的记录器则用之。
    real_exit = os._exit
    if monkeypatch_exit is None:
        os._exit = _fake_exit
    else:
        monkeypatch_exit(_fake_exit)
    try:
        # --- 超时路径：close() 阻塞在 gate 上永不返回 ---
        gate = threading.Event()

        class _HangingApp:
            def close(self):  # pragma: no cover - 故意挂住
                gate.wait()

        rep = _safe_close(_HangingApp(), timeout_s=0.05, exit_code=7)
        _check(rep["timed_out"] is True, f"超时未被判定（{rep}）")
        _check(rep["closed"] is False, f"超时路径 closed 应为 False（{rep}）")
        _check(recorded == [7], f"超时未以 exit_code=7 调 os._exit（记录 {recorded}）")

        # --- 正常路径：close() 立即返回 -> 不调 os._exit ---
        recorded.clear()

        class _OkApp:
            def close(self):
                return None

        rep2 = _safe_close(_OkApp(), timeout_s=5.0, exit_code=0)
        _check(rep2["closed"] is True and rep2["timed_out"] is False, f"正常路径报告错（{rep2}）")
        _check(recorded == [], f"正常路径不应调 os._exit（记录 {recorded}）")

        # --- close() 抛异常也要解脱主线程（不真退、不挂） ---
        class _RaisingApp:
            def close(self):
                raise RuntimeError("boom")

        rep3 = _safe_close(_RaisingApp(), timeout_s=5.0, exit_code=0)
        _check(rep3["closed"] is True and rep3["error"] and "boom" in rep3["error"],
               f"close() 抛异常路径报告错（{rep3}）")
        _check(recorded == [], f"close() 抛异常路径不应调 os._exit（记录 {recorded}）")
    finally:
        gate.set()  # 释放挂住的 daemon 线程（防其长期占用）
        os._exit = real_exit
    return {
        "timeout_path_exits": True,
        "timeout_exit_code": 7,
        "ok_path_no_exit": True,
        "error_path_returns": True,
    }


def selftest_main_loop_crash_log() -> dict:
    """新增断言 ②：`_log_main_loop_crash` 落盘路径（raise 后文件存在且含 traceback）。"""
    import tempfile  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        try:
            raise ValueError("synthetic-interrupt-at-23M-steps")
        except ValueError as exc:
            path = _log_main_loop_crash(out_dir, exc, iteration=950)
        _check(path.exists(), f"crash log 未落盘（{path}）")
        _check(path == _crash_log_path(out_dir), f"crash log 路径非 outputs/<out>/{MAIN_LOOP_CRASH_LOG}")
        text = path.read_text(encoding="utf-8")
        _check("Traceback (most recent call last)" in text, "crash log 缺 traceback 头")
        _check("ValueError" in text and "synthetic-interrupt-at-23M-steps" in text,
               "crash log 缺异常类型/消息")
        _check("iteration: 950" in text, "crash log 缺 iteration 字段")
        _check("timestamp:" in text, "crash log 缺时间戳")
        _check("selftest_main_loop_crash_log" in text, "crash log traceback 未含抛出点帧")
    return {"crash_log_written": True, "has_traceback": True, "has_iteration": True}


# ------------------------------------------------- σ 病态区夹紧 selftest（第二层修复）
def selftest_std_clamp_negative_std(seed: int = 0) -> dict:
    """新增断言 ①：负 σ（构造 -0.5）经夹紧层后 sample 不抛且 scale==STD_CLAMP_MIN。

    复现 crash 根因：rsl_rl ActorCritic 的裸 σ 被 PPO 推负 -> `Normal.sample` 抛
    `normal expects all elements of std >= 0.0`（tmp/_fixverify_main_loop_crash.log）。
    先断言**裸替身确实抛**（防「夹紧断言因平凡原因通过」——若裸替身都不抛，本测试失去区分度），
    再断言**夹紧替身不抛且分布 scale 逐位 == STD_CLAMP_MIN**，最后断言参数本体仍是原始负值
    （夹紧只作用于采样分布，不改写 σ 参数——漂移仪器才能观测原始漂移）。
    """
    torch.manual_seed(seed)
    n, dim = 5, ACTION_DIM
    obs = torch.zeros(n, dim)

    # (a) 裸替身：负 σ 采样必抛（复现崩溃）
    raw = _MockActorCriticStd(init_noise_std=-0.5)
    raised = False
    try:
        raw.update_distribution(obs)
        raw.distribution.sample()
    except (RuntimeError, ValueError):
        raised = True
    _check(raised, "裸替身负 σ 采样未抛（crash 根因未复现，断言失去区分度）")

    # (b) 夹紧替身：不抛，scale 逐位 == STD_CLAMP_MIN
    clamped = _MockClampedActorCritic(init_noise_std=-0.5)
    clamped.update_distribution(obs)
    scale = clamped.distribution.scale
    _check(bool((scale == STD_CLAMP_MIN).all()),
           f"夹紧后 scale 非 STD_CLAMP_MIN：{scale.reshape(-1)[:3].tolist()}")
    sample = clamped.distribution.sample()
    _check(sample.shape == (n, dim) and bool(torch.isfinite(sample).all()), "夹紧后 sample 形状/有限性异常")
    # (c) 参数本体未被改写（仍是原始 -0.5，漂移仪器可观测）
    _check(isinstance(clamped.std, nn.Parameter), "夹紧后 std 不再是 nn.Parameter")
    _check(float(clamped.std.reshape(-1)[0].item()) == -0.5, "夹紧改写了 σ 参数本体（应只作用于分布）")
    _check("std" in clamped.state_dict(), "夹紧后 state_dict 缺 std（ckpt 结构被破坏）")
    return {
        "raw_negative_std_raises": True,
        "clamped_scale": STD_CLAMP_MIN,
        "clamped_sample_ok": True,
        "param_untouched": True,
    }


def selftest_std_clamp_healthy_bitwise(seed: int = 0) -> dict:
    """新增断言 ②：健康 σ（1.0）经夹紧层后逐位不变（scale 与 log_prob 梯度均逐位）。

    「只下夹不上夹」的可比性保证：σ=1.0 ≥ STD_CLAMP_MIN 时 clamp 是恒等映射 ⇒ 分布 scale、
    `log_prob` 及其对 σ 的反传梯度都必须与**裸替身**逐位相同（不是「近似相等」）。
    """
    torch.manual_seed(seed)
    n, dim = 5, ACTION_DIM
    obs = torch.zeros(n, dim)
    actions = torch.randn(n, dim)

    def _scale_and_grad(cls):
        m = cls(init_noise_std=1.0)
        m.update_distribution(obs)
        lp = m.get_actions_log_prob(actions).sum()   # 标量化以便对 σ 求导
        g, = torch.autograd.grad(lp, m.std)
        return m.distribution.scale.detach().clone(), g.detach().clone()

    s_clamped, g_clamped = _scale_and_grad(_MockClampedActorCritic)
    s_raw, g_raw = _scale_and_grad(_MockActorCriticStd)

    _check(torch.equal(s_clamped, s_raw), "健康 σ 下夹紧层改动了 scale（非逐位）")
    _check(torch.equal(s_clamped, torch.ones(n, dim)), "健康 σ=1.0 的 scale 非全 1")
    _check(torch.equal(g_clamped, g_raw), "健康 σ 下夹紧层改动了 log_prob 对 σ 的梯度（非逐位）")
    return {
        "scale_bitwise_equal_raw": True,
        "scale_all_ones": True,
        "grad_bitwise_equal_raw": True,
    }


def selftest_std_clamp_all_arms() -> dict:
    """新增断言 ③：三臂策略类都挂了 σ 夹紧（配置/构造断言，防漏挂一臂）。

    两条独立检查：
      1. **类层**：direct/decoder 臂类（`ClampedStdActorCritic`）与 residual 臂类
         （`ResidualDecoderPolicy`）都继承 `ClampedStdPolicyMixin`，且
         `ClampedStdActorCritic` 继承 rsl_rl `ActorCritic`（非夹紧版不复用，防误配回裸基类）；
      2. **实例层**（本机替身，不依赖 rsl_rl）：负 σ 的夹紧替身 `act()` 不抛 —— 覆盖真实训练
         采样路径（ppo.py:260 `policy.act` -> sample）。
    """
    # (1) 三臂策略类都挂了夹紧混入
    _check(issubclass(ClampedStdActorCritic, ClampedStdPolicyMixin), "ClampedStdActorCritic 未挂夹紧混入")
    _check(issubclass(ResidualDecoderPolicy, ClampedStdPolicyMixin), "ResidualDecoderPolicy 未挂夹紧混入")
    _check(ClampedStdActorCritic is not _ActorCriticBase, "ClampedStdActorCritic 不应等于裸基类")
    if HAS_RSL_RL:  # pragma: no cover - 服务器
        _check(issubclass(ClampedStdActorCritic, _ActorCriticBase), "ClampedStdActorCritic 未继承 rsl_rl ActorCritic")
    else:  # 本机：_ActorCriticBase 降级为 nn.Module，仅断言是 nn.Module 子类
        _check(issubclass(ClampedStdActorCritic, nn.Module), "ClampedStdActorCritic 非 nn.Module 子类")
    # direct/decoder 两臂同用 ClampedStdActorCritic；residual 用 ResidualDecoderPolicy
    arm_classes = {
        "direct": ClampedStdActorCritic,
        "decoder": ClampedStdActorCritic,
        "residual": ResidualDecoderPolicy,
    }
    _check(set(arm_classes) == set(ARMS), f"三臂类表键 {set(arm_classes)} != {set(ARMS)}")
    for arm, cls in arm_classes.items():
        _check(issubclass(cls, ClampedStdPolicyMixin), f"{arm} 臂策略类未挂夹紧混入")
    # (2) 实例层：负 σ 经 act() 采样不抛（真实训练采样路径）
    m = _MockClampedActorCritic(init_noise_std=-0.5)
    a = m.act(torch.zeros(4, ACTION_DIM))
    _check(a.shape == (4, ACTION_DIM) and bool(torch.isfinite(a).all()), "夹紧替身 act() 输出异常")
    return {
        "arms_clamped": sorted(arm_classes),
        "clamped_class_subclasses_mixin": True,
        "clamped_act_ok_on_negative_std": True,
    }


# ------------------------------------------------- σ NaN 净化 + nan_probe selftest（第三层修复）
def selftest_std_clamp_nan_purified(seed: int = 0) -> dict:
    """新增断言 ①：NaN σ 经净化后 scale==STD_CLAMP_MIN 且 sample 不抛。

    复现第三层 crash：σ 轨迹全程健康却在 update 内部被写成 NaN
    （tmp/_d067_verify/freeze_sigmafix_iter950_crash.txt），旧 `clamp(min=1e-4)` 对 NaN 恒等
    （`NaN.clamp = NaN`）⇒ 保命带失效、`torch.normal` 抛 `std >= 0.0`。
    先断言**裸替身 NaN σ 采样必抛**（防断言因平凡原因通过），再断言**净化替身不抛且 scale
    逐位 == STD_CLAMP_MIN**，最后断言 σ 参数本体仍是 NaN（净化只作用于采样分布，不改写参数——
    溯源探针/仪器才能看到污染真相）。
    """
    torch.manual_seed(seed)
    n, dim = 5, ACTION_DIM
    obs = torch.zeros(n, dim)
    bad = float("nan")

    # (a) 裸替身：NaN σ 采样必抛（复现第三层崩溃）
    raw = _MockActorCriticStd(init_noise_std=bad)
    raised = False
    try:
        raw.update_distribution(obs)
        raw.distribution.sample()
    except (RuntimeError, ValueError):
        raised = True
    _check(raised, "裸替身 NaN σ 采样未抛（crash 根因未复现，断言失去区分度）")

    # (b) 净化替身：不抛，scale 逐位 == STD_CLAMP_MIN（全 NaN -> 全 STD_CLAMP_MIN）
    clamped = _MockClampedActorCritic(init_noise_std=bad)
    clamped.update_distribution(obs)
    scale = clamped.distribution.scale
    _check(bool(torch.isfinite(scale).all()), f"净化后 scale 仍含非有限值：{scale.reshape(-1)[:3].tolist()}")
    _check(bool((scale == STD_CLAMP_MIN).all()),
           f"净化后 scale 非 STD_CLAMP_MIN：{scale.reshape(-1)[:3].tolist()}")
    sample = clamped.distribution.sample()
    _check(sample.shape == (n, dim) and bool(torch.isfinite(sample).all()), "净化后 sample 形状/有限性异常")
    # (c) 参数本体仍是 NaN（净化不改写 σ 参数——仪器/探针可见污染真相）
    _check(isinstance(clamped.std, nn.Parameter), "净化后 std 不再是 nn.Parameter")
    _check(bool(torch.isnan(clamped.std).all()), "净化改写了 NaN σ 参数本体（应只作用于采样分布）")
    _check("std" in clamped.state_dict(), "净化后 state_dict 缺 std（ckpt 结构被破坏）")

    # (d) ±inf 同样被净化（旧 clamp 对 +inf 恒等、对 -inf 也保不住下限）
    for val in (float("inf"), float("-inf")):
        m = _MockClampedActorCritic(init_noise_std=val)
        m.update_distribution(obs)
        _check(bool((m.distribution.scale == STD_CLAMP_MIN).all()),
               f"σ={val} 未被净化到 STD_CLAMP_MIN：{m.distribution.scale.reshape(-1)[:3].tolist()}")
    return {
        "raw_nan_std_raises": True,
        "purified_scale": STD_CLAMP_MIN,
        "purified_sample_ok": True,
        "param_untouched_nan": True,
        "inf_purified": True,
    }


def selftest_std_clamp_finite_bitwise_vs_legacy(seed: int = 0) -> dict:
    """新增断言 ②：有限值路径与旧 `clamp(min=...)` 逐位一致（净化不改健康/负值路径）。

    覆盖三点：(1) `_clamp_std_param` 与 `x.clamp(min=STD_CLAMP_MIN)` 在含负值/零/健康值的
    张量上逐位相等；(2) 经 mixin 的分布 scale 与旧 clamp 逐位相等；(3) 健康 σ=1.0 时分布
    scale 逐位 == 原始 σ（旧语义不变）。
    """
    torch.manual_seed(seed)
    n = 6
    obs = torch.zeros(n, ACTION_DIM)
    x = torch.tensor([-0.5, 0.0, 1.0e-9, STD_CLAMP_MIN, 0.51, 3.7], dtype=torch.float32)
    x = x.repeat((ACTION_DIM + x.numel() - 1) // x.numel())[:ACTION_DIM].contiguous()  # 铺满 29 维

    new = _clamp_std_param(x)
    legacy = x.clamp(min=STD_CLAMP_MIN)
    _check(torch.equal(new, legacy), f"有限值路径与旧 clamp 非逐位一致：{new.tolist()} vs {legacy.tolist()}")
    _check(bool((new >= STD_CLAMP_MIN).all()), "净化后出现 < STD_CLAMP_MIN 的值")

    # 经 mixin：健康 σ 下分布 scale 逐位不变
    m = _MockClampedActorCritic(init_noise_std=1.0)
    m.update_distribution(obs)
    _check(torch.equal(m.distribution.scale, torch.ones(n, ACTION_DIM)), "健康 σ=1.0 经净化后 scale 非全 1")

    # 经 mixin：含负值的 σ 参数 -> scale 逐位 == 旧 clamp 结果
    m2 = _MockClampedActorCritic(init_noise_std=0.0)
    with torch.no_grad():
        m2.std.copy_(x)
    m2.update_distribution(obs)
    _check(torch.equal(m2.distribution.scale, x.clamp(min=STD_CLAMP_MIN).expand(n, ACTION_DIM)),
           "负值 σ 经净化后的 scale 与旧 clamp 非逐位一致")
    return {
        "finite_path_bitwise_legacy": True,
        "healthy_scale_unchanged": True,
        "negative_scale_bitwise_legacy": True,
    }


def selftest_nan_probe_line_and_trigger(seed: int = 0) -> dict:
    """新增断言 ③：nan_probe 行格式与触发（mock 一个含 NaN 的 loss_dict）。

    用最小 mock alg（`update` 返回含 NaN 的 loss_dict；policy 带 `std`）装仪器，断言：
      - 日志里出现 `tag == "nan_probe"` 行，字段齐全（name/n_nonfinite/min/max/first/
        first_index/shape/dtype/stage/update_calls）；
      - `first` 确为 NaN、`n_nonfinite` 计数正确、`name == "loss.<key>"`；
      - 探针**不阻断**：`update` 返回值原样返回、训练继续（不抛）；
      - 反例：全有限 loss_dict 时不产生 nan_probe 行（断言有区分度）。
    """
    import tempfile  # noqa: PLC0415

    class _MockAlg:
        def __init__(self, loss_value):
            self.policy = _MockActorCriticStd(init_noise_std=1.0)
            self.storage = SimpleNamespace(observations=None)
            self._loss = loss_value

        def update(self):
            return {"loss": torch.tensor([self._loss], dtype=torch.float32), "surrogate": torch.tensor(1.0)}

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        # --- 触发路径：loss 含 NaN ---
        alg = _MockAlg(float("nan"))
        install_std_trajectory_instrument(alg, out_dir, every=1, phase="selftest")
        ret = alg.update()
        _check(isinstance(ret, dict) and "loss" in ret, "探针改写了 update 返回值（应原样返回）")
        _check(bool(torch.isnan(ret["loss"]).all()), "mock update 返回值被改写")

        lines = (out_dir / STD_TRAJECTORY_LOG).read_text(encoding="utf-8").strip().splitlines()
        rows = [json.loads(ln) for ln in lines]
        probes = [r for r in rows if r.get("tag") == NAN_PROBE_TAG]
        _check(len(probes) == 1, f"nan_probe 行数应为 1，实为 {len(probes)}（rows={[r.get('tag') for r in rows]}）")
        p = probes[0]
        for field in ("name", "n_nonfinite", "n_total", "min", "max", "first", "first_index",
                      "shape", "dtype", "stage", "update_calls"):
            _check(field in p, f"nan_probe 行缺字段 {field!r}：{p}")
        _check(p["name"] == "loss.loss", f"nan_probe name 应为 loss.loss，实为 {p['name']!r}")
        _check(p["stage"] == "post", f"nan_probe stage 应为 post，实为 {p['stage']!r}")
        _check(p["n_nonfinite"] == 1 and p["n_total"] == 1, f"nan_probe 计数错：{p}")
        _check(isinstance(p["first"], float) and math.isnan(p["first"]), f"nan_probe first 非 NaN：{p['first']!r}")
        _check(p["first_index"] == 0, f"nan_probe first_index 应为 0，实为 {p['first_index']!r}")
        _check(p["min"] is None and p["max"] is None, f"全 NaN 张量的 min/max 应为 None：{p}")
        _check(p["shape"] == [1] and p["dtype"] == "torch.float32", f"nan_probe shape/dtype 错：{p}")
        # 其它 tag 仍在（pre/post σ 轨迹行未被破坏）
        _check(any(r.get("tag") == "pre" for r in rows) and any(r.get("tag") == "post" for r in rows),
               "σ 轨迹 pre/post 行缺失（仪器回归）")

    with tempfile.TemporaryDirectory() as tmp:
        # --- 反例：全有限 loss_dict -> 无 nan_probe 行 ---
        out_dir = Path(tmp)
        alg = _MockAlg(1.0)
        install_std_trajectory_instrument(alg, out_dir, every=1, phase="selftest")
        alg.update()
        rows = [json.loads(ln) for ln in (out_dir / STD_TRAJECTORY_LOG).read_text(encoding="utf-8").strip().splitlines()]
        _check(not any(r.get("tag") == NAN_PROBE_TAG for r in rows),
               f"全有限 loss 下不应产生 nan_probe 行：{[r.get('tag') for r in rows]}")
    return {
        "nan_probe_line_written": True,
        "fields_ok": True,
        "nonblocking": True,
        "finite_case_no_probe": True,
    }


# ------------------------------------------------- 第四层：update 内分步探针 selftest
class _MockAlgLayer4:
    """第四层 selftest 替身 alg：带 storage（advantages/returns/values）+ learning_rate + policy。

    `update()` 可选在内部把 `policy.std` 打成 NaN（模拟「update 内部把 σ 打成 NaN」的崩点），
    返回 loss_dict（可含 NaN）。storage 用 SimpleNamespace 复刻 rsl_rl RolloutStorage 的属性名。
    """

    def __init__(self, *, adv=None, ret=None, loss=None, nan_std_in_update: bool = False):
        self.policy = _MockActorCriticStd(init_noise_std=1.0)
        self.storage = SimpleNamespace(
            observations=None,
            advantages=adv,
            returns=ret,
            values=torch.tensor([0.5], dtype=torch.float32),
        )
        self.learning_rate = 3.0e-4
        self._loss = loss
        self._nan_std = nan_std_in_update

    def update(self):
        if self._nan_std:
            with torch.no_grad():
                self.policy.std.fill_(float("nan"))
        return {"loss": torch.tensor([self._loss], dtype=torch.float32), "surrogate": torch.tensor(1.0)}


def selftest_nan_probe_advantage_pre_mid(seed: int = 0) -> dict:
    """第四层新增断言 ①：advantage NaN 注入 -> `pre_mid` stage 的 nan_probe 行触发。

    复刻假说「首 NaN 在 advantage（GAE）而非 loss/backward」：storage.advantages 注入 NaN，
    断言探针在 update **之前**（stage=pre_mid）就报出 `storage.advantages`，且 loss 全有限时
    **不**误报 loss——即探针能区分「污染在 advantage」还是「污染在 loss」。
    """
    import tempfile  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        alg = _MockAlgLayer4(adv=torch.tensor([float("nan"), 1.0], dtype=torch.float32), loss=0.7)
        install_std_trajectory_instrument(alg, out_dir, every=1, phase="selftest")
        alg.update()
        rows = [json.loads(ln) for ln in
                (out_dir / STD_TRAJECTORY_LOG).read_text(encoding="utf-8").strip().splitlines()]
        probes = [r for r in rows if r.get("tag") == NAN_PROBE_TAG]
        _check(len(probes) == 1, f"advantage NaN 应触发恰好 1 条 nan_probe，实为 {len(probes)}")
        p = probes[0]
        _check(p["stage"] == "pre_mid", f"advantage NaN 应在 pre_mid 触发，实为 stage={p['stage']!r}")
        _check(p["name"] == "storage.advantages", f"pre_mid 探针 name 应为 storage.advantages，实为 {p['name']!r}")
        _check(p["n_nonfinite"] == 1 and p["n_total"] == 2, f"advantage 计数错：{p}")
        _check(isinstance(p["first"], float) and math.isnan(p["first"]), f"advantage first 非 NaN：{p['first']!r}")
        _check(p["first_index"] == 0, f"advantage first_index 应为 0，实为 {p['first_index']!r}")
        # 反例：advantage 全有限 -> 无 pre_mid 行（断言有区分度）
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        alg = _MockAlgLayer4(adv=torch.tensor([1.0, 2.0], dtype=torch.float32), loss=0.7)
        install_std_trajectory_instrument(alg, out_dir, every=1, phase="selftest")
        alg.update()
        rows = [json.loads(ln) for ln in
                (out_dir / STD_TRAJECTORY_LOG).read_text(encoding="utf-8").strip().splitlines()]
        _check(not any(r.get("tag") == NAN_PROBE_TAG and r.get("stage") == "pre_mid" for r in rows),
               "advantage 全有限时不应产生 pre_mid nan_probe 行")
    return {"adv_nan_pre_mid": True, "adv_name_ok": True, "finite_adv_no_probe": True}


def selftest_nan_probe_mag_loss_fields(seed: int = 0) -> dict:
    """第四层新增断言 ②：loss 幅值字段格式（`nan_probe_mag` 行，逐 key min/max/abs_max/mean）。

    断言 loss_dict 每个 key 都落一条幅值行，字段齐全且值正确；NaN key 的有限元素统计正确
    （min/max/abs_max/mean 只算有限元素）；并断言 `update.aux` 行含 kl_mean/kl_source/
    learning_rate_pre/learning_rate（kl 在当前 rsl_rl 版本恒 absent）。
    """
    import tempfile  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        alg = _MockAlgLayer4(adv=torch.tensor([1.0], dtype=torch.float32), loss=0.7)
        # 让 update 返回含 NaN 与有限元素混合的 loss_dict（便于验证有限元素统计口径）
        def _upd():
            return {
                "loss": torch.tensor([float("nan"), -2.0, 4.0], dtype=torch.float32),
                "surrogate": torch.tensor([0.5, 1.5], dtype=torch.float32),
            }
        alg.update = _upd
        install_std_trajectory_instrument(alg, out_dir, every=1, phase="selftest")
        alg.update()
        rows = [json.loads(ln) for ln in
                (out_dir / STD_TRAJECTORY_LOG).read_text(encoding="utf-8").strip().splitlines()]
        mags = [r for r in rows if r.get("tag") == NAN_PROBE_MAG_TAG and r.get("stage") == "post"]
        by_name = {r["name"]: r for r in mags}
        for key in ("loss.loss", "loss.surrogate"):
            _check(key in by_name, f"post 幅值行缺 {key}：{[r['name'] for r in mags]}")
            for field in ("name", "shape", "dtype", "n_total", "n_nonfinite", "min", "max",
                          "abs_max", "mean", "stage"):
                _check(field in by_name[key], f"{key} 幅值行缺字段 {field!r}")
        loss_row = by_name["loss.loss"]
        _check(loss_row["n_nonfinite"] == 1 and loss_row["n_total"] == 3, f"loss.loss 计数错：{loss_row}")
        _check(loss_row["min"] == -2.0 and loss_row["max"] == 4.0, f"loss.loss 有限元素 min/max 错：{loss_row}")
        _check(loss_row["abs_max"] == 4.0, f"loss.loss abs_max 应为 4.0：{loss_row}")
        _check(abs(loss_row["mean"] - 1.0) < 1e-6, f"loss.loss 有限元素 mean 应为 1.0：{loss_row}")
        sur_row = by_name["loss.surrogate"]
        _check(sur_row["n_nonfinite"] == 0 and sur_row["min"] == 0.5 and sur_row["max"] == 1.5,
               f"surrogate 幅值错：{sur_row}")
        # update.aux 行（kl_mean/kl_source/lr 前后）
        aux = [r for r in mags if r.get("name") == "update.aux"]
        _check(len(aux) == 1, f"update.aux 行应恰好 1 条，实为 {len(aux)}")
        a = aux[0]
        for field in ("kl_mean", "kl_source", "learning_rate_pre", "learning_rate"):
            _check(field in a, f"update.aux 缺字段 {field!r}：{a}")
        _check(a["kl_source"] == "absent", f"当前 rsl_rl 版本 kl_mean 应 absent，实为 {a['kl_source']!r}")
        _check(a["learning_rate"] == 3.0e-4, f"update.aux 应记到 learning_rate，实为 {a['learning_rate']!r}")
    return {"mag_loss_fields_ok": True, "finite_elems_stats_ok": True, "update_aux_ok": True}


def selftest_nan_probe_param_snapshot_flip(seed: int = 0) -> dict:
    """第四层新增断言 ③：参数快照对比（模拟 update 内 std 变 NaN -> post 快照捕获翻转）。

    复刻假说「某中间量 NaN -> loss NaN -> 反向染全部梯度 -> 参数全灭」：mock update 内部把
    `policy.std` 打成 NaN，断言 `param_snapshot` 行的 pre/post 快照捕获 `std` 从有限翻转为
    非有限（`param_nan_flipped` 含 "std"），且 pre 快照有限、post 快照 n_nonfinite 正确。
    另断言健康 update（std 不变 NaN）时 `param_nan_flipped` 为空（断言有区分度）。
    """
    import tempfile  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        alg = _MockAlgLayer4(adv=torch.tensor([1.0], dtype=torch.float32), loss=0.7,
                             nan_std_in_update=True)
        install_std_trajectory_instrument(alg, out_dir, every=1, phase="selftest")
        alg.update()
        rows = [json.loads(ln) for ln in
                (out_dir / STD_TRAJECTORY_LOG).read_text(encoding="utf-8").strip().splitlines()]
        snaps = [r for r in rows if r.get("name") == "param_snapshot"]
        _check(len(snaps) == 1, f"param_snapshot 行应恰好 1 条，实为 {len(snaps)}")
        s = snaps[0]
        _check("param_pre" in s and "param_post" in s, f"param_snapshot 缺 pre/post：{list(s)}")
        _check("std" in s["param_pre"] and "std" in s["param_post"], f"快照缺 std key：{s}")
        _check(s["param_pre"]["std"]["isfinite"] is True, f"pre std 应有限：{s['param_pre']['std']}")
        _check(s["param_post"]["std"]["isfinite"] is False, f"post std 应变非有限：{s['param_post']['std']}")
        _check(s["param_post"]["std"]["n_nonfinite"] == ACTION_DIM,
               f"post std 非有限个数应 == {ACTION_DIM}：{s['param_post']['std']}")
        _check("std" in s["param_nan_flipped"], f"param_nan_flipped 应含 std：{s['param_nan_flipped']}")
        _check(s["param_nan_flipped_n"] >= 1, f"param_nan_flipped_n 应 >=1：{s['param_nan_flipped_n']}")
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        alg = _MockAlgLayer4(adv=torch.tensor([1.0], dtype=torch.float32), loss=0.7,
                             nan_std_in_update=False)
        install_std_trajectory_instrument(alg, out_dir, every=1, phase="selftest")
        alg.update()
        rows = [json.loads(ln) for ln in
                (out_dir / STD_TRAJECTORY_LOG).read_text(encoding="utf-8").strip().splitlines()]
        s = [r for r in rows if r.get("name") == "param_snapshot"][0]
        _check(s["param_nan_flipped_n"] == 0, f"健康 update 不应有参数翻转：{s['param_nan_flipped']}")
    return {"param_snapshot_flip_captured": True, "pre_finite_post_nan": True, "healthy_no_flip": True}


# ------------------------------------------------- 第五层：消毒器 + critic 探针 selftest
def selftest_advantage_sanitizer(seed: int = 0) -> dict:
    """第五/六层新增断言：advantage/returns/values 消毒器（非有限置 0 + 写回行 + 健康零写回）。

    五路：
      (a) 注入 1 个 NaN advantage -> 该元素置 0、其余逐位不变、落 nan_probe（pre_mid）+
          nan_probe_sanitize 行（n_zeroed=1、purged_full=False）；
      (b) advantage 全 NaN -> 全置 0、purged_full=True、returns 不受影响；
      (c) 健康路径 -> advantages/returns/values **三张量逐位不变且对象引用不变**（零写回）、
          **无** sanitize 行（发射解锁件只在真出 NaN 时动作）；
      (d) **第六层：inference tensor 场景** —— 在 `torch.inference_mode()` 下造含 NaN 的
          advantages（复刻 rollout 产物）→ 消毒成功（重赋值后 storage 属性为**普通 tensor**
          且无 NaN）；并断言**旧就地路径**（`masked_fill_`）在此场景确实抛错以证明修复必要性；
      (e) **values 纳入消毒** —— 注入 NaN values -> 被置 0、落 sanitize 行（name=storage.values）。
    """
    import tempfile  # noqa: PLC0415

    # --- (a) 单 NaN advantage ---
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        alg = _MockAlgLayer4(adv=torch.tensor([float("nan"), 1.0, 2.0], dtype=torch.float32),
                             ret=torch.tensor([3.0, 4.0, 5.0], dtype=torch.float32), loss=0.7)
        ret_before = alg.storage.returns.clone()
        install_std_trajectory_instrument(alg, out_dir, every=1, phase="selftest")
        ret = alg.update()
        _check(isinstance(ret, dict) and "loss" in ret, "消毒器改写了 update 返回值")
        adv = alg.storage.advantages
        _check(adv[0].item() == 0.0, f"NaN advantage 应置 0，实为 {adv[0].item()!r}")
        _check(adv[1].item() == 1.0 and adv[2].item() == 2.0, f"非 NaN 元素被误改：{adv.tolist()}")
        _check(bool(torch.isfinite(adv).all()), f"消毒后 advantage 仍含非有限值：{adv.tolist()}")
        _check(not adv.is_inference(), "消毒后 advantage 仍标记为 inference tensor")
        _check(torch.equal(alg.storage.returns, ret_before), "returns 有限却被改写（应零写回）")
        rows = [json.loads(ln) for ln in
                (out_dir / STD_TRAJECTORY_LOG).read_text(encoding="utf-8").strip().splitlines()]
        san = [r for r in rows if r.get("tag") == NAN_PROBE_SANITIZE_TAG]
        _check(len(san) == 1, f"应恰好 1 条 sanitize 行，实为 {len(san)}")
        s = san[0]
        for field in ("name", "n_nonfinite", "n_total", "n_zeroed", "purged_full", "abs_max_before",
                      "first", "first_index", "first_coord", "stage"):
            _check(field in s, f"sanitize 行缺字段 {field!r}：{s}")
        _check(s["name"] == "storage.advantages", f"sanitize name 错：{s['name']!r}")
        _check(s["n_nonfinite"] == 1 and s["n_zeroed"] == 1 and s["n_total"] == 3, f"sanitize 计数错：{s}")
        _check(s["purged_full"] is False, f"单 NaN 不应 purged_full：{s}")
        _check(s["stage"] == "pre_mid", f"sanitize stage 应为 pre_mid：{s['stage']!r}")
        _check(isinstance(s["first"], float) and math.isnan(s["first"]), f"sanitize first 非 NaN：{s}")
        probes = [r for r in rows if r.get("tag") == NAN_PROBE_TAG and r.get("stage") == "pre_mid"]
        _check(len(probes) == 1 and probes[0]["name"] == "storage.advantages",
               f"应同时落 1 条 pre_mid nan_probe：{probes}")

    # --- (b) 全 NaN advantage -> purged_full ---
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        alg = _MockAlgLayer4(adv=torch.full((4,), float("nan"), dtype=torch.float32),
                             ret=torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32), loss=0.7)
        install_std_trajectory_instrument(alg, out_dir, every=1, phase="selftest")
        alg.update()
        adv = alg.storage.advantages
        _check(bool((adv == 0).all()), f"全 NaN advantage 应全置 0：{adv.tolist()}")
        rows = [json.loads(ln) for ln in
                (out_dir / STD_TRAJECTORY_LOG).read_text(encoding="utf-8").strip().splitlines()]
        s = [r for r in rows if r.get("tag") == NAN_PROBE_SANITIZE_TAG][0]
        _check(s["purged_full"] is True, f"全 NaN 应 purged_full=True：{s}")
        _check(s["n_zeroed"] == 4 and s["n_total"] == 4, f"全 NaN 计数错：{s}")

    # --- (c) 健康路径：三张量零写回（逐位 + 对象引用不变）、无 sanitize 行 ---
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        adv = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32)
        ret = torch.tensor([0.5, 0.25, 0.125], dtype=torch.float32)
        val = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        alg = _MockAlgLayer4(adv=adv.clone(), ret=ret.clone(), loss=0.7)
        alg.storage.values = val.clone()
        adv_obj, ret_obj, val_obj = alg.storage.advantages, alg.storage.returns, alg.storage.values
        install_std_trajectory_instrument(alg, out_dir, every=1, phase="selftest")
        alg.update()
        _check(torch.equal(alg.storage.advantages, adv), "健康 advantage 被改写（应逐位零写回）")
        _check(torch.equal(alg.storage.returns, ret), "健康 returns 被改写（应逐位零写回）")
        _check(torch.equal(alg.storage.values, val), "健康 values 被改写（应逐位零写回）")
        # 零写回 = 连对象引用都不变（重赋值路径在健康路径完全不触发）
        _check(alg.storage.advantages is adv_obj, "健康 advantage 对象引用被换（应零写回）")
        _check(alg.storage.returns is ret_obj, "健康 returns 对象引用被换（应零写回）")
        _check(alg.storage.values is val_obj, "健康 values 对象引用被换（应零写回）")
        rows = [json.loads(ln) for ln in
                (out_dir / STD_TRAJECTORY_LOG).read_text(encoding="utf-8").strip().splitlines()]
        _check(not any(r.get("tag") == NAN_PROBE_SANITIZE_TAG for r in rows),
               "健康路径不应有 sanitize 行")

    # --- (d) 第六层：inference tensor（rollout 产物）消毒 ---
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        with torch.inference_mode():   # 复刻 on_policy_runner.py:201 的 rollout 上下文
            adv_inf = torch.tensor([float("nan"), 1.0, 2.0], dtype=torch.float32)
        _check(adv_inf.is_inference(), "构造的 advantage 未标记为 inference tensor（场景不成立）")
        # 必要性证明：旧就地路径对 inference tensor 抛异常（注意 clone() 会清掉 inference 标记，
        # 必须直接对 inference tensor 就地写，才复刻旧 `val.masked_fill_` 的真实行为）
        with torch.inference_mode():
            legacy_probe = torch.tensor([float("nan"), 1.0, 2.0], dtype=torch.float32)
        legacy_raised = False
        try:
            legacy_probe.masked_fill_(torch.tensor([True, False, False]), 0.0)
        except RuntimeError as exc:
            legacy_raised = "InferenceMode" in str(exc)
        _check(legacy_raised, "旧就地 masked_fill_ 未对 inference tensor 抛错（修复必要性不成立）")
        alg = _MockAlgLayer4(adv=adv_inf, ret=torch.tensor([3.0, 4.0, 5.0], dtype=torch.float32), loss=0.7)
        install_std_trajectory_instrument(alg, out_dir, every=1, phase="selftest")
        alg.update()
        adv = alg.storage.advantages
        _check(not adv.is_inference(), "消毒后 advantage 仍为 inference tensor（重赋值未生效）")
        _check(bool(torch.isfinite(adv).all()), f"inference tensor 消毒后仍含非有限值：{adv.tolist()}")
        _check(adv[0].item() == 0.0 and adv[1].item() == 1.0 and adv[2].item() == 2.0,
               f"inference tensor 消毒值错：{adv.tolist()}")
        rows = [json.loads(ln) for ln in
                (out_dir / STD_TRAJECTORY_LOG).read_text(encoding="utf-8").strip().splitlines()]
        san = [r for r in rows if r.get("tag") == NAN_PROBE_SANITIZE_TAG]
        _check(len(san) == 1 and san[0]["name"] == "storage.advantages" and san[0]["n_zeroed"] == 1,
               f"inference tensor 场景应落 1 条 advantage sanitize 行：{san}")

    # --- (e) values 纳入消毒（第六层）---
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        alg = _MockAlgLayer4(adv=torch.tensor([1.0, 2.0], dtype=torch.float32),
                             ret=torch.tensor([1.0, 2.0], dtype=torch.float32), loss=0.7)
        alg.storage.values = torch.tensor([float("nan"), 0.5], dtype=torch.float32)
        install_std_trajectory_instrument(alg, out_dir, every=1, phase="selftest")
        alg.update()
        val = alg.storage.values
        _check(val[0].item() == 0.0 and val[1].item() == 0.5, f"NaN values 未置 0：{val.tolist()}")
        rows = [json.loads(ln) for ln in
                (out_dir / STD_TRAJECTORY_LOG).read_text(encoding="utf-8").strip().splitlines()]
        san = [r for r in rows if r.get("tag") == NAN_PROBE_SANITIZE_TAG]
        _check(len(san) == 1 and san[0]["name"] == "storage.values" and san[0]["n_zeroed"] == 1,
               f"values 应落 1 条 sanitize 行：{san}")
    return {
        "single_nan_zeroed": True,
        "full_purged": True,
        "healthy_bitwise_untouched": True,
        "inference_tensor_sanitized": True,
        "legacy_inplace_raises": True,
        "values_sanitized": True,
    }


def selftest_sanitize_obs_end(seed: int = 0) -> dict:
    """第七层新增断言：消毒器上移到观测端（五路 + 大张量 GPU 侧 + dict observations）。

    四路：
      (a) **大张量场景（超阈值）** —— 把 `NAN_PROBE_MAX_TENSOR_ELEMS` 临时压到 4，造一个
          numel=64（> 阈值）的含 NaN obs 张量：断言消毒器**不跳过**、该张量被消毒（NaN 置 0、
          其余逐位不变），且落 `nan_probe_sanitize` 行；同时断言**探针**（`_collect_*`）对超阈值
          张量仍**跳过**（无对应 `obs` 的 nan_probe 行）——证明「消毒不跳、探针仍跳」两条口径分离。
      (b) **dict 型 observations 逐 key 消毒** —— observations={"policy": 含 NaN, "critic": 有限}：
          断言 policy 被消毒、critic **对象引用不变**（未命中零写回），sanitize 行 name=
          `storage.observations.policy`。
      (c) **五 attr 全覆盖** —— 五个 attr 各注入 NaN，断言五条 sanitize 行 name 齐全。
      (d) **健康零写不变** —— 五路全有限：无 sanitize 行、五路对象引用全部不变（零写回）。
    """
    import tempfile  # noqa: PLC0415

    # --- (c) 五 attr 全覆盖 ---
    _check(
        _SANITIZE_STORAGE_ATTRS
        == ("advantages", "returns", "values", "privileged_observations", "observations"),
        f"消毒 attr 列表应恰为五路，实为 {_SANITIZE_STORAGE_ATTRS}",
    )
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        alg = _MockAlgLayer4(adv=torch.tensor([float("nan"), 1.0], dtype=torch.float32),
                             ret=torch.tensor([float("nan"), 1.0], dtype=torch.float32), loss=0.7)
        alg.storage.values = torch.tensor([float("nan"), 1.0], dtype=torch.float32)
        alg.storage.privileged_observations = torch.tensor([[float("nan"), 1.0]], dtype=torch.float32)
        alg.storage.observations = {"policy": torch.tensor([float("nan"), 1.0], dtype=torch.float32)}
        install_std_trajectory_instrument(alg, out_dir, every=1, phase="selftest")
        alg.update()
        rows = [json.loads(ln) for ln in
                (out_dir / STD_TRAJECTORY_LOG).read_text(encoding="utf-8").strip().splitlines()]
        names = {r["name"] for r in rows if r.get("tag") == NAN_PROBE_SANITIZE_TAG}
        for expected in ("storage.advantages", "storage.returns", "storage.values",
                         "storage.privileged_observations", "storage.observations.policy"):
            _check(expected in names, f"五路消毒缺 {expected!r}：{sorted(names)}")
        _check(bool(torch.isfinite(alg.storage.privileged_observations).all()),
               "privileged_observations 消毒后仍含非有限值")
        _check(bool(torch.isfinite(alg.storage.observations["policy"]).all()),
               "observations.policy 消毒后仍含非有限值")

    # --- (a) 大张量（超阈值）不跳过（GPU 侧路径）+ 探针仍跳过 ---
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        big_n = 64
        big = torch.ones(big_n, dtype=torch.float32)
        big[3] = float("nan")
        big = big.reshape(8, 8)   # 2 维：使 first_coord 有意义（flat idx 3 -> [0, 3]）
        alg = _MockAlgLayer4(adv=torch.tensor([1.0], dtype=torch.float32),
                             ret=torch.tensor([1.0], dtype=torch.float32), loss=0.7)
        alg.storage.observations = {"policy": big}
        _check(big.numel() > 4, "大张量夹具未超过压低的阈值（场景不成立）")
        saved = globals()["NAN_PROBE_MAX_TENSOR_ELEMS"]
        globals()["NAN_PROBE_MAX_TENSOR_ELEMS"] = 4   # 压低阈值，使 64 元素张量成为「超大张量」
        try:
            install_std_trajectory_instrument(alg, out_dir, every=1, phase="selftest")
            alg.update()
        finally:
            globals()["NAN_PROBE_MAX_TENSOR_ELEMS"] = saved
        rows = [json.loads(ln) for ln in
                (out_dir / STD_TRAJECTORY_LOG).read_text(encoding="utf-8").strip().splitlines()]
        san = [r for r in rows if r.get("tag") == NAN_PROBE_SANITIZE_TAG]
        _check(len(san) == 1 and san[0]["name"] == "storage.observations.policy",
               f"超阈值张量应被消毒（1 条 sanitize 行），实为 {san}")
        _check(san[0]["n_nonfinite"] == 1 and san[0]["first_index"] == 3,
               f"超阈值张量消毒计数/首索引错：{san[0]}")
        _check(san[0]["first_coord"] == [0, 3], f"超阈值张量 first_coord 错：{san[0]}")
        cleaned = alg.storage.observations["policy"]
        _check(bool(torch.isfinite(cleaned).all()), f"超阈值张量消毒后仍含非有限值：{cleaned.tolist()}")
        _check(cleaned[0, 3].item() == 0.0 and cleaned[0, 0].item() == 1.0,
               f"超阈值张量消毒值错：{cleaned.tolist()}")
        # 探针口径不变：`_collect_*` 对超阈值张量仍跳过（无 obs.* 的 nan_probe 行）
        _check(not any(r.get("tag") == NAN_PROBE_TAG and str(r.get("name", "")).startswith("obs.")
                       for r in rows),
               "探针本应跳过超阈值张量，却落了 obs.* nan_probe 行（探针跳过口径被破坏）")

    # --- (b) dict 型 observations 逐 key 消毒（未命中 key 零写回）---
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        pol = torch.tensor([float("nan"), 2.0], dtype=torch.float32)
        cri = torch.tensor([0.5, 0.25], dtype=torch.float32)
        alg = _MockAlgLayer4(adv=torch.tensor([1.0], dtype=torch.float32),
                             ret=torch.tensor([1.0], dtype=torch.float32), loss=0.7)
        alg.storage.observations = {"policy": pol, "critic": cri}
        cri_obj = cri
        install_std_trajectory_instrument(alg, out_dir, every=1, phase="selftest")
        alg.update()
        obs = alg.storage.observations
        _check(obs["policy"][0].item() == 0.0 and obs["policy"][1].item() == 2.0,
               f"dict observations 的 policy key 未消毒：{obs['policy'].tolist()}")
        _check(obs["critic"] is cri_obj, "dict observations 的有限 key 对象引用被换（应零写回）")
        rows = [json.loads(ln) for ln in
                (out_dir / STD_TRAJECTORY_LOG).read_text(encoding="utf-8").strip().splitlines()]
        san = [r for r in rows if r.get("tag") == NAN_PROBE_SANITIZE_TAG]
        _check(len(san) == 1 and san[0]["name"] == "storage.observations.policy",
               f"dict observations 应落 1 条 policy sanitize 行：{san}")

    # --- (d) 健康零写不变（五路对象引用全不变、无 sanitize 行）---
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        alg = _MockAlgLayer4(adv=torch.tensor([1.0], dtype=torch.float32),
                             ret=torch.tensor([1.0], dtype=torch.float32), loss=0.7)
        alg.storage.values = torch.tensor([0.1], dtype=torch.float32)
        alg.storage.privileged_observations = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
        alg.storage.observations = {"policy": torch.tensor([0.3], dtype=torch.float32),
                                    "critic": torch.tensor([0.4], dtype=torch.float32)}
        refs = {
            "advantages": alg.storage.advantages, "returns": alg.storage.returns,
            "values": alg.storage.values,
            "privileged_observations": alg.storage.privileged_observations,
            "observations": alg.storage.observations,
        }
        install_std_trajectory_instrument(alg, out_dir, every=1, phase="selftest")
        alg.update()
        for attr, obj in refs.items():
            _check(getattr(alg.storage, attr) is obj,
                   f"健康路径 {attr} 对象引用被换（应零写回）")
        rows = [json.loads(ln) for ln in
                (out_dir / STD_TRAJECTORY_LOG).read_text(encoding="utf-8").strip().splitlines()]
        _check(not any(r.get("tag") == NAN_PROBE_SANITIZE_TAG for r in rows),
               "健康路径不应有 sanitize 行")
    return {
        "five_attrs_covered": True,
        "oversized_tensor_sanitized": True,
        "probe_still_skips_oversized": True,
        "dict_observations_per_key": True,
        "healthy_zero_write": True,
    }


# ------------------------------------------------- rollout 期观测源消毒器 selftest（第八层）
class _MockObsEnv:
    """第八层 selftest 用的最小 env 桩（纯 Layer 1，零 isaaclab）。

    step/reset 返回 obs dict（policy/critic 两组张量，可注入非有限值）；step/reset 的调用与
    参数记进 `calls`（透传断言用）；`extra_attr`/`echo` 是供透传断言用的非 step/reset 成员。
    注意 `__class__` 重分类要求实例有 `__dict__`（普通类满足；Isaac env 亦然）。
    """

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self._step_i = 0
        self.reset_obs: dict = {}
        self.step_obs: dict = {}
        self.extra_attr = 42

    def step(self, action):
        self.calls.append(("step", action))
        self._step_i += 1
        return dict(self.step_obs), float(self._step_i), False, False, {"cost": 0.5}

    def reset(self, seed=None, env_ids=None):
        self.calls.append(("reset", seed, env_ids))
        return dict(self.reset_obs), {"seed": seed}

    def echo(self, x):
        return ("echo", x)


def selftest_obs_sanitize_dirty(seed: int = 0) -> dict:
    """第八层断言 ①：mock env 出 NaN/±inf obs -> 包装后全有限 + obs_sanitize 计数行 + 首例坐标。

    场景：reset 与 step 的 obs 均含非有限值（policy 3x3 含 NaN/+inf/-inf 各一、critic 1 个 NaN）。
    断言：返回 obs 全有限且逐元素映射正确（NaN->0、±inf->±1e4、有限值逐位不变）；
    std_trajectory.log 落 tag=obs_sanitize 行（逐 key：call/step/n_nonfinite/n_nan/n_posinf/
    n_neginf/n_total/cum/first/first_index/first_coord，首例坐标 [0,1]）；第二次命中 cum 逐 key
    累计；step 返回非 obs 位原样。
    """
    import tempfile  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        env = _MockObsEnv()
        dirty_p = torch.tensor([[1.0, float("nan"), 3.0],
                                [4.0, float("inf"), 6.0],
                                [float("-inf"), 8.0, 9.0]], dtype=torch.float32)
        dirty_c = torch.tensor([0.25, float("nan")], dtype=torch.float32)
        env.reset_obs = {"policy": dirty_p, "critic": dirty_c}
        env.step_obs = {"policy": dirty_p.clone(), "critic": dirty_c.clone()}
        base_cls = type(env)
        info = install_obs_sanitize_wrapper(env, torch, out_dir / STD_TRAJECTORY_LOG)
        _check(info["installed"] is True, f"包装应成功：{info}")
        _check(isinstance(env, base_cls), "包装后 isinstance（原类）兼容性丢失")
        obs, extras = env.reset(seed=7, env_ids=[0, 1])
        _check(extras == {"seed": 7}, f"reset extras 被改：{extras}")
        p = obs["policy"]
        _check(bool(torch.isfinite(p).all()), f"消毒后 policy obs 仍含非有限值：{p.tolist()}")
        _check(p[0, 1].item() == 0.0, f"NaN 应置 0：{p[0, 1].item()!r}")
        _check(p[1, 1].item() == OBS_SANITIZE_INF_CLIP, f"+inf 应映射 +1e4：{p[1, 1].item()!r}")
        _check(p[2, 0].item() == -OBS_SANITIZE_INF_CLIP, f"-inf 应映射 -1e4：{p[2, 0].item()!r}")
        _check(p[0, 0].item() == 1.0 and p[2, 2].item() == 9.0, "有限元素被误改（应逐位不变）")
        c = obs["critic"]
        _check(bool(torch.isfinite(c).all()) and c[1].item() == 0.0 and c[0].item() == 0.25,
               f"critic obs 消毒错：{c.tolist()}")
        rows = [json.loads(ln) for ln in
                (out_dir / STD_TRAJECTORY_LOG).read_text(encoding="utf-8").splitlines()]
        san = [r for r in rows if r.get("tag") == OBS_SANITIZE_TAG]
        _check(len(san) == 2, f"reset 两 key 命中应各落 1 行，实为 {len(san)}：{san}")
        by_key = {r["key"]: r for r in san}
        _check(set(by_key) == {"policy", "critic"}, f"计数行 key 应为 policy/critic：{sorted(by_key)}")
        rp = by_key["policy"]
        for field in ("call", "step", "n_nonfinite", "n_total", "n_nan", "n_posinf", "n_neginf",
                      "cum", "first", "first_index", "first_coord"):
            _check(field in rp, f"obs_sanitize 行缺字段 {field!r}：{rp}")
        _check(rp["call"] == "reset", f"首行 call 应为 reset：{rp['call']!r}")
        _check(rp["n_nonfinite"] == 3 and rp["n_total"] == 9, f"policy 计数错：{rp}")
        _check(rp["n_nan"] == 1 and rp["n_posinf"] == 1 and rp["n_neginf"] == 1,
               f"nan/inf 分计数错：{rp}")
        _check(rp["cum"] == 3, f"首拍 cum 应为 3：{rp['cum']}")
        _check(rp["first_index"] == 1 and math.isnan(rp["first"]), f"首例应为 flat idx 1 的 NaN：{rp}")
        _check(rp["first_coord"] == [0, 1], f"首例坐标应为 [0, 1]：{rp['first_coord']}")
        rc = by_key["critic"]
        _check(rc["n_nonfinite"] == 1 and rc["n_total"] == 2 and rc["cum"] == 1
               and rc["first_coord"] is None, f"critic 计数行错（1 维张量无坐标）：{rc}")
        # 第二次调用（step）：计数行继续落、cum 逐 key 累计、step 返回非 obs 位原样
        out2 = env.step(torch.zeros(2))
        _check(len(out2) == 5 and isinstance(out2[0], dict), f"step 返回结构被改：{type(out2)}")
        _check(bool(torch.isfinite(out2[0]["policy"]).all()), "step obs 未消毒")
        _check(out2[1] == 1.0 and out2[2] is False and out2[3] is False and out2[4] == {"cost": 0.5},
               f"step 返回非 obs 位被改：{out2[1:]}")
        rows2 = [json.loads(ln) for ln in
                 (out_dir / STD_TRAJECTORY_LOG).read_text(encoding="utf-8").splitlines()]
        san2 = [r for r in rows2 if r.get("tag") == OBS_SANITIZE_TAG]
        _check(len(san2) == 4, f"step 两 key 应再各落 1 行（共 4），实为 {len(san2)}")
        rp2 = [r for r in san2 if r["key"] == "policy" and r["call"] == "step"][0]
        _check(rp2["cum"] == 6, f"cum 应逐 key 累计（3+3=6）：{rp2}")
        state = env._obs_sanitize_state
        _check(state["cum"]["policy"] == 6 and state["cum"]["critic"] == 2
               and state["hits"] == 4 and state["steps"] == 2,
               f"state 计数错：cum={state['cum']} hits={state['hits']} steps={state['steps']}")
    return {
        "dirty_sanitized_finite": True,
        "count_lines_per_key": True,
        "first_coord_recorded": True,
        "cum_accumulates": True,
        "non_obs_positions_untouched": True,
    }


def selftest_obs_sanitize_healthy_bitwise(seed: int = 0) -> dict:
    """第八层断言 ②：健康 obs 逐位不变 + 零计数（张量对象引用不变、无 obs_sanitize 行）。

    健康路径（全有限）下净化器完全不动作：返回的张量是**原对象**（零写回）、逐位相等，
    日志文件甚至不应被创建（无任何 obs_sanitize 行可写）。
    """
    import tempfile  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        env = _MockObsEnv()
        p = torch.tensor([[1.25, -2.5, 0.0], [3.75, -4.125, 6.5]], dtype=torch.float32)
        c = torch.tensor([0.5, -0.75], dtype=torch.float32)
        env.reset_obs = {"policy": p, "critic": c}
        env.step_obs = {"policy": p.clone(), "critic": c.clone()}
        rp, rc = env.step_obs["policy"], env.step_obs["critic"]
        install_obs_sanitize_wrapper(env, torch, out_dir / STD_TRAJECTORY_LOG)
        obs, _ = env.reset()
        _check(obs["policy"] is p, "健康 reset obs 张量对象引用被换（应零写回）")
        _check(obs["critic"] is c, "健康 reset obs 张量对象引用被换（应零写回）")
        _check(torch.equal(obs["policy"], p) and torch.equal(obs["critic"], c), "健康 reset obs 逐位被改")
        out = env.step(torch.zeros(2))
        _check(out[0]["policy"] is rp and out[0]["critic"] is rc, "健康 step obs 对象引用被换（应零写回）")
        _check(torch.equal(out[0]["policy"], rp), "健康 step obs 逐位被改")
        log_file = out_dir / STD_TRAJECTORY_LOG
        rows = ([json.loads(ln) for ln in log_file.read_text(encoding="utf-8").splitlines()]
                if log_file.exists() else [])
        _check(not any(r.get("tag") == OBS_SANITIZE_TAG for r in rows),
               "健康路径不应有 obs_sanitize 行")
        state = env._obs_sanitize_state
        _check(state["hits"] == 0 and not state["cum"] and state["steps"] == 2,
               f"健康路径计数应全零（steps=2 只计调用数）：{state}")
    return {
        "healthy_bitwise_untouched": True,
        "zero_object_swap": True,
        "zero_count_lines": True,
    }


def selftest_obs_sanitize_passthrough(seed: int = 0) -> dict:
    """第八层断言 ③：透传性——isinstance/属性访问/普通方法/step-reset 参数与返回其余位原样 + 幂等。

    重分类包装后：`isinstance(env, 原类)` 恒真、非 step/reset 的属性与方法原生可用、
    step/reset 的参数按原对象透传到内层实现、返回值除 obs 位外原样；重复安装幂等跳过。
    """
    import tempfile  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        env = _MockObsEnv()
        env.reset_obs = {"policy": torch.zeros(1)}
        env.step_obs = {"policy": torch.zeros(1)}
        base_cls = type(env)
        info = install_obs_sanitize_wrapper(env, torch, out_dir / STD_TRAJECTORY_LOG)
        _check(info["installed"] is True and type(env) is not base_cls,
               f"应已重分类到动态子类：{info}")
        _check(isinstance(env, base_cls), "isinstance（原类）应为真（动态子类关系）")
        _check(env.extra_attr == 42, "未知属性访问未透传（原生继承失败）")
        _check(env.echo(3) == ("echo", 3), "普通方法调用未透传")
        env.reset(seed=11, env_ids=[2])
        _check(any(call[0] == "reset" and call[1] == 11 and call[2] == [2] for call in env.calls),
               f"reset 参数未透传：{env.calls}")
        action = torch.tensor([0.5, -0.5])
        out = env.step(action)
        _check(any(call[0] == "step" and call[1] is action for call in env.calls),
               f"step 参数未透传（action 应原对象）：{env.calls}")
        _check(len(out) == 5 and out[1] == 1.0 and out[2] is False and out[3] is False
               and out[4] == {"cost": 0.5}, f"step 返回非 obs 位被改：{out}")
        # 幂等：重复安装不再二次包装
        info2 = install_obs_sanitize_wrapper(env, torch, out_dir / "another.log")
        _check(info2.get("already") is True and info2.get("installed") is False,
               f"重复安装应幂等跳过：{info2}")
    return {
        "isinstance_compat": True,
        "attr_and_method_passthrough": True,
        "args_and_rest_of_return_untouched": True,
        "reinstall_idempotent": True,
    }


def selftest_critic_forward_probe(seed: int = 0) -> dict:
    """第五层新增断言 ②：critic 前向探针（mock evaluate 出 NaN -> 记录行含 critic_obs 幅值）。

    复刻已定案根因链的第一环（critic values 出 1 个 NaN）：mock policy.evaluate 返回含 NaN 的
    values，断言落 `tag=nan_probe` 行（stage=critic_forward，name=critic.values，含 first_coord）
    且该次 critic_obs 逐 key（含 height_scan）落 `nan_probe_mag` 幅值行；反例：values 全有限时
    无 critic_forward 行。**探针不改 evaluate 返回值**（原样透传）。
    """
    import tempfile  # noqa: PLC0415

    class _MockPolicyCritic(_MockActorCriticStd):
        """带只读 evaluate 的 mock policy（返回 NaN 与否可控）。"""

        def __init__(self, nan_values: bool):
            super().__init__(init_noise_std=1.0)
            self._nan_values = nan_values

        def evaluate(self, critic_observations):
            base = torch.tensor([[0.5, 0.25]], dtype=torch.float32)
            if self._nan_values:
                base[0, 1] = float("nan")
            return base

    class _MockAlgCritic:
        def __init__(self, nan_values: bool):
            self.policy = _MockPolicyCritic(nan_values)
            self.storage = SimpleNamespace(
                observations=None,
                advantages=torch.tensor([1.0], dtype=torch.float32),
                returns=torch.tensor([1.0], dtype=torch.float32),
                values=torch.tensor([0.5], dtype=torch.float32),
            )

        def update(self):
            return {"loss": torch.tensor([0.5], dtype=torch.float32)}

    critic_obs = {
        "policy": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        "height_scan": torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        alg = _MockAlgCritic(nan_values=True)
        info = install_std_trajectory_instrument(alg, out_dir, every=1, phase="selftest")
        _check(info.get("critic_probe") is True, f"critic 探针应包装成功：{info}")
        vals = alg.policy.evaluate(critic_obs)
        _check(bool(torch.isnan(vals).any()), "evaluate 返回值被探针改写（应原样透传）")
        rows = [json.loads(ln) for ln in
                (out_dir / STD_TRAJECTORY_LOG).read_text(encoding="utf-8").strip().splitlines()]
        cf = [r for r in rows if r.get("tag") == NAN_PROBE_TAG and r.get("stage") == "critic_forward"]
        _check(len(cf) == 1, f"应恰好 1 条 critic_forward nan_probe，实为 {len(cf)}")
        p = cf[0]
        _check(p["name"] == "critic.values", f"critic_forward name 错：{p['name']!r}")
        _check(p["n_nonfinite"] == 1 and p["n_total"] == 2, f"critic values 计数错：{p}")
        _check(p["first_coord"] == [0, 1], f"critic values first_coord 应为 [0,1]：{p['first_coord']!r}")
        mags = {r["name"]: r for r in rows if r.get("tag") == NAN_PROBE_MAG_TAG
                and r.get("stage") == "critic_forward"}
        for key in ("critic_obs.policy", "critic_obs.height_scan"):
            _check(key in mags, f"critic_obs 幅值行缺 {key}：{list(mags)}")
            for field in ("min", "max", "abs_max", "mean", "n_nonfinite", "n_total"):
                _check(field in mags[key], f"{key} 幅值行缺字段 {field!r}")
        hs = mags["critic_obs.height_scan"]
        _check(abs(hs["abs_max"] - 0.3) < 1e-6 and hs["n_nonfinite"] == 0, f"height_scan 幅值错：{hs}")

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        alg = _MockAlgCritic(nan_values=False)
        install_std_trajectory_instrument(alg, out_dir, every=1, phase="selftest")
        alg.update()   # 先跑一次 update 以建日志（有限路径零记录，日志仅含 σ 轨迹行）
        alg.policy.evaluate(critic_obs)
        rows = [json.loads(ln) for ln in
                (out_dir / STD_TRAJECTORY_LOG).read_text(encoding="utf-8").strip().splitlines()]
        _check(not any(r.get("tag") == NAN_PROBE_TAG and r.get("stage") == "critic_forward" for r in rows),
               "values 全有限时不应有 critic_forward 行")
    return {"critic_nan_recorded": True, "critic_obs_mag_recorded": True, "finite_no_record": True}


def selftest_nan_probe_snapshot_guard(seed: int = 0) -> dict:
    """第五层新增断言 ③：快照守卫（mock 快照抛异常 -> update 不受影响）。

    复刻上一轮 reviewer 的 should-fix：`_nan_probe_param_snapshot` 若抛异常，必须被 wrapper
    吞掉（降级空快照），`orig_update` 照常执行并原样返回。断言 update 返回值不变、无异常冒泡、
    日志仍落 param_snapshot 行（空快照也能写）。
    """
    import tempfile  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        alg = _MockAlgLayer4(adv=torch.tensor([1.0], dtype=torch.float32), loss=0.7)
        install_std_trajectory_instrument(alg, out_dir, every=1, phase="selftest")

        def _boom(policy):
            raise RuntimeError("mock snapshot failure")

        orig_fn = globals()["_nan_probe_param_snapshot"]
        globals()["_nan_probe_param_snapshot"] = _boom
        try:
            ret = alg.update()   # 不应抛
        finally:
            globals()["_nan_probe_param_snapshot"] = orig_fn
        _check(isinstance(ret, dict) and "loss" in ret, "快照抛异常时 update 返回值被破坏")
        rows = [json.loads(ln) for ln in
                (out_dir / STD_TRAJECTORY_LOG).read_text(encoding="utf-8").strip().splitlines()]
        snaps = [r for r in rows if r.get("name") == "param_snapshot"]
        _check(len(snaps) == 1, f"快照抛异常时仍应落 1 条 param_snapshot 行：{len(snaps)}")
        _check(snaps[0]["param_pre"] == {} and snaps[0]["param_post"] == {},
               f"降级快照应为空 dict：{snaps[0]}")
    return {"snapshot_exc_guarded": True, "update_unaffected": True, "degraded_snapshot_logged": True}


# --------------------------------------- adapter env 维分块前向 selftest（OOM 修复）
def _load_author_adapter_layer() -> SimpleNamespace:
    """延迟加载 eval_author_v0_decoder 的 Layer 1 纯 torch 件（本机可导入，零 isaaclab）。

    返回 SimpleNamespace(AuthorPolicyAdapter, STATE_DIM)；缺件即 fail loud（真实
    AuthorPolicyAdapter 是分块等价断言的证据主体，mock 替代会削弱证明力）。
    """
    try:
        from isaac.eval_author_v0_decoder import AuthorPolicyAdapter, STATE_DIM
    except ImportError:  # pragma: no cover（本机走此支）
        from apt_g1.isaac.eval_author_v0_decoder import AuthorPolicyAdapter, STATE_DIM  # type: ignore[no-redef]
    return SimpleNamespace(AuthorPolicyAdapter=AuthorPolicyAdapter, STATE_DIM=int(STATE_DIM))


class _MockAuthorTransformer(nn.Module):
    """本机替身 author transformer（满足 AuthorPolicyAdapter 构造/前向契约的最小件）。

    契约（eval_author_v0_decoder.py）：构造期校验 `pos.shape[1] - n_ctx == win`
    （:547-552）；前向 `model(codes, states, codes, ctx) -> (N, t, 64, vocab)`
    （:630-635）。结构 = 码嵌入 + state 投影 + ctx 投影 + 位置表 -> **因果累积均值**
    （模拟因果注意力「末帧输出依赖窗内全部前序帧」）-> vocab 头。窗口内任一帧变化
    或窗口长度 t 变化都会改变末帧 logits——正是「分块窗长钉桩」等价断言需要的性质。
    前向对 batch 维逐行独立（与生产 author transformer 同性质）。
    """

    def __init__(
        self,
        win: int = 200,
        n_ctx: int = 2,
        d_model: int = 24,
        vocab: int = 8,
        token_dim: int = TOKEN_DIM,
        state_dim: int = 36,
        ctx_dim: int = 13,
        seed: int = 3,
    ) -> None:
        super().__init__()
        self.n_ctx = int(n_ctx)
        g = torch.Generator().manual_seed(int(seed))
        self.register_buffer("pos", torch.randn(1, win + n_ctx, d_model, generator=g) * 0.05)
        self.code_emb = nn.Embedding(vocab, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)
        self.ctx_proj = nn.Linear(ctx_dim, d_model)
        self.head = nn.Linear(d_model, vocab)
        with torch.no_grad():
            self.code_emb.weight.normal_(0.0, 0.5, generator=g)
            self.state_proj.weight.normal_(0.0, 0.2, generator=g)
            self.state_proj.bias.zero_()
            self.ctx_proj.weight.normal_(0.0, 0.2, generator=g)
            self.ctx_proj.bias.zero_()
            self.head.weight.normal_(0.0, 0.5, generator=g)
            self.head.bias.normal_(0.0, 0.1, generator=g)

    def forward(self, codes, states, intent, ctx) -> torch.Tensor:
        t = int(codes.shape[1])
        h = (
            self.code_emb(codes)                                        # (N, t, 64, d)
            + self.state_proj(states).unsqueeze(2)                      # (N, t, 1, d)
            + self.ctx_proj(ctx)[:, None, None, :]                      # (N, 1, 1, d)
            + self.pos[:, self.n_ctx : self.n_ctx + t, :].unsqueeze(2)  # (1, t, 1, d)
        )  # (N, t, 64, d)
        # 因果累积均值：末帧依赖窗内全部前序帧（t 变 ⇒ 末帧变）
        w = torch.arange(1, t + 1, dtype=h.dtype, device=h.device).view(1, t, 1, 1)
        h = h.cumsum(dim=1) / w
        return self.head(h)  # (N, t, 64, vocab)


def _build_chunk_test_stack(num_envs: int, chunk_envs: int, seed: int = 3):
    """构建分块等价对照栈：(单 adapter 参考, 分块 wrapper)，mock transformer 权重共享。

    两栈同一份模型实例（无状态前向）+ 同参构造（code_min/vocab/token 统计/alpha/bound），
    初始状态均为全 reset（standing/neutral 填充）⇒ 唯一差异就是「分块与否」。
    """
    lay = _load_author_adapter_layer()
    model = _MockAuthorTransformer(seed=seed)
    token_mean = torch.zeros(TOKEN_DIM)
    token_std = torch.pow(torch.tensor(2.0), (torch.arange(TOKEN_DIM) % 4 - 2).float())
    kw = dict(
        code_min=-4,
        vocab=8,
        token_mean=token_mean,
        token_std=token_std,
        token_alpha=1.0,
        token_bound="none",
        device="cpu",
    )
    ref = lay.AuthorPolicyAdapter(model, num_envs=int(num_envs), **kw)
    sizes = [chunk_envs] * (num_envs // chunk_envs)
    if num_envs % chunk_envs:
        sizes.append(num_envs % chunk_envs)
    wrapped = _ChunkedAuthorAdapter(
        [lay.AuthorPolicyAdapter(model, num_envs=s, **kw) for s in sizes],
        chunk_envs=int(chunk_envs),
        num_envs=int(num_envs),
    )
    return ref, wrapped


def _chunk_stack_check_step(ref, wrapped, state_rows, ctx_vec, tag: str) -> torch.Tensor:
    """两栈同输入各 step 一步，输出与状态（n_valid/codes/states）逐位一致。"""
    a_ref = ref.step(state_rows, ctx=ctx_vec)
    a_wrp = wrapped.step(state_rows, ctx=ctx_vec)
    _check(torch.equal(a_ref, a_wrp), f"{tag}: 分块与全 batch adapter.step 输出非逐位一致")
    nv_wrp = torch.cat([c.n_valid for c in wrapped.chunks])
    _check(torch.equal(nv_wrp, ref.n_valid), f"{tag}: 分块栈 n_valid 与参考不一致（reset 路由/推进错位）")
    codes_wrp = torch.cat([c.codes for c in wrapped.chunks], dim=0)
    _check(torch.equal(codes_wrp, ref.codes), f"{tag}: 分块栈 codes 窗与参考不一致")
    states_wrp = torch.cat([c.states for c in wrapped.chunks], dim=0)
    _check(torch.equal(states_wrp, ref.states), f"{tag}: 分块栈 states 窗与参考不一致")
    return a_ref


def selftest_adapter_chunk_equivalence(seed: int = 0) -> dict:
    """新增断言 ①：adapter env 维分块前向 ≡ 全 batch（逐位，含自回归状态滚动）。

    以**真实 AuthorPolicyAdapter**（eval_author_v0_decoder Layer 1，本机可导入）+
    mock transformer 建两套同源栈：chunk=0（单 adapter num_envs=12，参考）vs chunk=4
    （_ChunkedAuthorAdapter，3 块）。序列：连续 5 步（状态滚动同步）-> **整块部分
    reset**（env 8..11 = 第 2 块全部 env，制造 n_valid 非均匀）-> 再连续 3 步 ->
    全量 reset -> 1 步。n_valid 非均匀段正是 `window_len()` 的 batch max 耦合生效处：
    显式断言「块内 max < 全局 max」证明 naive 分块必然偏离、t_global 钉桩是必要的，
    并以逐位一致证明钉桩后的等价性。
    """
    g = torch.Generator().manual_seed(int(seed))
    ref, wrapped = _build_chunk_test_stack(12, 4, seed=3)
    _check(wrapped.num_envs == 12 and len(wrapped.chunks) == 3, "分块栈结构不符（12 envs / 3 块）")
    ctx_vec = np.linspace(-0.5, 0.5, 13, dtype=np.float32)

    # --- 连续 5 步（含自回归滚动：_last_idx 进下一帧窗）---
    for i in range(5):
        st = torch.randn(12, wrapped.chunks[0].states.shape[-1], generator=g)
        _chunk_stack_check_step(ref, wrapped, st, ctx_vec, f"step{i}")

    # --- 整块部分 reset：env 8..11（第 2 块全部）---
    ids = torch.tensor([8, 9, 10, 11], dtype=torch.long)
    ref.reset(ids)
    wrapped.reset(ids)
    for i in range(3):
        st = torch.randn(12, wrapped.chunks[0].states.shape[-1], generator=g)
        _chunk_stack_check_step(ref, wrapped, st, ctx_vec, f"post_reset_step{i}")
        # 钉桩必要性证据：块内 n_valid max < 全局 max（naive 分块用块内 max ⇒ 非等价；
        # _step_window_len 给出的是下一步推进后的全局应有帧数，含非均匀 n_valid 全局耦合）
        t_global = wrapped._step_window_len()
        chunk2_local = int(wrapped.chunks[2].n_valid.max().item())
        _check(chunk2_local < t_global, f"post_reset_step{i}: 测试构造失效（块内 max 应 < 全局 max）")

    # --- 全量 reset（None 路由）后再走一步 ---
    ref.reset(None)
    wrapped.reset(None)
    st = torch.randn(12, wrapped.chunks[0].states.shape[-1], generator=g)
    _chunk_stack_check_step(ref, wrapped, st, ctx_vec, "post_full_reset_step")
    return {
        "bitwise_equal": True,
        "state_buffers_equal": True,
        "window_pin_load_bearing": True,
        "num_envs": 12,
        "chunk_envs": 4,
    }


def selftest_adapter_chunk_cli_default() -> dict:
    """新增断言 ②：--adapter-chunk-envs 默认 1024 生效于 CLI；0=不分块可显式覆盖。

    （2026-09-24 性能补丁：默认 128 -> 1024，见模块 docstring「adapter 前向精度档」。）
    """
    ap = build_args()
    dflt = vars(ap.parse_args([]))["adapter_chunk_envs"]
    _check(
        dflt == DEFAULT_ADAPTER_CHUNK_ENVS == 1024,
        f"--adapter-chunk-envs 默认值应为 1024，实得 {dflt}",
    )
    off = vars(ap.parse_args(["--adapter-chunk-envs", "0"]))["adapter_chunk_envs"]
    _check(off == 0, "--adapter-chunk-envs 0（不分块）解析失败")
    explicit = vars(ap.parse_args(["--adapter-chunk-envs", "64"]))["adapter_chunk_envs"]
    _check(explicit == 64, "--adapter-chunk-envs 64 解析失败")
    return {"default": int(dflt), "off_override": int(off), "explicit": int(explicit)}


def selftest_adapter_chunk_tail_block(seed: int = 1) -> dict:
    """新增断言 ③：batch 非 chunk 整数倍时尾块处理（10 envs / chunk=4 -> [4,4,2]）。

    覆盖：块划分正确；跨块的**非均匀部分 reset**（env 0=首块、env 9=尾块各一，reset
    路由换算块内局部 id）；尾块（2 envs）在 t_global 钉桩下与全 batch 逐位一致。
    """
    g = torch.Generator().manual_seed(int(seed))
    ref, wrapped = _build_chunk_test_stack(10, 4, seed=3)
    _check(
        [c.num_envs for c in wrapped.chunks] == [4, 4, 2],
        f"尾块划分应为 [4,4,2]，实得 {[c.num_envs for c in wrapped.chunks]}",
    )
    ctx_vec = np.linspace(-0.5, 0.5, 13, dtype=np.float32)
    state_dim = wrapped.chunks[0].states.shape[-1]
    for i in range(4):
        st = torch.randn(10, state_dim, generator=g)
        _chunk_stack_check_step(ref, wrapped, st, ctx_vec, f"tail_step{i}")
    # 跨块非均匀 reset：env 0（首块局部 0）+ env 9（尾块局部 1）
    ids = torch.tensor([0, 9], dtype=torch.long)
    ref.reset(ids)
    wrapped.reset(ids)
    _check(
        wrapped.chunks[2].n_valid.tolist() == [4, 0],
        f"尾块 reset 路由错：n_valid={wrapped.chunks[2].n_valid.tolist()}（应 [4,0]）",
    )
    for i in range(3):
        st = torch.randn(10, state_dim, generator=g)
        _chunk_stack_check_step(ref, wrapped, st, ctx_vec, f"tail_post_reset_step{i}")
    return {"bitwise_equal": True, "chunk_layout": [4, 4, 2], "cross_block_partial_reset": True}


# ------------------------------ adapter 前向精度档 selftest（性能补丁：精度换速度）
def _build_non_tie_adapter_stack():
    """构建「非近平局」mock 双栈：(单 adapter 参考, 分块 wrapper)，同权重实例隔离。

    在 `_MockAuthorTransformer` 上做两步改造，使 argmax 在 bf16 舍入下**可证明不翻转**：
      1) head.weight ×0.01：h 依赖项压到远小于固定赢家间隔的量级；
      2) head.bias = 固定赢家模式（vocab 下标 3 为 +8，其余 -4）⇒ logits 间隔 ~12，
         远大于 bf16 在 |logit|~8 处的 ulp（≈0.0625）⇒ bf16/fp32 argmax 必然一致。
    两栈各自 `_MockAuthorTransformer(seed=3)` 构造（同种子同权重序 ⇒ 权重逐位一致，
    实例隔离，互不共享 model 对象——bf16 档的实例级钉桩不影响 fp32 参考）。
    """
    lay = _load_author_adapter_layer()
    token_mean = torch.zeros(TOKEN_DIM)
    token_std = torch.pow(torch.tensor(2.0), (torch.arange(TOKEN_DIM) % 4 - 2).float())
    kw = dict(
        code_min=-4,
        vocab=8,
        token_mean=token_mean,
        token_std=token_std,
        token_alpha=1.0,
        token_bound="none",
        device="cpu",
    )

    def build_adapter(num_envs: int):
        model = _MockAuthorTransformer(seed=3)
        with torch.no_grad():
            model.head.weight.mul_(0.01)
            bias = torch.full((8,), -4.0)
            bias[3] = 8.0
            model.head.bias.copy_(bias)
        return lay.AuthorPolicyAdapter(model, num_envs=int(num_envs), **kw)

    ref = build_adapter(8)
    wrapped = _ChunkedAuthorAdapter([build_adapter(4), build_adapter(4)], chunk_envs=4, num_envs=8)
    return ref, wrapped


def selftest_adapter_dtype_bf16_autocast(seed: int = 2) -> dict:
    """新增断言 ④：bf16 档 autocast 前向可用 + argmax 输出形状正确 + 与 fp32 档 argmax 一致。

    三层证据：
      a) **合成 logits 秩同构**：非近平局合成 logits（赢家 6~8、输家 -4±1，最小间隔
         ~6 >> bf16 ulp）的 fp32 argmax 与 bf16 舍入后 argmax 100% 一致——这是
         「argmax 在 fp32 logits 上做」与 bf16 路径等价的数学根据；
      b) **非近平局 mock 端到端**：bf16 档（autocast 包 model 前向 + logits 边界升
         fp32）与 fp32 档对同一 state 序列逐步输出**逐位一致**（act 由 argmax idx 唯一
         决定 ⇒ 输出一致 ⇔ argmax 一致），且走分块 wrapper（生产形态，逐块钉桩路径）；
      c) **钉桩卫生**：bf16 步内各底层 adapter 的 `.model` 已替换为 autocast 替身、
         步后逐个还原为原对象；替身输出 logits dtype == fp32（argmax fp32 口径直证）。
    """
    # ---- a) 合成 logits（非近平局）----
    g = torch.Generator().manual_seed(int(seed))
    synth = torch.randn(16, TOKEN_DIM, 8, generator=g) - 4.0      # 输家基线 -4±1
    win = torch.randint(0, 8, (16, TOKEN_DIM), generator=g)
    winner_val = 6.0 + 2.0 * torch.rand(16, TOKEN_DIM, generator=g)  # 赢家 6~8
    synth = synth.scatter(2, win.unsqueeze(-1), winner_val.unsqueeze(-1))
    am_fp = synth.argmax(dim=-1)
    am_bf = synth.bfloat16().float().argmax(dim=-1)
    agree = int((am_fp == am_bf).sum())
    _check(
        agree == am_fp.numel(),
        f"合成 logits bf16/fp32 argmax 一致率应 100%：{agree}/{int(am_fp.numel())}",
    )
    # ---- b) 非近平局 mock 端到端（fp32 参考 vs bf16 档，同权重双栈）----
    fp_stack = _build_non_tie_adapter_stack()[1]
    bf_stack = _build_non_tie_adapter_stack()[1]
    bf_bases = _base_adapters(bf_stack)
    orig_models = [ad.model for ad in bf_bases]
    ctx_vec = np.linspace(-0.5, 0.5, 13, dtype=np.float32)
    state_dim = int(fp_stack.chunks[0].states.shape[-1])
    for i in range(3):
        st = torch.randn(8, state_dim, generator=g)
        with _adapter_dtype_ctx(fp_stack, "fp32"):
            a_fp = fp_stack.step(st, ctx=ctx_vec)
        with _adapter_dtype_ctx(bf_stack, "bf16"):
            for ad in bf_bases:
                _check(isinstance(ad.model, _AutocastModelForward), f"step{i}: bf16 步内 .model 应为 autocast 替身")
            a_bf = bf_stack.step(st, ctx=ctx_vec)
        _check(a_bf.shape == (8, TOKEN_DIM), f"step{i}: bf16 档 argmax 输出形状应 (8,64)，得 {tuple(a_bf.shape)}")
        _check(bool(torch.isfinite(a_bf).all()), f"step{i}: bf16 档输出含非有限值")
        _check(torch.equal(a_fp, a_bf), f"step{i}: 非近平局 mock 下 bf16 档与 fp32 档 argmax 输出应逐位一致")
    for ad, m in zip(bf_bases, orig_models):
        _check(ad.model is m, "bf16 步后 .model 未还原为原对象（实例级钉桩泄漏）")
    # ---- c) 替身直证：autocast 边界升 fp32 + 形状契约 ----
    lay = _load_author_adapter_layer()
    probe = _AutocastModelForward(_MockAuthorTransformer(seed=3), "cpu", torch.bfloat16)
    out = probe(
        torch.zeros(2, 5, TOKEN_DIM, dtype=torch.long),
        torch.randn(2, 5, lay.STATE_DIM),
        torch.zeros(2, 5, TOKEN_DIM, dtype=torch.long),
        torch.randn(2, 13),
    )
    _check(out.dtype == torch.float32, f"替身输出 logits 应升回 fp32，实得 {out.dtype}")
    _check(tuple(out.shape) == (2, 5, TOKEN_DIM, 8), f"替身输出形状异常：{tuple(out.shape)}")
    return {
        "synthetic_argmax_agree": agree / float(am_fp.numel()),
        "e2e_bitwise_equal": True,
        "output_shape_ok": True,
        "model_patched_and_restored": True,
        "wrapper_logits_fp32": True,
    }


def selftest_adapter_dtype_tf32_restore() -> dict:
    """新增断言 ⑤：tf32 档 `torch.backends.cuda.matmul.allow_tf32` 进/出恢复（finally）。

    三路断言：进入后为 True；正常退出恢复进前值；异常路径（finally）同样恢复。
    标志位是纯进程级 flag（无 CUDA 也可读写），本机 CPU 即可验证。
    """
    dummy = SimpleNamespace(device=torch.device("cpu"))
    pre = bool(torch.backends.cuda.matmul.allow_tf32)
    with _adapter_dtype_ctx(dummy, "tf32"):
        _check(bool(torch.backends.cuda.matmul.allow_tf32) is True, "tf32 档进入后 allow_tf32 应为 True")
    _check(bool(torch.backends.cuda.matmul.allow_tf32) == pre, "tf32 档正常退出后 allow_tf32 未恢复进前值")
    # 异常路径：翻转初值再抛异常，finally 仍须恢复
    pre2 = not pre
    torch.backends.cuda.matmul.allow_tf32 = pre2
    try:
        with _adapter_dtype_ctx(dummy, "tf32"):
            raise RuntimeError("probe: tf32 异常路径")
    except RuntimeError:
        pass
    _check(bool(torch.backends.cuda.matmul.allow_tf32) == pre2, "tf32 档异常路径未在 finally 恢复 allow_tf32")
    torch.backends.cuda.matmul.allow_tf32 = pre   # 收尾还原原始值（不污染同进程后续断言）
    return {"enter_true": True, "exit_restored": True, "exc_restored": True, "pre_flag": pre}


def selftest_adapter_dtype_cli_default() -> dict:
    """新增断言 ⑥：--adapter-dtype 默认 bf16、--adapter-chunk-envs 默认 1024 生效于 CLI。"""
    ap = build_args()
    dflt_dtype = vars(ap.parse_args([]))["adapter_dtype"]
    _check(
        dflt_dtype == DEFAULT_ADAPTER_DTYPE == "bf16",
        f"--adapter-dtype 默认值应为 bf16，实得 {dflt_dtype!r}",
    )
    dflt_chunk = vars(ap.parse_args([]))["adapter_chunk_envs"]
    _check(
        dflt_chunk == DEFAULT_ADAPTER_CHUNK_ENVS == 1024,
        f"--adapter-chunk-envs 默认值应为 1024，实得 {dflt_chunk}",
    )
    got_tf32 = vars(ap.parse_args(["--adapter-dtype", "tf32"]))["adapter_dtype"]
    _check(got_tf32 == "tf32", "--adapter-dtype tf32 解析失败")
    got_fp32 = vars(ap.parse_args(["--adapter-dtype", "fp32"]))["adapter_dtype"]
    _check(got_fp32 == "fp32", "--adapter-dtype fp32 解析失败")
    import io  # noqa: PLC0415（仅自测用：吞掉 argparse 拒绝时的 usage/error 噪音）

    err_sink = io.StringIO()
    with contextlib.redirect_stderr(err_sink):
        try:
            ap.parse_args(["--adapter-dtype", "fp16"])
            _check(False, "--adapter-dtype fp16 应被 choices 拒绝（未拒绝）")
        except SystemExit:
            pass
    return {
        "default_dtype": str(dflt_dtype),
        "default_chunk_envs": int(dflt_chunk),
        "override_tf32": str(got_tf32),
        "override_fp32": str(got_fp32),
        "invalid_rejected": True,
    }


def run_selftest() -> None:
    """本机 CPU 全量自测（不 import isaaclab；唯一例外：adapter 分块断言经延迟
    import 加载 eval_author_v0_decoder 的 Layer 1 纯 torch 件，见
    `_load_author_adapter_layer`——真实 AuthorPolicyAdapter 是等价断言的证据主体）。"""
    torch.manual_seed(0)
    res = {
        "zero_init_identity": selftest_zero_init_identity(),
        "gradient_isolation": selftest_gradient_isolation(),
        "reward_math": selftest_reward_math(),
        "arm_table": selftest_arm_table(),
        "identity_envelope": selftest_identity_envelope(),
        "token_affine_roundtrip": selftest_token_affine_roundtrip(),
        "residual_vs_decoder_arm_decoder_input": selftest_residual_vs_decoder_arm_decoder_input(),
        "ctx_terrain_contract": selftest_ctx_terrain_contract(),
        "official_obs_slice_guard": selftest_official_obs_slice_guard(),
        # 执行/反馈仿射一致性（CVGL 探针根因修复）
        "residual_action_scale_gain_identity": selftest_residual_action_scale_gain_identity(),
        "residual_action_scale_legacy_counterexample": selftest_residual_action_scale_legacy_counterexample(),
        "residual_action_scale_shape_source": selftest_residual_action_scale_shape_source(),
        # 关闭死锁看门狗 + 打断事件落盘仪器（faulthandler 栈实证修复）
        "safe_close_timeout": selftest_safe_close_timeout(),
        "main_loop_crash_log": selftest_main_loop_crash_log(),
        # σ 病态区夹紧（第二层修复：负 σ 导致 Normal.sample 抛异常）
        "std_clamp_negative_std": selftest_std_clamp_negative_std(),
        "std_clamp_healthy_bitwise": selftest_std_clamp_healthy_bitwise(),
        "std_clamp_all_arms": selftest_std_clamp_all_arms(),
        # σ NaN 净化 + update 内首 NaN 溯源探针（第三层修复：update 内部把 σ 打成 NaN）
        "std_clamp_nan_purified": selftest_std_clamp_nan_purified(),
        "std_clamp_finite_bitwise_vs_legacy": selftest_std_clamp_finite_bitwise_vs_legacy(),
        "nan_probe_line_and_trigger": selftest_nan_probe_line_and_trigger(),
        # 第四层：update 内分步 NaN 溯源探针（advantage/幅值/参数快照）
        "nan_probe_advantage_pre_mid": selftest_nan_probe_advantage_pre_mid(),
        "nan_probe_mag_loss_fields": selftest_nan_probe_mag_loss_fields(),
        "nan_probe_param_snapshot_flip": selftest_nan_probe_param_snapshot_flip(),
        # 第五层：advantage/returns 消毒器 + critic 前向探针 + 快照守卫（should-fix）
        "advantage_sanitizer": selftest_advantage_sanitizer(),
        "critic_forward_probe": selftest_critic_forward_probe(),
        "nan_probe_snapshot_guard": selftest_nan_probe_snapshot_guard(),
        # 第七层：消毒器上移到观测端（五路 + 大张量 GPU 侧检测 + dict observations）
        "sanitize_obs_end": selftest_sanitize_obs_end(),
        # 第八层：rollout 期观测源消毒器（第七层 update 侧消毒的上游补全）
        "obs_sanitize_dirty": selftest_obs_sanitize_dirty(),
        "obs_sanitize_healthy_bitwise": selftest_obs_sanitize_healthy_bitwise(),
        "obs_sanitize_passthrough": selftest_obs_sanitize_passthrough(),
        # residual 臂 author adapter env 维分块前向（iter0 CUDA OOM 修复）
        "adapter_chunk_equivalence": selftest_adapter_chunk_equivalence(),
        "adapter_chunk_cli_default": selftest_adapter_chunk_cli_default(),
        "adapter_chunk_tail_block": selftest_adapter_chunk_tail_block(),
        # residual 臂 adapter 前向精度档（性能补丁：精度换速度，默认 bf16/1024）
        "adapter_dtype_bf16_autocast": selftest_adapter_dtype_bf16_autocast(),
        "adapter_dtype_tf32_restore": selftest_adapter_dtype_tf32_restore(),
        "adapter_dtype_cli_default": selftest_adapter_dtype_cli_default(),
    }
    print("[d067] selftest 明细: " + json.dumps(res, ensure_ascii=False), flush=True)
    print("D067_SELFTEST_PASS", flush=True)


# ---------------------------------------------------------------------------
# Layer 2: isaac（以下 import 都在函数体内；模块 import 期零 isaaclab 依赖）
# ---------------------------------------------------------------------------
def _import_train_g1_decoder():
    try:
        from isaac import train_g1_decoder as td
    except ImportError:  # pragma: no cover（本机走此支）
        from apt_g1.isaac import train_g1_decoder as td
    return td


def _import_heavy() -> SimpleNamespace:
    """App 起来后的重型 import：复用 train_g1_decoder._import_heavy 并补齐 D067 需要件。

    复用父本单份组装（torch/np/gym/ManagerBasedRLEnv/RslRl*Cfg/EnvWrapper/
    OnPolicyRunner/mdp/file_md5/make_env_cfg/SonicDecoderActionTerm[Cfg]），
    追加本文件所需：ObsTerm/RewTerm/SceneEntityCfg（obs/奖励覆写）、AuthorPolicyAdapter
    相关件（author token 源）、SonicTorchDecoder（残差臂 policy 内冻结 decoder）。
    """
    td = _import_train_g1_decoder()
    hv = td._import_heavy()
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import RewardTermCfg as RewTerm
    from isaaclab.managers import SceneEntityCfg

    try:  # 服务器执行根平铺 import / 仓库根包 import 双兼容
        from isaac.g1_velocity_decoder_env import SONIC_DECODER_ONNX  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        from apt_g1.isaac.g1_velocity_decoder_env import SONIC_DECODER_ONNX  # type: ignore[no-redef]

    try:
        from isaac.eval_author_v0_decoder import (  # noqa: PLC0415
            AuthorPolicyAdapter,
            TOKEN_SCALE_DEFAULT,
            build_state,
            ctx_from_command,
            load_author_ckpt,
        )
    except ImportError:  # pragma: no cover
        from apt_g1.isaac.eval_author_v0_decoder import (  # type: ignore[no-redef]
            AuthorPolicyAdapter,
            TOKEN_SCALE_DEFAULT,
            build_state,
            ctx_from_command,
            load_author_ckpt,
        )
    try:
        from isaac.sonic_decoder_torch import SonicTorchDecoder  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        from apt_g1.isaac.sonic_decoder_torch import SonicTorchDecoder  # type: ignore[no-redef]

    # sonic_scale 单一事实源（与 sonic_proprio_hist 反归一化同一份）：残差臂执行端
    # JointPositionAction scale 覆写要用它。与 sonic_action_term.py:462 的调用同源。
    try:  # noqa: PLC0415
        from isaac.sonic_action_term import _sonic_scale_isaac  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        from apt_g1.isaac.sonic_action_term import _sonic_scale_isaac  # type: ignore[no-redef]

    hv.ObsTerm = ObsTerm
    hv.RewTerm = RewTerm
    hv.SceneEntityCfg = SceneEntityCfg
    hv.SONIC_DECODER_ONNX = SONIC_DECODER_ONNX
    hv.AuthorPolicyAdapter = AuthorPolicyAdapter
    hv.TOKEN_SCALE_DEFAULT = TOKEN_SCALE_DEFAULT
    hv.build_state = build_state
    hv.ctx_from_command = ctx_from_command
    hv.load_author_ckpt = load_author_ckpt
    hv.SonicTorchDecoder = SonicTorchDecoder
    hv._sonic_scale_isaac = _sonic_scale_isaac
    return hv


# ------------------------------------------------------------- 冻结 decoder 分支
def make_frozen_decoder_branch(decoder, token_dim: int = TOKEN_DIM) -> nn.Module:
    """把 SonicTorchDecoder 包成 (N,994)->(N,29) 的 nn.Module 分支。

    运算序列与 `sonic_action_term.wrap_sonic_torch_decoder` 逐字相同：先对 token 片做
    FSQ 量化，再 forward 全 994 维（冻结 decoder 的一部分；「无 VAE」指不做 VAE 解码）。
    """

    class _FrozenSonicDecoderBranch(nn.Module):
        def __init__(self, inner) -> None:
            super().__init__()
            self.decoder = inner
            self.token_dim = int(token_dim)

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            tokens_q = self.decoder.quantize_tokens(obs[:, : self.token_dim])
            return self.decoder.forward(torch.cat([tokens_q, obs[:, self.token_dim :]], dim=1))

    return _FrozenSonicDecoderBranch(decoder)


# ------------------------------------------------------------- max_forward 奖励覆写
def _max_forward_reward_func(env, asset_cfg, positive_only: bool = True):
    """isaaclab RewardTerm func：root_lin_vel_b[:,0] 正向项（weight 由 RewTerm 施加）。

    返回 raw 前向速度（positive_only 时 clamp(min=0)），RewardManager 再乘 RewTerm.weight。
    与 Layer 1 的 forward_vel_reward 同口径（此处的 weight 交给 manager）。
    """
    vx_b = env.scene[asset_cfg.name].data.root_lin_vel_b[:, 0]
    if positive_only:
        vx_b = torch.clamp(vx_b, min=0.0)
    return vx_b


def _apply_max_forward_recipe(cfg, hv, weight_forward: float, positive_only: bool = True) -> dict:
    """把 max_forward 奖励配方覆写到 cfg.rewards（D067 残差臂专用）。

    做法（对照 §5y「奖励=机体坐标前进速度最大化 + 官方正则、命令跟踪项移除」）：
      1. 移除命令跟踪项（track_lin_vel_xy_exp / track_ang_vel_z_exp -> None）；
      2. 正则项权重按 MAX_FORWARD_RECIPE 逐项覆写（防摔/姿态/动作平滑）；
      3. 新增 `max_forward_vel` 项（func = _max_forward_reward_func，weight = --forward-weight）。
    对 cfg.rewards.* 用直接属性访问（不做 hasattr 静默跳过；官方 term 名漂移首启即报错，
    同 g1_velocity_decoder_env._adapt_g1_sonic_asset 纪律）。
    """
    rewards = cfg.rewards
    removed = []
    for term in MAX_FORWARD_REMOVED_TERMS:
        if getattr(rewards, term, None) is not None:
            setattr(rewards, term, None)
            removed.append(term)
    reg_written = {}
    for name, spec in MAX_FORWARD_RECIPE.items():
        if spec["kind"] == "task":
            continue
        term_cfg = getattr(rewards, name, None)
        if term_cfg is None:
            # 配方表内的项却不存在/为 None = 官方 term 名漂移，显式失败（不静默漏适配）
            raise RuntimeError(
                f"rewards.{name} 不存在/为 None（官方 term 名漂移？配方表 {list(MAX_FORWARD_RECIPE)}）"
            )
        term_cfg.weight = float(spec["weight"])
        reg_written[name] = float(spec["weight"])

    # 新增前进项（dynamic attr：RewardManager 按 cfg.__dict__ 枚举 term，同 obs manager 惯例）
    func = (lambda env, asset_cfg: _max_forward_reward_func(env, asset_cfg, positive_only))
    setattr(
        rewards,
        "max_forward_vel",
        hv.RewTerm(func=func, weight=float(weight_forward), params={"asset_cfg": hv.SceneEntityCfg("robot")}),
    )
    return {"removed": removed, "regularizers": reg_written, "forward_weight": float(weight_forward)}


# ------------------------------------------------------------- author_token 观测项
def load_token_stats(path: str | os.PathLike, *, floor: float = TOKEN_STD_FLOOR) -> tuple[torch.Tensor, torch.Tensor]:
    """读 `--token-stats` npz 的 mean/std（float32，CPU，std 施加 floor）。

    与 `eval_author_v0_decoder.author_token_stats`（:489-498）与
    `sonic_action_term.SonicDecoderActionTerm`（:443-453，floor=token_std_floor）
    **同式同 floor**，故 `residual_token_obs` 的前向仿射与 author adapter 的逆仿射
    消费的是同一份统计（token 值域契约 M2）。
    """
    stats = np.load(str(path))
    mean = torch.as_tensor(np.asarray(stats["mean"], dtype=np.float32), device="cpu").reshape(-1)
    std = torch.as_tensor(np.asarray(stats["std"], dtype=np.float32), device="cpu").reshape(-1)
    if mean.shape != (TOKEN_DIM,) or std.shape != (TOKEN_DIM,):
        raise ValueError(f"token_stats 的 mean/std 须为 ({TOKEN_DIM},)，收到 {tuple(mean.shape)}/{tuple(std.shape)}")
    return mean, torch.clamp(std, min=float(floor))


# ------------------------------------------------- adapter env 维分块前向（OOM 修复）
# 实证：residual 臂 iter0 崩于 author 全 batch 前向的码嵌入分配（1024env×200帧窗
# ≈3.75GiB，train_author_v0.py:421 F.embedding）；本节把 N 个 env 拆成 ceil(N/C) 个
# 独立 AuthorPolicyAdapter 实例逐块前向（峰值 ~1/M），**不改 adapter 本体文件**。
@contextlib.contextmanager
def _pinned_window_len(adapter, t_frames: int):
    """把 adapter 实例的 `window_len` 临时钉到 t_frames（上下文管理器，退出即还原）。

    为什么需要：`AuthorPolicyAdapter.window_len()`（eval_author_v0_decoder.py:603-606）
    的 t = n_valid 对**整个 adapter batch** 取 max——这是该类唯一的 batch 维耦合。分块
    后每块只看块内 max，与全 batch 不等价；钉 t_global（各块全局 max = 单 adapter 的
    window_len()）即恢复逐位等价。实现 = 实例属性覆写（Python 实例 __dict__ 优先于类
    方法），**不触碰 adapter 本体文件**（轴 B 生产件）；步内即还原，不留驻留状态。
    """
    sentinel = object()
    old = adapter.__dict__.get("window_len", sentinel)
    adapter.window_len = lambda: int(t_frames)
    try:
        yield
    finally:
        if old is sentinel:
            adapter.__dict__.pop("window_len", None)
        else:
            adapter.__dict__["window_len"] = old


class _ChunkedAuthorAdapter:
    """AuthorPolicyAdapter 的 env 维分块前向包装（residual 臂 CUDA OOM 修复）。

    结构：持 M = ceil(N/C) 个**独立 AuthorPolicyAdapter 实例**（每实例 num_envs≤C，
    env 区间不重叠），`step` 逐块调用 `adapter.step(st_chunk, ctx_chunk)` 后 cat。

    **为何分块 ≡ 全 batch（逐位）**：adapter 的自回归状态全部按 env 维独立——
    codes/states/ctx/n_valid/_last_idx 逐 env 切片（eval_author_v0_decoder.py:579-588）、
    `_push` 逐 env 滚动（:608-614）、`reset` 只动指定 env（:650-658）、step 的模型前向
    对 batch 逐行独立 ⇒ 不重叠拆分后每 env 的状态演化与前向输入完全不变。唯一 batch 维
    耦合 `window_len()` 的处理见 `_step_window_len` + `_pinned_window_len`（每步把各块
    窗长钉到「`_push` 推进后」的全局 n_valid max，= 单 adapter 同步的 window_len()）。

    已知边界（如实声明）：模型前向逐位一致以「同输入同结果」为前提；不同 batch 形状
    理论上可能选到不同 GEMM 分块策略产生末位舍入差（argmax 仅近平局时可能翻转）。
    CPU 与常见 CUDA 路径实测逐位一致；selftest 断言 ①③ 以真实 AuthorPolicyAdapter
    + mock transformer 逐位验证（含整块部分 reset 的非均匀 n_valid 场景与尾块）。
    """

    def __init__(self, chunks: list, chunk_envs: int, num_envs: int) -> None:
        if not chunks:
            raise ValueError("chunks 须非空")
        sizes = [int(c.num_envs) for c in chunks]
        if int(chunk_envs) <= 0 or any(s <= 0 for s in sizes):
            raise ValueError(f"chunk_envs 与各块 env 数须 > 0：chunk_envs={chunk_envs} sizes={sizes}")
        if int(num_envs) != sum(sizes):
            raise ValueError(f"各块 env 数之和 {sum(sizes)} != 总 env 数 {num_envs}")
        if sizes[0] != int(chunk_envs):
            raise ValueError(f"首块 env 数 {sizes[0]} 应等于 chunk_envs={chunk_envs}（仅尾块可小）")
        self.chunks = list(chunks)
        self.chunk_envs = int(chunk_envs)
        self.num_envs = int(num_envs)
        # 同一 device 假设（生产恒单 GPU；t_global 归约要求同 device）
        self.device = torch.device(chunks[0].n_valid.device)
        self.win = int(chunks[0].win)

    # -- 统计代理（residual_token_obs 的同源一致性守卫读这两个属性；各块同参构造）--
    @property
    def token_mean(self):
        return self.chunks[0].token_mean

    @property
    def token_std(self):
        return self.chunks[0].token_std

    def _step_window_len(self) -> int:
        """本步（各块 `_push` 之后）应有的前向帧数 t_global（逐位等价的关键）。

        `AuthorPolicyAdapter.step` 的顺序是**先 `_push`（n_valid +1，clamp win）再读
        `window_len()`**（eval_author_v0_decoder.py:624-625），故分块侧必须按**推进后**
        的全局 max 取值：
            t = max_e( min(v_e + 1, win) ) = min( max_e(v_e) + 1, win )
        其中 v_e 为推进前各 env 的 n_valid（min/max 可交换：单调映射）。
        torch.stack 一次归约：每控制步只做一次 GPU->CPU 同步（与单 adapter 的
        `window_len().item()` 同开销量级）。若错用推进前的 max，第 2 步起各块窗长
        会比单 adapter 少 1 帧（selftest 断言 ① 的连续步检查抓的正是这个）。
        """
        n_max = int(torch.stack([ad.n_valid.max() for ad in self.chunks]).max().item())
        return max(1, min(n_max + 1, self.win))

    @staticmethod
    def _ctx_for(ctx, off: int, n: int):
        """把全量 ctx 切到本块：一维（广播向量）原样传，二维按 env 行切片。"""
        if ctx is None:
            return None
        if torch.is_tensor(ctx):
            return ctx if ctx.ndim == 1 else ctx[off : off + n]
        arr = np.asarray(ctx)
        return ctx if arr.ndim == 1 else arr[off : off + n]

    def reset(self, env_ids=None) -> None:
        """按全局 env id 路由 reset 到所属块（换算块内局部 id）；None = 全部。"""
        if env_ids is None:
            for ad in self.chunks:
                ad.reset(None)
            return
        ids = torch.as_tensor(env_ids, device=self.device).reshape(-1).to(torch.long)
        for k, ad in enumerate(self.chunks):
            lo = k * self.chunk_envs
            local = ids[(ids >= lo) & (ids < lo + int(ad.num_envs))] - lo
            if int(local.numel()) > 0:
                ad.reset(local)

    def step(self, state_rows, ctx=None) -> torch.Tensor:
        """逐块 step（块内窗长统一钉 t_global）后 cat -> (N, 64)。"""
        st = torch.as_tensor(state_rows)
        if int(st.shape[0]) != self.num_envs:
            raise ValueError(f"state_rows 首维须 = num_envs={self.num_envs}，得 {tuple(st.shape)}")
        t_global = self._step_window_len()
        outs = []
        off = 0
        for ad in self.chunks:
            n = int(ad.num_envs)
            with _pinned_window_len(ad, t_global):
                outs.append(ad.step(st[off : off + n], ctx=self._ctx_for(ctx, off, n)))
            off += n
        return torch.cat(outs, dim=0)


# --------------------------------------- adapter 前向精度档（性能补丁：精度换速度）
# 实测：1024env 全批量 adapter 前向 chunk128 ≈179s/it（5000it 不可行），精确 KV cache
# 不可行（v0 位置编码窗口相对+滑动驱逐，缓存每步失效）。本节三件只改「前向算什么精度」，
# 不改奖励/网络/超参/adapter 本体；口径声明见模块 docstring「adapter 前向精度档」。
class _AutocastModelForward:
    """bf16 档的 model 前向替身：autocast 包前向，logits 在边界升回 fp32。

    为何要替身而不是直接包整个 `adapter.step`：AuthorPolicyAdapter.step 内部
    `logits = self.model(...)` 后**紧接 argmax**（本体文件是轴 B 生产件不可改，argmax
    无法外移）——把 autocast 精确钉在 model 前向上、logits 于 autocast 边界 `.float()`，
    即同时得到「matmul bf16 / embedding fp32 由 autocast 决定，不手动控制具体算子」
    与「argmax 在 fp32 logits 上做」（bf16 logits 与其 fp32 提升严格序同构，argmax
    逐位一致；升 fp32 只是保住 step 内 argmax/逆仿射的输入 dtype 契约，selftest 断言 ④
    以合成 logits + 非近平局 mock 直证）。替身是普通可调用对象（非 nn.Module）：
    adapter 冻结推理、无参数注册需求，避免 _modules/state_dict 语义副作用。
    """

    def __init__(self, model, device_type: str, dtype: torch.dtype) -> None:
        self._inner = model
        self._device_type = str(device_type)
        self._dtype = dtype

    def __call__(self, *args, **kwargs):
        with torch.autocast(device_type=self._device_type, dtype=self._dtype):
            out = self._inner(*args, **kwargs)
        return out.float()


def _base_adapters(adapter) -> list:
    """枚举 adapter（单实例或 _ChunkedAuthorAdapter）底层的 AuthorPolicyAdapter 列表。"""
    chunks = getattr(adapter, "chunks", None)
    return list(chunks) if chunks is not None else [adapter]


@contextlib.contextmanager
def _adapter_dtype_ctx(adapter, dtype_mode: str):
    """residual 臂 `adapter.step` 的精度档上下文（性能补丁唯一改动点=前向精度）。

    - "fp32"：什么都不做（旧行为逐字，无任何附加开销）；
    - "bf16"：把底层每个 AuthorPolicyAdapter 的 `.model` 临时替换为
      `_AutocastModelForward`（autocast(bf16) 包前向 + logits 边界升 fp32），步内即
      还原（同 `_pinned_window_len` 的实例级钉桩先例，不改 adapter 本体文件、不留
      驻留状态，异常路径同样还原）；
    - "tf32"：进入时 `torch.backends.cuda.matmul.allow_tf32=True`，退出 finally 恢复
      进前值（仅本 wrapper 范围生效，不全局泄漏；selftest 断言 ⑤ 直证进/出/异常三路）。
    """
    mode = str(dtype_mode)
    if mode not in ADAPTER_DTYPES:
        raise ValueError(f"--adapter-dtype 须 ∈ {ADAPTER_DTYPES}，收到 {mode!r}")
    if mode == "fp32":
        yield
        return
    if mode == "tf32":
        old = bool(torch.backends.cuda.matmul.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old
        return
    # bf16：实例级钉桩底层 model（chunked 时逐块替换）
    dev_type = torch.device(adapter.device).type
    bases = _base_adapters(adapter)
    saved = [ad.model for ad in bases]
    for ad in bases:
        ad.model = _AutocastModelForward(ad.model, dev_type, torch.bfloat16)
    try:
        yield
    finally:
        for ad, m in zip(bases, saved):
            ad.model = m


def _log_adapter_perf_first_step(env, adapter, t0: float) -> None:
    """首步仪器：residual_token_obs 装载后首个 adapter step 打一行 [adapter-perf]。

    CUDA 侧同步后再读钟（首步含预热/内核选择，量级判读用，非稳态基准）；失败只降级
    为不同步计时，绝不打断观测计算。
    """
    try:
        if torch.device(getattr(adapter, "device", "cpu")).type == "cuda":
            torch.cuda.synchronize()
    except Exception:  # pragma: no cover（仪器绝不打断训练）
        pass
    dt = time.perf_counter() - t0
    n_chunks = len(getattr(adapter, "chunks", []) or [adapter])
    print(
        f"[adapter-perf] dtype={env._d067_adapter_dtype} adapter_chunk_envs="
        f"{env._d067_adapter_chunk_envs} num_envs={int(adapter.num_envs)} chunks={n_chunks} "
        f"first_adapter_step_s={dt:.3f}（首步含 CUDA 预热，供首跑判读量级）",
        flush=True,
    )


def residual_token_obs(
    env,
    ckpt_path: str,
    token_stats: str,
    token_alpha: float,
    token_bound: str,
    intent_pin_max: bool,
    intent_pin_value: float,
    terrain: str = "plane",
    *,
    token_mean: torch.Tensor | None = None,
    token_std: torch.Tensor | None = None,
    author_loader,
    adapter_factory,
    ctx_builder,
    ctx_builder_batch,
    state_builder,
    adapter_chunk_envs: int = 0,
    adapter_dtype: str = DEFAULT_ADAPTER_DTYPE,
):
    """policy 观测项：(N, 64) 冻结 author adapter 产出的 **token 坐标**（观测常量，无梯度）。

    设计依据（假设 A1）：author adapter 是**有状态自回归**件——若放进 policy.forward，
    PPO 的 evaluate 阶段会重跑并污染滚动窗。故必须 env 侧每控制步算一次、作为观测
    喂给 policy。token 走 obs ⇒ 与 rollout buffer 同步存储，evaluate 阶段读的是同一值。

    **值域契约（M2）**：`adapter.step` 返回的是归一化动作 a = (tok-mean)/(alpha*std)
    （eval_author_v0_decoder.AuthorPolicyAdapter.step:638-642），**不是** token 坐标。
    喂 decoder 分支（会做 FSQ 量化）前必须补前向仿射
    `tok = mean + alpha*std*(tanh(a) if token_bound=='tanh' else a)`
    （`forward_token_affine`；权威式 sonic_action_term.py:205-215 / sonic_lora_policy.py:157-161）。
    `token_mean`/`token_std` 由调用侧从**与 decoder 分支同一份** `--token-stats` npz 显式
    传入（`_token_obs_params` 经 `load_token_stats`）；装载 adapter 时与 adapter 内部统计
    做**逐位一致性断言**（不同源即 fail loud）。

    地形契约（M1）：`terrain` 先经 `ctx_terrain` 映射到 ctx 契约名
    （rough/stairs/stones/discrete -> rough_paper），再进 ctx 构造，杜绝
    `ctx_from_command` 首帧 ValueError。

    依赖经 params 显式注入（`author_loader`/`adapter_factory`/`ctx_builder`/
    `ctx_builder_batch`/`state_builder`，见 `_attach_token_obs`）——避免模块级全局转发名，
    便于单测替换。

    实现要点：
      - 懒建 `AuthorPolicyAdapter`（冻结）挂 `env._d067_token_adapter`；
      - **每控制步只推一帧**：以 `episode_length_buf` 快照去重（ObservationManager 可能
        在一步内被多次 compute，若无去重会多推帧污染自回归窗）；
      - state 由 `robot.data` 直取（**不走 obs**——避免本项在 obs 计算中递归调用
        observation_manager）：joint_pos（SONIC 序，绝对）+ root_quat_wxyz + env 局部
        trans，与 eval_author_v0_decoder._run_cond_main 同口径（build_state 内做
        Isaac->MuJoCo perm 重排）；
      - ctx 命令槽：`intent_pin_max=True` 钉 `intent_pin_value`（默认 1.0，假设 A2），
        否则读 env 当前速度命令 vx（逐 env）；
      - 返回 token.detach()（不建图，梯度不达 token 源）。
    构造期容忍：ObservationManager 在 env 构造早期会无条件调本项一次填维度，此时
    `episode_length_buf` 可能尚未创建——用 getattr 探测，缺件时返回零 token 占位
    （其后 env.reset() 后的正式计算覆盖）。
    """
    terrain_ctx = ctx_terrain(terrain)
    adapter = getattr(env, "_d067_token_adapter", None)
    if adapter is None:
        if token_mean is None or token_std is None:
            raise ValueError(
                "residual_token_obs 需要显式 token_mean/token_std（与 decoder 分支同一份 "
                "--token-stats npz；见 `_token_obs_params`）"
            )
        model, model_cfg = author_loader(ckpt_path, device=str(env.device))
        token_mean_t = torch.as_tensor(token_mean, dtype=torch.float32, device=env.device).reshape(-1)
        token_std_t = torch.as_tensor(token_std, dtype=torch.float32, device=env.device).reshape(-1)
        # 性能补丁（精度换速度，见模块 docstring「adapter 前向精度档」）：前向精度档
        # fail-loud 校验，非法档拒绝开训。
        dtype_mode = str(adapter_dtype)
        if dtype_mode not in ADAPTER_DTYPES:
            raise ValueError(f"--adapter-dtype 须 ∈ {ADAPTER_DTYPES}，收到 {dtype_mode!r}")
        # OOM 修复（见模块 docstring「env 维分块前向」）：--adapter-chunk-envs > 0 且小于
        # 总 env 数时，把 N 个 env 拆成 ceil(N/C) 个独立 adapter 实例逐块前向（峰值 ~1/M）；
        # 0 = 不分块（单 adapter，旧行为逐字）；C >= N 时分块无意义，同样走单 adapter。
        chunk_envs = int(adapter_chunk_envs)
        n_total = int(env.num_envs)
        if chunk_envs < 0:
            raise ValueError(f"--adapter-chunk-envs 须 >= 0（0=不分块），收到 {chunk_envs}")
        if chunk_envs > 0 and chunk_envs < n_total:
            sizes = [chunk_envs] * (n_total // chunk_envs)
            if n_total % chunk_envs:
                sizes.append(n_total % chunk_envs)   # 尾块（非整数倍时）
            adapter = _ChunkedAuthorAdapter(
                [
                    adapter_factory(
                        model,
                        model_cfg,
                        token_mean_t,
                        token_std_t,
                        num_envs=s,
                        device=str(env.device),
                        token_alpha=token_alpha,
                        token_bound=token_bound,
                    )
                    for s in sizes
                ],
                chunk_envs=chunk_envs,
                num_envs=n_total,
            )
        else:
            sizes = [n_total]
            adapter = adapter_factory(
                model,
                model_cfg,
                token_mean_t,
                token_std_t,
                num_envs=env.num_envs,
                device=str(env.device),
                token_alpha=token_alpha,
                token_bound=token_bound,
            )
        # 同一份统计守卫：adapter 内部统计（含 floor）须与传入的前向仿射统计逐位一致
        a_mean = getattr(adapter, "token_mean", None)
        a_std = getattr(adapter, "token_std", None)
        if a_mean is not None and a_std is not None:
            am = torch.as_tensor(a_mean, dtype=torch.float32, device=token_mean_t.device).reshape(-1)
            as_ = torch.as_tensor(a_std, dtype=torch.float32, device=token_std_t.device).reshape(-1)
            if not (torch.equal(am, token_mean_t) and torch.equal(as_, token_std_t)):
                raise RuntimeError(
                    "adapter 内部 token 统计与传入的前向仿射统计不一致（必须同源同一份 npz）："
                    f"mean maxdiff={float((am - token_mean_t).abs().max())} "
                    f"std maxdiff={float((as_ - token_std_t).abs().max())}"
                )
        env._d067_token_adapter = adapter
        env._d067_token_cache = None
        env._d067_token_step_key = None
        env._d067_token_pin_max = bool(intent_pin_max)
        env._d067_token_pin_value = float(intent_pin_value)
        env._d067_token_terrain = str(terrain_ctx)
        env._d067_token_mean = token_mean_t
        env._d067_token_std = token_std_t
        env._d067_token_alpha = float(token_alpha)
        env._d067_token_bound = str(token_bound)
        # 性能补丁仪器（首步 [adapter-perf] 行的数据源）：精度档 + 分块 + 首步待计时
        env._d067_adapter_dtype = dtype_mode
        env._d067_adapter_chunk_envs = chunk_envs
        env._d067_adapter_perf_pending = True
        print(
            f"[d067] author token 源已装载：ckpt={ckpt_path} token_stats={token_stats} "
            f"vocab={model_cfg.get('vocab')} code_min={model_cfg.get('code_min')} "
            f"intent_pin_max={intent_pin_max} intent_pin_value={intent_pin_value} "
            f"token_bound={token_bound} terrain={terrain!r}->ctx_terrain={terrain_ctx!r} "
            f"adapter_chunk_envs={chunk_envs} 块划分={sizes} adapter_dtype={dtype_mode}",
            flush=True,
        )

    ep = getattr(env, "episode_length_buf", None)
    if ep is None:
        # 构造期兜底：返回零 token 占位（维度正确；reset 后正式计算覆盖）
        return torch.zeros(env.num_envs, TOKEN_DIM, device=env.device)

    # 每控制步去重（同一步内多次 compute 只推一帧）
    key = ep.detach().clone()
    prev_key = getattr(env, "_d067_token_step_key", None)
    if prev_key is not None and torch.equal(prev_key, key):
        cached = getattr(env, "_d067_token_cache", None)
        if cached is not None:
            return cached

    reset_ids = (ep == 0).nonzero(as_tuple=False).squeeze(-1)
    if reset_ids.numel() > 0:
        adapter.reset(reset_ids)

    robot = env.scene["robot"]
    joint_ids_t = getattr(env, "_d067_joint_ids_t", None)
    if joint_ids_t is None:
        try:
            from isaac.sonic_action_term import resolve_sonic_joint_ids  # noqa: PLC0415
        except ImportError:  # pragma: no cover
            from apt_g1.isaac.sonic_action_term import resolve_sonic_joint_ids  # type: ignore[no-redef]

        ids, _identity = resolve_sonic_joint_ids(robot.joint_names)
        joint_ids_t = torch.as_tensor(ids, dtype=torch.long, device=env.device)
        env._d067_joint_ids_t = joint_ids_t

    # state = joint_pos（SONIC 序，绝对）+ root quat_wxyz + env 局部 trans
    jp = robot.data.joint_pos[:, joint_ids_t]
    quat = robot.data.root_quat_w
    trans = robot.data.root_pos_w - env.scene.env_origins
    st = state_builder(jp, quat, trans, jp_absolute=True)

    if env._d067_token_pin_max:
        ctx = ctx_builder(env._d067_token_pin_value, env._d067_token_terrain)
    else:
        cmd = env.command_manager.get_term("base_velocity").vel_command_b[:, 0]
        ctx = ctx_builder_batch(cmd, env._d067_token_terrain)
    # 性能补丁：首步仪器待打 -> 先取 t0（只在 adapter.step 计时，不含仿射/观测组装）
    perf_t0 = time.perf_counter() if getattr(env, "_d067_adapter_perf_pending", False) else None
    with torch.no_grad():
        # 精度档上下文（fp32=逐字旧行为；bf16=autocast 包前向+logits 边界升 fp32；
        # tf32=matmul 标志进/出恢复）。仅包 adapter.step 前向，仿射/缓存逻辑不动。
        with _adapter_dtype_ctx(adapter, getattr(env, "_d067_adapter_dtype", "fp32")):
            raw = adapter.step(st, ctx=ctx)
        # M2：adapter.step 返回归一化动作 a（逆仿射产物），补前向仿射 -> token 坐标，
        # 否则 decoder 分支的 FSQ 量化吃到 a 而非 tok（先验破坏）。
        tok = forward_token_affine(
            raw,
            env._d067_token_mean,
            env._d067_token_std,
            env._d067_token_alpha,
            env._d067_token_bound,
        )
    if perf_t0 is not None:
        env._d067_adapter_perf_pending = False
        _log_adapter_perf_first_step(env, adapter, perf_t0)
    tok = tok.detach()
    env._d067_token_cache = tok
    env._d067_token_step_key = key
    return tok


def _token_obs_params(hv, cli, terrain: str) -> tuple[dict, str]:
    """residual_token_obs 的 params（含注入的可调用件，避免模块级全局转发名）。

    token_mean/std 由**与 decoder 分支同一份** `--token-stats` npz 读出（M2 值域契约）；
    terrain 在此先经 `ctx_terrain` 映射（M1 地形契约，幂等），再进 ctx 构造器。
    返回 (params, ctx_terrain_name)。
    """
    if not cli.token_stats:
        raise RuntimeError("residual_token_obs 需要 --token-stats（前向仿射 mean/std 与 decoder 分支同源）")
    token_mean, token_std = load_token_stats(cli.token_stats)
    ctx_terrain_name = ctx_terrain(terrain)
    return {
        "ckpt_path": cli.author_ckpt,
        "token_stats": cli.token_stats,
        "token_alpha": cli.token_alpha if cli.token_alpha is not None else 1.0,
        "token_bound": cli.token_bound or "tanh",
        "intent_pin_max": bool(cli.intent_pin_max),
        "intent_pin_value": float(cli.intent_pin_value),
        "terrain": terrain,
        "token_mean": token_mean,
        "token_std": token_std,
        "author_loader": hv.load_author_ckpt,
        "adapter_factory": lambda model, cfg, mean, std, **kw: hv.AuthorPolicyAdapter(
            model,
            code_min=int(cfg["code_min"]),
            vocab=int(cfg["vocab"]),
            token_mean=mean,
            token_std=std,
            token_scale=hv.TOKEN_SCALE_DEFAULT,
            **kw,
        ),
        "ctx_builder": lambda pin_value, terr: hv.ctx_from_command(pin_value, 0.0, 0.0, terrain=terr),
        "ctx_builder_batch": lambda vx, terr: np.stack(
            [hv.ctx_from_command(float(v), 0.0, 0.0, terrain=terr) for v in np.asarray(vx).reshape(-1)]
        ),
        "state_builder": hv.build_state,
        # OOM 修复：adapter env 维分块前向块大小（0=不分块；缺省随 CLI 默认 1024）
        "adapter_chunk_envs": int(getattr(cli, "adapter_chunk_envs", DEFAULT_ADAPTER_CHUNK_ENVS)),
        # 性能补丁：adapter 前向精度档（fp32/bf16/tf32，缺省随 CLI 默认 bf16）
        "adapter_dtype": str(getattr(cli, "adapter_dtype", DEFAULT_ADAPTER_DTYPE)),
    }, ctx_terrain_name


# ------------------------------------------------------------- rsl_rl 残差策略
try:  # pragma: no cover - 服务器有 rsl_rl；本机走 except（降级 nn.Module 仅供 import）
    from rsl_rl.modules import ActorCritic as _ActorCriticBase

    HAS_RSL_RL = True
except ImportError:  # pragma: no cover - 本机（无训练 venv）
    _ActorCriticBase = nn.Module  # type: ignore[assignment,misc]
    HAS_RSL_RL = False

RESIDUAL_POLICY_CLASS_NAME = "ResidualDecoderPolicy"
# direct/decoder 两臂的策略类名：rsl_rl ActorCritic + σ 夹紧混入（见上方 ClampedStdPolicyMixin）。
# 注册进 rsl_rl 命名空间后由 runner `eval(class_name)` 解析；`__init__` 逐字继承 ActorCritic
# ⇒ 网络/初始化/超参全同 D064，仅 `update_distribution` 多一层病态区 σ 下夹。
CLAMPED_ACTOR_CRITIC_CLASS_NAME = "ClampedStdActorCritic"


class ClampedStdActorCritic(ClampedStdPolicyMixin, _ActorCriticBase):  # type: ignore[misc]
    """rsl_rl `ActorCritic` + σ 下夹（direct/decoder 臂，D064 逐字网络 + 病态区夹紧）。

    除 `update_distribution`（经 `ClampedStdPolicyMixin`）外不覆写任何东西：构造、actor/critic
    MLP、init_noise_std 语义、参数量与初始化分布均与 rsl_rl ActorCritic 逐字一致。健康轨迹
    （σ≥STD_CLAMP_MIN）下夹紧是恒等映射，行为与 D064 ActorCritic 逐位相同。
    """


class ResidualDecoderPolicy(ClampedStdPolicyMixin, _ActorCriticBase):  # type: ignore[misc]
    """rsl_rl ActorCritic 变体：μ = 冻结 decoder(cat[token, proprio]) + r(官方 obs)。

    - 动作分布 N(μ, σ)，动作维 29（= 关节目标，与 direct 臂同口径）。
    - `token_slice` = policy obs 里 64 维 `author_token` 的切片；`proprio_slice` = 930 维
      `sonic_proprio` 的切片；二者由 train 侧按 observation_manager 运行期 term 序算出
      （不硬编码位置）。
    - `official_obs_dim` = policy obs 总维 - 930 - 64 = 原 D064 policy 八项（含 height_scan）；
      r 的输入即该切片（假设 A5）。
    - 冻结分支在 `ResidualMuPath` 内（decoder no_grad）；**只有 residual 与 critic 可训**，
      σ 默认冻结（`sigma_trainable=False`，§5y「PPO 只训 r」）；基类 actor 干被冻结且不参与
      前向（本臂无独立 actor MLP，μ 由冻结分支 + r 构成）。
    - 可训集合摘除由 train 侧 `_prune_frozen_params` 落地（同 D065 口径）。
    - σ 病态区下夹经 `ClampedStdPolicyMixin.update_distribution`（本臂 `update_distribution`
      覆写在调用 super() 时命中 mixin 的夹紧上下文；健康区逐位不变）。
    """

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int = ACTION_DIM,
        *,
        decoder: nn.Module | None = None,
        token_slice: tuple[int, int] | None = None,
        proprio_slice: tuple[int, int] | None = None,
        official_obs_dim: int | None = None,
        residual_hidden: tuple[int, ...] = RESIDUAL_HIDDEN,
        activation: str = RESIDUAL_ACTIVATION,
        init_noise_std: float = 1.0,
        sigma: float | None = None,
        sigma_trainable: bool = False,
        **kwargs,
    ) -> None:
        if not HAS_RSL_RL:
            raise RuntimeError(
                "ResidualDecoderPolicy 需要 rsl_rl（rsl-rl-lib 2.3.3 经典 API ActorCritic）；"
                "本机（Windows）无训练 venv，请在服务器 .venv_isaac 内实例化。"
            )
        if decoder is None:
            raise ValueError("ResidualDecoderPolicy 需要冻结 decoder（见 _build_residual_policy_kwargs）")
        if token_slice is None or proprio_slice is None:
            raise ValueError("ResidualDecoderPolicy 需要 token_slice/proprio_slice（obs 切片）")
        if int(num_actions) != ACTION_DIM:
            raise ValueError(f"动作维冻结为 {ACTION_DIM}，收到 {num_actions}")
        super().__init__(
            num_actor_obs=int(num_actor_obs),
            num_critic_obs=int(num_critic_obs),
            num_actions=ACTION_DIM,
            actor_hidden_dims=[512, 256, 128],  # 占位（本臂不用基类 actor 干）
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            init_noise_std=float(init_noise_std),
        )
        # 基类 actor 干不参与本臂前向：冻结（不进优化器，防污染可训集合）
        for p in self.actor.parameters():
            p.requires_grad_(False)
        self.actor_frozen = True

        self.token_start, self.token_width = int(token_slice[0]), int(token_slice[1])
        self.proprio_start, self.proprio_width = int(proprio_slice[0]), int(proprio_slice[1])
        if self.token_width != TOKEN_DIM:
            raise ValueError(f"token_slice 宽度须为 {TOKEN_DIM}，收到 {self.token_width}")
        if self.proprio_width != PROPRIO_DIM:
            raise ValueError(f"proprio_slice 宽度须为 {PROPRIO_DIM}，收到 {self.proprio_width}")
        if int(self.token_width) + int(self.proprio_width) != DECODER_OBS_DIM:
            raise ValueError("token+proprio 宽度应等于 decoder 输入维 994")

        # r 输入维：缺省 = 官方 policy obs 切片（total - proprio - token），假设 A5
        if official_obs_dim is None:
            official_obs_dim = int(num_actor_obs) - self.proprio_width - self.token_width
        self.official_obs_dim = int(official_obs_dim)
        if self.official_obs_dim <= 0:
            raise ValueError(f"official_obs_dim 推得 {self.official_obs_dim}（num_actor_obs={num_actor_obs}）")
        self.mu_path = ResidualMuPath(
            decoder, ResidualActionMLP(self.official_obs_dim, ACTION_DIM, residual_hidden, activation)
        )

        sigma0 = float(init_noise_std) if sigma is None else float(sigma)
        self.std = nn.Parameter(torch.full((ACTION_DIM,), sigma0), requires_grad=bool(sigma_trainable))
        self.sigma_frozen = not bool(sigma_trainable)

    @staticmethod
    def _flat_obs(obs):
        if isinstance(obs, dict):
            return obs["policy"]
        return obs

    def decoder_obs(self, flat: torch.Tensor) -> torch.Tensor:
        """从 policy obs 切出 (N,994) = cat[author_token(64), sonic_proprio(930)]。"""
        return assemble_decoder_obs(
            flat,
            (self.token_start, self.token_width),
            (self.proprio_start, self.proprio_width),
        )

    def mu(self, obs) -> torch.Tensor:
        """μ(29) = decoder(cat[token, proprio]) + r(官方 obs 切片)。"""
        flat = self._flat_obs(obs)
        dec_obs = self.decoder_obs(flat)
        r_obs = flat[:, : self.official_obs_dim]
        return self.mu_path(dec_obs, r_obs)

    def update_distribution(self, observations) -> None:
        # σ 病态区下夹：本臂自带 update_distribution（MRO 会盖掉 mixin 的同名方法），
        # 故显式经 mixin 的夹紧上下文构造 Normal（健康区恒等，见 ClampedStdPolicyMixin）。
        with self._std_clamped():
            mu = self.mu(observations)
            std = self.std.to(mu.device).expand_as(mu)
            self.distribution = torch.distributions.Normal(mu, std)

    def act(self, observations, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations) -> torch.Tensor:
        return self.mu(observations)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    def trainable_report(self) -> dict:
        def _count(mod):
            ps = list(mod.parameters())
            return {
                "n_tensors": len(ps),
                "n_trainable": sum(1 for p in ps if p.requires_grad),
                "n_elements": int(sum(p.numel() for p in ps)),
                "n_trainable_elements": int(sum(p.numel() for p in ps if p.requires_grad)),
            }

        return {
            "sigma": {"value": float(self.std.detach().reshape(-1)[0].item()), "frozen": bool(self.sigma_frozen)},
            "actor_placeholder": _count(self.actor),
            "critic": _count(self.critic),
            "decoder": _count(self.mu_path.decoder),
            "residual": _count(self.mu_path.residual),
            "only_residual_trainable": self.mu_path.only_residual_trainable(),
        }


def register_residual_policy_in_rsl_rl(cls=ResidualDecoderPolicy, name: str = RESIDUAL_POLICY_CLASS_NAME) -> str:
    """把残差策略类注入 rsl_rl 类名解析命名空间（同 sonic_lora_policy 手法）。

    rsl_rl 经典 API 用 `eval(cfg["policy"]["class_name"])` 在 runner 模块命名空间解析；
    挂到 rsl_rl.runners.on_policy_runner / rsl_rl.runners / rsl_rl.modules 即可生效。
    调用方随后必须断言 runner 内 policy 实例类型（防静默回落成 ActorCritic 把残差臂训成 direct 臂）。
    """
    import importlib

    targets = []
    for mod_name in ("rsl_rl.runners.on_policy_runner", "rsl_rl.runners", "rsl_rl.modules"):
        try:  # pragma: no cover - 服务器有 rsl_rl
            targets.append(importlib.import_module(mod_name))
        except ImportError:
            continue
    if not targets:
        raise RuntimeError("rsl_rl 不可导入——无法注册残差策略类（请在 isaaclab venv 内运行）")
    for mod in targets:
        setattr(mod, name, cls)
    return name


def register_arm_policy_classes_in_rsl_rl() -> dict:
    """把三臂策略类统一注入 rsl_rl 命名空间，返回 {arm: 类名}（三臂同经 σ 夹紧层）。

    - direct/decoder：`ClampedStdActorCritic`（D064 ActorCritic 逐字 + σ 下夹）；
    - residual：`ResidualDecoderPolicy`（μ 路径不变 + σ 下夹）。
    统一入口保证「三臂配置表都挂了夹紧」由**同一处构造**落地，避免漏挂一臂。
    """
    return {
        "direct": register_residual_policy_in_rsl_rl(
            ClampedStdActorCritic, CLAMPED_ACTOR_CRITIC_CLASS_NAME),
        "decoder": register_residual_policy_in_rsl_rl(
            ClampedStdActorCritic, CLAMPED_ACTOR_CRITIC_CLASS_NAME),
        "residual": register_residual_policy_in_rsl_rl(),
    }


# ------------------------------------------------------------- env / policy 装配
def _policy_obs_slot(env, torch_mod, term_name: str, width_expected: int) -> dict:
    """定位 policy obs 里某 term 的切片（运行期 term 序，不硬编码位置）。

    与 eval_author_v0_decoder._policy_obs_cmd_slot 同款：按 observation_manager
    active_terms/group_obs_term_dim 逐 term 维度累加得 start/width；term 缺位即显式失败。
    """
    om = env.observation_manager
    names = [str(n) for n in om.active_terms["policy"]]
    dims = list(om.group_obs_term_dim["policy"])
    if len(names) != len(dims):
        raise RuntimeError(f"policy 组 names/dims 错位：{len(names)} vs {len(dims)}")
    if term_name not in names:
        raise RuntimeError(f"policy 组无 {term_name!r} 观测项（现有 {names}）——残差臂接线失败")
    idx = names.index(term_name)
    start = 0
    for term_dims in dims[:idx]:
        start += int(torch_mod.tensor(list(term_dims)).prod())
    width = int(torch_mod.tensor(list(dims[idx])).prod())
    if width != int(width_expected):
        raise RuntimeError(f"{term_name} 宽度 {width} != {width_expected}")
    return {"term": term_name, "index": int(idx), "start": int(start), "width": int(width), "policy_terms": names}


def _attach_token_obs(cfg, hv, cli, terrain: str) -> dict:
    """把 64 维 `author_token` 观测项追加到 cfg.observations.policy 末位（残差臂）。

    返回地形契约/仿射信息（进身份信封）：`ctx_terrain` = 实际进 ctx 的地形名
    （经 CTX_TERRAIN_MAP 映射），`token_stats` = 前向仿射统计路径。
    """
    # configclass 的 policy 组是 dataclass 实例；直接 setattr 新 term（obs manager 按
    # __dict__ 枚举，见 tmp/isaac_ref/observation_manager.py:391-393）。字段序 = 追加序。
    params, ctx_terrain_name = _token_obs_params(hv, cli, terrain)
    setattr(cfg.observations.policy, "author_token", hv.ObsTerm(func=residual_token_obs, params=params))
    # critic 特权组同步追加（**有意的不对称**：critic 可比 actor 多看 token，SF-1 声明；
    # policy/critic 同 func 同参，故 critic 看到的 token 与 actor 逐位一致，仅多这一项）
    if getattr(cfg.observations, "critic", None) is not None:
        setattr(cfg.observations.critic, "author_token", hv.ObsTerm(func=residual_token_obs, params=params))
    return {
        "ctx_terrain": ctx_terrain_name,
        "env_terrain": str(terrain),
        "token_stats": cli.token_stats,
        "token_bound": params["token_bound"],
        "token_alpha": params["token_alpha"],
        "adapter_chunk_envs": params["adapter_chunk_envs"],
        "adapter_dtype": params["adapter_dtype"],
        "privileged_critic": True,
    }


def _override_residual_action_scale(cfg, hv) -> dict:
    """把残差臂 JointPositionAction 的 scale 覆写为**逐关节 sonic_scale 向量**（执行/反馈互逆）。

    **只改本臂 cfg 实例**（`cfg.actions.joint_pos`），不动 `g1_velocity_decoder_env.py`
    的 `DirectActionsCfg` 默认值（保 direct 臂 D064 逐字可比）；因此 C 臂（D065）默认仍是
    标量 0.5，其潜伏缺陷只登记不修（见模块 docstring「D065 潜伏缺陷警示」）。

    做法：`cfg.actions.joint_pos.scale = {关节名: sonic_scale[j]}`（isaaclab 2.1.0 只接受
    float|dict，见 `build_sonic_scale_map` docstring）。use_default_offset=True 不动（offset
    = default，与反馈端 `q_des-default` 消项一致）。同时断言覆写落地（配置对象自检，
    fail loud，不静默保留 0.5）。

    返回 {"scale_map": ..., "n_joints": 29, "source": ..., "legacy_scalar": 0.5}（进打印）。
    """
    action_term = getattr(cfg.actions, "joint_pos", None)
    if action_term is None:
        raise RuntimeError(
            "残差臂 cfg.actions.joint_pos 不存在（lora_policy 变体应 = DirectActionsCfg 的 "
            "JointPositionActionCfg）——动作通路接线漂移，拒绝开训"
        )
    if not bool(getattr(action_term, "use_default_offset", False)):
        raise RuntimeError(
            "残差臂 joint_pos.use_default_offset 应为 True（offset=default；否则反馈端 "
            "(q_des-default)/sonic_scale 与执行端仿射不互逆）"
        )
    sonic_scale = np.asarray(hv._sonic_scale_isaac(), dtype=np.float64).reshape(-1)
    if sonic_scale.shape != (ACTION_DIM,):
        raise RuntimeError(f"sonic_scale 应为 ({ACTION_DIM},)，收到 {tuple(sonic_scale.shape)}")
    scale_map = build_sonic_scale_map(sonic_scale)
    action_term.scale = scale_map
    # 覆写落地自检（配置对象级；运行期 action term 的 _scale 另在 _run 断言）
    if action_term.scale != scale_map:
        raise RuntimeError("joint_pos.scale 覆写未落地（仍为旧值？）")
    if action_term.scale == RESIDUAL_LEGACY_ACTION_SCALE:
        raise RuntimeError("joint_pos.scale 仍等于旧标量 0.5（覆写失效）")
    return {
        "scale_map": scale_map,
        "n_joints": int(sonic_scale.shape[0]),
        "source": SONIC_SCALE_SOURCE,
        "legacy_scalar": RESIDUAL_LEGACY_ACTION_SCALE,
    }


def assert_residual_action_scale(env, hv, *, action_term_name: str = "joint_pos") -> dict:
    """运行期守卫：action term 实际 `_scale` 逐关节 == sonic_scale（fail loud）。

    这是执行/反馈互逆的**唯一硬证据**——cfg 层覆写对、但运行期 term 解析漂移（关节名不
    命中/资产序不同）时，`_scale` 会退回 ones 或错序，闭环增益即 ≠1。

    读取与比对（对齐 isaaclab 2.1.0 实际存储）：
      - dict scale 经 `resolve_matching_names_values` 落到 `_scale = ones(num_envs, action_dim)`
        的 **asset 关节序**列（joint_actions.py:83-86），故 `_scale` 是 (num_envs, 29) 张量；
        取第 0 行（并断言各行一致，逐 env 无差异）。
      - 生效 scale 重排到 SONIC 序后须逐位 == `_sonic_scale_isaac()`：优先用 term 自身
        `_joint_names` 建 {名: scale} 映射（对 term 关节序稳健），缺名时退回
        `resolve_sonic_joint_ids(asset.joint_names)`（反馈端 `sonic_proprio_hist` 把
        processed_actions 重排到 SONIC 序用的同一函数，故这是闭环互逆的正确判据）。
    返回 {"n_joints", "gain_min", "gain_max", "bitwise": True}。
    """
    term = env.action_manager.get_term(action_term_name)
    scale = getattr(term, "_scale", None)
    if scale is None:
        raise RuntimeError(f"action term {action_term_name!r} 无 _scale（isaaclab 版本漂移？）")
    t = hv.torch
    if not t.is_tensor(scale):
        # 标量 float（未覆写/覆写失效）-> 显式失败，不静默当向量
        raise RuntimeError(
            f"action term {action_term_name!r} 的 _scale 是标量 {scale!r}（应逐关节向量）——"
            "执行/反馈仿射不互逆，拒绝开训"
        )
    if scale.ndim == 2:
        # (num_envs, action_dim)：逐 env 必须同值（本臂 scale 与环境无关）
        if not bool((scale == scale[0:1]).all()):
            raise RuntimeError("action term _scale 逐 env 不一致（本臂 scale 应与环境无关）")
        col = scale[0]
    elif scale.ndim == 1:
        col = scale
    else:
        raise RuntimeError(f"action term _scale 维度异常：{tuple(scale.shape)}")
    col = col.detach().to("cpu", t.float64)

    # SONIC 序生效 scale：优先按 term 自身关节名映射（对 term 关节序稳健）；缺名则退回
    # asset 序（resolve_sonic_joint_ids，与 sonic_proprio_hist 重排 processed_actions 同式）。
    term_names = getattr(term, "_joint_names", None)
    sonic_names = list(sonic_joint_names())
    if term_names is not None and len(term_names) == int(col.numel()):
        name_to_scale = {str(n): float(v) for n, v in zip(term_names, col.tolist())}
        missing = [n for n in sonic_names if n not in name_to_scale]
        if missing:
            raise RuntimeError(f"action term 关节名缺 SONIC 关节 {missing}（scale 映射不完整）")
        got_sonic = np.array([name_to_scale[n] for n in sonic_names], dtype=np.float64)
    else:
        try:  # noqa: PLC0415
            from isaac.sonic_action_term import resolve_sonic_joint_ids  # noqa: PLC0415
        except ImportError:  # pragma: no cover
            from apt_g1.isaac.sonic_action_term import resolve_sonic_joint_ids  # type: ignore[no-redef]
        asset = env.scene["robot"]
        ids, _identity = resolve_sonic_joint_ids(asset.joint_names)
        ids_t = t.as_tensor(ids, dtype=t.long, device=col.device)
        got_sonic = col[ids_t].numpy()  # SONIC 序的生效 scale
    sonic_scale = np.asarray(hv._sonic_scale_isaac(), dtype=np.float64).reshape(-1)
    if sonic_scale.shape != (ACTION_DIM,):
        raise RuntimeError(f"sonic_scale 应为 ({ACTION_DIM},)，收到 {tuple(sonic_scale.shape)}")
    if got_sonic.shape != (ACTION_DIM,):
        raise RuntimeError(f"生效 scale（SONIC 序）形状 {got_sonic.shape} != ({ACTION_DIM},)")
    exp = sonic_scale.astype(np.float64)
    if not np.array_equal(got_sonic, exp):
        raise RuntimeError(
            "action term 生效 scale（SONIC 序）!= sonic_scale（执行/反馈仿射不互逆）："
            f"maxdiff={float(np.max(np.abs(got_sonic - exp)))} "
            f"argmax={int(np.argmax(np.abs(got_sonic - exp)))}"
        )
    gain = exec_feedback_gain(got_sonic, exp)  # 逐关节，应恒 1
    if not np.array_equal(gain, np.ones(ACTION_DIM)):
        raise RuntimeError(f"闭环增益非恒 1（{float(gain.min())}..{float(gain.max())}）")
    return {
        "n_joints": int(got_sonic.shape[0]),
        "gain_min": float(gain.min()),
        "gain_max": float(gain.max()),
        "bitwise": True,
    }


def _build_residual_env_cfg(hv, cli, device: str, suffix: str):
    """残差臂 env cfg：direct 动作通路（29 维）+ max_forward 奖励 + author_token 观测。

    env 侧走 sibling 既有 `action="lora_policy"` 变体（= direct 臂动作通路逐字 + policy 组
    末位 930 维 decoder 本体历史观测），残差臂的唯一 env 侧增量 = 追加 `author_token` 观测项。
    """
    td = _import_train_g1_decoder()
    cfg, how = td._build_env_cfg(
        hv.make_env_cfg, hv.mdp, hv.SonicDecoderActionTermCfg, cli.terrain, "lora_policy", cli, device, suffix
    )
    # 执行/反馈仿射互逆（CVGL 探针根因修复）：JointPositionAction scale 覆写为逐关节
    # sonic_scale（与 sonic_proprio_hist 反归一化同源），闭环增益恒 1。只改本臂 cfg 实例。
    scale_info = _override_residual_action_scale(cfg, hv)
    recipe = _apply_max_forward_recipe(cfg, hv, cli.forward_weight)
    token_info = _attach_token_obs(cfg, hv, cli, cli.terrain)
    print(
        f"[d067] residual env cfg via {how} + max_forward 奖励 + author_token 观测：{recipe}；"
        f"token_info={json.dumps(token_info, ensure_ascii=False)}；"
        f"residual_action_scale={json.dumps(scale_info, ensure_ascii=False)}",
        flush=True,
    )
    return cfg, how, recipe, token_info, scale_info


def _build_residual_policy_kwargs(hv, env, cli, device: str) -> tuple[dict, dict]:
    """构造残差臂 policy kwargs：冻结 decoder 分支 + token/proprio 切片（运行期定位）。

    SF-2 守卫：`official_obs_dim`（r 的输入宽度）由 `official_obs_slice` 按**运行期
    term 序**推出（policy 总宽 − sonic_proprio − author_token），并断言 official 段是
    policy obs 前缀（起始=0）；不再让 `flat[:, :official_obs_dim]` 的隐式前缀假设裸奔。
    """
    onnx_path = cli.onnx_path or hv.SONIC_DECODER_ONNX
    decoder = hv.SonicTorchDecoder(onnx_path, device=device)
    branch = make_frozen_decoder_branch(decoder)
    for p in branch.parameters():
        p.requires_grad_(False)
    branch.eval()
    token_slot = _policy_obs_slot(env, hv.torch, "author_token", TOKEN_DIM)
    proprio_slot = _policy_obs_slot(env, hv.torch, "sonic_proprio", PROPRIO_DIM)
    om = env.observation_manager
    off = official_obs_slice(om.active_terms["policy"], om.group_obs_term_dim["policy"])
    kwargs = {
        "decoder": branch,
        "token_slice": (token_slot["start"], token_slot["width"]),
        "proprio_slice": (proprio_slot["start"], proprio_slot["width"]),
        "official_obs_dim": int(off["width"]),
        "sigma_trainable": bool(cli.sigma_trainable),
        "sigma": float(cli.sigma) if cli.sigma is not None else None,
    }
    info = {
        "token_slot": token_slot,
        "proprio_slot": proprio_slot,
        "official_slice": off,
        "onnx_path": onnx_path,
        "decoder_frozen": True,
    }
    return kwargs, info


def _prune_frozen_params(optimizer) -> dict:
    """把 requires_grad=False 的参数从优化器 param_groups 摘掉（同 D065 手法）。

    落地「PPO 只训 r（+critic）」：decoder/actor 占位/σ（冻结时）摘除，只剩 residual
    与 critic。**摘除须先于 runner.load**（resume 时 optimizer state 按已摘除的组存）。
    """
    removed, kept = [], 0
    for grp in optimizer.param_groups:
        keep = [p for p in grp["params"] if p.requires_grad]
        removed.extend([p for p in grp["params"] if not p.requires_grad])
        grp["params"] = keep
        kept += len(keep)
    return {"n_removed_tensors": len(removed), "n_kept_tensors": kept}


def _resolve_ppo_and_policy(runner) -> tuple[object, object, object]:
    """从 rsl_rl runner 取 (alg, optimizer, policy)（属性名兼容查找，缺即显式失败）。"""
    alg = getattr(runner, "alg", None) or getattr(runner, "algorithm", None)
    if alg is None:
        raise RuntimeError(f"runner 上找不到 alg/algorithm（{type(runner).__name__}）")
    optimizer = getattr(alg, "optimizer", None) or getattr(alg, "optim", None)
    if optimizer is None:
        raise RuntimeError(f"PPO 上找不到 optimizer/optim（{type(alg).__name__}）")
    policy = getattr(alg, "policy", None)
    if policy is None:
        raise RuntimeError(f"PPO 上找不到 policy（{type(alg).__name__}）")
    return alg, optimizer, policy


# ------------------------------------------------- update 内首 NaN 溯源探针（第三层修复）
# 实证：σ 轨迹全程健康却在某次 update 内部变 NaN（tmp/_d067_verify/freeze_sigmafix_iter950_crash.txt）
# ⇒ 需要「update 内部」的观测点，而非只有 pre/post 两行。探针**只记录不阻断**（保命带在，
# 训练继续），让后续崩点自然复现时留下完整现场（哪个 stage/哪个量先出非有限值）。
NAN_PROBE_TAG = "nan_probe"
# 第四层：幅值趋势行（逐 key/逐段 min/max/abs_max/mean + 参数快照）。**与首 NaN 行分 tag**：
# 首 NaN 行的语义是「有非有限量才落一行」，幅值行是「每次都落」的趋势量，混在一起会破坏
# 既有 `nan_probe` 断言（全有限时不应有 nan_probe 行）与 reviewer 核对口径。
NAN_PROBE_MAG_TAG = "nan_probe_mag"
# 第五层：advantage/returns 消毒器写回行（**与首 NaN 行分 tag**：首 NaN 行「有非有限才落」，
# 消毒行「消毒写回才落」，且带 n_zeroed/purged_full 等写回语义字段）。健康路径（全有限）零写回。
NAN_PROBE_SANITIZE_TAG = "nan_probe_sanitize"
# probe 张量时跳过的超大 obs key（防把整段 height_scan（4096×187）反复 copy 到 CPU 造成 OOM/慢）。
NAN_PROBE_MAX_TENSOR_ELEMS = 1 << 24


def _to_cpu_tensor(obj) -> torch.Tensor | None:
    """尽力把 obj 转成 CPU float32 张量；非张量/无张量元素返回 None（探针 best-effort，不抛）。"""
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", torch.float32)
    if isinstance(obj, (tuple, list)):
        tensors = [t for t in obj if isinstance(t, torch.Tensor)]
        if not tensors:
            return None
        try:
            return torch.cat([t.detach().to("cpu", torch.float32).reshape(-1) for t in tensors])
        except Exception:  # noqa: BLE001（形状不一/空张量 -> 跳过）
            return None
    return None


def _nan_probe_obs_slices(policy) -> dict:
    """policy 段的切片方案（按 policy 类可及属性推；推不出则整体当一段）。

    - residual 臂（`ResidualDecoderPolicy`）：official = `[:, :official_obs_dim]`；
      proprio = `[:, proprio_start:proprio_start+930]`；token = `[:, token_start:token_start+64]`
      （三者由 policy 构造期按运行期 term 序算出，见 `_build_residual_policy_kwargs`）。
    - 其它臂 / 属性缺位：`official_obs_slice` 推不出 ⇒ 只记整体 `policy`（整体已能定位污染）。
    返回 `{name: (start, width)}`（width=None 表示到末尾）。
    """
    out: dict[str, tuple[int, int | None]] = {}
    official = getattr(policy, "official_obs_dim", None)
    if official is not None:
        try:
            out["official"] = (0, int(official))
        except (TypeError, ValueError):  # pragma: no cover（防御：非整数宽度）
            pass
    p_start = getattr(policy, "proprio_start", None)
    p_width = getattr(policy, "proprio_width", None)
    if p_start is not None and p_width is not None:
        try:
            out["proprio"] = (int(p_start), int(p_width))
        except (TypeError, ValueError):  # pragma: no cover
            pass
    t_start = getattr(policy, "token_start", None)
    t_width = getattr(policy, "token_width", None)
    if t_start is not None and t_width is not None:
        try:
            out["token"] = (int(t_start), int(t_width))
        except (TypeError, ValueError):  # pragma: no cover
            pass
    return out


def _collect_nan_probe_candidates(alg, stage: str, result=None) -> list[tuple[str, object]]:
    """收集该 stage 可及的非有限量候选（name, tensor）；**永不抛**（探针 best-effort）。

    探测点（尽量覆盖「污染从哪来」）：
      - `pre`：`alg.storage.observations` 逐 key（policy 段按 `_nan_probe_obs_slices` 切分）；
      - `pre_mid`：`alg.storage.advantages/returns/values`（GAE 输出，update 的输入）——
        首 NaN 落在 advantage 计算还是 loss/backward 的分界点；
      - `post`：返回的 loss_dict 逐 key + `alg.policy.std` 参数本体 + 策略输出
        （`policy.mu(...)` 优先、否则 `policy.actor(...)`，可及才跑；仅对 policy 段 obs）。
    """
    cands: list[tuple[str, object]] = []
    policy = getattr(alg, "policy", None)

    if stage == "pre_mid":
        return _collect_advantage_candidates(alg)

    if stage == "pre":
        storage = getattr(alg, "storage", None)
        obs = getattr(storage, "observations", None) if storage is not None else None
        if isinstance(obs, dict):
            for key, val in obs.items():
                # 大张量跳过判断必须在 _to_cpu_tensor **之前**——否则超大 obs
                # 先被 GPU→CPU 全量 copy 后才跳过，白付一次拷贝+同步（reviewer 实测抓出）。
                if isinstance(val, torch.Tensor) and \
                        val.numel() > NAN_PROBE_MAX_TENSOR_ELEMS:
                    continue
                t = _to_cpu_tensor(val)
                if t is None:
                    continue
                if t.numel() > NAN_PROBE_MAX_TENSOR_ELEMS:
                    # 非张量入参（如 list/numpy）转 CPU 后仍超阈值则跳过，防拖慢/OOM；
                    # policy 段（含 930 proprio）远小于阈值，不受影响。
                    continue
                # policy 段先按切片探（official/proprio/token，定位到段），再退整体兜底；
                # 其它 key 无切片口径，直接整体探。
                if key == "policy" and t.dim() == 2:
                    width = int(t.shape[-1])
                    for part, (start, w) in _nan_probe_obs_slices(policy).items():
                        end = width if w is None else min(int(start) + int(w), width)
                        if 0 <= int(start) < end:
                            cands.append((f"obs.policy[{part}]", t[:, int(start) : end]))
                cands.append((f"obs.{key}", t))
        else:
            t = _to_cpu_tensor(obs)
            if t is not None:
                cands.append(("obs", t))

    if stage == "post":
        if isinstance(result, dict):
            for key, val in result.items():
                t = _to_cpu_tensor(val)
                if t is not None:
                    cands.append((f"loss.{key}", t))
        elif result is not None:
            t = _to_cpu_tensor(result)
            if t is not None:
                cands.append(("result", t))
        std = getattr(policy, "std", None)
        t_std = _to_cpu_tensor(std)
        if t_std is not None:
            cands.append(("policy.std", t_std))
        # 策略输出：mu 优先（residual 臂 μ 路径），否则 actor 干（direct/decoder 臂）
        obs_policy = None
        storage = getattr(alg, "storage", None)
        raw_obs = getattr(storage, "observations", None) if storage is not None else None
        if isinstance(raw_obs, dict) and isinstance(raw_obs.get("policy"), torch.Tensor):
            obs_policy = raw_obs["policy"]
        if obs_policy is not None and policy is not None:
            with torch.no_grad():
                for attr, name in (("mu", "policy.mu"), ("actor", "policy.actor")):
                    fn = getattr(policy, attr, None)
                    if not callable(fn):
                        continue
                    try:
                        out = fn(obs_policy)
                    except Exception:  # noqa: BLE001（探针 best-effort：前向失败不影响训练）
                        continue
                    t = _to_cpu_tensor(out)
                    if t is not None:
                        cands.append((name, t))
                    break
    return cands


def _collect_advantage_candidates(alg) -> list[tuple[str, object]]:
    """advantage/returns 探针候选（第四层 stage=`pre_mid`）：从 `alg.storage` 取 update 的输入量。

    rsl_rl 2.3.3 `RolloutStorage`（服务器 .venv_isaac 源码核对）属性名：
      `advantages` / `returns` / `values`（rl 训练类型下均为 (T, N, 1) 张量）；
      `action_mean`(mu) / `action_sigma`(sigma) 亦有，但 sigma 已在 post 快照覆盖，此处不重复。
    取不到（属性缺位 / 非张量）静默跳过该 key；**永不抛**。大张量按同一阈值跳过（copy 前置判断）。
    这些量是 `compute_returns` 的输出、update 内 `advantages_batch`/`returns_batch` 的来源——
    若在此处即非有限，则首 NaN 在 advantage 计算（GAE）而非 loss/backward。
    """
    cands: list[tuple[str, object]] = []
    storage = getattr(alg, "storage", None)
    if storage is None:
        return cands
    for attr in ("advantages", "returns", "values"):
        val = getattr(storage, attr, None)
        if not isinstance(val, torch.Tensor):
            continue
        if val.numel() > NAN_PROBE_MAX_TENSOR_ELEMS:
            continue   # copy 前置判断：超阈值先跳，不做 GPU→CPU 全量 copy
        t = _to_cpu_tensor(val)
        if t is None:
            continue
        cands.append((f"storage.{attr}", t))
    return cands


def _collect_mag_candidates(alg, stage: str, result=None) -> list[tuple[str, object]]:
    """幅值趋势行的候选（第四层）：逐 key/逐段的可及量，**含全有限量**（与首 NaN 探针互补）。

    - `pre`：`alg.storage.observations` 逐 key；policy 段按 official/proprio/token 切片**逐段**
      记幅值（量级趋势从此可见）。
    - `pre_mid`：advantages/returns/values（见 `_collect_advantage_candidates`）。
    - `post`：返回的 loss_dict 逐 key + `policy.std` 本体 + 策略输出（mu/actor）。
    复用 `_to_cpu_tensor` 与大张量跳过口径；**永不抛**（best-effort）。
    """
    if stage == "pre_mid":
        return _collect_advantage_candidates(alg)

    policy = getattr(alg, "policy", None)
    cands: list[tuple[str, object]] = []

    if stage == "pre":
        storage = getattr(alg, "storage", None)
        obs = getattr(storage, "observations", None) if storage is not None else None
        if isinstance(obs, dict):
            for key, val in obs.items():
                if isinstance(val, torch.Tensor) and \
                        val.numel() > NAN_PROBE_MAX_TENSOR_ELEMS:
                    continue   # copy 前置判断
                t = _to_cpu_tensor(val)
                if t is None or t.numel() > NAN_PROBE_MAX_TENSOR_ELEMS:
                    continue
                if key == "policy" and t.dim() == 2:
                    width = int(t.shape[-1])
                    for part, (start, w) in _nan_probe_obs_slices(policy).items():
                        end = width if w is None else min(int(start) + int(w), width)
                        if 0 <= int(start) < end:
                            cands.append((f"obs.policy[{part}]", t[:, int(start):end]))
                cands.append((f"obs.{key}", t))
        else:
            t = _to_cpu_tensor(obs)
            if t is not None:
                cands.append(("obs", t))

    if stage == "post":
        if isinstance(result, dict):
            for key, val in result.items():
                t = _to_cpu_tensor(val)
                if t is not None:
                    cands.append((f"loss.{key}", t))
        elif result is not None:
            t = _to_cpu_tensor(result)
            if t is not None:
                cands.append(("result", t))
        t_std = _to_cpu_tensor(getattr(policy, "std", None))
        if t_std is not None:
            cands.append(("policy.std", t_std))
    return cands


def _first_nonfinite_probe(name: str, t: torch.Tensor) -> dict | None:
    """首个非有限量的探针记录（全有限 -> None）。

    字段：name / shape / dtype / n_nonfinite / min / max / first（首个非有限值）/ first_index
    （扁平下标 + 多维坐标，便于回查是哪个 env/哪个通道先炸）。min/max 用 `nanmin/nanmax`
    （全 NaN 时退回 None）。**只读**，不改张量。
    """
    if t.numel() == 0:
        return None
    finite = torch.isfinite(t)
    n_bad = int((~finite).sum().item())
    if n_bad == 0:
        return None
    flat = t.reshape(-1)
    idx = int((~torch.isfinite(flat)).nonzero(as_tuple=False)[0].item())
    fin_mask = torch.isfinite(flat)
    t_min = float(flat[fin_mask].min().item()) if bool(fin_mask.any()) else None
    t_max = float(flat[fin_mask].max().item()) if bool(fin_mask.any()) else None
    coords = None
    if t.dim() > 1:
        coords = [int(c) for c in torch.unravel_index(torch.tensor(idx), t.shape)]
    return {
        "name": str(name),
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "n_nonfinite": n_bad,
        "n_total": int(flat.numel()),
        "min": t_min,
        "max": t_max,
        "first": float(flat[idx].item()),
        "first_index": idx,
        "first_coord": coords,
    }


def _finite_mag_stats(name: str, t: torch.Tensor) -> dict:
    """张量幅值统计（**只记录不抛**，第四层幅值趋势行用）。

    字段：name / shape / dtype / n_total / n_nonfinite / min / max / abs_max / mean
    （min/max/abs_max/mean 只对**有限元素**算；全 NaN/inf 时置 None）。与
    `_first_nonfinite_probe` 的 min/max 口径一致（都用有限元素），但**全有限时也返回记录**
    （首 NaN 探针只记非有限量），故用于「量级趋势」而非「崩点定位」。
    """
    if t.numel() == 0:
        return {
            "name": str(name), "shape": list(t.shape), "dtype": str(t.dtype),
            "n_total": 0, "n_nonfinite": 0,
            "min": None, "max": None, "abs_max": None, "mean": None,
        }
    flat = t.reshape(-1)
    finite = torch.isfinite(flat)
    n_bad = int((~finite).sum().item())
    fin = flat[finite]
    if bool(fin.numel()):
        t_min = float(fin.min().item())
        t_max = float(fin.max().item())
        t_absmax = float(fin.abs().max().item())
        t_mean = float(fin.mean().item())
    else:
        t_min = t_max = t_absmax = t_mean = None
    return {
        "name": str(name),
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "n_total": int(flat.numel()),
        "n_nonfinite": n_bad,
        "min": t_min,
        "max": t_max,
        "abs_max": t_absmax,
        "mean": t_mean,
    }


# 参数快照只取各首末 nn.Linear 的 weight（actor/critic 各 3 层 MLP，取首末即可看「权重是否
# 同刻全灭」；全量 29 维 std + 6 个 weight 足够印证「loss NaN 染全部梯度」单点假说，且开销可控）。
_NAN_PROBE_PARAM_SNAP_KEYS = ("std", "actor_first_weight", "actor_last_weight",
                              "critic_first_weight", "critic_last_weight")


def _linear_ends(module) -> dict:
    """从 nn.Sequential（或任意 module）里取**首个/末个** `nn.Linear` 的 weight（可及才给）。

    返回 {"first_weight": Tensor|None, "last_weight": Tensor|None}；找不到返回全 None。
    只取 weight 本体（detach），不做前向，不建图。
    """
    linears: list[nn.Module] = []
    try:
        for m in module.modules():
            if isinstance(m, nn.Linear):
                linears.append(m)
    except Exception:  # noqa: BLE001（探针 best-effort）
        return {"first_weight": None, "last_weight": None}
    if not linears:
        return {"first_weight": None, "last_weight": None}
    return {"first_weight": linears[0].weight, "last_weight": linears[-1].weight}


def _nan_probe_param_snapshot(policy) -> dict:
    """参数快照：`policy.std` 全量 + actor/critic 各首末 weight 的 isfinite/范数（**只读不抛**）。

    为什么用「快照对比」而非 backward 钩子（指令明令不做钩子，侵入风险）：update 内部不可
    侵入，但 orig_update 前后各拍一张参数快照即可回答「梯度把哪些参数打成了 NaN」——若 std
    与 actor/critic 权重**同刻**变 NaN，则印证「某中间量 NaN → loss NaN → 反向染全部梯度」
    的单点假说（clip_grad_norm_(1.0) 对 NaN 无效，故参数全灭）。

    返回 {key: {isfinite: bool, n_nonfinite: int, norm: float|None, n_total: int}}；某 key
    取不到（非 scalar σ / 无 actor/critic 属性）则整体静默跳过该 key（不写 null 占位，
    保持记录简洁）。**纯只读**（detach + 范数），不改任何参数。
    """
    out: dict = {}
    std = getattr(policy, "std", None)
    if isinstance(std, torch.Tensor):
        out["std"] = _param_snap_entry(std)
    actor = getattr(policy, "actor", None)
    if actor is not None:
        ends = _linear_ends(actor)
        if ends["first_weight"] is not None:
            out["actor_first_weight"] = _param_snap_entry(ends["first_weight"])
        if ends["last_weight"] is not None:
            out["actor_last_weight"] = _param_snap_entry(ends["last_weight"])
    critic = getattr(policy, "critic", None)
    if critic is not None:
        ends = _linear_ends(critic)
        if ends["first_weight"] is not None:
            out["critic_first_weight"] = _param_snap_entry(ends["first_weight"])
        if ends["last_weight"] is not None:
            out["critic_last_weight"] = _param_snap_entry(ends["last_weight"])
    return out


def _param_snap_entry(param: torch.Tensor) -> dict:
    """单个参数的快照项（isfinite / n_nonfinite / 有限元素 L2 范数 / 元素数）；**只读**。"""
    with torch.no_grad():
        t = param.detach().reshape(-1).to(torch.float32)
        finite = torch.isfinite(t)
        n_bad = int((~finite).sum().item())
        fin = t[finite]
        norm = float(torch.linalg.vector_norm(fin).item()) if bool(fin.numel()) else None
        return {
            "isfinite": n_bad == 0,
            "n_nonfinite": n_bad,
            "norm": norm,
            "n_total": int(t.numel()),
        }


def _nan_probe_param_diff(pre: dict, post: dict) -> dict:
    """参数快照 diff：给出「pre 有限 -> post 非有限」的 key（同刻变 NaN 的印证）；只记录。"""
    flipped = []
    for key, post_entry in post.items():
        pre_entry = pre.get(key)
        if pre_entry is None:
            continue
        if pre_entry.get("isfinite") and not post_entry.get("isfinite"):
            flipped.append(key)
    return {"param_nan_flipped": flipped, "param_nan_flipped_n": len(flipped)}


# ------------------------------------------------- 存储消毒器（第五/六/七层修复）
# 第七层：消毒范围从 GAE 链三张量扩到**观测端**两路（critic 重算路径的污染源，见模块
# docstring 第七层段）。`privileged_observations` 是张量、`observations` 是 dict（policy/critic
# 两组）——由 `_sanitize_storage_tensor` 按运行期类型分派。
_SANITIZE_STORAGE_ATTRS = (
    "advantages", "returns", "values",           # GAE 链（第五/六层）
    "privileged_observations", "observations",   # 观测端（第七层：ppo.update 内 critic 重算输入）
)


def _sanitize_tensor_gpu(t: torch.Tensor, name: str) -> tuple[dict | None, torch.Tensor]:
    """GPU 侧检测 + 消毒**单个**张量；返回 `(写回记录 | None, 消毒后张量)`。

    **第七层核心（大张量检测改 GPU 侧）**：检测 = GPU 侧 `isfinite` 归约
    `bad_n = int((~torch.isfinite(t)).sum())`——**零全量拷贝**，只把 1 个标量同步回 CPU；
    因此**超阈值张量（obs ≈37M 元素 > NAN_PROBE_MAX_TENSOR_ELEMS）也照常检测，不再跳过**。
    - `bad_n == 0`（健康）：返回 `(None, t)`——**原张量对象**，调用方零写回、对象引用不变；
    - `bad_n > 0`：GPU 侧 `nan_to_num` 重赋值（返回**普通 tensor**，绕开 inference-tensor
      就地限制，第六层口径）+ 统计。`first`/`first_coord` 在 GPU 侧 `nonzero` 找**首个**
      非有限索引，只回传该索引与单值（**只拷小片**），不回传全量数据。

    为什么不能「copy 前置跳过」：消毒器要保证 update 的输入**无 NaN**，跳过大张量就等于对
    obs 端（本次污染的真正源头）不设防；检测/消毒都在 GPU 侧做完，只有标量/索引过 CPU。
    """
    if not isinstance(t, torch.Tensor) or not t.is_floating_point():
        return None, t
    with torch.no_grad():
        bad_n = int((~torch.isfinite(t)).sum().item())   # GPU 侧归约，单标量同步，零全量拷贝
        if bad_n == 0:
            return None, t
        n_total = int(t.numel())
        flat = t.reshape(-1)
        # 首个非有限索引：GPU 侧 nonzero -> 只回传索引（小张量），不拷全量数据
        idx = int((~torch.isfinite(flat)).nonzero(as_tuple=False)[0].item())
        first_val = float(flat[idx].item())
        clean = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)   # GPU 侧重赋值，零全量拷贝
        abs_max_before = float(clean.abs().max().item()) if bad_n < n_total else None
        coords = None
        if t.dim() > 1:
            coords = [int(c) for c in torch.unravel_index(torch.tensor(idx, device=t.device), t.shape)]
        rec = {
            "name": str(name),
            "n_nonfinite": bad_n,
            "n_total": n_total,
            "n_zeroed": bad_n,
            "purged_full": bad_n == n_total,   # 全量 NaN -> actor 梯度 0、critic 被拉向 0（非空学习步）
            "abs_max_before": abs_max_before,
            "first": first_val,
            "first_index": idx,
            "first_coord": coords,
        }
        return rec, clean


def _sanitize_storage_tensor(alg, attr: str) -> list[dict]:
    """对 `alg.storage.<attr>` 消毒：非有限元素置 0；返回写回记录列表（健康/不可及 -> 空列表）。

    **第六层修复（inference-tensor 写回）**：`storage.advantages` 由 rollout 在
    `torch.inference_mode()` 下产生（on_policy_runner.py:201）⇒ 它是 **inference tensor**，
    对它的**就地** `masked_fill_` 会抛 `RuntimeError: Inplace update to inference tensor
    outside InferenceMode is not allowed`（第六跑实证：300 次消毒全失败，returns 恰好非
    inference tensor 故成功）。修法 = **重赋值路径**：`torch.nan_to_num(t, nan=0.0,
    posinf=0.0, neginf=0.0)` 返回**普通 tensor**（`is_inference()==False`），
    `setattr(storage, attr, ...)` 绕开 inference-tensor 的就地限制。rsl_rl 2.3.3
    `mini_batch_generator` 每次调用**现读** `self.advantages`（服务器 .venv_isaac 源码核对），
    故重赋值对 `update` 可见（就地路径已删除，见下方注释）。

    **第七层（观测端 + 类型分派 + 大张量不跳过）**：
      - `privileged_observations`（张量）与 `observations`（**dict**：policy/critic 两组）一并
        消毒——前者正是 `ppo.update()` 内 critic **现算**的输入（第七跑实证污染源），消毒
        values 管不到重算，必须在上游 obs 端净化；
      - dict 则**逐 key** 走同一 GPU 侧路径，命中才整 dict 重赋值（未命中零写回）；
      - 检测/消毒全在 **GPU 侧**（见 `_sanitize_tensor_gpu`），**超阈值大张量不再跳过**。

    **只在检测到非有限时才写**（健康路径逐位零改动、零写回，对象引用亦不变；仅付一次 GPU 侧
    归约同步 `bad_n` 的必要代价）。

    口径（见模块 docstring 第五/六/七层）：单个离群置 0 对归一化统计量偏置可忽略；全量 NaN
    置 0 ⇒ 权重存活但 actor 梯度 0、critic 被拉向 0（**非空学习步**，见 docstring 口径修正）。

    旧就地路径（已删，仅留说明）：`val.masked_fill_(~finite, 0.0)`——对普通张量可行，但对
    inference tensor 抛异常（第六层实证），故弃用；改用 `nan_to_num` 重赋值。
    **永不抛**（探针/消毒器 best-effort）。
    """
    storage = getattr(alg, "storage", None)
    if storage is None:
        return []
    val = getattr(storage, attr, None)
    if isinstance(val, dict):
        # observations 是 dict（policy/critic 两组）：逐 key 消毒，命中才整 dict 重赋值。
        recs: list[dict] = []
        new_val = dict(val)
        changed = False
        for key, sub in val.items():
            rec, clean = _sanitize_tensor_gpu(sub, f"storage.{attr}.{key}")
            if rec is None:
                continue
            recs.append(rec)
            new_val[key] = clean
            changed = True
        if changed:
            setattr(storage, attr, new_val)
        return recs
    if not isinstance(val, torch.Tensor) or not val.is_floating_point() or val.numel() == 0:
        return []
    rec, clean = _sanitize_tensor_gpu(val, f"storage.{attr}")
    if rec is None:
        return []   # 健康路径：不写、不记（发射解锁件只在真出 NaN 时动作）
    setattr(storage, attr, clean)
    return [rec]


def _sanitize_storage_attrs(alg) -> list[dict]:
    """消毒 `_SANITIZE_STORAGE_ATTRS` 五路（orig_update 前调用）；返回写回记录列表。

    `advantages`/`returns`/`values` = GAE 链的输入（values）与输出；`privileged_observations`/
    `observations` = **观测端**（第七层：`ppo.update()` 内 critic 用 privileged_observations
    **现算** values，是第七跑坐实的污染源——消毒 values 管不到重算，必须在上游 obs 端净化）。
    各项独立 best-effort：任一项失败不影响其余，也**不影响 update**（异常在此吞掉并警告）。
    返回空列表 = 健康路径（零写回）。
    """
    out: list[dict] = []
    for attr in _SANITIZE_STORAGE_ATTRS:
        try:
            out.extend(_sanitize_storage_tensor(alg, attr))
        except Exception as exc:  # noqa: BLE001（消毒器 best-effort：绝不打断训练）
            print(f"[WARN] 存储消毒失败（attr={attr}，忽略）：{type(exc).__name__}: {exc}", flush=True)
            continue
    return out


# ------------------------------------------------- rollout 期观测源消毒器（第八层）
# 实证（r4 里程碑）：update 侧消毒器 1634 次/1254it，purged_full=411（33% update 空学习步）——
# obs NaN 在 rollout 期进 buffer 后 update 侧只能救权重、救不了该次 update（归一化已放大成全量），
# 且策略直接吃 NaN obs -> 动作 NaN -> 物理更炸（正反馈）。治本 = 在 obs 进入训练管线前净化：
# 本件在 env 出口（step/reset 返回值）逐 key 净化 + 计数，**只统计与净化，不改动作语义**；
# update 侧五路消毒器（上）保留 = 防御纵深。语义/包装手法/性能见模块 docstring 第八层段。
OBS_SANITIZE_TAG = "obs_sanitize"   # 计数行 tag（落 std_trajectory.log，与 update 侧各 tag 分离）
OBS_SANITIZE_INF_CLIP = 1.0e4       # ±inf 映射目标（保号有限大值，供学习信号；信封登记）


def install_obs_sanitize_wrapper(env, torch_mod, log_path, *, nan_value: float = 0.0,
                                 inf_clip: float = OBS_SANITIZE_INF_CLIP,
                                 phase: str = "rollout") -> dict:
    """给 env 本体打「只读+净化」包装：step/reset 返回的 obs 在出口逐 key 净化（第八层）。

    包装手法 = **动态子类 + `__class__` 重赋值**（同一对象、零 rebound）：
      `type("ObsSanitized_<原类名>", (type(env),), {...})` 只覆写 `step`/`reset`，随后
      `env.__class__ = 子类`。相比 `__getattr__` 透传 wrapper：
      - **isinstance 强兼容**：`isinstance(env, 原类)` 恒真（子类关系）——任何对 env 做
        isinstance/属性访问/方法调用的既有路径（`RslRlVecEnvWrapper`、
        `assert_residual_action_scale`、runner 内部）零感知；
      - **identity 不变**：env 还是同一个对象，不存在「旧引用绕过包装」的问题；
      - 未知属性**原生继承**，无需 `__getattr__` 转发（更快、也不会漏）。

    净化语义（与 update 侧第五~七层「置 0」口径**不同**，信封 `obs_sanitize` 字段登记）：
      - 检测：逐 float key GPU 侧 `isfinite` 归约（单标量同步、零全量拷贝）——**健康路径唯一
        开销**，零写回、原 dict/张量原样返回；
      - 命中：`nan_to_num(nan=0, posinf=+inf_clip, neginf=-inf_clip)`（NaN->0；±inf 保号映射
        ±1e4 有限大值——rollout 出口保号给策略/训练留学习信号）+ 落 `tag=obs_sanitize` 计数行
        （逐 key 逐命中：n_nonfinite/n_nan/n_posinf/n_neginf/n_total/cum/first/first_index/
        first_coord）+ stdout WARN（逐 key 限频一次）。
      - 只统计与净化，不改动作/奖励/terminated/truncated/extras 语义；step/reset 抛出的异常
        原样冒泡（**不吞**），仅消毒器自身失败时 best-effort（WARN + obs 原样放行，不打断训练，
        与既有探针/消毒器口径一致）。

    返回 {"log": 路径, "installed": True, "base_class"/"wrapped_class": 类名, nan/posinf/neginf}；
    已包装时幂等返回 {"installed": False, "already": True}（不二次包装）。
    """
    already = getattr(env, "_obs_sanitize_state", None)
    if already is not None:
        return {"log": str(already.get("log_path", log_path)), "installed": False, "already": True}
    th = torch_mod
    log_path = Path(log_path)
    state = {
        "steps": 0,            # 包装后 step/reset 调用总数（日志行 step 字段）
        "hits": 0,             # 命中次数（某 key 某次调用出非有限 = 1 次）
        "lines": 0,            # 已落日志行数
        "cum": {},             # 逐 key 累计非有限元素数
        "warned": set(),       # 已发过 stdout WARN 的 key（限频；命中可能高频防刷屏）
        "log_path": str(log_path),
    }
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:  # 目录建不了等首行写入再暴露（_emit_line 的 OSError 路径兜底）
        pass

    def _emit_line(rec: dict) -> None:
        line = json.dumps(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "phase": phase,
                "tag": OBS_SANITIZE_TAG,
                "step": state["steps"],
                **rec,
            },
            ensure_ascii=False,
        )
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        state["lines"] += 1

    def _sanitize_one(name, t, call: str):
        """单张量 GPU 侧检测+净化；返回 (净化后张量 | 原张量, 记录 | None)。

        检测 = `(~isfinite).sum()` 单标量同步（健康路径唯一开销、零写回）；命中才 nonzero 找
        **首个**非有限索引（只拷小片）+ `nan_to_num` 重造（与 `_sanitize_tensor_gpu` 同型，
        但 ±inf 保号映射 ±inf_clip 而非置 0——rollout 出口语义，见 docstring）。
        """
        if not isinstance(t, th.Tensor) or not t.is_floating_point():
            return t, None
        with th.no_grad():
            bad_n = int((~th.isfinite(t)).sum().item())   # 健康路径唯一开销：一次归约 + 单标量同步
            if bad_n == 0:
                return t, None
            flat = t.reshape(-1)
            idx = int((~th.isfinite(flat)).nonzero(as_tuple=False)[0].item())
            first_val = float(flat[idx].item())
            n_nan = int(th.isnan(t).sum().item())
            n_posinf = int((t == float("inf")).sum().item())
            n_neginf = int((t == float("-inf")).sum().item())
            clean = th.nan_to_num(t, nan=nan_value, posinf=inf_clip, neginf=-inf_clip)
            coords = None
            if t.dim() > 1:
                coords = [int(c) for c in th.unravel_index(th.tensor(idx, device=t.device), t.shape)]
            rec = {
                "key": str(name),
                "call": call,
                "n_nonfinite": bad_n,
                "n_total": int(t.numel()),
                "n_nan": n_nan,
                "n_posinf": n_posinf,
                "n_neginf": n_neginf,
                "first": first_val,
                "first_index": idx,
                "first_coord": coords,
                "cum": state["cum"].get(str(name), 0) + bad_n,
            }
            return clean, rec

    def _sanitize_obs(obs, call: str):
        """obs dict 逐 key 净化；健康路径原 dict 原样返回（零写回），命中才换新 dict。"""
        if not isinstance(obs, dict):
            return obs
        new_obs = None
        for key, val in obs.items():
            clean, rec = _sanitize_one(key, val, call)
            if rec is None:
                if new_obs is not None:
                    new_obs[key] = val
                continue
            state["hits"] += 1
            state["cum"][rec["key"]] = rec["cum"]
            try:
                _emit_line(rec)
            except OSError as exc:  # 仪器 best-effort：写日志失败只警告，不阻断训练
                print(f"[WARN] obs_sanitize 写日志失败（key={rec['key']}）: {exc}", flush=True)
            if rec["key"] not in state["warned"]:
                state["warned"].add(rec["key"])
                print(
                    f"[WARN] obs_sanitize: rollout obs['{rec['key']}'] 非有限 {rec['n_nonfinite']}/"
                    f"{rec['n_total']}（nan={rec['n_nan']} +inf={rec['n_posinf']} "
                    f"-inf={rec['n_neginf']}）first={rec['first']} "
                    f"first_coord={rec['first_coord']} -> nan=0 / ±inf=±{inf_clip:g}"
                    f"（净化后出口，训练继续；逐次计数见 {log_path.name} tag={OBS_SANITIZE_TAG}）",
                    flush=True,
                )
            if new_obs is None:
                new_obs = dict(obs)
            new_obs[key] = clean
        return obs if new_obs is None else new_obs

    def _wrap_return(out, call: str):
        """step/reset 返回值只净化第 0 位（obs），其余（reward/terminated/truncated/extras）原样。"""
        try:
            if isinstance(out, tuple) and len(out) > 0:
                clean0 = _sanitize_obs(out[0], call)
                return out if clean0 is out[0] else (clean0,) + out[1:]
            if isinstance(out, list) and len(out) > 0:
                clean0 = _sanitize_obs(out[0], call)
                return out if clean0 is out[0] else [clean0] + out[1:]
            return _sanitize_obs(out, call)   # 裸 obs（非序列）也兜住
        except Exception as exc:  # noqa: BLE001（消毒器 best-effort：绝不打断训练）
            print(f"[WARN] obs_sanitize 消毒失败（call={call}，obs 原样放行）：{type(exc).__name__}: {exc}", flush=True)
            return out

    orig_cls = type(env)

    def _wrapped_step(self, action, *args, **kwargs):
        state["steps"] += 1
        out = orig_cls.step(self, action, *args, **kwargs)
        return _wrap_return(out, "step")

    def _wrapped_reset(self, *args, **kwargs):
        state["steps"] += 1
        out = orig_cls.reset(self, *args, **kwargs)
        return _wrap_return(out, "reset")

    wrapped_cls = type(
        f"ObsSanitized_{orig_cls.__name__}",
        (orig_cls,),
        {
            "step": _wrapped_step,
            "reset": _wrapped_reset,
            # 类属性挂 state：selftest/事后审计可经 `env._obs_sanitize_state` 读计数
            "_obs_sanitize_state": state,
            "__doc__": f"{orig_cls.__name__} 的第八层 obs 消毒重分类子类（step/reset 出口净化，identity 不变）",
        },
    )
    env.__class__ = wrapped_cls   # 失败即显式异常（fail loud；可 --no-obs-sanitize 关闭本件）
    return {
        "log": str(log_path),
        "installed": True,
        "base_class": orig_cls.__name__,
        "wrapped_class": wrapped_cls.__name__,
        "nan": float(nan_value),
        "posinf": float(inf_clip),
        "neginf": float(-inf_clip),
    }


# ------------------------------------------------- critic 前向探针（第五层溯源件）
def _critic_obs_mag_records(critic_obs) -> list[dict]:
    """critic_obs 逐 key 幅值记录（min/max/abs_max/mean/n_nonfinite）；**只读**，best-effort。

    critic 组含**无噪** height_scan 187 维（重点看：若它出 inf/NaN 则根因在特权观测）。大张量按
    同一阈值**copy 前置跳过**。返回空列表表示无可探量。
    """
    out: list[dict] = []
    try:
        if isinstance(critic_obs, dict):
            for key, val in critic_obs.items():
                if isinstance(val, torch.Tensor) and val.numel() > NAN_PROBE_MAX_TENSOR_ELEMS:
                    continue   # copy 前置判断
                t = _to_cpu_tensor(val)
                if t is None or t.numel() > NAN_PROBE_MAX_TENSOR_ELEMS:
                    continue
                out.append(_finite_mag_stats(f"critic_obs.{key}", t))
        else:
            if isinstance(critic_obs, torch.Tensor) and \
                    critic_obs.numel() > NAN_PROBE_MAX_TENSOR_ELEMS:
                return out
            t = _to_cpu_tensor(critic_obs)
            if t is not None and t.numel() <= NAN_PROBE_MAX_TENSOR_ELEMS:
                out.append(_finite_mag_stats("critic_obs", t))
    except Exception:  # noqa: BLE001（探针 best-effort）
        return out
    return out


def _critic_forward_probe(critic_obs, values) -> tuple[dict | None, list[dict]]:
    """只读检查 `evaluate` 返回 values 的 isfinite；返回 (values 首 NaN 探针, critic_obs 幅值列表)。

    **全有限时返回 (None, [])**（不记录、不改值）；发现非有限时给出 values 的
    `_first_nonfinite_probe`（含 first_coord）+ 该次 critic_obs 逐 key 幅值。**绝不改 values /
    critic_obs**（溯源件只记录）。values 为 (values, ...) 变体时取首个张量。
    """
    v = values
    if isinstance(v, (tuple, list)):
        v = next((x for x in v if isinstance(x, torch.Tensor)), None)
    if not isinstance(v, torch.Tensor) or v.numel() == 0:
        return None, []
    try:
        if bool(torch.isfinite(v).all()):
            return None, []
    except Exception:  # noqa: BLE001（非浮点/异常 -> 不判非有限，静默）
        return None, []
    rec = None
    tv = _to_cpu_tensor(v)
    if tv is not None:
        rec = _first_nonfinite_probe("critic.values", tv)
    return rec, _critic_obs_mag_records(critic_obs)


def _install_critic_forward_probe(alg, probe_fn) -> bool:
    """只读包装 `alg.policy.evaluate`：每次调用后把 (critic_obs, values) 交给 `probe_fn` 探测。

    返回是否包装成功。**不改 evaluate 的入参/返回值/异常语义**（原样透传；probe_fn 内部自带
    try/except）。policy 无 `evaluate`（非 rsl_rl 变体）时静默返回 False。防重复包装（重复
    install 时用标记属性跳过）。
    """
    policy = getattr(alg, "policy", None)
    if policy is None:
        return False
    orig_evaluate = getattr(policy, "evaluate", None)
    if not callable(orig_evaluate):
        return False
    if getattr(orig_evaluate, "_d067_critic_probe", False):
        return True   # 已包装过，避免叠包

    def _wrapped_evaluate(*args, **kwargs):
        critic_obs = None
        if args:
            critic_obs = args[0]
        else:
            critic_obs = kwargs.get("critic_observations", kwargs.get("observations"))
        values = orig_evaluate(*args, **kwargs)   # 异常原样冒泡（探针不吞）
        probe_fn(critic_obs, values)
        return values

    _wrapped_evaluate._d067_critic_probe = True
    try:
        policy.evaluate = _wrapped_evaluate
    except Exception as exc:  # noqa: BLE001（探针 best-effort：包装失败不影响训练）
        print(f"[WARN] critic 前向探针包装失败（忽略）：{type(exc).__name__}: {exc}", flush=True)
        return False
    return True


# ------------------------------------------------- σ 漂移仪器（update 前后各记一行）
def _read_alg_scalar(alg, names: Sequence[str]) -> float | None:
    """按 `names` 顺序在 `alg` 上取首个可及的标量（float/int/0维张量），取不到返回 None。

    用途：`kl_mean`（rsl_rl 2.3.3 里是 `update` 的**局部变量**，不落 `self`，故此处恒取不到——
    见模块 docstring 的 kl 口径）与 `learning_rate`（adaptive 分支会改写 `self.learning_rate`，
    可后验反映 KL 是否爆炸）。**只读不抛**（best-effort）。
    """
    for name in names:
        try:
            val = getattr(alg, name, None)
        except Exception:  # noqa: BLE001（属性访问器可能抛，探针不打断）
            continue
        if val is None:
            continue
        if isinstance(val, torch.Tensor):
            if val.numel() != 1:
                continue
            return float(val.detach().to(torch.float32).item())
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _read_alg_scalar_with_source(alg, names: Sequence[str]) -> tuple[float | None, str]:
    """同 `_read_alg_scalar`，但返回 (值, 来源名)；取不到返回 (None, "absent")。

    用于 `update.aux` 行的 `kl_source`：`"absent"` 表示当前 rsl_rl 版本无此属性（kl 是局部
    变量），`"kl_mean"`/`"kl"` 表示真读到了（未来 rsl_rl 版本若落盘即可自动接入）。
    """
    for name in names:
        try:
            val = getattr(alg, name, None)
        except Exception:  # noqa: BLE001
            continue
        if val is None:
            continue
        if isinstance(val, torch.Tensor) and val.numel() == 1:
            return float(val.detach().to(torch.float32).item()), name
        if isinstance(val, (int, float)):
            return float(val), name
    return None, "absent"


def _policy_std_stats(policy) -> dict | None:
    """读 policy 的 σ（`self.std`，可能为 None/非 Parameter）统计：min/mean/max/负值个数。

    直接读**参数本体**（非夹紧后的分布 std），故能观测到 σ 滑向负值的全过程。非 "scalar"
    σ 变体（无 `std` 属性）返回 None，仪器静默跳过（不干扰其它变体）。
    """
    std = getattr(policy, "std", None)
    if std is None:
        return None
    with torch.no_grad():
        s = std.detach().reshape(-1).to(torch.float32)
        return {
            "min": float(s.min().item()),
            "mean": float(s.mean().item()),
            "max": float(s.max().item()),
            "n_negative": int((s < 0).sum().item()),
            "dim": int(s.numel()),
        }


def install_std_trajectory_instrument(alg, out_dir: Path, every: int = 1, phase: str = "train") -> dict:
    """包住 `alg.update`，每次把 σ 统计追加一行到 `outputs/<out>/std_trajectory.log`。

    rsl_rl 主循环每 iter 调一次 `alg.update()`（on_policy_runner.py:262），故包住它即可得到
    「每 iter 的 σ 轨迹」。每次记 pre（update 前）/post（update 后）两行；update 抛异常时记
    crash 行再原样 re-raise（不吞异常）——crash 行正好是 σ 变负被夹紧兜住的那一瞬。
    `every` 控制抽样（默认每 iter；设 N 则每 N 个 update 记一次，但 crash 行始终记）。

    **第三层修复追加**：同一 wrapper 内挂「首 NaN 溯源探针」——orig_update 前探 obs_batch
    （逐 key + policy 段按 official/proprio/token 切片），后探 loss_dict 逐 key + `policy.std`
    参数本体 + 策略输出（mu/actor，可及才跑）；首个非有限量以 `tag=nan_probe` 行落同一日志
    （字段 name/n_nonfinite/min/max/first/first_index/shape/dtype + stage），**只记录不阻断**。

    **第四层追加（update 内分步溯源）**：同一 wrapper 再挂三类**幅值趋势**行
    （`tag=nan_probe_mag`，与首 NaN 行分 tag）：
      ① `stage=pre_mid`：advantage/returns/values（`alg.storage.*`）——首 NaN 在 GAE 还是
         loss/backward 的分界点；
      ② `stage=post`：loss_dict 逐 key 的 min/max/abs_max/mean + 一行 `update.aux`
         （`kl_mean`/`kl_source`/`learning_rate_pre`/`learning_rate`，见模块 docstring 的 kl 口径）；
      ③ `stage=post` 参数快照对比：`param_snapshot` 行（`policy.std` 全量 + actor/critic
         首末 weight 的 isfinite/norm，pre/post 两拍 + `param_nan_flipped`）。
    另在 `stage=pre` 追加 obs 逐 key/逐段（official/proprio/token）幅值行。**全部只读**。

    **第五层追加（advantage/returns 消毒器 + critic 前向探针）**：
      - `_wrapped_update` 在 orig_update **之前**消毒 `storage.advantages/returns/values`（非有限
        置 0，`tag=nan_probe_sanitize` 行；健康路径零写回）；
      - 仪器安装时只读包装 `alg.policy.evaluate`：values 非有限时落 `tag=nan_probe`
        行（`stage=critic_forward`）+ 该次 critic_obs 逐 key 幅值（含无噪 height_scan 187）。

    **第六层修复（inference-tensor 写回）**：消毒器改用**重赋值**路径
    （`setattr(storage, attr, torch.nan_to_num(t, ...))`）——rollout 在 `torch.inference_mode()`
    下产生的 `storage.advantages` 是 inference tensor，第五层的**就地** `masked_fill_` 对其抛
    `Inplace update to inference tensor outside InferenceMode is not allowed`（第六跑实证：300 次
    消毒全失败）⇒ 改用 `nan_to_num` 返回的普通 tensor 重赋值，`update` 每次现读 `storage.<attr>`
    故可见。`values` 一并纳入消毒（GAE 输入/源头）。健康路径仍零写。

    **第七层修复（消毒器上移到观测端）**：消毒范围扩为五路
    （`_SANITIZE_STORAGE_ATTRS` = advantages/returns/values/**privileged_observations**/
    **observations**）——`ppo.update()` 内部用 `privileged_observations` **现算** critic
    （`ppo.py:263`），第七跑实证该 obs 的单个 NaN 就是 value_loss 再 NaN 的源头；消毒 values
    管不到重算，必须在上游 obs 端净化。**大张量检测改 GPU 侧**：`_sanitize_storage_tensor`
    对超阈值张量（obs ≈37M 元素 > `NAN_PROBE_MAX_TENSOR_ELEMS`）**不再跳过**，检测 = GPU 侧
    `isfinite` 归约（单标量同步、零全量拷贝），命中才 GPU 侧 `nan_to_num` 重赋值 + 只拷小片
    统计（first/first_coord）；`observations` 是 dict 则逐 key 走同一路径。健康路径**零写回**，
    代价 = 一次 GPU 侧归约同步。探针类 `_collect_*` 的大张量跳过逻辑**保持不变**（日志性能保护）。

    返回 {"log": 路径, "wrapped": True, "sanitize": True, "critic_probe": bool}（供身份信封/打印）。
    **update 数值路径唯一改动 = 消毒器对非有限元素的置 0 写回**（健康路径零改动）；critic 探针只读。
    """
    log_path = Path(out_dir) / STD_TRAJECTORY_LOG
    state = {"calls": 0, "nan_probes": 0, "nan_probe_mags": 0, "sanitized": 0, "critic_nan": 0}
    orig_update = alg.update

    def _emit(tag: str, stats: dict | None, **extra) -> None:
        if stats is None:
            return
        try:
            _emit_line(tag, stats, **extra)
        except OSError as exc:  # 仪器 best-effort：写日志失败只警告，不阻断训练、不吞真异常
            print(f"[WARN] std_trajectory 写日志失败（tag={tag}）: {exc}", flush=True)

    def _emit_line(tag: str, stats: dict, **extra) -> None:
        line = json.dumps(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "phase": phase,
                "tag": tag,
                "update_calls": state["calls"],
                **stats,
                **extra,
            },
            ensure_ascii=False,
        )
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def _probe(stage: str, result=None) -> None:
        """首个非有限量落 nan_probe 行（每 update 每 stage 最多一条；只记录不阻断）。"""
        try:
            for name, tensor in _collect_nan_probe_candidates(alg, stage, result):
                rec = _first_nonfinite_probe(name, tensor)
                if rec is None:
                    continue
                state["nan_probes"] += 1
                _emit(NAN_PROBE_TAG, rec, stage=stage)
                print(
                    f"[WARN] nan_probe: 首个非有限量 stage={stage} name={name} "
                    f"n_nonfinite={rec['n_nonfinite']}/{rec['n_total']} first={rec['first']} "
                    f"idx={rec['first_index']}（σ 净化保命带已兜底，训练继续）",
                    flush=True,
                )
                return
        except Exception as exc:  # noqa: BLE001（探针绝不打断训练）
            print(f"[WARN] nan_probe 探测失败（stage={stage}，忽略）：{type(exc).__name__}: {exc}", flush=True)

    def _probe_mag(stage: str, result=None) -> None:
        """幅值趋势行（第四层）：逐 key/逐段 min/max/abs_max/mean（含全有限量），只记录不阻断。"""
        try:
            for name, tensor in _collect_mag_candidates(alg, stage, result):
                rec = _finite_mag_stats(name, tensor)
                state["nan_probe_mags"] += 1
                _emit(NAN_PROBE_MAG_TAG, rec, stage=stage)
        except Exception as exc:  # noqa: BLE001（探针绝不打断训练）
            print(f"[WARN] nan_probe_mag 探测失败（stage={stage}，忽略）：{type(exc).__name__}: {exc}", flush=True)

    def _emit_update_aux(pre_lr) -> None:
        """`update.aux` 行：kl_mean（可及才记）+ learning_rate 前后值（adaptive 分支可后验量）。"""
        try:
            kl_val, kl_src = _read_alg_scalar_with_source(alg, ("kl_mean", "kl"))
            post_lr = _read_alg_scalar(alg, ("learning_rate", "lr"))
            state["nan_probe_mags"] += 1
            _emit(
                NAN_PROBE_MAG_TAG,
                {
                    "name": "update.aux",
                    "kl_mean": kl_val,
                    "kl_source": kl_src,
                    "learning_rate_pre": pre_lr,
                    "learning_rate": post_lr,
                },
                stage="post",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] nan_probe_mag update.aux 失败（忽略）：{type(exc).__name__}: {exc}", flush=True)

    def _emit_param_snapshot(pre_snap: dict, post_snap: dict) -> None:
        """`param_snapshot` 行：pre/post 参数快照 + 「有限→非有限」翻转 key 列表（第四层 ③）。"""
        try:
            diff = _nan_probe_param_diff(pre_snap, post_snap)
            state["nan_probe_mags"] += 1
            _emit(
                NAN_PROBE_MAG_TAG,
                {"name": "param_snapshot", "param_pre": pre_snap, "param_post": post_snap, **diff},
                stage="post",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] nan_probe_mag param_snapshot 失败（忽略）：{type(exc).__name__}: {exc}", flush=True)

    def _emit_sanitize(rec: dict) -> None:
        """`nan_probe_sanitize` 行：消毒器写回记录（**与首 NaN 行分 tag**，带写回语义字段）。"""
        try:
            state["sanitized"] += 1
            _emit(NAN_PROBE_SANITIZE_TAG, rec, stage="pre_mid")
            print(
                f"[WARN] advantage 消毒器: {rec['name']} 非有限 {rec['n_nonfinite']}/{rec['n_total']} "
                f"-> 置 0（purged_full={rec['purged_full']}，本 update 继续）",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001（消毒器 best-effort）
            print(f"[WARN] nan_probe_sanitize 记录失败（忽略）：{type(exc).__name__}: {exc}", flush=True)

    def _sanitize() -> None:
        """orig_update 前消毒五路（advantages/returns/values/privileged_observations/observations，非有限置 0，重赋值路径）；健康路径零写回。"""
        for rec in _sanitize_storage_attrs(alg):
            _emit_sanitize(rec)

    def _critic_probe(critic_obs, values) -> None:
        """critic 前向只读探针：values 非有限 -> `tag=nan_probe` 行（stage=critic_forward）+ obs 幅值。

        **只记录不阻断、不改 values/critic_obs**；全有限时零记录。
        """
        try:
            rec, obs_recs = _critic_forward_probe(critic_obs, values)
            if rec is None:
                return
            state["nan_probes"] += 1
            state["critic_nan"] += 1
            _emit(NAN_PROBE_TAG, rec, stage="critic_forward")
            for orec in obs_recs:
                state["nan_probe_mags"] += 1
                _emit(NAN_PROBE_MAG_TAG, orec, stage="critic_forward")
            print(
                f"[WARN] nan_probe: critic 前向 values 非有限 {rec['n_nonfinite']}/{rec['n_total']} "
                f"first_coord={rec['first_coord']}（该次 critic_obs 幅值已落 nan_probe_mag）",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001（探针绝不打断训练）
            print(f"[WARN] nan_probe critic 前向探测失败（忽略）：{type(exc).__name__}: {exc}", flush=True)

    def _wrapped_update(*args, **kwargs):
        state["calls"] += 1
        record = (state["calls"] % max(1, int(every)) == 0)
        if record:
            _emit("pre", _policy_std_stats(alg.policy))
        _probe("pre")        # ① update 前的 obs_batch（逐 key + policy 段切片）
        _probe_mag("pre")    # ④ obs 逐 key/逐段幅值趋势
        # 第四层：update 前拍参数快照 + 读 lr + 探 advantage/returns（GAE 输出，update 的输入）
        pre_lr = _read_alg_scalar(alg, ("learning_rate", "lr"))
        # 探针绝不打断训练：快照失败降级为空快照（下游 diff 仍能跑，不会误报翻转）
        try:
            pre_snap = _nan_probe_param_snapshot(getattr(alg, "policy", None))
        except Exception as exc:  # noqa: BLE001（探针绝不打断训练）
            print(f"[WARN] nan_probe 参数快照(pre)失败（忽略）：{type(exc).__name__}: {exc}", flush=True)
            pre_snap = {}
        _probe("pre_mid")        # ①' advantage/returns/values（首 NaN 在 GAE 还是 loss/backward 的分界）
        _probe_mag("pre_mid")
        _sanitize()              # 第五层：advantage/returns 非有限置 0（健康路径零写回）
        try:
            result = orig_update(*args, **kwargs)
        except BaseException:  # noqa: BLE001（crash 行必记；异常原样冒泡，不吞）
            _emit("crash", _policy_std_stats(alg.policy))
            _probe("post")   # 崩点现场：σ 参数本体 + 策略输出（可及才跑）
            raise
        if record:
            _emit("post", _policy_std_stats(alg.policy))
        _probe("post", result)        # ② loss_dict 逐 key ③ update 后 policy.std / 策略输出
        _probe_mag("post", result)    # ② loss 逐 key 幅值升级
        _emit_update_aux(pre_lr)      # ②' kl_mean（可及才记）+ lr 前后
        # ③ 参数快照对比：std + actor/critic 首末 weight（同刻变 NaN -> 印证 loss NaN 染全部梯度）
        try:
            post_snap = _nan_probe_param_snapshot(getattr(alg, "policy", None))
        except Exception as exc:  # noqa: BLE001（探针绝不打断训练）
            print(f"[WARN] nan_probe 参数快照(post)失败（忽略）：{type(exc).__name__}: {exc}", flush=True)
            post_snap = {}
        _emit_param_snapshot(pre_snap, post_snap)
        return result

    # 第五层：critic 前向探针——只读包装 `alg.policy.evaluate`（values 非有限才落行）。
    critic_probe = _install_critic_forward_probe(alg, _critic_probe)

    alg.update = _wrapped_update
    return {"log": str(log_path), "wrapped": True, "sanitize": True, "critic_probe": bool(critic_probe)}


def _make_runner_cfg_dict(hv, cli, out_dir: Path, device: str, policy_kwargs: dict | None) -> dict:
    """rsl_rl runner cfg dict：PPO 配方逐字段继承 D064 官方 G1 rough（保可比性）。

    residual 臂改 policy.class_name + 透传 policy_kwargs（照 D065 的
    `runner_cfg.to_dict()` + `dict["policy"].update(...)` 手法）；三臂 policy 类
    统一经 register_arm_policy_classes_in_rsl_rl 设 class_name——direct/decoder
    = ClampedStdActorCritic（rsl_rl ActorCritic 逐字继承 + σ 病态区夹紧，健康区
    逐位等价；2026-09-23 σ 变负 crash 修复）、residual = ResidualDecoderPolicy
    （自带夹紧）。
    """
    RslRlPpoActorCriticCfg = hv.RslRlPpoActorCriticCfg
    RslRlPpoAlgorithmCfg = hv.RslRlPpoAlgorithmCfg
    RslRlOnPolicyRunnerCfg = hv.RslRlOnPolicyRunnerCfg

    runner_cfg = RslRlOnPolicyRunnerCfg()
    runner_cfg.seed = cli.seed
    runner_cfg.device = device
    runner_cfg.num_steps_per_env = 24
    runner_cfg.max_iterations = cli.max_iterations
    runner_cfg.save_interval = 50
    runner_cfg.experiment_name = out_dir.name
    runner_cfg.empirical_normalization = False
    runner_cfg.policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    runner_cfg.algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    runner_cfg.run_name = f"{cli.arm}_{cli.terrain}"
    cfg_dict = runner_cfg.to_dict()
    # 三臂统一走 σ 夹紧层：direct/decoder -> ClampedStdActorCritic；residual ->
    # ResidualDecoderPolicy（两者都继承 ClampedStdPolicyMixin）。见 register_arm_policy_classes_in_rsl_rl。
    arm_classes = register_arm_policy_classes_in_rsl_rl()
    cfg_dict["policy"]["class_name"] = arm_classes[cli.arm]
    if cli.arm == "residual":
        cfg_dict["policy"].update(policy_kwargs or {})
    return cfg_dict


def _default_out_dir(arm: str, terrain: str, seed: int) -> Path:
    return REPO_ROOT / "apt_g1" / "outputs" / f"d067_{arm}_{terrain}_s{seed}"


# ------------------------------------------------------------- 训练主回路
def _run(cli, out_dir: Path, launcher_args) -> None:
    """App 起来后的主逻辑（三臂共用；residual 臂额外接线冻结分支 + 残差策略）。"""
    hv = _import_heavy()
    td = _import_train_g1_decoder()
    torch_mod = hv.torch
    device = getattr(launcher_args, "device", "cuda:0")
    reward_mode = resolve_reward_mode(cli.arm, cli.reward_mode)

    torch_mod.manual_seed(cli.seed)
    hv.np.random.seed(cli.seed)
    print(
        f"[CFG] out={out_dir} arm={cli.arm} terrain={cli.terrain} reward_mode={reward_mode} "
        f"num_envs={cli.num_envs} iters={cli.max_iterations} seed={cli.seed} device={device}",
        flush=True,
    )

    # ---- 身份信封（含三臂表项 + 奖励配方 + intent-pin + 地形/仿射契约声明） ----
    intent_pin = {
        "intent_pin_max": bool(cli.intent_pin_max),
        "value": float(cli.intent_pin_value) if cli.arm == "residual" else None,
    }
    envelope = build_identity_envelope(
        cli.arm,
        reward_mode,
        author_ckpt=cli.author_ckpt,
        onnx_path=cli.onnx_path,
        token_stats=cli.token_stats,
        intent_pin=intent_pin,
        terrain=cli.terrain,
    )
    (out_dir / "run.json").write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[d067] run.json -> {out_dir / 'run.json'}", flush=True)

    # ---- env cfg ----
    token_info = None
    scale_info = None
    if cli.arm == "residual":
        if not cli.author_ckpt:
            raise RuntimeError("--arm residual 需要 --author-ckpt（token 源 = D061a author v0 ckpt）")
        cfg, cfg_how, recipe, token_info, scale_info = _build_residual_env_cfg(hv, cli, device, suffix="main")
    else:
        cfg, cfg_how = td._build_env_cfg(
            hv.make_env_cfg, hv.mdp, hv.SonicDecoderActionTermCfg, cli.terrain, cli.arm, cli, device, suffix="main"
        )
        recipe = None
        if reward_mode == "max_forward":
            recipe = _apply_max_forward_recipe(cfg, hv, cli.forward_weight)
    print(f"[d067] env cfg via {cfg_how}（reward_mode={reward_mode} recipe={recipe}）", flush=True)
    if token_info is not None:
        # 地形契约/仿射统计进信封（reviewer M1/M2 要求）
        envelope["ctx_terrain"] = token_info["ctx_terrain"]
        envelope["env_terrain"] = token_info["env_terrain"]
        envelope["token_affine"]["token_bound"] = token_info["token_bound"]
        envelope["token_affine"]["token_alpha"] = token_info["token_alpha"]
        envelope["token_affine"]["token_stats"] = token_info["token_stats"]
        # 性能补丁登记（精度换速度）：adapter 前向精度档/分块默认与 fp32 口径偏差声明
        # （见模块 docstring「adapter 前向精度档」），进 run.json 供事后审计。
        envelope["adapter_perf"] = {
            "adapter_dtype": str(token_info.get("adapter_dtype", DEFAULT_ADAPTER_DTYPE)),
            "adapter_chunk_envs": int(token_info.get("adapter_chunk_envs", DEFAULT_ADAPTER_CHUNK_ENVS)),
            "declaration": (
                "冻结 80M transformer 纯推理 argmax 生成 token：bf16 数值近似可接受、"
                "token 流偶发近平局翻转不影响 RL 适配；与轴 A/B（D061a/D064）fp32 口径"
                "的偏差显式声明（不改奖励/网络/超参/adapter 本体）"
            ),
        }
        # 执行/反馈仿射信封（值/来源/逐关节 min-max）；运行期 action term _scale 守卫
        # 在 env 构造后追加 bitwise/gain 实测（见下）。
        envelope[RESIDUAL_ACTION_SCALE_FIELD] = residual_action_scale_envelope(
            hv._sonic_scale_isaac(),
            source=str((scale_info or {}).get("source", SONIC_SCALE_SOURCE)),
        )
        (out_dir / "run.json").write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")

    # asymmetric 退化判定（同 D064）
    has_critic_group = getattr(cfg.observations, "critic", None) is not None
    asym_ok, rsl_ver = td._rsl_rl_supports_dict_obs()
    if has_critic_group and not asym_ok:
        cfg.observations.critic = None
        has_critic_group = False
        print(f"[WARN] rsl_rl {rsl_ver} 不支持 dict obs，退化单组（丢 critic）", flush=True)

    env = hv.ManagerBasedRLEnv(cfg=cfg)
    # ---- 第八层：rollout 期观测源消毒器（--obs-sanitize 默认开，--no-obs-sanitize 关）----
    # 位置契约：env 构造后、runner 构造前完成包装（覆盖下方首次 reset 与 runner 内 rollout 的
    # 全部 step/reset 出口）；重分类 = 同一对象（identity 不变），下方既有 env 引用路径
    # （assert_residual_action_scale / _build_residual_policy_kwargs / EnvWrapper / smoke）零感知。
    # 语义声明：NaN->0、±inf->±1e4 保号（与 update 侧置 0 口径不同）；update 侧五路消毒器
    # 保留（防御纵深）。性能：健康路径每 key 一次 GPU isfinite 归约标量同步（见模块 docstring）。
    envelope["obs_sanitize"] = {
        "enabled": bool(getattr(cli, "obs_sanitize", True)),
        "nan": 0.0,
        "posinf": OBS_SANITIZE_INF_CLIP,
        "neginf": -OBS_SANITIZE_INF_CLIP,
        "declaration": (
            "rollout 期 obs 消毒（第七层 update 侧消毒的上游补全，r4 里程碑 purged_full=411/1254it "
            "实证 33% update 空学习步）：env.step/reset 出口逐 key nan_to_num（NaN->0、±inf->±1e4 "
            "保号供学习信号，与 update 侧置 0 口径不同）+ 计数落 std_trajectory.log"
            "（tag=obs_sanitize）；健康路径逐位零改动（每 key 一次 GPU 归约标量同步）；"
            "update 侧五路消毒器保留（防御纵深）。--no-obs-sanitize 关闭。"
        ),
    }
    if getattr(cli, "obs_sanitize", True):
        obs_san = install_obs_sanitize_wrapper(env, torch_mod, out_dir / STD_TRAJECTORY_LOG)
        envelope["obs_sanitize"]["log"] = obs_san["log"]
        envelope["obs_sanitize"]["wrapped_class"] = obs_san.get("wrapped_class")
        print(f"[d067] obs 消毒器（第八层）已包装 env（{obs_san.get('base_class')} -> "
              f"{obs_san.get('wrapped_class')}）-> {obs_san['log']}（tag={OBS_SANITIZE_TAG}）", flush=True)
    else:
        print("[d067] obs 消毒器（第八层）已关闭（--no-obs-sanitize）", flush=True)
    (out_dir / "run.json").write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
    obs, _ = env.reset()
    if not (isinstance(obs, dict) and "policy" in obs):
        raise RuntimeError(f"env obs 应为含 'policy' 的 dict，实际 {type(obs)}")
    policy_obs = obs["policy"]
    obs_dim_policy = int(policy_obs.shape[-1])
    obs_dim_critic = int(obs["critic"].shape[-1]) if (isinstance(obs, dict) and obs.get("critic") is not None) else 0
    action_dim = int(env.action_manager.total_action_dim)
    print(f"[d067] obs(policy)={obs_dim_policy} obs(critic)={obs_dim_critic or '-'} action={action_dim}", flush=True)

    # ---- policy 接线（residual：冻结 decoder 分支 + token/proprio 切片） ----
    policy_kwargs = None
    wiring = None
    if cli.arm == "residual":
        # 动作通路守卫：残差臂必须落在 direct 动作通路（29 维 JointPositionAction）。
        # 若 make_env_cfg 签名回退把 lora_policy 变体拿成别的动作通路，此处即显式失败
        # （否则 μ(29) 与 env 期望动作维错位，训练静默崩）。
        _check(action_dim == ACTION_DIM, f"residual 臂 env 动作维应 {ACTION_DIM}，实 {action_dim}")
        # 执行/反馈仿射互逆的**运行期硬证据**：action term 实际 _scale 逐位 == sonic_scale
        # （cfg 覆写对但 term 解析漂移 -> 此处即 fail loud，拒绝用错误闭环开训）。
        scale_guard = assert_residual_action_scale(env, hv)
        print(f"[d067] residual 执行/反馈仿射守卫：{json.dumps(scale_guard, ensure_ascii=False)}", flush=True)
        policy_kwargs, wiring = _build_residual_policy_kwargs(hv, env, cli, device)
        print(f"[d067] residual 接线 {json.dumps(wiring, ensure_ascii=False)}", flush=True)
        # 身份信封补记实际接线（env 运行期才能确定）
        envelope["residual_wiring"] = wiring
        envelope[RESIDUAL_ACTION_SCALE_FIELD]["runtime_guard"] = scale_guard
        envelope[RESIDUAL_ACTION_SCALE_FIELD]["cfg_override"] = {
            "n_joints": int((scale_info or {}).get("n_joints", 0)),
            "legacy_scalar": RESIDUAL_LEGACY_ACTION_SCALE,
        }
        envelope["obs_dim_policy"] = obs_dim_policy
        envelope["obs_dim_critic"] = obs_dim_critic
        envelope["action_dim"] = action_dim
        (out_dir / "run.json").write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")

    runner_cfg_dict = _make_runner_cfg_dict(hv, cli, out_dir, device, policy_kwargs)

    if cli.smoke:
        _smoke(cli=cli, out_dir=out_dir, env=env, policy_obs=policy_obs, obs_dim_policy=obs_dim_policy,
               action_dim=action_dim, hv=hv, runner_cfg_dict=runner_cfg_dict, device=device)
        return

    # ---- 正式训练 ----
    wrapped = hv.EnvWrapper(env)
    runner = hv.OnPolicyRunner(wrapped, runner_cfg_dict, log_dir=str(out_dir), device=device)
    if cli.arm == "residual":
        _alg, optimizer, policy = _resolve_ppo_and_policy(runner)
        if not isinstance(policy, ResidualDecoderPolicy):
            raise RuntimeError(
                f"runner 内 policy 类型 {type(policy).__name__} != ResidualDecoderPolicy"
                "（策略类注册失败，拒绝把残差臂训成 direct 臂）"
            )
        prune = _prune_frozen_params(optimizer)
        print(f"[d067] 可训集合（只 r + critic）：{prune}；{policy.trainable_report()}", flush=True)
    if cli.resume:
        if not cli.ckpt:
            raise RuntimeError("--resume 需要 --ckpt <path>")
        runner.load(str(Path(cli.ckpt).resolve()))
        print(f"[d067] resumed from {cli.ckpt}", flush=True)
    # σ 漂移仪器：包住 alg.update，每 iter 记 σ 的 min/mean/max/负值个数（三臂共用）。
    # 需在 runner.load 之后装（load 会重建 optimizer，但 alg/policy 对象同一；包 update 不受影响）。
    _alg_i, _opt_i, _policy_i = _resolve_ppo_and_policy(runner)
    std_instrument = install_std_trajectory_instrument(_alg_i, out_dir, every=1, phase=cli.arm)
    print(f"[d067] σ 漂移仪器 -> {std_instrument['log']}（初始 σ={_policy_std_stats(_policy_i)}）", flush=True)
    runner.learn(num_learning_iterations=runner_cfg_dict["max_iterations"])
    final_ckpt = out_dir / "model_final.pt"
    runner.save(str(final_ckpt))
    print(f"[d067] done. final ckpt -> {final_ckpt}", flush=True)
    env.close()


def _smoke(*, cli, out_dir, env, policy_obs, obs_dim_policy, action_dim, hv, runner_cfg_dict, device) -> None:
    """D067 冒烟：三臂 obs/action 形状 + residual 零初始化恒等 + learn x2 + ckpt 落盘。"""
    n = cli.num_envs
    _check(n == SMOKE_NUM_ENVS, f"smoke num_envs 应为 {SMOKE_NUM_ENVS}")
    _check(policy_obs.ndim == 2 and policy_obs.shape[0] == n, "policy obs 形状异常")
    print(f"[d067] smoke a) obs={obs_dim_policy} action={action_dim} PASS", flush=True)

    if cli.arm == "residual":
        token_slot = _policy_obs_slot(env, hv.torch, "author_token", TOKEN_DIM)
        proprio_slot = _policy_obs_slot(env, hv.torch, "sonic_proprio", PROPRIO_DIM)
        _check(action_dim == ACTION_DIM, f"residual 臂 action 维应 {ACTION_DIM}，实 {action_dim}")
        # 执行/反馈仿射互逆（smoke 侧再断言一次：env 已构造，term._scale 可实测）
        guard = assert_residual_action_scale(env, hv)
        _check(guard["bitwise"] and guard["gain_min"] == 1.0 and guard["gain_max"] == 1.0,
               f"smoke 执行/反馈仿射守卫不通过：{guard}")
        print(f"[d067] smoke b) residual 槽 token={token_slot} proprio={proprio_slot} "
              f"仿射守卫={guard} PASS", flush=True)

    wrapped = hv.EnvWrapper(env)
    runner = hv.OnPolicyRunner(wrapped, runner_cfg_dict, log_dir=str(out_dir), device=device)
    if cli.arm == "residual":
        _alg, optimizer, policy = _resolve_ppo_and_policy(runner)
        _check(isinstance(policy, ResidualDecoderPolicy), "runner policy 非 ResidualDecoderPolicy")
        # 零初始化恒等：μ == 冻结 decoder(cat[token,proprio]) 逐位
        dec_obs = policy.decoder_obs(policy._flat_obs(policy_obs))
        _check(policy.mu_path.residual.is_zero(policy_obs[:, : policy.official_obs_dim], atol=0.0),
               "残差非零初始化")
        with hv.torch.no_grad():
            _check(hv.torch.equal(policy.mu(policy_obs), policy.mu_path.decoder(dec_obs)),
                   "μ 与冻结 decoder 输出非逐位一致")
        prune = _prune_frozen_params(optimizer)
        _check(policy.mu_path.only_residual_trainable(), "梯度隔离不通过（除 r 外仍有可训参数）")
        print(f"[d067] smoke b2) 零初始化恒等 + 梯度隔离 PASS {prune}", flush=True)
    _alg_s, _opt_s, _policy_s = _resolve_ppo_and_policy(runner)
    install_std_trajectory_instrument(_alg_s, out_dir, every=1, phase=f"smoke_{cli.arm}")
    runner.learn(num_learning_iterations=SMOKE_ITERS)
    ckpt = out_dir / "model_smoke.pt"
    runner.save(str(ckpt))
    _check(ckpt.exists() and ckpt.stat().st_size > 0, "ckpt 未落盘")
    print(f"[d067] smoke c) learn x{SMOKE_ITERS} PASS -> {ckpt}", flush=True)
    env.close()
    print(
        f"SMOKE PASS: arm={cli.arm} reward_mode={resolve_reward_mode(cli.arm, cli.reward_mode)} "
        f"obs={obs_dim_policy} action={action_dim} learn={SMOKE_ITERS}it ckpt={ckpt.name}",
        flush=True,
    )


# ------------------------------------------------------------- CLI
def build_args() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="D067 G2 输出端并行残差臂训练入口（三臂对照 + max-forward 奖励）"
    )
    ap.add_argument("--arm", choices=ARMS, default="residual",
                    help="direct/decoder（D064 逐字对照臂）/residual（a=decoder(token)+r(s)）")
    ap.add_argument("--terrain", choices=TERRAINS, default="rough")
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-iterations", type=int, default=3000)
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--reward-mode", choices=REWARD_MODES, default=DEFAULT_REWARD_MODE,
                    help="auto：residual->max_forward、direct/decoder->track（D064 可比）")
    ap.add_argument("--forward-weight", type=float, default=DEFAULT_FORWARD_WEIGHT,
                    help="max_forward 前进速度项权重")
    ap.add_argument("--author-ckpt", default="", help="D061a author v0 ckpt（residual 臂 token 源，必填）")
    ap.add_argument("--intent-pin-max", action=argparse.BooleanOptionalAction, default=True,
                    help="推理期把 author ctx 命令槽钉最大速度档（默认开）")
    ap.add_argument("--intent-pin-value", type=float, default=DEFAULT_INTENT_PIN_VALUE,
                    help="钉档值（默认 1.0 = 官方 G1 rough lin_vel_x 上界，见假设 A2）")
    ap.add_argument("--token-stats", default="", help="官方 g1-mode token 统计 npz（decoder 仿射/author 逆仿射）")
    ap.add_argument("--adapter-chunk-envs", type=int, default=DEFAULT_ADAPTER_CHUNK_ENVS,
                    help="residual 臂 author adapter env 维分块前向块大小（CUDA OOM 修复；"
                         "0=不分块；默认 1024（性能补丁拉大，bf16 下峰值可行；"
                         "OOM 时可降回 512/128）)")
    ap.add_argument("--adapter-dtype", choices=ADAPTER_DTYPES, default=DEFAULT_ADAPTER_DTYPE,
                    help="residual 臂 author adapter 前向精度档（性能补丁：精度换速度；"
                         "默认 bf16=autocast 包前向+logits 边界升 fp32；tf32=matmul 标志"
                         "仅 wrapper 范围生效；fp32=旧行为逐字。口径声明见模块 docstring）")
    ap.add_argument("--onnx-path", default="", help="SONIC ONNX decoder（残差臂冻结分支；缺省工厂默认）")
    ap.add_argument("--token-alpha", type=float, default=None)
    ap.add_argument("--token-bound", choices=("none", "tanh"), default=None)
    ap.add_argument("--sigma", type=float, default=None, help="残差臂 σ 值（缺省=init_noise_std=1.0）")
    ap.add_argument("--sigma-trainable", action=argparse.BooleanOptionalAction, default=False,
                    help="残差臂 σ 是否可训（默认关 = PPO 只训 r）")
    ap.add_argument("--obs-sanitize", action=argparse.BooleanOptionalAction, default=True,
                    help="rollout 期观测源消毒器（第八层，默认开）：env.step/reset 返回的 obs "
                         "在出口逐 key 净化（NaN->0、±inf->±1e4 保号）+ 计数落 std_trajectory.log"
                         "（tag=obs_sanitize）；update 侧五路消毒器保留（防御纵深）")
    ap.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--smoke", action="store_true", help="冒烟：num_envs=8、iters=2")
    ap.add_argument("--selftest", action="store_true", help="本机纯 torch 自测（不 import isaaclab）")
    return ap


def main() -> None:
    cli = build_args().parse_args()

    # ---- 纯本机分支：绝不 import isaac ----
    if cli.selftest:
        run_selftest()
        return

    if cli.resume and not cli.ckpt:
        build_args().error("--resume 需要 --ckpt <path>")
    if cli.arm == "residual" and not cli.author_ckpt:
        build_args().error("--arm residual 需要 --author-ckpt <D061a ckpt>（token 源）")
    if cli.arm == "residual" and not cli.token_stats:
        build_args().error("--arm residual 需要 --token-stats <npz>（author 逆仿射 token 统计）")
    if cli.arm == "decoder" and not cli.token_stats:
        build_args().error("--arm decoder 需要 --token-stats <npz>（token 仿射统计）")

    if cli.smoke:
        cli.num_envs = SMOKE_NUM_ENVS
        cli.max_iterations = SMOKE_ITERS

    out_dir = Path(cli.output_dir) if cli.output_dir else _default_out_dir(cli.arm, cli.terrain, cli.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # AppLauncher 启动链（app 起来之前不 import isaaclab 栈）
    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args()
    launcher_args.num_envs = cli.num_envs
    launcher_args.headless = cli.headless
    app_launcher = AppLauncher(launcher_args)
    simulation_app = app_launcher.app

    interrupted = False
    try:
        # 打断事件落盘仪器：主训练循环（含 smoke）外的 BaseException 守卫——把完整
        # traceback 写 outputs/<out>/main_loop_crash.log 再 raise（修复前该异常会被
        # finally 的 close 死锁吞掉，见模块 docstring「Isaac Sim 关闭路径死锁修复」）。
        try:
            _run(cli, out_dir, launcher_args)
        except BaseException as exc:  # noqa: BLE001
            _log_main_loop_crash(out_dir, exc)
            raise
    except Exception as exc:  # noqa: BLE001
        interrupted = True  # 超时强退须带非零 rc，防「异常打断+close 挂死」被伪装成成功
        if cli.smoke:
            print(f"SMOKE FAIL: {type(exc).__name__}: {exc}", flush=True)
        raise
    finally:
        # 看门狗替代裸 close()：Isaac Sim 关闭忙等（simulation_context.py:743）时强退，
        # 不再把进程卡死；正常关闭则由本函数正常返回。interrupted=True 时 exit_code=1
        # （reviewer should-fix：异常在飞的超时强退不得以 rc=0 掩盖挂死事件）。
        _safe_close(simulation_app, exit_code=1 if interrupted else 0)


if __name__ == "__main__":
    main()