# ResearchClawBench（RCBench）调研报告

> 调研日期：2026-08-20。数据来源：GitHub 仓库与文件树（当日 HEAD）、官方 leaderboard.json（2026-08-20 快照）、arXiv:2606.07591 论文全文、GitHub issues、知乎公开讨论。
> 调研动机：评估"AI agent 自主科研"这一方向的工具生态，并为本仓（APT-RL / Unitree G1 / Isaac Lab）选型 3 个可用的 agent。

---

## 0. TL;DR

1. **ResearchClawBench 是目前"agent 能否独立做科研"最系统的公开 benchmark**：40 个来自真实发表论文的任务、10 个领域、专家标注的加权 rubric（checklist）、多模态 LLM 判分，50 分 = 追平原论文，>70 = 超越原论文。
2. **当前没有任何系统接近人类论文水平**：截至 2026-08-20 榜首 AutoSciRub（Codex GPT-5.6 底座）34.2 分；40 个任务里只有 7 个被任一系统打到 ≥50，≥70 的为 0。论文期（2026-06）最强 Claude Code 仅 21.5。
3. **失败模式不是"跑不动代码"，而是"科学上跑偏"**：280 次运行的错误分析显示，失败集中在实验设计偏离、证据链不匹配、科学核心缺失——不是基础设施崩溃。这对我们用 agent 做RL实验的启示是：**agent 写代码可靠，但实验设计和结论归因必须人来把关**（与本仓 HANDOFF 结论的归因纪律一致）。
4. **分数与花钱多少只有弱正相关**：$18.3/run 的 Qiushi（30.2 分）被 $2.3/run 的 OpenEvo（32.7 分）碾压；性价比最高的是 GLM-5.2（20.7 分，$1.24/run）。
5. **对本仓的选型结论**：Claude Code（主力 coding agent，RCB 内置 agent 中最强）+ Codex CLI（第二引擎，当前榜首生态的底座）+ ResearchClaw（文献综述/实验管理技能型助手）。三者已在本机配置到可用，见 §9。

---

## 1. 项目是什么

| 项 | 内容 |
|---|---|
| 定位 | 评测 AI coding agent 能否**独立完成端到端科研**（读原始数据 → 写代码 → 出图 → 写出出版级报告），再以同行评审式 rubric 对照**真实人类论文**严格判分 |
| 出品 | InternScience（上海 AI 实验室系，一作 Wanghan Xu，上交联系邮箱）；论文 arXiv:2606.07591（2026-06-09） |
| 仓库 | github.com/InternScience/ResearchClawBench，MIT 协议，创建于 2026-03-18 |
| 热度 | 244 stars / 23 forks；11 条 issue（质量高，见 §7）；**当天仍在更新**（2026-08-20 有社区提交刷新） |
| 口号 | "From Re-Discovery to New-Discovery"：重发现（复现论文=50 分）到新发现（超越论文>70 分） |
| 镜像 | HuggingFace 数据集（含 16 个额外社区任务）、ModelScope、BenchLM 收录 |

一句话：**它不测模型"知道什么"，只测 agent"能做出什么"**——给一个装好数据/文献/说明书的科研工作区，让 agent 无人值守地做出 `report/report.md`，然后请一个"严格的 LLM 审稿人"拿着从原论文提取的 checklist 逐项打分。

---

## 2. 评测方法学（怎么打分）

### 2.1 两阶段流水线

- **Stage 1 自主科研**：agent 拿到 workspace（`data/` 原始数据、`related_work/` 参考文献 PDF、`task_info.json` 任务说明；目标论文 `target_study/` **不给 agent 看**），必须独立完成探索数据 → 写分析代码 → 出 PNG 图 → 产出 `report/report.md`。工作区隔离，全工具权限，无人应答。
- **Stage 2 参照式评测**：多模态 LLM judge 拿（任务说明 + agent 报告 + rubric checklist + 目标论文），对每个 checklist 项打 0–100 分，加权求和。

### 2.2 Rubric（checklist）设计

每个任务的 checklist 由领域专家从原论文核心贡献提取，每项含：
- **type**：`text`（方法/结论）或 `image`（图像证据，需多模态视觉比对原图）；
- **keywords**：判分必须逐条核验的技术要点（如"ROC-AUC 提升""Monte Carlo 积分"）；
- **weight**：重要性权重。
  例：Physics_000（多壳层二十面体团簇）3 个 image 项各带 3 条 keyword、权重 0.3/0.4/0.3，要求复现 Caspar-Klug 三角剖分公式、Mackay 魔数序列 [1,13,55,147,309]、最优失配值 0.04/0.14 等。

### 2.3 双模式判分刻度（0–100，50=追平论文）

- **Mode A 目标优化**（定量结果）：0=缺失 → 41–50=指标与论文相当 → 51–60 略好 → 91–100 突破性超越。
- **Mode B 诊断分析**（定性机理）：0=空话 → 41–50=分析深度与论文相当 → 91–100=突破性洞见。
- judge prompt 明确写入：**"50 意味着和真实发表论文一样好，这是高 bar"**；对 AI 生成内容高度怀疑、长报告不加分、空话不给分。Substance over style。

### 2.4 统一 agent 协议（instructions_tmpl.py）

所有 agent 收到完全相同的提示词，要点：无人应答、禁止只输出计划不调工具（ReAct 无工具调用=任务终止）、禁止提问、报错就 debug、`report/report.md` 写完才算完、图必须 PNG。另配执行预算（ResearchHarness 默认 500 轮 / 3 小时 / 输入 128k token、96k 触发上下文压缩）。

### 2.5 论文的四个补充维度

除 rubric 外还评 Comprehensiveness/Depth/Instruction Following/Professionalism：**Professionalism 常被打到 70+，但与 rubric 分弱相关**——"排版漂亮"和"科学内容对"是两回事。这是该 benchmark 刻度设计里最值得学习的点。

---

## 3. 仓库逐层拆解

```
ResearchClawBench/  (8875 个条目)
├── tasks/               40 个任务 ×（task_info.json + data/ + related_work/ + target_study/{paper.pdf, checklist.json, images/}）
├── evaluation/          Flask + SSE 流式 UI + 判分引擎（score.py 用 structai 的 LLMAgent 逐项判分）
│   ├── agents.json      9 个 agent 预设（见 §5），加自己的 agent 只需加一条 JSON
│   ├── instructions_tmpl.py  统一提示词模板
│   └── run_task.py / server.py / score.py / config.py
├── eval_configs/        4 个 YAML 示例（单任务/混合重复/全任务/Qwen thinking）
├── rcb-eval / rcb-clear CLI 批量评测入口 + 重复输入清理
├── workspaces/          运行时生成（gitignored）
└── tests/  CONTRIBUTING.md  LICENSE(MIT)
```

### 3.1 40 个任务清单（10 领域 × 4）

| 任务 | 内容概要 |
|---|---|
| Astronomy_000 | 贝叶斯统计约束超轻玻色子（黑洞超辐射排除区间） |
| Astronomy_001 | 早期暗能量(EDE)能否缓解 Hubble 张力 |
| Astronomy_002 | ~1% 精度测量哈勃常数 |
| Astronomy_003 | 双黑洞系统（质量比/自旋/偏心率）演化 |
| Chemistry_000 | Kolmogorov–Arnold 型 GNN 分子性质预测 |
| Chemistry_001 | 蛋白/核酸/小分子统一深度学习框架 |
| Chemistry_002 | HADDOCK3 蛋白–配体对接流程 |
| Chemistry_003 | 原子构型 → 势能与性质预测 |
| Earth_000 | 19 个全球冰川区域质量平衡（233 组观测融合） |
| Earth_001 | NOAA 人工增雨记录因果分析 |
| Earth_002 | 热带气旋+海平面复合风险指数 |
| Earth_003 | ERA5 再分析 5 个高空变量 |
| Energy_000 | 锂电池宏观实验数据建模 |
| Energy_001 | 英国能源系统（拓扑+机组）优化 |
| Energy_002 | 非洲绿氢输欧平准化成本地理模型 |
| Energy_003 | 147 栋建筑电/热/冷负荷+PV 传感数据 |
| Information_000 | 统一自回归多模态理解/生成 |
| Information_001 | 免训练 MLLM 细粒度感知增强 |
| Information_002 | 15 篇量子多体 Hartree-Fock 多步解析计算 |
| Information_003 | 恶意网络流量检测 |
| Life_000 | 蛋白序列→水凝胶粘附强度回归 |
| Life_001 | 患者个性化新抗原疫苗肽段优化 |
| Life_002 | 蛋白复合物结构→功能分析 |
| Life_003 | 纳米孔电信号（FAST5/POD5）分析 |
| Material_000 | AI 搜索引擎加速材料发现 |
| Material_001 | 多模态材料数据（结构/成分/晶体图/显微图） |
| Material_002 | MPtrj 数据集预训练机器学习势 |
| Material_003 | AI 引导 vitrimer 聚合物逆向设计 |
| Math_000 | 视频帧级检测→多目标跟踪 |
| Math_001 | 光滑凸优化算法设计与证明 |
| Math_002 | 多智能体路径规划 MAPF（最大任务，2551 文件） |
| Math_003 | IMO 级几何题机器求解 |
| Neuroscience_000 | 小鼠攻击/嗅探行为帧级分类 |
| Neuroscience_001 | 果蝇视叶运动通路连接组分析 |
| Neuroscience_002 | 电镜过分割体→神经元重建 |
| Neuroscience_003 | 单细胞 readout 子集选择 |
| Physics_000 | 多壳层二十面体团簇稳定结构（Caspar-Klug） |
| Physics_001 | 魔角石墨烯门控输运 |
| Physics_002 | 随机量子线路采样保真度（XEB）估计 |
| Physics_003 | 石墨烯中红外泵浦超快动力学 |

注：全部为 **dry-lab 任务**（数据/代码/文献即可完成），不涉及湿实验——论文自己把这列为首要局限。

### 3.2 数据构建管线

高质量论文 → 专家提炼任务指令 → 设计 rubric → 收集数据与文献 → **人类复现验证**（确保 checklist 每项真的可达成）。"人类先复现一遍"这一步是它与一般自造 benchmark 的最大差别。

---

## 4. 榜单与强度（2026-08-20 快照，34 个系统）

### 4.1 总榜（均分 / 单次成本 / 单次时长 / 底座模型）

| # | 系统 | 均分 | $/run | 秒/run | 底座 |
|---|---|---|---|---|---|
| 1 | AutoSciRub（社区） | **34.2** | 3.62 | 2414 | Codex GPT-5.6-Sol |
| 2 | OpenEvo（社区） | 32.7 | 2.32 | 619 | GPT-5.5 |
| 3 | Qiushi Engine（社区） | 30.2 | **18.30** | 4042 | GPT-5.5 |
| 4 | InnoClaw（社区） | 28.8 | 7.10 | 1895 | GPT-5.5 |
| 5 | Open Science（社区） | 22.8 | 3.44 | 1031 | Claude-Opus-4.8 |
| 6 | **Claude Code**（内置最强） | 21.5 | 5.25 | 1575 | Claude-Opus-4.6 |
| 7 | ResearchHarness (Opus-4.8) | 21.1 | 3.97 | 1192 | Claude-Opus-4.8 |
| 8 | ResearchHarness (Opus-4.7) | 20.7 | 5.06 | 1518 | Claude-Opus-4.7 |
| 9 | ResearchHarness (**GLM-5.2**) | 20.7 | **1.24** | 2540 | GLM-5.2 |
| 10 | ResearchHarness (Opus-4.6) | 19.9 | 6.05 | 1814 | Claude-Opus-4.6 |
| 11 | ResearchHarness (MiniMax-M3) | 19.8 | 0.45 | 2565 | MiniMax-M3 |
| 12 | EvoScientist 0.1.1 | 18.8 | 4.08 | 2177 | GPT-5.4 |
| 13 | ResearchHarness (Qwen3.7-Max) | 18.7 | 0.42 | 668 | Qwen3.7-Max |
| 14 | **Codex CLI** | 18.4 | 2.01 | 1073 | GPT-5.4 |
| 15–33 | ResearchHarness 各 LLM | 18.2→12.4 | 0.03–2.12 | 390–2081 | GLM-5.1/Qwen/Kimi/Gemini/DeepSeek/GPT/MiMo/Grok/Hy3 等 |
| — | **OpenClaw** | 16.6 | 0.69 | 366 | GPT-5.4 |
| — | **ResearchClaw** | 16.3 | 0.82 | 435 | GPT-5.4 |
| — | EvoScientist 0.0.4 | 15.5 | 1.19 | 637 | GPT-5.4 |
| — | **ARIS Codex** | 13.6 | 0.74 | 396 | GPT-5.4 |
| — | **Nanobot** | 12.8 | 0.49 | 260 | GPT-5.4 |

（完整 34 行含每个 LLM 的明细，可直接读 Home 仓库 `data/leaderboard.json`。）

### 4.2 Frontier（每任务全系统最高分）

均 41.2 / 最高 65.6 / 最低 22.2；**≥50 的任务 7/40，≥70 的 0/40**。也就是说 33/40 的任务上，没有任何系统追平过原论文。

### 4.3 分领域强度（头部系统）

| 领域 | AutoSciRub | OpenEvo | Qiushi | InnoClaw | Claude Code |
|---|---|---|---|---|---|
| Physics | 40.9 | **51.3** | 38.6 | 45.2 | 32.3 |
| Astronomy | 39.8 | 44.8 | 39.5 | 41.6 | 30.2 |
| Math | **38.9** | 27.6 | 34.8 | 27.9 | 27.5 |
| Chemistry | **34.8** | 27.5 | 22.7 | 18.0 | 9.3 |
| Energy | 33.7 | 33.9 | 30.6 | 30.4 | 21.7 |
| Information | 29.5 | **38.2** | 31.4 | 24.1 | 25.0 |
| Neuroscience | **34.0** | 20.1 | 16.1 | 16.3 | **5.5** |

规律：**物理/天文等"数据+第一性原理"任务好做，神经科学/化学等"需要领域隐性知识和复杂预处理"的任务全员溃败**（Claude Code 神经科学均分 5.5）。头部系统间任务难度高度一致（任务级两两相关 0.64–0.86，中位 0.79）——说明差距主要在能力而非运气。

### 4.4 Pass@5（稳定性）

6 个 ResearchHarness LLM × 每任务 5 次重复：单任务内分数波动不小（例 Opus-4.8 Astronomy_000 五次 26.6/23.6/23.5/22.6/21.1，std≈1.8；有的任务 std 更大）。**单次跑分不可信，重复统计是必须的**——这和我们做 RL 实验"多 seed 汇报"的纪律完全同构。

---

## 5. 逐个 Agent 深评（内置 9 + 社区 4）

> 启动命令均来自 `evaluation/agents.json`；`<PROMPT>`/`<WORKSPACE>` 为占位符。

### 5.1 Claude Code（Anthropic 官方 CLI）
- **命令**：`claude --dangerously-skip-permissions -p <PROMPT> --output-format stream-json --verbose`
- **强度**：论文期七 agent 第一（21.5，Opus-4.6），当前总榜第 6、内置 agent 第一；40 任务赢 14 个。高分但**不统治**。
- **成本/时长**：$5.25/run、1575s——论文点名它是"高分+高成本+长时长"三高选手，独自抬高了 cost-score 相关性。
- **能力画像**：长时程 coding/调试最稳，报告专业度（Professionalism）高；神经科学类需要隐性领域知识的任务崩盘（5.5）。
- **扩展性**：MCP、subagents、hooks、skills、AGENTS.md/CLAUDE.md 项目记忆，生态最厚。
- **风评**：公认 coding agent 第一梯队；知乎 auto-research 讨论中被当参照系。

### 5.2 Codex CLI（OpenAI 官方 CLI）
- **命令**：`codex exec --full-auto <PROMPT>`
- **强度**：本体 18.4（GPT-5.4），并非最高；**但当前总榜前四里三个（AutoSciRub/OpenEvo/Qiushi/InnoClaw 中除 Open Science 外）都构建在 Codex/GPT-5.5-5.6 生态上**——Codex 是当下最强科研 harness 的底座。
- **成本/时长**：$2.01/run、1073s，便宜耐造。
- **扩展性**：AGENTS.md 原生支持、MCP、sandbox 模式分级；`--full-auto` 适合无人值守。
- **评**：作为"第二引擎"价值高——与 Claude 交叉验证实验设计/结论，正好对冲 RCB 揭示的"单 agent 系统性跑偏"风险。

### 5.3 ARIS Codex（Auto-claude-code-research-in-sleep）
- **命令**：RCB 中仅支持导入历史 run，不可一键启动。
- 13.6 分、$0.74、396s。卖点是"睡觉时让 Claude Code 自动跑研究"的无人值守编排。对本仓的启发是**夜间批量跑 probe/扫参**的编排模式。

### 5.4 OpenClaw（前身 Clawdbot/Moltbot，自托管 agent 网关）
- **命令**：`openclaw agent --agent main --timeout 3600 --message <PROMPT>`
- 16.6 分、$0.69、366s。强项是渠道（WhatsApp/Telegram/Discord…）+ 常驻守护进程 + 定时任务，本质是"个人助理网关"。RCB 任务上表现平平；但**"常驻 + 推送 + 定时"模式很适合监控服务器上 E 系列训练任务**（nohup 起的 PPO 跑完了推送结果）。安装较重（gateway+channels），本次未选。

### 5.5 Nanobot（HKUDS 超轻量 agent）
- **命令**：`nanobot agent -m <PROMPT> -w <WORKSPACE>`
- 12.8 分、$0.49、260s（最快）。可靠的多步工具执行，无花活，分数低。适合当"最小可用对照组"。

### 5.6 EvoScientist（自进化 AI 科学家）
- **命令**：`evosci --ui cli --workdir ... --auto-approve -p ...`
- v0.1.1 18.8（$4.08/2177s）vs v0.0.4 15.5——**版本迭代涨 3.3 分**，是少数可看到持续进化的系统。进化式搜索"科研想法→实验→报告"。重、慢，适合离线无人值守场景。

### 5.7 ResearchClaw（本地优先 Research OS，本仓选型之一，详见 §9）
- **命令**：`researchclaw agent -m <PROMPT> -w <WORKSPACE>`（RCB 预设）
- 16.3 分、$0.82、435s——刷分不是它的价值。**内置技能正对我们 workflow**：arxiv / literature_review / citation_network / paper_summarizer / experiment_tracker / research_notes / figure_generator / cron；工具含 semantic_scholar_search、bibtex 管理、data_describe、run_shell。本地文件优先，支持 openai/anthropic/gemini/dashscope/deepseek/ollama 等多 provider。

### 5.8 LingTai（"AI 组织"基座）
- **命令**：`cd <WORKSPACE> && lingtai-tui -p <PROMPT>`
- agents.json 有预设但 leaderboard 无成绩（论文亦未评测）。多 agent 组织化协作方向，观望。

### 5.9 ResearchHarness（官方轻量基线 harness）
- **命令**：`python3 ResearchHarness/run_agent.py <PROMPT> --workspace-root ...`（pip 包 `researchharness`）
- 官方"控制变量"工具：固定工具集（Bash/terminal 会话/webfetch/readpdf/…）、128k 输入预算 + **96k 触发的上下文压缩**、500 轮上限。用它横向测了 17+ 个裸 LLM。
- 结论性观察：同 harness 下 Opus-4.8 21.1 vs Grok-4.3 12.4，**harness 拉不开模型本身差距**；GLM-5.2（20.7，$1.24）与 MiniMax-M3（19.8，$0.45）证明国产模型 auto-research 已逼近 Claude 系——与知乎讨论一致。

### 5.10–5.13 社区提交系统（当前第一梯队）
- **AutoSciRub**（34.2，榜一）：浙大 zjunlp 出品，rubric 自动化框架 + Codex GPT-5.6-Sol；$3.62、2414s。
- **OpenEvo**（32.7）：浙大 CompLifeLab 进化计算框架 + GPT-5.5；$2.32、619s——**头部里最便宜最快，性价比标杆**。
- **Qiushi Engine**（30.2）：商业系统（oxelra.com），$18.30/run 全场最贵、4042s 全场最慢——"高投入≠高回报"的实证。
- **InnoClaw**（28.8）与 **Open Science**（22.8，Claude-Opus-4.8 底座）：社区/科研机构提交，完整 40 任务结果。

---

## 6. Token 使用与成本分析

- RCB 公开口径是 **cost_usd（按 API 计费折算）+ duration**，不直接公开 token 计数；判分侧还要另付 judge 调用（每任务按 checklist 逐项 + 图像）。
- **成本跨度 600 倍**：$0.03（Hy3-Preview）到 $18.30（Qiushi）；同为 GPT-5.5 底座，OpenEvo $2.32 vs Qiushi $18.30。
- **性价比榜**（≥17 分里每分单价最低）：GLM-5.2（$1.24/20.7）≈ Qwen3.7-Max（$0.42/18.7）≈ DeepSeek-V4-Pro（$0.28/17.1）> MiniMax-M3（$0.45/19.8）>> Claude Code（$5.25/21.5）。
- **论文结论**：score 与资源投入仅弱正相关，且该相关性主要被 Claude Code 抬高；Pareto 前沿的"膝盖"附近是便宜模型（GPT-5.5 RH $1.82/17.0 一档）。
- **长任务上下文管理是硬约束**：ResearchHarness 96k 压缩触发、128k 上限、16k 输出上限；RCB CLI 还专门提醒 thinking 模式要把 `max_output_tokens` 开大。对我们跑 Isaac Lab 长日志分析的启示：**喂日志前先切片/摘要，别硬塞**。
- 时长：快档（Nanobot 260s / ARIS 396s / ResearchClaw 435s）是"少做事所以快"；头部系统普遍 600–2500s，Qiushi 4042s。

---

## 7. 风评

- **GitHub**：244★/23 fork，5 个月从发布到 30+ 系统上榜，社区提交活跃（AutoSciRub/OpenEvo/InnoClaw 均为外部团队主动送测）。11 条 issue 讨论质量高：#10 锚定式 rubric 对开放式研究的有效性之争、#9 judge 只看前 5 张图的 bug、#5 agent 能否读 target_study（防作弊边界）、#6 新 OpenAI judge 模型兼容——维护者均回复并修复。
- **知乎**：《Claude v.s. GLM：谁是最强的 Auto-Research 大模型？》等帖将其当主要刻度：观点一，最高才 21.5 分（论文期），"谈自主科学发现为时过早"；观点二，该基准是"一把刻度清晰的尺子"，GLM/MiniMax 已逼近 Claude。两派都认可 benchmark 本身质量。
- **HN/Reddit**：无显著讨论（同名系 **ClawBench**（TIGER-AI-Lab，153 个日常在线任务）是完全不同的项目，别混淆）。
- **总体判断**：学术圈认可度 > 社区热度；属于"做实事、低营销"的 benchmark，且仍在快速迭代（当天还有更新）。

---

## 8. 扩展性

| 维度 | 评价 |
|---|---|
| 接入新 agent | **极好**：`evaluation/agents.json` 加一条 JSON（cmd 里用 `<PROMPT>`/`<WORKSPACE>` 占位符），重启即用 |
| 新增任务 | **好**：HF Space 上传 zip → 校验 → PR 到 HF 数据集 → `download_tasks.py` 拉进本地 `tasks/` 自动发现（已积累 16 个社区任务） |
| 批量评测 | **好**：`rcb-eval` YAML（并发/重复/自动判分/md 报告），agent 与 judge 模型解耦，`--dry-run` 校验 |
| 磁盘管理 | `rcb-clear` 清理 CLI 批量跑产生的重复任务输入 |
| 代码栈 | 轻（Flask + 原生 JS + structai），无重框架，二开门槛低 |
| 缺点 | 偏 Linux/bash（Windows 裸跑会踩脚本坑）；judge 依赖外部多模态 API key；榜单数据与主仓分离（Home 仓库）；只测 dry-lab |

---

## 9. 本仓选型与配置实录（2026-08-20）

### 9.1 为什么选这三个（对 APT-RL / G1 / Isaac Lab 工作流）

| 选型 | 角色 | 依据 |
|---|---|---|
| **Claude Code** | 主力科研 coding agent | RCB 内置 agent 最强（21.5），长时程代码/调试最稳；本机已重度使用（~/.claude 有历史配置）；AGENTS.md 项目记忆可直接继承本仓规范 |
| **Codex CLI** | 第二引擎 / 榜首生态底座 | 本体 18.4 不算顶尖，但**当前 RCB 前四几乎全是 Codex/GPT-5.5-5.6 生态**；便宜（$2/run）；与 Claude 交叉验证实验设计，对冲"单 agent 系统性跑偏" |
| **ResearchClaw** | 文献综述 + 实验管理技能型助手 | RCB 得分平庸（16.3）但技能栈正对我们：literature_review / citation_network / paper_summarizer / experiment_tracker / cron；本地优先，多 provider |
| **AutoSciRub**（2026-08-21 补装） | **我们方向实测冠军**：rubric 驱动的科研执行/验证插件，装在 Codex 宿主上 | 按"我们方向"代理任务集（Information+Math 8 任务）重算榜单：**AutoSciRub 34.2 第一**（核心感知/规划 5 任务子集 31.4 亦第一），拿下 Math_002（多智能体路径规划）41.0、Math_003 30.8 两个单任务冠军；详见 §9.4 |

### 9.1.1 我们方向的子集排名（2026-08-21 补算）

RCB 无机器人领域，取最接近的代理集（Information=深度学习方法，Math_000/M002=跟踪与规划）：

| 代理集 | 第一名 | 第二名 | 第三名 |
|---|---|---|---|
| Information+Math 全 8 任务 | **AutoSciRub 34.2** | Qiushi 33.1（闭源商业，不可装） | OpenEvo 32.9 |
| 核心感知/规划 5 任务（I000/I001/I003/M000/M002） | **AutoSciRub 31.4** | OpenEvo 31.1 | Qiushi 30.1 |
| 单任务冠军（8 任务中） | AutoSciRub×2（M002 41.0、M003 30.8） | OpenEvo×2（I001 50.8、I002 51.1） | Qiushi×3（闭源） |

注意 Qiushi（oxelra.com）为商业闭源系统无法安装，故可装范围内 AutoSciRub 是明确冠军，OpenEvo 为开源第二。

备选说明：OpenClaw（常驻网关，适合服务器任务监控，但安装重、RCB 分低）与 EvoScientist（自进化科研，重且慢）暂不选；gemini CLI 已修复但 OAuth 需交互重登，备用。

### 9.2 修复与配置记录（全部已完成并验证）

本机 agent CLI 原本大面积损坏，根因两层：① npm 全局 prefix 指向非标准路径 `C:\home\zyz\.npm-global`，且生成的 bash/cmd shim 内路径被 Anaconda 环境污染成 `C:\ProgramData\Anaconda3\Library\c\home\...`；② claude-code≥2.1 改为原生 exe 入口（`bin/claude.exe`），旧 bashrc 函数还指向已不存在的 `cli.js`。已做：

1. `npm i -g @anthropic-ai/claude-code @openai/codex @google/gemini-cli` 重装（2.1.237 / 0.148.0 / 0.56.0）。
2. `~/.bashrc` 三个函数改为正确入口（claude→原生 exe，gemini→`bundle/gemini.js`，codex→node 直调 `bin/codex.js`）。
3. 重写 `C:\home\zyz\.npm-global\` 下 9 个 shim（bash/cmd/ps1 × claude/codex/gemini），Git Bash / PowerShell / cmd.exe 三 shell 全部验证通过。
4. **codex**：已登录（API key 方式），`codex login status` ✅。
5. **claude**：端到端 ping 返回 OK ✅。注意：本机 claude 配置的路由模型是 `deepseek-v4-flash[1m]`（用户自配），当前会自动回退并正常工作；如要切回官方模型改 `~/.claude/settings.json`。
6. **researchclaw 0.1.0a2**：GitHub 源码安装（源码在 `C:\Users\zyz\Documents\tools\ResearchClaw`；注意 PyPI 上的 `researchclaw` 包已弃用并指向可疑的改名 `cr-cli`，**不要从 PyPI 装**）+ `researchclaw init --defaults --accept-security` 完成初始化（`~/.researchclaw`）+ models 配置（见下）。
7. 已知遗留坑：`command-code` 包在 PATH 里提供名为 `cmd` 的 shim，**会劫持 Git Bash 里的 `cmd` 调用**（它自己是坏的）；本次未动它（用户已装的工具不擅自删），遇到 `cmd //c` 报 node loader 错误时用 `/c/Windows/System32/cmd.exe` 全路径即可绕开。npm 重装包会重新生成 shim，若再次损坏，重跑修复脚本 `C:\Users\zyz\Documents\tools\fix_npm_shims.sh`。

### 9.3 使用建议（贴合本仓纪律）

- 三个 agent 都会自动读仓库根 `AGENTS.md`（claude/codex 原生支持，researchclaw 有同名机制），本仓"服务器训练、本机只检视、实验编号、中文日志"的约束对它们直接生效。
- 典型用法：
  - Claude Code：`claude "检视 apt_g1/isaac/ppo_core.py 最近改动，对照 EXPERIMENT_TRACKER 检查有没有未登记的实验"`（本机代码检视，勿让它起训练）。
  - Codex CLI：`codex exec "读 refine-logs/STAGE_SUMMARY_2026-08-13.md，独立复述当前蒸馏路径的边界结论，指出与 FINAL_REPORT 表述不一致处"`——**交叉验证归因**。
  - ResearchClaw：文献侧 `researchclaw papers search "action pretrained transformer humanoid"` → literature_review 技能生成综述初稿到 refine-logs；实验侧用 experiment_tracker 技能维护 Run/Data 台账。
- RCB 的教训直接转成我们的操作纪律：agent 产出**必须人审实验设计与结论归因**（它最大的失败模式是把协议做偏还写得头头是道）；报告专业度与内容正确性弱相关，别被排版骗了。

### 9.4 AutoSciRub 安装与实测记录（2026-08-21）

**性质**：AutoSciRub 不是独立 agent，是 zjunlp 的**插件包**（六阶段：rubric skeleton 归纳 → 文献 grounding → 任务-数据探查 → 准则合成 → rubric 引导执行 → 逐条验证 → 定向修改），跑在宿主 agent 上。榜单成绩即 "Codex CLI + AutoSciRub" 组合，因此宿主用我们已修好的 codex。

**安装实录**（源码在 `C:\Users\zyz\Documents\tools\AutoSciRub`）：
1. `git clone https://github.com/zjunlp/AutoSciRub`，`python scripts/validate.py` → validation ok。
2. 官方 `install.sh codex` 在 codex 0.148 下报 "marketplace root does not contain a supported manifest"——**坑：新版 codex 要传仓库根目录而不是 `.agents/plugins`**。成功命令：`codex plugin marketplace add C:\Users\zyz\Documents\tools\AutoSciRub` + `codex plugin add autoscirub@autoscirub-local`（已顺手把清单里多余的 `interface` 字段去掉、authentication 改 ON_USE）。
3. 状态确认：`codex plugin list` 显示 `autoscirub@autoscirub-local installed, enabled 0.1.0`。

**两个 Windows 坑（已解决/已绕）**：
- codex 0.148 删除了 `--full-auto` 参数（RCB 仓库 agents.json 里的预设命令是老版本的），本机可用：`codex exec -s workspace-write "<任务>"`。
- codex 沙箱在 Windows 上首次运行报"无法创建终端进程、读取 `~/.codex/cap_sid` 失败"——那是 6 月份的陈旧会话文件，删掉即愈；若再遇到沙箱终端问题，可在隔离目录用 `--dangerously-bypass-approvals-and-sandbox`（注意范围控制）。

**端到端实测**（tmp/autoscirub_smoke，只跑第 1 阶段）：给一句我们自己的课题"在 Isaac Lab 平坦地形上复现 APT-RL 基线，对比冻结/微调解码器对 G1 行走速度与航向稳定性的影响"，插件正确产出 `.autoscirub/rubric_skeleton.json`（task_id=apt_rl_g1_flat_decoder_comparison，goals G1 复现基线 / G2 冻结解码器条件 / G3 微调条件 / G4 行走速度对比），未越权跑文献检索。

**用法**：在任务目录下自然语言触发，例如
`codex exec -s workspace-write "用 autoscirub 技能：induce the rubric, guide execution, verify criterion by criterion, run targeted revision"`（全流程），或只触发某一阶段。状态文件落 `.autoscirub/`（rubric_skeleton/literature_grounding/task_data_profile/executable_rubric/verification_report/revisions/）。
**可选增强 key**（均非必需，arXiv 检索免 key）：`SEMANTIC_SCHOLAR_API_KEY`（提额）、`TAVILY_API_KEY`（网页检索）、`MINERU_API_TOKEN`（高精度 PDF 解析）、`OPENALEX_EMAIL`（polite pool）；项目级检索配置抄 `config/sources.example.yaml` → `.autoscirub/config.yaml`。

### 9.5 第二/三名追装与 ZCode 集成（2026-08-21）

**追装结果**：
- **Qiushi（子集第二 33.1）**：核查 oxelra.com——纯商业闭源（杭州奥思拉科技），无 API/CLI/自部署/注册入口，**无法接入**，仅能人工联系厂商。
- **OpenEvo（子集第三 32.9）**：GitHub `CompLifeLab-ZJU/OpenEvo` **仍 404**（直查/组织列表/PyPI/全网均无，疑似评测后改名或撤库），暂不可装；其强项恰是多模态感知（Information_001/002 双单任务冠军），回归后应优先补装。**另注（2026-08-21）**：用户找到的 `deepelementlab/openevo` 经核实**不是**该项目的迁移（作者 deepelement.ai、定位是 agent 记忆层、代码零关联），系重名；已顺手装下作本机 agent 舰队的跨会话记忆层（`evo serve` @127.0.0.1:8765，Claude Code 插件 `openevo-memory` 已装、OpenClaw 插件已注册，写入 E35 结论→语义检索召回实测通过），详见技能卡 `/agent-openevo`。
- **InnoClaw（总榜第四 28.8，遂为可装第二名）**：源码装于 `C:\Users\zyz\Documents\tools\InnoClaw`（Next.js+SQLite，Node 24，1091 包）。已配 `.env.local`（WORKSPACE_ROOTS=本仓，OPENAI key 复用 codex）、`drizzle-kit migrate` 建库、dev server 冒烟 3s HTTP 200 ✅。定位：自托管科研工作区（文献多角色研讨/RAG 引用问答/200+ 科学技能/Shell-Slurm-K8s 远程实验门控）。安全注意：含 shell 执行能力，只许 localhost。

**ZCode 集成**（`C:\Users\zyz\.zcode\skills\`，新会话生效）：7 个技能——`agent-codex-autoscirub`（冠军调用卡）、`agent-claude-code`、`agent-researchclaw`、`agent-innoclaw`、`agent-qiushi`（闭源指引卡）、`agent-openevo`（下线观察卡，含找回检查清单）、`research-agents`（统一路由：按任务类型分流 + 速查命令 + 纪律）。

---

## 10. 局限与注意事项

- 只测 dry-lab；wet-lab、真实机器人硬件在环（我们的 Isaac Lab 训练其实也超出其覆盖范围——RCB 任务都是"给定数据集出报告"，不含长时程 GPU 训练运维）。
- 判分是 LLM judge（默认 GPT-5.1），rubric 锚定式打分对"开放式新发现"有系统性偏差（issue #10 的核心争论）；50 分锚点合理性依赖"人类复现验证"环节的质量。
- 榜单混装"官方跑的"与"社区自报"的结果（README 已标注），横向比较需留意底座模型差异。
- 我们引用其数字时应注明快照日期（本报告为 2026-08-20），该榜单仍在快速变化。

## 11. 参考

- 仓库：<https://github.com/InternScience/ResearchClawBench>；榜单数据：<https://github.com/InternScience/ResearchClawBench-Home/blob/main/data/leaderboard.json>
- 论文：<https://arxiv.org/abs/2606.07591>；主页：<https://internscience.github.io/ResearchClawBench-Home/>
- HF 数据集：<https://huggingface.co/datasets/InternScience/ResearchClawBench>（+16 社区任务）
- 知乎讨论：《Claude v.s. GLM：谁是最强的 Auto-Research 大模型？》<https://zhuanlan.zhihu.com/p/2058484335074863040>
- 各 agent：Claude Code <https://code.claude.com/docs/en/overview> | Codex CLI <https://developers.openai.com/codex/cli> | OpenClaw <https://openclaw.ai/> | Nanobot <https://github.com/HKUDS/nanobot> | EvoScientist <https://github.com/EvoScientist/EvoScientist> | ResearchClaw <https://github.com/ymx10086/ResearchClaw> | AutoSciRub <https://github.com/zjunlp/AutoSciRub> | OpenEvo <https://github.com/CompLifeLab-ZJU/OpenEvo> | InnoClaw <https://github.com/SpectrAI-Initiative/InnoClaw>
