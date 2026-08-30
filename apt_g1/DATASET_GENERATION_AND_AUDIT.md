# APT-RL x SONIC G1：原始数据集生成、处理与审计说明

更新时间：2026-08-17  
适用目录：`apt_g1/data/exp_all3`  
主要源码：`drive_exp3.py`、`build_exp3_dataset.py`、`train_token_vae_e27.py`、`train_token_vae_e35.py`、`train_token_vae_e39.py`

## 先回答三个问题

### 1. 这个数据集到底是什么？

`exp_all3` 不是人工动作捕捉数据，也不是 RL 训练过程中自动产生的轨迹。

它记录的是：**官方 SONIC 闭环部署程序在不同命令下，每个控制帧输出的 64 维 token**，以及该帧对应的本体感觉和命令。后续实验用这些 token 训练一个 VAE，使 RL 不直接输出关节目标，而是输出 16 维潜变量 `z`。

运行时的关系是：

```text
RL policy 输出 z (16 维)
    -> 冻结 VAE：z + 步态相位 + 可选速度/方向条件 -> token (64 维)
    -> 冻结 SONIC decoder：token + proprio 历史 -> 关节目标 (29 DoF)
```

因此，`exp_all3` 的角色是“学习 SONIC token 运动流形”的训练资产。它不是 APT-RL 论文所需的自洽轨迹优化力矩数据。

### 2. 数据最初从哪里来？

数据来自官方 C++ planner/encoder/decoder 驱动的闭环控制会话。采集时，部署程序应当在其日志目录中写出三类 CSV：

```text
官方闭环控制（约 50 Hz）
    ├─ policy_input.csv        每帧 token + proprio
    ├─ commands.csv            命令流
    └─ logs/token_state.csv    独立 token 记录，用于交叉检查
```

`drive_exp3.py` 只负责向部署程序发送键盘事件；它不负责生成 token，也不负责把 CSV 转为 NPY。

### 3. 当前能否从头复现？

**不能完全从本机重现。** 当前 `C:\Users\zyz\Documents\gr00t\apt_g1\data` 仅保留 `exp_all3` 成品，`exp1_raw`、`exp2_raw`、`exp3_raw` 不在本机。

这意味着：

- 可以检查成品数组是否自洽；
- 可以从源码准确说明 CSV 如何被转换为 NPY；
- 不能重新验证原始控制会话、CSV 行时间、命令与 token 的真实同步关系；
- 不能重新执行构建脚本并证明当前 68,093 行一定由当时的三份原始 CSV 生成。

下面的“已验证”指当前源码或现存 NPY 可以直接检查。“无法验证”指需要已缺失的原始 CSV 或远程运行环境。

---

## 1. 完整流程图

```text
阶段 A：采集

官方 planner / encoder / decoder
    -> 在闭环控制过程中每帧生成 token 和 policy 输入
    -> 记录 policy_input.csv、commands.csv、logs/token_state.csv

阶段 B：构建 exp_all3

exp1_raw + exp2_raw + exp3_raw
    -> build_exp3_dataset.py
    -> 对每个批次：读取、校验 token、按行数重采样命令、生成标签
    -> 按 exp1 -> exp2 -> exp3 顺序拼接
    -> token.npy / proprio.npy / cmd.npy / mode.npy / speed.npy / angle_bin.npy

阶段 C：VAE 训练前处理

token.npy + mode.npy (+ angle_bin.npy)
    -> 以 WALK(mode=2) token 拟合 2D PCA
    -> 每行计算相位 phi
    -> 将 token 变成 10 帧因果窗口
    -> E27：相位条件 VAE
    -> E35：相位 + phase-rate 速度 bin + 方向 bin 条件 VAE
    -> E39：E35 + 方向/速度对抗解耦头
```

关键点：阶段 B 只做文件转换和标签构造，**不会调用 planner，不会推导新 token，也不会执行 RL**。

---

## 2. 阶段 A：原始闭环采集如何进行

### 2.1 `drive_exp3.py` 做了什么

`drive_exp3.py` 使用 PTY（伪终端）启动远程/临时部署脚本：

```python
proc = subprocess.Popen(["bash", "/tmp/g1deploy_exp3.sh"], ...)
```

它将部署终端输出写入：

```text
/tmp/exp3/deploy.log
```

并维护一个 `events` 列表。每次进入一个预设阶段时，脚本记录：

```json
[墙钟秒数, "阶段名"]
```

最后写到：

```text
/tmp/exp3/events.json
```

源码：`drive_exp3.py:16-27`、`drive_exp3.py:47-70`、`drive_exp3.py:208-209`。

### 需要特别区分的事实

- `events.json` 记录的是脚本的阶段开始时间，不是每个数据帧的时间戳。
- `drive_exp3.py` 没有把某个键盘事件绑定到 `policy_input.csv` 的某一行。
- 真正写 CSV 的程序是 `/tmp/g1deploy_exp3.sh` 启动的部署进程；这个脚本当前不在版本库中。

因此，`drive_exp3.py` 可以说明“采集意图和命令顺序”，不能单独证明“第 k 行 token 对应哪次按键”。

### 2.2 exp3 的命令序列

`drive_exp3.py` 按墙钟时间执行以下过程：

| 顺序 | 阶段 | 时长 | 输入 | 目的 |
|---:|---|---:|---|---|
| 1 | 等待启动 | 最多 120 s | 无 | 等待部署进程输出并检查是否存活 |
| 2 | `start_control` | 12 s | `]` | 启动控制 |
| 3 | `planner_mode` | 6 s | Enter | 进入 planner 模式 |
| 4 | `idle_12s` | 12 s | 无 | 初始静置 |
| 5 | `walk_fwd_20s` | 20 s | mode `2`，持续 `w` | 前向 WALK |
| 6 | `walk_heading_right_sweep_40s` | 40 s | 持续 `w`，间歇 `q` | 右向航向扫描 |
| 7 | `walk_heading_left_sweep_40s` | 40 s | 持续 `w`，间歇 `e` | 左向航向扫描 |
| 8 | `walk_strafe_right_20s` | 20 s | mode `2`，持续 `.` | 右横移 |
| 9 | `walk_strafe_left_20s` | 20 s | mode `2`，持续 `,` | 左横移 |
| 10 | `slow_speed_plus_20s` | 20 s | mode `1`，`0` 四次，再持续 `w` | 慢走较高速度 |
| 11 | `slow_speed_minus_20s` | 20 s | mode `1`，`9` 六次，再持续 `w` | 慢走较低速度 |
| 12 | 停止 | 约 15 s | `r`、`o` | 空闲并停止 |

动作段之间通常有 8 s 的 `idle` 段。右/左航向扫描的 `q/e` 触发取决于 `int(time.time() * 4) % 3 == 0`，因此它依赖墙钟调度，不是严格固定的每 N 帧一次。

源码：`drive_exp3.py:81-213`。

### 2.3 原始 CSV 的预期结构

构建脚本假定每个原始批次目录具有：

```text
expX_raw/
    policy_input.csv
    commands.csv
    logs/
        token_state.csv
```

| 原始文件 | 读取类型 | 使用方式 | 最终是否直接保存 |
|---|---|---|---|
| `policy_input.csv` | `float32` | 前 64 列作为 token；第 65 列以后作为 proprio | 是 |
| `commands.csv` | `float64` | 提取 planner、mode、方向向量、speed、height | 是，转换成多个 NPY |
| `logs/token_state.csv` | `float64`，跳过首行 | 取第 6-69 列作 token 对照检查 | 否，仅用于验证 |

源码：`build_exp3_dataset.py:31-35`、`build_exp3_dataset.py:79-86`。

### `policy_input.csv`

构建代码隐含的结构是：

```text
列 0..63       64 维 SONIC token
列 64..末尾    930 维 proprio
```

现存成品可确认：`token.npy` 为 64 维，`proprio.npy` 为 930 维，因此一行 `policy_input` 至少应有 994 列。

但是，构建脚本本身**没有**断言它恰好为 994 列，也没有记录每一列的物理语义。

### `commands.csv`

构建代码实际读取的列如下。下表列号从 0 开始：

| `commands.csv` 列 | 变量 | 用途 |
|---:|---|---|
| 5 | `planner` | 写入 `cmd.npy` 的最后一维 |
| 7 | `mode` | 写入 `mode.npy`，也编码成 5 维 one-hot |
| 8:11 | `mdir` | 三维运动方向；用于 `cmd.npy` 和 `angle_bin.npy` |
| 11:14 | `fdir` | 三维朝向方向；写入 `cmd.npy` |
| 14 | `speed` | 写入 `speed.npy` 和 `cmd.npy` |
| 15 | `height` | 写入 `cmd.npy` |

代码没有解释其余列，也没有验证 mode 值是否只属于预期集合。

---

## 3. 阶段 B：如何从原始 CSV 生成 `exp_all3`

入口脚本是 `build_exp3_dataset.py`。部署时它使用 Linux 路径：

```python
RAW = "/home/cvgluser/ros2_data/apt_g1/data"
OUT = "/home/cvgluser/ros2_data/apt_g1/data/exp_all3"
```

源码：`build_exp3_dataset.py:12-14`。

### 3.1 步骤 1：读取 CSV

`load_csv()` 的行为：

1. 读取所有文本行。
2. 跳过空行。
3. 按逗号分割。
4. 若最后一个字段为空，去掉它。
5. 将其余字段全部转为 `float`，最后转为指定的 NumPy dtype。

`policy_input.csv` 与 `commands.csv` 不跳过首行；`token_state.csv` 固定跳过首行。也就是说，前两个 CSV 必须没有表头，或其首行必须可以解析为浮点数。

源码：`build_exp3_dataset.py:17-28`、`build_exp3_dataset.py:31-35`。

### 3.2 步骤 2：token 交叉检查

脚本把：

```python
policy_input[:, :64]
```

和：

```python
token_state[:, 5:69]
```

作比较。

### 情况 A：两者行数相同

计算全体元素的最大绝对差：

```python
max(abs(policy_input[:, :64] - token_state[:, 5:69]))
```

### 情况 B：两者行数不同

从 `token_state` 的起点 0 到最多 1999 之间尝试偏移量 `off`。如果存在一个偏移使：

```python
max(abs(policy_input[:, :64] - token_state[off:off+n, 5:69])) < 1e-3
```

则记为对齐成功。

### 情况 C：没有找到偏移

脚本仅打印 WARNING，并继续使用 `policy_input[:, :64]` 构建数据集。

### 这一步实际保证了什么

它只是在“两个 token 记录通道是否数值相同”这个问题上提供弱验证。它没有保证：

- 两个记录器使用同一时间戳；
- 偏移超过 1999 行时仍可找到；
- 每个 CSV 的列数正确；
- 未通过验证时构建会停止。

最终 token 的唯一权威来源仍是 `policy_input.csv` 的前 64 列。

源码：`build_exp3_dataset.py:36-53`。

### 3.3 步骤 3：按行数比例把命令映射到 token 帧

这是构建流程最重要、也最需要谨慎理解的一步。

设：

```text
n = policy_input 的行数 = token/proprio 帧数
m = commands 的行数
```

对于第 `i` 个 token 帧，脚本取的命令行号为：

```text
j = floor(i * m / n)
```

再执行：

```python
cm = commands[j]
```

例如：

- 当 `n == m`，第 i 个 token 使用第 i 个命令，等价于逐行对齐。
- 当 `m < n`，同一命令行会复制给多个 token 帧。
- 当 `m > n`，部分命令行不会被用到。

这个方法的前提是：两个文件覆盖同一段会话，且它们从开始到结束没有显著启动延迟、停止空窗、丢帧或时钟漂移。

它**不是**时间戳 join。因此，即使两份文件都看起来“总时长差不多”，局部命令仍可能错位。例如，部署启动后若 `commands.csv` 先记录 2 秒空闲而 `policy_input.csv` 后开始记录，按行数比例映射会把整个动作段的标签推迟或提前。

源码：`build_exp3_dataset.py:54-56`。

### 3.4 步骤 4：构造 mode、cmd 和方向 bin

对对齐后的每一行命令 `cm`，脚本构造：

```text
mode     = cm[7]
mdir     = cm[8:11]
fdir     = cm[11:14]
speed    = cm[14]
height   = cm[15]
planner  = cm[5]
```

### mode one-hot

固定模式表为：

```python
MODES = [0, 1, 2, 17, 18]
```

因此 `mode=2` 被编码为：

```text
[0, 0, 1, 0, 0]
```

### 14 维 `cmd`

拼接顺序固定为：

```text
cmd = [
    mode_one_hot(5),
    mdir(3),
    fdir(3),
    speed(1),
    height(1),
    planner(1),
]
```

总维度为：

```text
5 + 3 + 3 + 1 + 1 + 1 = 14
```

### 8 类方向 `angle_bin`

脚本只使用 `mdir` 的 x、y 分量：

```python
angle = atan2(mdir_y, mdir_x)
angle_bin = floor((angle + pi) / (2*pi) * 8) % 8
```

其中：

```text
bin 4 = +x 前向
```

这里没有做方向向量归一化，也没有为接近零的向量定义专门类别。由于 `angle_bin` 在 `float64` 的原始命令上计算，而 `cmd.npy` 中的方向后来被保存为 `float32`，极接近分箱边界的行可能无法由保存后的 `cmd.npy` 完全复算。

源码：`build_exp3_dataset.py:57-86`。

### 3.5 步骤 5：按批次顺序拼接并写 NPY

脚本按以下固定顺序检查目录：

```text
exp1_raw -> exp2_raw -> exp3_raw
```

每个存在的批次调用一次 `process()`，然后按行方向拼接：

```python
np.concatenate(arrs, axis=0)
```

最终写入：

```text
exp_all3/
    token.npy
    proprio.npy
    cmd.npy
    mode.npy
    speed.npy
    angle_bin.npy
    meta_modes.npy
```

`meta_modes.npy` 固定写入 `[0, 1, 2, 17, 18]`。

### 当前脚本的行为风险

若某个预期原始目录不存在，脚本只打印：

```text
skip expX: ... missing
```

并继续写 `exp_all3`。只要至少存在一个批次，它就不会报错。这意味着一次误运行可能用不完整数据覆盖原有成品。

脚本也不保存：

- 每个批次的行区间；
- 原始 CSV 文件 hash；
- 采集时间；
- source batch / episode ID；
- 构建脚本版本。

源码：`build_exp3_dataset.py:89-119`。

---

## 4. 当前成品 `exp_all3` 的实际状态

以下结果直接读取当前本机的 NPY 文件得到。

| 文件 | 形状 | dtype | 已检查结果 |
|---|---:|---|---|
| `token.npy` | `(68093, 64)` | `float32` | 全为有限值；范围 `[-0.875, 0.8125]`；全部处于 `1/16` 格点 |
| `proprio.npy` | `(68093, 930)` | `float32` | 全为有限值 |
| `cmd.npy` | `(68093, 14)` | `float32` | 全为有限值；前 5 维每行恰有一个 one-hot 为 1 |
| `mode.npy` | `(68093,)` | `int64` | 全为有限值 |
| `speed.npy` | `(68093,)` | `float32` | 全为有限值 |
| `angle_bin.npy` | `(68093,)` | `int64` | 全为有限值，取值在 0 到 7 |
| `meta_modes.npy` | `(5,)` | `int32` | `[0, 1, 2, 17, 18]` |

mode 分布：

| mode | 行数 |
|---:|---:|
| 0 | 19,993 |
| 1 | 21,541 |
| 2（WALK） | 18,295 |
| 17 | 6,009 |
| 18 | 2,255 |

方向 bin 4 有 47,781 行，占 70.2%。因此数据明显偏向前向；方向条件模型的整体重构误差不能代表稀有方向也有相同质量。

历史记录把三个来源批次分别记为 20,838、32,675、14,580 行，总和为 68,093。由于当前数据没有 `source_id`，且原始批次目录缺失，这个分段当前只能视为历史记录，不能在本机重新证明。

---

## 5. 阶段 C：VAE 训练前如何处理这些数据

### 5.1 为什么需要 PCA 相位

VAE 不能只看到一个静态 token。走路 token 随步态周期变化，同一个动作风格在左脚支撑和右脚支撑时 token 不同。

E27 的做法是：

1. 只取 `mode == 2` 的 WALK token。
2. 计算它们的均值和协方差矩阵。
3. 取协方差最大的两个主成分 `V2`。
4. 将所有 token 投影到该二维平面。
5. 用 `atan2(PC2, PC1)` 把二维坐标转成相位 `phi`。

```text
phi_t = atan2(((token_t - mean) @ V2)_1,
              ((token_t - mean) @ V2)_0)
```

模型实际接收的是：

```text
[sin(phi), cos(phi)]
```

而不是直接接收角度，避免 `-pi` 与 `pi` 的数值断点。

源码：`train_token_vae_e27.py:92-110`。

### 5.2 10 帧因果窗口是什么

每行训练样本都由“当前 token 和前 9 个 token”构成：

```text
x[t] = [token[t-9], ..., token[t-1], token[t]]
y[t] = token[t]
```

窗口长度为 10，因此输入维度为：

```text
10 x 64 = 640
```

数据集开头没有历史 token 时，前面补零。

```text
t = 0: [0, 0, ..., 0, token[0]]
t = 1: [0, 0, ..., token[0], token[1]]
```

代码直接在拼接后的 68,093 行上生成窗口，不知道 exp1、exp2、exp3 或动作段的边界。因此某一批次末尾与下一批次开头之间也可能形成跨来源窗口。

源码：`train_token_vae_e27.py:73-78`。

### 5.3 E27：相位条件 VAE

E27 的网络关系：

```text
10 x 64 token window
    -> encoder
    -> mu(16), logvar(16)
    -> reparameterization
    -> z(16)

z(16) + [sin(phi), cos(phi)]
    -> decoder
    -> 当前 token(64)
```

训练设置：

| 项目 | 值 |
|---|---|
| 窗口长度 | 10 |
| token 维度 | 64 |
| latent 维度 | 16 |
| hidden 维度 | 256 |
| 优化器 | AdamW，`lr=1e-3`，`weight_decay=1e-5` |
| epoch | 30 |
| 损失 | `reconstruction MSE + 0.1 * KL` |
| train / validation | 随机打乱行后 90% / 10% |
| train batch | 512 |
| validation batch | 1024 |

每当 validation reconstruction MSE 改善时，保存 `vae.pt`。训练结束后，脚本计算所有 WALK 窗口的 `mu` 平均值并保存为 `z_walk.npy`，用于 RL 的潜变量 warm start。

源码：`train_token_vae_e27.py:111-172`。

### E27 的验证集为什么不能当作严格泛化结果

步骤顺序是：

```text
先构造全部重叠窗口
    -> 再随机按行切 90/10
```

所以验证集中的一个窗口通常和训练集中的相邻窗口共享 9/10 个 token。PCA 也在切分前用全体 WALK token 拟合。

这使 `val_mae` 适合作为“同一批轨迹上的插值重构误差”，不适合作为“未见 episode、未见命令段或未见采集批次的泛化误差”。

### 5.4 E35：显式方向和速度条件

E35 的 decoder 输入从：

```text
z + phase
```

扩展为：

```text
z + phase + speed_embedding + direction_embedding
```

其中：

```text
direction bin = angle_bin.npy 中的 0..7
speed bin     = PCA phase-rate 的三分位分箱，不是 speed.npy
```

速度 bin 的生成方法是：

1. 对相邻 token 的 PCA 相位做环绕差分。
2. 取差分绝对值作为 `rate`。
3. 只在 WALK 行上计算 `rate` 的 1/3 和 2/3 分位数。
4. 用两个阈值把所有行的 rate 分为 0、1、2 三类。

因此，E35 中“速度”实际更接近 **token 周期的相位推进快慢**，而非 `commands.csv` 的 `speed` 字段。

源码：`train_token_vae_e35.py:94-140`、`train_token_vae_e35.py:142-202`。

### 5.5 E39：方向和速度对抗解耦

E39 使用与 E35 相同的显式条件，但加入两个训练期分类头：

```text
dir_head:   z -> 8 个方向类别
speed_head: z -> 3 个 phase-rate 速度类别
```

每个 batch：

1. 先固定 VAE，通过 detached `z` 更新两个分类头 3 次。
2. 再更新 VAE，使它在重构 token 的同时让分类头更难从 `z` 猜出方向和速度。

损失为：

```text
loss = rec + 0.1 * KL - 3.0 * CE_dir - 3.0 * CE_speed
```

直观含义：方向和速度应由显式条件提供，不应暗藏在 `z` 中；`z` 只保留不能由这些条件表达的步态风格或残差。

分类头仅是训练工具，运行时只加载 VAE。

源码：`train_token_vae_e39.py:118-208`。

---

## 6. 数据条件与运行时条件的一个重要不一致

训练 E35/E39 时：

```text
speed bin = PCA phase-rate 的三分位数分箱
```

运行 Isaac 环境时：

```text
speed bin = cmd_vx 在 [0, vx_max] 的等宽 bucketize
direction bin = atan2(cmd_vy, cmd_vx)
```

因此，训练期 `vb=2` 的含义是“相位推进率最高的一组 token”，而运行期 `vb=2` 的含义是“命令前向速度落入最高的等宽区间”。二者没有由当前代码证明为同一件事。

这不是数组损坏，而是条件变量的语义不一致。它会使速度条件 VAE 的运行时行为难以解释：模型看到的 bin 标签在训练和推理阶段代表不同划分规则。

相关代码：

- 训练：`train_token_vae_e35.py:124-129`、`train_token_vae_e39.py:129-133`
- 推理：`isaac/apt_flat_env.py:504-521`

建议把方向/速度分箱函数写成一个共享模块，训练、数据检查与推理全部调用它，并将阈值保存进版本化 manifest。

---

## 7. reward 参数与数据集的边界

reward 不参与 CSV 采集，也不参与 `exp_all3` 的构建。它是 Isaac RL 环境中用来更新 policy 的评分函数。

默认 reward 由以下项组成：

```text
reward =
    1.0 * track_xy
  + yaw_scale * track_yaw
  + 0.1 * upright
  + 0.5 * height
  + stillness
  - termination_penalty * terminated
  + 可选项
```

| 项 | 默认参数 | 作用 |
|---|---|---|
| `track_xy` | 权重 `1.0`，`vel_sigma2=0.25` | 奖励 body-frame 前向速度贴近 `cmd_vx` |
| `track_yaw` | `yaw_scale=0.5`，`yaw_sigma2=0.25` | 奖励 yaw 角速度贴近命令 |
| `upright` | 权重 `0.1` | 奖励投影重力接近竖直 |
| `height` | 权重 `0.5` | 奖励根部高度接近 0.76 m |
| `stillness` | `stillness_vx_scale=0.05` | 惩罚过大线速度和 roll/pitch 角速度 |
| `termination` | `fall_height=0.2`，`termination_penalty=-10` | 根部高度低于 0.2 m 或非有限值时终止 |
| `heading_scale` | 默认 0 | 可选：奖励速度方向与命令方向一致 |
| `progress_scale` | 默认 0 | 可选：奖励前向速度 |
| `anti_stop_scale` | 默认 0 | 可选：惩罚低于阈值的前向速度 |
| `yaw_rate_penalty` | 默认 0 | 可选：直接惩罚过大 yaw rate |

对数据审计最重要的解释是：**改变 reward 不会修复原始命令/token 对齐，也不会改变已经写入的 `exp_all3`。** reward 只会改变后续 policy 在冻结 VAE/SONIC 先验上的选择方式。

源码：`isaac/apt_flat_env.py:133-176`、`isaac/apt_flat_env.py:197-232`、`isaac/apt_flat_env.py:756-812`。

---

## 8. 审计发现：问题、后果、应如何试错

| 优先级 | 问题 | 当前后果 | 下一步应验证或修复什么 |
|---|---|---|---|
| P1 | 命令用行数比例映射，没有时间戳 | 启动延迟、停机空窗、丢帧会导致标签错位 | 每行写单调时间戳、帧号与 session ID；以时间戳 join |
| P1 | 缺失原始批次时静默跳过，仍覆盖输出 | 可生成不完整 `exp_all3` | 将缺失批次、列宽错误、token 校验失败改为直接报错 |
| P1 | token 检查不是 fail-fast | 错列或错位 token 可能进入成品 | 断言行数、列数、token 差和偏移范围 |
| P1 | 重叠窗口随机行切分 | `val_mae` 明显偏乐观 | 按完整 episode/source batch 切 train/val/test；PCA 仅由 train 拟合 |
| P1 | 训练/推理速度 bin 语义不一致 | 速度条件不能直接解释为命令速度学习 | 共享同一个分箱函数与阈值 metadata |
| P2 | `angle_bin` 和保存后的 `cmd` 精度不同 | 少量 bin 无法精确复算 | 归一化方向，对边界做 epsilon snap，保存原始角度 |
| P2 | 没有 source/episode 边界 | 窗口和 phase diff 可跨批次、跨动作段 | 保存 source_id、episode_id、phase_id；边界处重新开始窗口 |
| P2 | `z_walk.npy` 在训练最后一轮直接导出 | 可能不是最佳 `vae.pt` 对应的 latent | reload 最优 checkpoint 后再导出 `z_walk.npy` |
| P2 | 采集依赖未版本化 `/tmp/g1deploy_exp3.sh` | 采集配置无法审计 | 将启动脚本、模型版本、配置和日志路径纳入版本控制/manifest |

### 推荐的试错顺序

不要先调 VAE 网络层数，也不要先调 reward。应按以下顺序排除更基础的问题：

1. **先证明记录正确。** 比较 `policy_input` 与 `token_state`，检查列数、值域、token 格点、时间戳连续性。
2. **再证明标签正确。** 用同一时间轴将 commands 对齐到 token，抽取动作切换前后几秒人工检查。
3. **再证明样本划分正确。** 确保 VAE 验证集没有共享同一 episode 的重叠窗口。
4. **再比较模型。** 在固定、可复现的数据版本上比较 E27/E35/E39 的重构和闭环行为。
5. **最后调 RL reward。** 这时 reward 的变化才可以解释为控制偏好变化，而不是数据错位的偶然补偿。

---

## 9. 下一版数据集应如何生成

建议的最小可复现流程：

```text
采集前
    生成 run_id
    记录 git commit、部署脚本 hash、ONNX 模型 hash、控制频率和配置

采集时
    policy_input / commands / token_state 每行都写：timestamp、frame_id、run_id
    键盘事件、模式切换、跌倒、重置写入同一时钟域

采集后
    不覆盖原始 CSV
    写 manifest.json：文件名、行数、hash、采集起止时间、事件摘要
    运行 fail-fast 校验：列数、有限值、token diff、采样间隔、命令覆盖率

构建时
    按 timestamp join commands 和 token
    写 source_id、episode_id、原始行号
    禁止窗口、差分、训练/验证切分跨越 episode 边界
    将 mode/方向/速度分箱函数和阈值与数据一同版本化

训练时
    仅用 train set 拟合 PCA 和 bin 阈值
    按 episode 或来源批次隔离 validation/test
    reload 最优 checkpoint 后导出 z_walk
```

## 10. 最终结论

`exp_all3` 是一个内部自洽、可继续用于现有 E27/E35/E39 实验的 68,093 帧 token 数据资产。它的字段构造、维度和后续 VAE 前处理可以由当前源码完整解释。

但它当前不是严格可追溯、可从头复建的数据集：原始 CSV、统一时间戳、source/episode ID、部署脚本版本和构建 manifest 都缺失。因此，基于它得到的闭环实验结果可以作为历史实验资产；涉及数据覆盖、条件语义或跨轨迹泛化的结论，应在新一版带完整 provenance 的数据集上重新验证。
