"""TO42 learned-regime-selection gate——纯 torch 状态机（无 isaaclab 依赖）。

角色（SCRIPT_MAP 登记）：**MODULE**（被 apt_flat_env / to42_selftest 消费）。
把论文 gait-gate 语义（2 Hz 决策门 + 0.5 s 锁存 + 切换布尔）物化到冻结解码器的
{vb0, vb1} 二元 regime 选择上（TO42_PLAN §3 逐字）。独立成模块的原因：G0 门
要求在无 Isaac 环境可单测（negative cases 先行），训练/评测用的状态机与自检
覆盖的状态机必须是同一份代码。

两臂语义（TO42_PLAN §3/§4，预注册）：
- ``lsel``：决策边界（每 hold_steps 控制步，50 Hz 下 25 步 = 0.5 s）处采纳策略
  Bernoulli 位作为提案，与当前不同才切换；两次边界之间状态锁存（= 论文
  "选中后锁定 0.5 s"）。gate 布尔只在真切换步为 True（与 env._update_gate 的
  _gate_tick 语义一致：只报 actual decision，不报空转时钟）。
- ``fbkt``：每步 state = clamp(bucketize(cmd_v), 0, 1)——eval 网格（v ≤ 0.325
  < 0.533）上与冻结 bucketize 逐位一致；gate 恒 False；策略位被忽略（配对
  基线臂：obs/action 结构相同，唯一差异 = 选择由冻结函数还是策略产生）。
- reset：state = clamp(bucketize(新 cmd), 0, 1)（自然 bin 中性起步），count
  归零（reset 后首个决策边界在第 hold_steps 步）。
"""

from __future__ import annotations

import torch

N_SEL = 2  # Rung 1 selector 值域 = {vb0, vb1}（vb2 不进 Rung 1，保 TO41 可比）


def vae_speed_edges(vx_max: float, n_bins: int = 3) -> torch.Tensor:
    """冻结 bucketize 边界（= env/decft 的 linspace(0, vx_max, n+1)[1:-1]）。

    vx_max=0.8, n=3 → [0.2667, 0.5333]：TO41 记录的 [0.267, 0.533] 边界。
    """
    return torch.linspace(0.0, vx_max, n_bins + 1)[1:-1]


def natural_vb(cmd_v, vx_max: float = 0.8, n_bins: int = 3):
    """冻结自然条件分配（TO41 的 natural bucketize），不钳制——供核对/记录。

    接受 torch tensor 或 numpy 数组（v3 教训：checker 传 np.array 时
    `cmd_v.device` 直接 AttributeError，杀死了整个 checker 阶段）。"""
    cmd_t = torch.as_tensor(cmd_v, dtype=torch.float32)
    edges = vae_speed_edges(vx_max, n_bins).to(cmd_t.device)
    return torch.bucketize(cmd_t, edges).clamp(0, n_bins - 1)


class To42Gate:
    """per-env 二元 regime 选择状态机（lsel / fbkt 两模式共用一份代码）。"""

    def __init__(
        self,
        num_envs: int,
        device,
        hold_steps: int = 25,
        mode: str = "lsel",
        vx_max: float = 0.8,
        n_bins: int = 3,
        n_sel: int = N_SEL,
    ):
        if mode not in ("lsel", "fbkt"):
            raise ValueError(f"bad to42 mode: {mode!r}")
        if hold_steps <= 0:
            raise ValueError("hold_steps must be positive")
        self.mode = mode
        self.hold_steps = int(hold_steps)
        self.n_sel = int(n_sel)
        self.device = device
        self.edges = vae_speed_edges(vx_max, n_bins).to(device)
        self.state = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.gate = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids: torch.Tensor, cmd_v: torch.Tensor) -> None:
        nat = torch.bucketize(cmd_v, self.edges).clamp(0, self.n_sel - 1)
        self.state[env_ids] = nat
        self.count[env_ids] = 0
        self.gate[env_ids] = False

    def step(self, cmd_v: torch.Tensor, sel_bit: torch.Tensor | None = None):
        """推进一步：返回 (state, gate)。fbkt 忽略 sel_bit；lsel 边界处采纳。"""
        self.count += 1
        if self.mode == "fbkt":
            self.state = torch.bucketize(cmd_v, self.edges).clamp(0, self.n_sel - 1)
            self.gate[:] = False
            return self.state, self.gate
        boundary = (self.count % self.hold_steps) == 0
        if sel_bit is None:
            raise ValueError("lsel mode requires the policy sel bit")
        proposed = sel_bit.clamp(0, self.n_sel - 1)
        changed = boundary & (proposed != self.state)
        self.state = torch.where(changed, proposed, self.state)
        self.gate = changed
        return self.state, self.gate
