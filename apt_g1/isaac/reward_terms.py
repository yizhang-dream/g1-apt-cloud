"""D048j 进度奖励封顶项（纯 torch 函数模块，无 isaaclab 依赖）。

背景（D048j 预注册设计）：``apt_flat_env._get_rewards`` 的 progress 项原为
``progress_scale * torch.clamp(base_lin_vel[:, 0], 0.0, 1.0)`` —— 封顶上界
1.0 与当前命令无关，与 track_xy = exp(-(vx-cmd)^2/0.25)（权重 1.0）联合后，
cmd=0.4 时静态最优 vx ≈ 0.534（结构性超速），且 vx=0.62 处每步反高 0.033
（相对恰跟踪 cmd 的 f 值；stillness_vx_scale=0.05 计入）——超速的奖励偏置
来源。修正 = 旗标开启时封顶上界改为当前命令 ``cmd.clamp(min=0)``：静态最优
恰为 cmd（0.534 → cmd；0.62 处 -0.033 → +0.187 每步，偏置反转；同上
stillness_vx_scale=0.05 计入）。

默认关（flag off）= 与旧内联公式逐位一致（REW_CONTRACT_VER 2）；开启 =
REW_CONTRACT_VER 3（实例属性，见 AptFlatG1Env.__init__）。
"""

from __future__ import annotations

import torch

__all__ = ["progress_bonus"]


def progress_bonus(
    vx: torch.Tensor, cmd: torch.Tensor | float, cap_cmd: bool
) -> torch.Tensor:
    """Forward-progress bonus 项。

    Args:
        vx: 机体系前向速度（任意 shape）。
        cmd: 当前 vx 命令。张量时与 ``vx`` 逐元素（可广播）；标量自动广播。
        cap_cmd: False → ``clamp(vx, 0, 1)``（历史行为，逐位一致）；
            True → ``min(clamp(vx, 0, 1), clamp(cmd, min=0))``（D048j：
            封顶上界 = 当前命令，静态最优恰为 cmd）。
    """
    capped = torch.clamp(vx, 0.0, 1.0)
    if not cap_cmd:
        return capped
    if not torch.is_tensor(cmd):
        cmd = torch.tensor(cmd, dtype=capped.dtype, device=capped.device)
    return torch.minimum(capped, cmd.clamp(min=0.0))
