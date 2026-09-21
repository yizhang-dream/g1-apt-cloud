"""Terrain configuration helpers for the APT Isaac env."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.terrains import (
    HfDiscreteObstaclesTerrainCfg,
    HfPyramidStairsTerrainCfg,
    HfRandomUniformTerrainCfg,
    HfSteppingStonesTerrainCfg,
    MeshRandomGridTerrainCfg,
    TerrainGeneratorCfg,
    TerrainImporterCfg,
)

# D066 C2 门（§5w）climb_box 地形高度档：对齐 D060 语料 terrain_desc
# climbing_box{height:0.5}（build/d060_windows.py:109，源 2 爬障 0.5 m 箱）。
# 网格高度取 (0.4, 0.5) = 0.5 m 量级（单值 0.5 是语料侧箱高，地形侧留一档变化
# 以免 187 点 patch 退化为常量；量级对齐是单变量，见下方 climb_box 分支注释）。
CLIMB_BOX_HEIGHT_RANGE = (0.4, 0.5)


def make_terrain_importer_cfg(
    terrain_type: str = "plane",
    noise: float = 0.04,
    seed: int | None = 0,
) -> TerrainImporterCfg:
    """Return a TerrainImporterCfg for flat or rough (random heightfield) ground.

    ``seed`` fixes the random heightfield so that aux / noaux comparisons and
    repeated evaluations share the exact same terrain; pass None for random.
    """
    if terrain_type == "plane":
        return TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="average",
                restitution_combine_mode="average",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            debug_vis=False,
        )
    if terrain_type == "rough":
        return TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=seed,
                size=(8.0, 8.0),
                border_width=20.0,
                num_rows=10,
                num_cols=20,
                horizontal_scale=0.1,
                vertical_scale=0.005,
                slope_threshold=0.75,
                use_cache=False,
                sub_terrains={
                    "random_rough": HfRandomUniformTerrainCfg(
                        proportion=1.0,
                        noise_range=(0.0, noise),
                        noise_step=0.01,
                        border_width=0.25,
                    ),
                },
            ),
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="average",
                restitution_combine_mode="average",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            debug_vis=False,
        )
    if terrain_type == "rough_paper":
        # Gate-0 "paper-shaped" rough (PAPER_TERRAIN_SPEC.md): symmetric
        # U(-noise, +noise) cells (has pits, mean ~= 0) sampled at 0.2 m and
        # spline-upsampled to the 0.1 m physics grid, matching APT-RL's
        # "min U(-0.06,-0.02) / max U(0.02,0.06), downsampled scale 0.2".
        # Per-instance min/max sampling is not expressible with the stock cfg
        # (cell-uniform +-noise is the slightly harder conservative reading).
        return TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=seed,
                size=(8.0, 8.0),
                border_width=20.0,
                num_rows=10,
                num_cols=20,
                horizontal_scale=0.1,
                vertical_scale=0.005,
                slope_threshold=0.75,
                use_cache=False,
                sub_terrains={
                    "random_rough_paper": HfRandomUniformTerrainCfg(
                        proportion=1.0,
                        noise_range=(-noise, noise),
                        noise_step=0.01,
                        downsampled_scale=0.2,
                        border_width=0.25,
                    ),
                },
            ),
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="average",
                restitution_combine_mode="average",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            debug_vis=False,
        )
    if terrain_type == "rough_sym":
        # Gate-0 control cell: symmetric U(-noise, +noise) like rough_paper but
        # at the default 0.1 m cell size -- isolates "pits" from "0.2 m feature
        # size" in the paper-shape vs our-shape difficulty gap.
        return TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=seed,
                size=(8.0, 8.0),
                border_width=20.0,
                num_rows=10,
                num_cols=20,
                horizontal_scale=0.1,
                vertical_scale=0.005,
                slope_threshold=0.75,
                use_cache=False,
                sub_terrains={
                    "random_rough_sym": HfRandomUniformTerrainCfg(
                        proportion=1.0,
                        noise_range=(-noise, noise),
                        noise_step=0.01,
                        border_width=0.25,
                    ),
                },
            ),
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="average",
                restitution_combine_mode="average",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            debug_vis=False,
        )
    if terrain_type in ("stairs", "stairs_hi", "stones", "discrete"):
        if terrain_type == "stairs":
            sub_terrain = {
                "stairs": HfPyramidStairsTerrainCfg(
                    proportion=1.0,
                    step_height_range=(0.04, 0.08),
                    step_width=0.35,
                    platform_width=1.0,
                    border_width=0.25,
                ),
            }
        elif terrain_type == "stairs_hi":
            sub_terrain = {
                "stairs_hi": HfPyramidStairsTerrainCfg(
                    proportion=1.0,
                    step_height_range=(0.08, 0.14),
                    step_width=0.35,
                    platform_width=1.0,
                    border_width=0.25,
                ),
            }
        elif terrain_type == "stones":
            sub_terrain = {
                "stones": HfSteppingStonesTerrainCfg(
                    proportion=1.0,
                    stone_height_max=0.06,
                    stone_width_range=(0.25, 0.4),
                    stone_distance_range=(0.3, 0.5),
                    holes_depth=-0.5,
                    platform_width=1.0,
                    border_width=0.25,
                ),
            }
        else:
            sub_terrain = {
                "discrete": HfDiscreteObstaclesTerrainCfg(
                    proportion=1.0,
                    obstacle_height_range=(0.05, 0.10),
                    obstacle_width_range=(0.15, 0.35),
                    num_obstacles=10,
                    platform_width=1.0,
                    border_width=0.25,
                ),
            }
        return TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=seed,
                size=(8.0, 8.0),
                border_width=20.0,
                num_rows=10,
                num_cols=20,
                horizontal_scale=0.1,
                vertical_scale=0.005,
                slope_threshold=0.75,
                use_cache=False,
                sub_terrains=sub_terrain,
            ),
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="average",
                restitution_combine_mode="average",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            debug_vis=False,
        )
    if terrain_type == "climb_box":
        # D066 C2 门（§5w）四族之一：语料侧 climb_box=0.5 m 箱
        # （build/d060_windows.py:109 climbing_box{height:0.5}）在 Isaac 栈的
        # 地形对应物（本文件此前无该族，工厂亦不认——D066 补件）。
        # 配方 = 官方 MeshRandomGridTerrainCfg boxes 原语（tmp/rough_terrain_v210.py:37-39
        # 官方 G1 rough 的 boxes 写法逐字：grid_width=0.45、platform_width=2.0），
        # 唯一变量 = grid_height_range 抬到 CLIMB_BOX_HEIGHT_RANGE（0.5 m 量级，
        # 对齐语料 height=0.5）；其余 generator 参数（8x8/边 20/10x20/0.1/0.005/0.75/
        # 无缓存 + average 摩擦材质）与既有 Hf 族同源，保证族间可比。
        return TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=seed,
                size=(8.0, 8.0),
                border_width=20.0,
                num_rows=10,
                num_cols=20,
                horizontal_scale=0.1,
                vertical_scale=0.005,
                slope_threshold=0.75,
                use_cache=False,
                sub_terrains={
                    "climb_box": MeshRandomGridTerrainCfg(
                        proportion=1.0,
                        grid_width=0.45,
                        grid_height_range=CLIMB_BOX_HEIGHT_RANGE,
                        platform_width=2.0,
                    ),
                },
            ),
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="average",
                restitution_combine_mode="average",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            debug_vis=False,
        )
    raise ValueError(f"unknown terrain_type {terrain_type}")
