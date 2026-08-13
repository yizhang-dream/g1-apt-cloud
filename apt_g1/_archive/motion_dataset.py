"""Flat-ground G1 motion dataset helpers.

Stage 1 of the APT-RL process is a physically grounded state-action dataset.
When SONIC replaces the representation-learning stage, this module is used to
tokenize reference motions and to produce short state-action episodes for
inspection or auxiliary training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MotionEpisode:
    states: np.ndarray
    actions: np.ndarray
    commands: np.ndarray
    skill_id: int


class MotionDataset:
    """A directory of .npz motion episodes."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.episodes: list[MotionEpisode] = []

    def load(self) -> "MotionDataset":
        for path in sorted(self.root.glob("*.npz")):
            data = np.load(path)
            self.episodes.append(
                MotionEpisode(
                    states=data["states"],
                    actions=data["actions"],
                    commands=data.get("commands", np.zeros((len(data["states"]), 3))),
                    skill_id=int(data.get("skill_id", 0)),
                )
            )
        return self

    def iter_windows(self, window: int = 3):
        """Yield (state_window, action) slices for TVAE-style training."""
        for episode in self.episodes:
            for t in range(len(episode.states) - window):
                yield (
                    episode.states[t : t + window],
                    episode.actions[t + window - 1],
                )


def collect_episode_from_sim(env, reference_actions, skill_id: int):
    """Record one episode by replaying reference actions in a simulator.

    This is a placeholder for BONES-SEED replay or VR teleoperation data.
    The environment must return observations that contain joint states.
    """
    obs = env.reset()
    states = []
    actions = []
    commands = []
    for action in reference_actions:
        next_obs, _, terminated, _ = env.step(action)
        states.append(obs)
        actions.append(action)
        commands.append(np.zeros(3, dtype=np.float32))
        obs = next_obs
        if terminated:
            break
    return MotionEpisode(
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        commands=np.asarray(commands, dtype=np.float32),
        skill_id=skill_id,
    )
