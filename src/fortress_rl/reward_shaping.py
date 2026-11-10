"""Potential-based reward shaping using hidden goal geometry."""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym

from .env import GoalBounds


class PotentialDistanceReward(gym.Wrapper):
    """Add potential-based progress reward without changing observations.

    The goal region and player position are used only to calculate reward.
    The policy continues to receive exactly the wrapped environment's original
    observation.
    """

    def __init__(
        self,
        env: gym.Env,
        goal_bounds: GoalBounds,
        *,
        gamma: float = 0.99,
        scale: float = 0.05,
    ) -> None:
        super().__init__(env)
        if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be finite and between 0 and 1")
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("scale must be finite and greater than 0")
        self.goal_bounds = goal_bounds
        self.gamma = float(gamma)
        self.scale = float(scale)
        self._potential: float | None = None

    def _distance(self, position: tuple[float, float, float]) -> float:
        deltas = tuple(
            max(low - value, 0.0, value - high)
            for value, low, high in zip(
                position,
                self.goal_bounds.minimum,
                self.goal_bounds.maximum,
            )
        )
        return math.sqrt(sum(delta * delta for delta in deltas))

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        distance = self._distance(tuple(info["position"]))
        self._potential = -distance
        shaped_info = dict(info)
        shaped_info["distance_to_goal"] = distance
        return observation, shaped_info

    def step(self, action: Any):
        if self._potential is None:
            raise RuntimeError("Call reset() before step()")

        observation, terminal_reward, terminated, truncated, info = self.env.step(action)
        distance = self._distance(tuple(info["position"]))
        next_potential = -distance
        if terminated:
            # All terminal states are absorbing for the potential calculation.
            next_potential = 0.0

        shaping_reward = self.scale * (
            self.gamma * next_potential - self._potential
        )
        self._potential = next_potential
        shaped_info = dict(info)
        shaped_info["distance_to_goal"] = distance
        shaped_info["reward_terminal"] = float(terminal_reward)
        shaped_info["reward_shaping"] = float(shaping_reward)
        return (
            observation,
            float(terminal_reward + shaping_reward),
            terminated,
            truncated,
            shaped_info,
        )
