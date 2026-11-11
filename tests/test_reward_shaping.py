import unittest

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from fortress_rl.env import GoalBounds
from fortress_rl.reward_shaping import PotentialDistanceReward


class PositionEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )
        self.position = (0.0, 64.0, 0.0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position = (0.0, 64.0, 0.0)
        return np.zeros(1, dtype=np.float32), {"position": self.position}

    def step(self, action):
        self.position = (2.0, 64.0, 0.0)
        return (
            np.zeros(1, dtype=np.float32),
            0.0,
            False,
            False,
            {"position": self.position, "is_success": False},
        )


class PotentialDistanceRewardTests(unittest.TestCase):
    def test_progress_reward_uses_distance_but_leaves_observation_unchanged(self):
        env = PotentialDistanceReward(
            PositionEnv(), GoalBounds((10, 64, 0), (10, 64, 0)), gamma=0.9, scale=0.5
        )
        observation, _ = env.reset()
        next_observation, reward, terminated, truncated, info = env.step(0)

        # Distance changes 10 -> 8, so F = scale * (gamma * -8 - (-10)) = 1.4.
        self.assertEqual(observation.shape, next_observation.shape)
        self.assertEqual(reward, 1.4)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["distance_to_goal"], 8.0)
        self.assertEqual(info["reward_terminal"], 0.0)
        self.assertAlmostEqual(info["reward_shaping"], 1.4)

    def test_requires_reset_before_step(self):
        env = PotentialDistanceReward(
            PositionEnv(), GoalBounds((10, 64, 0), (10, 64, 0))
        )
        with self.assertRaisesRegex(RuntimeError, "reset"):
            env.step(0)


if __name__ == "__main__":
    unittest.main()
