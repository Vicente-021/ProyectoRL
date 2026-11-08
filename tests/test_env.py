import unittest
from types import SimpleNamespace

import numpy as np

from fortress_rl.env import GoalBounds, NetherFortressEnv, SpawnPoint


class FakeCraftGround:
    """Small deterministic stand-in for unit-testing the wrapper contract."""

    def __init__(self):
        self._alive = False
        self.x, self.y, self.z = 0.5, 64.0, 0.5
        self.yaw = 0.0
        self.health = 20.0
        self.is_dead = False
        self.is_on_ground = True
        self.is_in_lava = False
        self.commands = []
        self.pending_commands = []

    @property
    def is_alive(self):
        return self._alive

    def _observation(self):
        blocks = [
            SimpleNamespace(
                x=x,
                y=y,
                z=z,
                translation_key=(
                    "block.minecraft.nether_brick_fence"
                    if (x, y, z) == (0, 64, 0)
                    else "block.minecraft.air"
                ),
            )
            for x in range(-1, 2)
            for y in range(63, 66)
            for z in range(-1, 2)
        ]
        full = SimpleNamespace(
            x=self.x,
            y=self.y,
            z=self.z,
            yaw=self.yaw,
            health=self.health,
            is_dead=self.is_dead,
            is_on_ground=self.is_on_ground,
            is_in_lava=self.is_in_lava,
            surrounding_blocks=blocks,
        )
        return {"full": full, "pov": np.zeros((64, 64, 3), dtype=np.uint8)}

    def reset(self, *, seed=None, options=None):
        self._alive = True
        self.x, self.y, self.z = 0.5, 64.0, 0.5
        self.yaw = 0.0
        return self._observation(), {}

    def add_commands(self, commands):
        self.commands.extend(commands)
        self.pending_commands.extend(commands)

    def step(self, action):
        for command in self.pending_commands:
            if " tp @p " in command:
                _, x, y, z, yaw, _pitch = command.split()[-6:]
                self.x, self.y, self.z = float(x), float(y), float(z)
                self.yaw = float(yaw)
        self.pending_commands.clear()
        if action["forward"]:
            self.z += 0.25
        if action["back"]:
            self.z -= 0.25
        if action["left"]:
            self.x += 0.25
        if action["right"]:
            self.x -= 0.25
        if action["camera"][1]:
            self.yaw += float(action["camera"][1])
        return self._observation(), 0.0, False, False, {}

    def close(self):
        self._alive = False


class NetherFortressEnvTests(unittest.TestCase):
    def make_env(self, **kwargs):
        base = FakeCraftGround()
        env = NetherFortressEnv(
            base,
            GoalBounds((0.0, 63.0, 0.75), (1.0, 65.0, 1.0)),
            **kwargs,
        )
        return env, base

    def test_reset_returns_declared_stacked_observation(self):
        env, _ = self.make_env(frame_stack=3)
        observation, info = env.reset(seed=7)
        self.assertEqual(observation.shape, env.observation_space.shape)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertFalse(info["is_success"])

    def test_forward_macro_can_reach_terminal_goal(self):
        env, _ = self.make_env(action_repeat=2)
        env.reset(seed=7)
        observation, reward, terminated, truncated, info = env.step(
            NetherFortressEnv.ACTION_FORWARD
        )
        self.assertEqual(reward, 1.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["is_success"])
        self.assertNotIn("full", info)
        self.assertNotIn("pov", info)
        self.assertTrue(env.observation_space.contains(observation))

    def test_timeout_is_truncated_with_zero_reward(self):
        env, _ = self.make_env(action_repeat=1, max_episode_steps=1)
        env.goal_bounds = GoalBounds((100.0, 100.0, 100.0), (101.0, 101.0, 101.0))
        env.reset(seed=7)
        _, reward, terminated, truncated, _ = env.step(
            NetherFortressEnv.ACTION_FORWARD
        )
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_spawn_command_targets_the_selected_dimension_and_heading(self):
        spawn = SpawnPoint(10, 70, -5, yaw=90, pitch=0)
        self.assertEqual(
            spawn.teleport_command(),
            "execute in minecraft:the_nether run tp @p 10 70 -5 90 0",
        )

    def test_first_reset_waits_until_spawn_teleport_is_observed(self):
        spawn = SpawnPoint(0.5, 64.0, 0.5, yaw=90, pitch=0)
        env, base = self.make_env(spawn_points=[spawn])
        _, info = env.reset(seed=7)
        self.assertEqual(info["position"], (spawn.x, spawn.y, spawn.z))
        self.assertEqual((base.x, base.y, base.z), (spawn.x, spawn.y, spawn.z))


if __name__ == "__main__":
    unittest.main()
