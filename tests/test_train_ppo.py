import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from fortress_rl import train_ppo
from fortress_rl import train_dqn_sb3
from fortress_rl.env import GoalBounds, SpawnPoint


class PPOTrainingConfigTests(unittest.TestCase):
    def test_defaults_match_the_documented_pilot_fortress(self):
        args = train_ppo.build_parser().parse_args([])

        self.assertEqual(args.world_seed, "12345")
        self.assertEqual(args.spawn, train_ppo.PILOT_SPAWN)
        self.assertEqual(args.goal_min, train_ppo.PILOT_GOAL_MIN)
        self.assertEqual(args.goal_max, train_ppo.PILOT_GOAL_MAX)
        self.assertEqual(args.total_timesteps, 100_000)
        self.assertEqual(args.gae_lambda, 0.95)
        self.assertEqual(args.image_size, (640, 360))
        self.assertFalse(args.hide_window)
        self.assertEqual(args.preview, train_ppo.os.name == "nt")

    def test_dqn_pilot_defaults_use_same_fortress_with_shorter_budget(self):
        args = train_dqn_sb3.build_parser().parse_args([])

        self.assertEqual(args.world_seed, train_ppo.PILOT_WORLD_SEED)
        self.assertEqual(args.total_timesteps, 20_000)
        self.assertEqual(args.goal_min, train_ppo.PILOT_GOAL_MIN)
        self.assertEqual(args.goal_max, train_ppo.PILOT_GOAL_MAX)
        self.assertFalse(args.reward_shaping)

    def test_goal_distance_is_zero_inside_target_bounds(self):
        distance = train_dqn_sb3._distance_to_goal(
            (72.0, 60.0, 674.0),
            (66.0, 55.0, 668.0),
            (78.0, 68.0, 680.0),
        )

        self.assertEqual(distance, 0.0)

    def test_environment_factory_uses_the_requested_minecraft_setup(self):
        args = train_ppo.build_parser().parse_args(
            [
                "--world-seed",
                "67890",
                "--spawn",
                "1",
                "70",
                "2",
                "--yaw",
                "90",
                "--goal-min",
                "3",
                "55",
                "4",
                "--goal-max",
                "5",
                "68",
                "6",
                "--port",
                "8024",
            ]
        )
        marker = object()
        with patch.object(
            train_ppo,
            "make_craftground_env",
            return_value=marker,
        ) as factory:
            result = train_ppo.create_environment(args)

        self.assertIs(result, marker)
        factory.assert_called_once_with(
            world_seed="67890",
            goal_bounds=GoalBounds((3.0, 55.0, 4.0), (5.0, 68.0, 6.0)),
            spawn_points=[SpawnPoint(1.0, 70.0, 2.0, yaw=90.0)],
            port=8024,
            image_size=(640, 360),
            action_repeat=4,
            frame_stack=4,
            max_episode_steps=500,
            verbose_gradle=True,
        )

    @patch("fortress_rl.train_ppo.subprocess.run")
    @patch("fortress_rl.train_ppo.shutil.which")
    def test_toolchain_check_accepts_jdk21_and_cmake_44(self, which, run):
        which.side_effect = lambda name: {
            "java": "java.exe",
            "javac": "javac.exe",
            "cmake": "cmake.exe",
        }.get(name)
        run.side_effect = [
            Mock(
                returncode=0,
                stdout="",
                stderr='openjdk version "21.0.10"',
            ),
            Mock(
                returncode=0,
                stdout="cmake version 4.4.5",
                stderr="",
            ),
        ]

        train_ppo.validate_craftground_toolchain()

    def test_craftground_visible_window_patch_is_idempotent(self):
        original = (
            "        if (!hasMinimizedWindow) {\n"
            "            GLFW.glfwIconifyWindow(window.handle)\n"
            "            hasMinimizedWindow = true\n"
            "        }"
        )
        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "EnvironmentInitializer.kt"
            source_path.write_text(original, encoding="utf-8")

            train_ppo.patch_craftground_window_source(source_path)
            patched_source = source_path.read_text(encoding="utf-8")
            train_ppo.patch_craftground_window_source(source_path)

        self.assertIn("CRAFTGROUND_KEEP_WINDOW_VISIBLE", patched_source)
        self.assertNotIn(original, patched_source)


if __name__ == "__main__":
    unittest.main()
