import unittest
from types import SimpleNamespace

from fortress_rl.baseline import run_random_baseline


class FakeDiscreteEnv:
    ACTION_NAMES = ("a", "b", "c")

    def __init__(self, *, fail_on_step=False, never_done=False):
        self.action_space = SimpleNamespace(n=len(self.ACTION_NAMES))
        self.fail_on_step = fail_on_step
        self.never_done = never_done
        self.actions = []
        self.reset_seeds = []
        self.close_count = 0
        self.episode_steps = 0

    def reset(self, *, seed=None):
        self.reset_seeds.append(seed)
        self.episode_steps = 0
        return None, {}

    def step(self, action):
        if self.fail_on_step:
            raise RuntimeError("simulated failure")
        self.actions.append(action)
        self.episode_steps += 1
        done = not self.never_done and self.episode_steps == 2
        return None, 0.0, False, done, {"is_success": False, "failure_reason": None}

    def close(self):
        self.close_count += 1


class RandomBaselineTests(unittest.TestCase):
    def test_seed_reproduces_actions_and_metrics(self):
        first = FakeDiscreteEnv()
        second = FakeDiscreteEnv()

        first_result = run_random_baseline(
            first, episodes=2, max_episode_steps=4, seed=17
        )
        second_result = run_random_baseline(
            second, episodes=2, max_episode_steps=4, seed=17
        )

        self.assertEqual(first.actions, second.actions)
        self.assertEqual(first.reset_seeds, [17, 18])
        self.assertEqual(first.reset_seeds, second.reset_seeds)
        self.assertEqual([episode.policy_steps for episode in first_result.episodes], [2, 2])
        self.assertEqual(first_result.summary["total_policy_steps"], 4)
        self.assertEqual(sum(first_result.action_counts.values()), 4)
        self.assertEqual(first.close_count, 1)
        self.assertEqual(second.close_count, 1)

    def test_runner_step_limit_truncates_episode(self):
        env = FakeDiscreteEnv(never_done=True)
        result = run_random_baseline(
            env, episodes=1, max_episode_steps=3, seed=0
        )

        self.assertEqual(result.episodes[0].policy_steps, 3)
        self.assertTrue(result.episodes[0].truncated)
        self.assertEqual(result.episodes[0].failure_reason, "runner_step_limit")
        self.assertEqual(env.close_count, 1)

    def test_environment_is_closed_if_step_raises(self):
        env = FakeDiscreteEnv(fail_on_step=True)
        with self.assertRaisesRegex(RuntimeError, "simulated failure"):
            run_random_baseline(env, episodes=1, max_episode_steps=2, seed=0)
        self.assertEqual(env.close_count, 1)

    def test_rejects_non_positive_episode_count(self):
        env = FakeDiscreteEnv()
        with self.assertRaisesRegex(ValueError, "episodes must be >= 1"):
            run_random_baseline(env, episodes=0, max_episode_steps=2, seed=0)
        self.assertEqual(env.close_count, 1)


if __name__ == "__main__":
    unittest.main()
