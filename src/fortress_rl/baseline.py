"""Reproducible uniform-random baseline and CraftGround benchmark runner."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any

import numpy as np

from .env import GoalBounds, SpawnPoint, make_craftground_env


@dataclass(frozen=True)
class EpisodeResult:
    episode: int
    reset_seed: int
    policy_steps: int
    reward: float
    success: bool
    terminated: bool
    truncated: bool
    failure_reason: str | None
    reset_seconds: float
    interaction_seconds: float
    elapsed_seconds: float


@dataclass(frozen=True)
class BaselineResult:
    policy_seed: int
    wall_seconds: float
    episodes: tuple[EpisodeResult, ...]
    action_counts: dict[str, int]

    @property
    def total_interaction_seconds(self) -> float:
        return sum(episode.interaction_seconds for episode in self.episodes)

    @property
    def startup_seconds(self) -> float:
        return self.episodes[0].reset_seconds if self.episodes else 0.0

    @property
    def summary(self) -> dict[str, int | float]:
        episode_count = len(self.episodes)
        total_steps = sum(episode.policy_steps for episode in self.episodes)
        successes = sum(episode.success for episode in self.episodes)
        steady_state_seconds = max(0.0, self.wall_seconds - self.startup_seconds)
        return {
            "episodes": episode_count,
            "successes": successes,
            "success_rate": successes / episode_count if episode_count else 0.0,
            "mean_policy_steps": (
                fmean(episode.policy_steps for episode in self.episodes)
                if episode_count
                else 0.0
            ),
            "mean_episode_seconds": (
                fmean(episode.elapsed_seconds for episode in self.episodes)
                if episode_count
                else 0.0
            ),
            "total_reset_seconds": sum(
                episode.reset_seconds for episode in self.episodes
            ),
            "total_interaction_seconds": self.total_interaction_seconds,
            "startup_seconds": self.startup_seconds,
            "total_policy_steps": total_steps,
            "wall_seconds": self.wall_seconds,
            "policy_steps_per_second": (
                total_steps / self.total_interaction_seconds
                if self.total_interaction_seconds
                else 0.0
            ),
            "end_to_end_policy_steps_per_second": (
                total_steps / self.wall_seconds if self.wall_seconds else 0.0
            ),
            "steady_state_policy_steps_per_second": (
                total_steps / steady_state_seconds if steady_state_seconds else 0.0
            ),
            "episodes_per_second": (
                episode_count / self.wall_seconds if self.wall_seconds else 0.0
            ),
        }


def run_random_baseline(
    env: Any,
    *,
    episodes: int,
    max_episode_steps: int,
    seed: int,
) -> BaselineResult:
    """Run uniformly random discrete actions and always close ``env``.

    The NumPy generator controls the action sequence. Each Gymnasium reset also
    receives a deterministic, episode-specific seed. The step limit is a
    runner-side safety bound in addition to the environment's own truncation.
    """
    try:
        return _run_random_baseline(
            env,
            episodes=episodes,
            max_episode_steps=max_episode_steps,
            seed=seed,
        )
    finally:
        env.close()


def _run_random_baseline(
    env: Any,
    *,
    episodes: int,
    max_episode_steps: int,
    seed: int,
) -> BaselineResult:
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    if max_episode_steps < 1:
        raise ValueError("max_episode_steps must be >= 1")

    action_count = int(env.action_space.n)
    if action_count < 1:
        raise ValueError("env.action_space.n must be >= 1")
    action_names = tuple(
        getattr(env, "ACTION_NAMES", tuple(str(i) for i in range(action_count)))
    )
    if len(action_names) != action_count:
        action_names = tuple(str(i) for i in range(action_count))

    rng = np.random.default_rng(seed)
    action_counts = {name: 0 for name in action_names}
    episode_results: list[EpisodeResult] = []
    run_started = perf_counter()

    for episode_index in range(episodes):
        reset_seed = seed + episode_index
        episode_started = perf_counter()
        reset_started = perf_counter()
        _, info = env.reset(seed=reset_seed)
        reset_seconds = perf_counter() - reset_started
        interaction_started = perf_counter()
        terminated = False
        truncated = False
        policy_steps = 0
        episode_reward = 0.0

        while not (terminated or truncated) and policy_steps < max_episode_steps:
            action = int(rng.integers(action_count))
            action_counts[action_names[action]] += 1
            _, reward, terminated, truncated, info = env.step(action)
            policy_steps += 1
            episode_reward += float(reward)

        interaction_seconds = perf_counter() - interaction_started
        # Enforce the runner's cap even when used with a non-conforming env.
        if not (terminated or truncated) and policy_steps >= max_episode_steps:
            truncated = True
            info = dict(info)
            info["failure_reason"] = info.get("failure_reason") or "runner_step_limit"

        episode_results.append(
            EpisodeResult(
                episode=episode_index + 1,
                reset_seed=reset_seed,
                policy_steps=policy_steps,
                reward=episode_reward,
                success=bool(info.get("is_success", episode_reward > 0.0)),
                terminated=bool(terminated),
                truncated=bool(truncated),
                failure_reason=info.get("failure_reason"),
                reset_seconds=reset_seconds,
                interaction_seconds=interaction_seconds,
                elapsed_seconds=perf_counter() - episode_started,
            )
        )

    wall_seconds = perf_counter() - run_started
    return BaselineResult(
        policy_seed=seed,
        wall_seconds=wall_seconds,
        episodes=tuple(episode_results),
        action_counts=action_counts,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a uniform-random CraftGround baseline and throughput benchmark."
    )
    parser.add_argument("--world-seed", default="12345")
    parser.add_argument("--seed", type=int, default=0, help="Random policy/reset seed")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--port", type=int, default=8023)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--spawn", type=float, nargs=3, default=(95.5, 61.0, 736.5))
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument(
        "--goal-min", type=float, nargs=3, default=(66.0, 55.0, 668.0)
    )
    parser.add_argument(
        "--goal-max", type=float, nargs=3, default=(78.0, 68.0, 680.0)
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report path (for example results/random_baseline.json)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.max_episode_steps < 1:
        parser.error("--episodes and --max-episode-steps must be >= 1")
    if args.image_size < 1 or args.action_repeat < 1 or args.frame_stack < 1:
        parser.error("--image-size, --action-repeat and --frame-stack must be >= 1")

    spawn = SpawnPoint(*args.spawn, yaw=args.yaw)
    env = make_craftground_env(
        world_seed=args.world_seed,
        goal_bounds=GoalBounds(tuple(args.goal_min), tuple(args.goal_max)),
        spawn_points=[spawn],
        port=args.port,
        image_size=(args.image_size, args.image_size),
        action_repeat=args.action_repeat,
        frame_stack=args.frame_stack,
        max_episode_steps=args.max_episode_steps,
    )
    result = run_random_baseline(
        env,
        episodes=args.episodes,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
    )

    report = {
        "config": {
            "world_seed": args.world_seed,
            "policy_seed": args.seed,
            "episodes": args.episodes,
            "max_episode_steps": args.max_episode_steps,
            "action_repeat": args.action_repeat,
            "frame_stack": args.frame_stack,
            "spawn": list(args.spawn),
            "yaw": args.yaw,
            "goal_min": list(args.goal_min),
            "goal_max": list(args.goal_max),
        },
        "summary": result.summary,
        "action_counts": result.action_counts,
        "episode_results": [asdict(episode) for episode in result.episodes],
    }

    summary = result.summary
    print(f"Random baseline: {summary['episodes']} episodes")
    print(f"Success rate: {summary['success_rate']:.1%}")
    print(f"Mean policy steps: {summary['mean_policy_steps']:.1f}")
    print(f"Mean episode time: {summary['mean_episode_seconds']:.2f} s")
    print(
        "Step throughput (excludes reset): "
        f"{summary['policy_steps_per_second']:.2f} policy steps/s"
    )
    print(
        "End-to-end throughput (includes startup/resets): "
        f"{summary['end_to_end_policy_steps_per_second']:.2f} policy steps/s"
    )
    print(
        "Steady-state throughput (excludes initial startup): "
        f"{summary['steady_state_policy_steps_per_second']:.2f} policy steps/s"
    )
    print(f"Action counts: {result.action_counts}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
