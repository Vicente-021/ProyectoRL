"""Run a time-boxed Stable-Baselines3 DQN pilot in real Minecraft."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Sequence

from .env import GoalBounds, NetherFortressEnv, SpawnPoint
from .preview import make_craftground_preview_callback
from .reward_shaping import PotentialDistanceReward
from .train_ppo import (
    PILOT_GOAL_MAX,
    PILOT_GOAL_MIN,
    PILOT_SPAWN,
    PILOT_WORLD_SEED,
    configure_craftground_window,
    create_environment,
    validate_craftground_toolchain,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entrena DQN de Stable-Baselines3 en Minecraft vía CraftGround."
    )
    parser.add_argument("--world-seed", default=PILOT_WORLD_SEED)
    parser.add_argument(
        "--spawn", type=float, nargs=3, default=PILOT_SPAWN, metavar=("X", "Y", "Z")
    )
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument(
        "--goal-min", type=float, nargs=3, default=PILOT_GOAL_MIN, metavar=("X", "Y", "Z")
    )
    parser.add_argument(
        "--goal-max", type=float, nargs=3, default=PILOT_GOAL_MAX, metavar=("X", "Y", "Z")
    )
    parser.add_argument("--port", type=int, default=8023)
    parser.add_argument(
        "--image-size", type=int, nargs=2, default=(640, 360), metavar=("WIDTH", "HEIGHT")
    )

    parser.add_argument("--total-timesteps", type=int, default=20_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=1_000)
    parser.add_argument("--exploration-fraction", type=float, default=0.5)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    parser.add_argument(
        "--reward-shaping",
        action="store_true",
        help="Usa shaping potencial por distancia; registra esta ejecución como ablación.",
    )
    parser.add_argument("--shaping-scale", type=float, default=0.05)

    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--checkpoint-freq", type=int, default=5_000)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--quiet-gradle", action="store_true")
    parser.add_argument("--audio", action="store_true")
    parser.add_argument("--hide-window", action="store_true")
    parser.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=os.name == "nt",
        help="Muestra en vivo el POV que recibe el agente.",
    )
    return parser


def _distance_to_goal(
    position: tuple[float, float, float],
    goal_min: tuple[float, float, float],
    goal_max: tuple[float, float, float],
) -> float:
    deltas = tuple(
        max(low - value, 0.0, value - high)
        for value, low, high in zip(position, goal_min, goal_max)
    )
    return math.sqrt(sum(delta * delta for delta in deltas))


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.total_timesteps < 1:
        parser.error("--total-timesteps debe ser mayor que cero")
    if args.batch_size < 2 or args.buffer_size < args.batch_size:
        parser.error("--buffer-size debe ser >= --batch-size y el batch al menos 2")
    if args.learning_starts < 0:
        parser.error("--learning-starts no puede ser negativo")
    if args.train_freq < 1 or args.gradient_steps < 1:
        parser.error("--train-freq y --gradient-steps deben ser positivos")
    if args.target_update_interval < 1:
        parser.error("--target-update-interval debe ser positivo")
    if args.learning_rate <= 0 or not 0 <= args.gamma <= 1:
        parser.error("Revisa --learning-rate y --gamma")
    if not 0 < args.exploration_fraction <= 1:
        parser.error("--exploration-fraction debe estar en (0, 1]")
    if not 0 <= args.exploration_final_eps <= 1:
        parser.error("--exploration-final-eps debe estar entre 0 y 1")
    if args.reward_shaping and (
        not math.isfinite(args.shaping_scale) or args.shaping_scale <= 0
    ):
        parser.error("--shaping-scale debe ser un número finito mayor que cero")
    if args.action_repeat < 1 or args.frame_stack < 1:
        parser.error("--action-repeat y --frame-stack deben ser positivos")
    if args.max_episode_steps < 1 or args.checkpoint_freq < 1:
        parser.error("--max-episode-steps y --checkpoint-freq deben ser positivos")
    if not 1 <= args.port <= 65_535:
        parser.error("--port debe estar entre 1 y 65535")
    if any(dimension < 64 for dimension in args.image_size):
        parser.error("Cada dimensión de --image-size debe ser al menos 64")
    if args.resume_from is not None and not args.resume_from.is_file():
        parser.error(f"No existe el checkpoint indicado: {args.resume_from}")
    try:
        GoalBounds(tuple(args.goal_min), tuple(args.goal_max))
    except ValueError as error:
        parser.error(str(error))


def train(args: argparse.Namespace) -> Path:
    try:
        from stable_baselines3 import DQN
        from stable_baselines3.common.callbacks import (
            BaseCallback,
            CallbackList,
            CheckpointCallback,
        )
        from stable_baselines3.common.monitor import Monitor
    except ImportError as error:
        raise SystemExit(
            "Faltan dependencias. Instala el proyecto con `.[minecraft,training]`."
        ) from error

    validate_craftground_toolchain()
    if not args.audio:
        os.environ["ALSOFT_DRIVERS"] = "null"
    configure_craftground_window(args.hide_window or args.preview)

    if args.run_dir is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        args.run_dir = Path("runs") / f"dqn_sb3_{run_id}"
    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    initial_distance = _distance_to_goal(
        tuple(args.spawn), tuple(args.goal_min), tuple(args.goal_max)
    )

    class RouteProgressCallback(BaseCallback):
        """Track privileged distance-to-goal for evaluation only, not policy input."""

        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.episode_best = initial_distance
            self.overall_best = initial_distance
            self.episodes = 0

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            dones = self.locals.get("dones", [])
            for info, done in zip(infos, dones):
                position = info.get("position")
                if position is not None:
                    distance = _distance_to_goal(
                        tuple(position), tuple(args.goal_min), tuple(args.goal_max)
                    )
                    self.episode_best = min(self.episode_best, distance)
                    self.overall_best = min(self.overall_best, distance)
                if done:
                    progress = max(0.0, 1.0 - self.episode_best / initial_distance)
                    self.episodes += 1
                    print(
                        f"[progreso DQN] episodio={self.episodes} "
                        f"reducción_distancia={progress:.0%} "
                        f"éxito={bool(info.get('is_success', False))}"
                    )
                    self.episode_best = initial_distance
            return True

        def _on_training_end(self) -> None:
            progress = max(0.0, 1.0 - self.overall_best / initial_distance)
            print(
                "Mejor reducción de distancia a la meta "
                f"(proxy de progreso, no distancia de ruta): {progress:.0%}"
            )

    base_environment = create_environment(args)
    environment = base_environment
    if args.reward_shaping:
        environment = PotentialDistanceReward(
            base_environment,
            base_environment.goal_bounds,
            gamma=args.gamma,
            scale=args.shaping_scale,
        )
    try:
        monitored_environment = Monitor(
            environment,
            filename=str(args.run_dir / "monitor"),
            info_keywords=("is_success",),
        )
        if args.resume_from is not None:
            model = DQN.load(
                str(args.resume_from),
                env=monitored_environment,
                device=args.device,
                tensorboard_log=str(args.run_dir / "tensorboard"),
            )
        else:
            model = DQN(
                "MlpPolicy",
                monitored_environment,
                learning_rate=args.learning_rate,
                buffer_size=args.buffer_size,
                learning_starts=args.learning_starts,
                batch_size=args.batch_size,
                tau=1.0,
                gamma=args.gamma,
                train_freq=args.train_freq,
                gradient_steps=args.gradient_steps,
                target_update_interval=args.target_update_interval,
                exploration_fraction=args.exploration_fraction,
                exploration_final_eps=args.exploration_final_eps,
                policy_kwargs={"net_arch": [256, 256]},
                tensorboard_log=str(args.run_dir / "tensorboard"),
                seed=args.seed,
                device=args.device,
                verbose=1,
            )

        callbacks = [
            CheckpointCallback(
                save_freq=args.checkpoint_freq,
                save_path=str(checkpoint_dir),
                name_prefix="dqn_sb3_nether_fortress",
                verbose=1,
            ),
            RouteProgressCallback(),
        ]
        if args.preview:
            callbacks.append(
                make_craftground_preview_callback(
                    base_environment,
                    "Minecraft 1.21 - Vista DQN (CraftGround)",
                )
            )
        callback = CallbackList(callbacks)

        config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        config["algorithm"] = "DQN-SB3"
        config["reward_mode"] = (
            "potential_distance_shaping" if args.reward_shaping else "sparse_terminal"
        )
        (args.run_dir / "config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )

        print(
            "Entrenamiento DQN de Stable-Baselines3 "
            f"(seed del mundo: {args.world_seed}; pasos: {args.total_timesteps})."
        )
        try:
            model.learn(
                total_timesteps=args.total_timesteps,
                callback=callback,
                tb_log_name="dqn_sb3_nether_fortress",
                reset_num_timesteps=args.resume_from is None,
            )
        except KeyboardInterrupt:
            interrupted_path = args.run_dir / "dqn_sb3_nether_fortress_interrupted.zip"
            model.save(str(interrupted_path))
            print(f"Interrupción recibida; modelo guardado en: {interrupted_path}")
            raise

        model_path = args.run_dir / "dqn_sb3_nether_fortress_final.zip"
        model.save(str(model_path))
        print(f"Modelo guardado en: {model_path}")
        print(f"Métricas y checkpoints: {args.run_dir}")
        return model_path
    finally:
        environment.close()


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)
    train(args)


if __name__ == "__main__":
    main()
