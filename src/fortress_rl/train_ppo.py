"""Train a PPO policy directly in Minecraft through CraftGround."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
from datetime import datetime
from importlib.metadata import distribution
from pathlib import Path
from typing import Sequence

import numpy as np

from .env import GoalBounds, NetherFortressEnv, SpawnPoint, make_craftground_env
from .preview import make_craftground_preview_callback


# Pilot fortress documented in TODO.md. If the world seed changes, supply the
# matching spawn and goal coordinates as well.
PILOT_WORLD_SEED = "12345"
PILOT_SPAWN = (95.5, 61.0, 736.5)
PILOT_GOAL_MIN = (66.0, 55.0, 668.0)
PILOT_GOAL_MAX = (78.0, 68.0, 680.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Entrena PPO en Minecraft Java 1.21.0 usando CraftGround y el "
            "wrapper NetherFortressEnv."
        )
    )
    parser.add_argument("--world-seed", default=PILOT_WORLD_SEED)
    parser.add_argument(
        "--spawn",
        type=float,
        nargs=3,
        default=PILOT_SPAWN,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument(
        "--goal-min",
        type=float,
        nargs=3,
        default=PILOT_GOAL_MIN,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--goal-max",
        type=float,
        nargs=3,
        default=PILOT_GOAL_MAX,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--port", type=int, default=8023)
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=(640, 360),
        metavar=("WIDTH", "HEIGHT"),
        help="Resolución de Minecraft; usa 64 64 para reducir el coste de renderizado.",
    )

    parser.add_argument(
        "--total-timesteps",
        "--timesteps",
        dest="total_timesteps",
        type=int,
        default=100_000,
    )
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)

    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument(
        "--quiet-gradle",
        action="store_true",
        help="Oculta la salida de Gradle; por defecto se muestra el progreso de Minecraft.",
    )
    parser.add_argument(
        "--audio",
        action="store_true",
        help="Activa el audio del cliente; por defecto se usa el backend silencioso.",
    )
    parser.add_argument(
        "--hide-window",
        action="store_true",
        help="Deja minimizada la ventana de CraftGround (modo más discreto).",
    )
    parser.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=os.name == "nt",
        help="Muestra en vivo el POV de CraftGround en una ventana de vista previa.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed del algoritmo PPO.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint-freq", type=int, default=10_000)
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Reanuda desde un modelo/checkpoint .zip; timesteps indica pasos adicionales.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Directorio de salida; por defecto se crea uno fechado dentro de runs/.",
    )
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.total_timesteps <= 0:
        parser.error("--total-timesteps debe ser mayor que cero")
    if args.n_steps < 2:
        parser.error("--n-steps debe ser al menos 2")
    if args.batch_size < 2 or args.batch_size > args.n_steps:
        parser.error("--batch-size debe estar entre 2 y --n-steps")
    if args.n_steps % args.batch_size != 0:
        parser.error("--n-steps debe ser divisible por --batch-size")
    if args.n_epochs < 1:
        parser.error("--n-epochs debe ser mayor que cero")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate debe ser mayor que cero")
    if not math.isfinite(args.gamma) or not 0.0 <= args.gamma <= 1.0:
        parser.error("--gamma debe estar entre 0 y 1")
    if not math.isfinite(args.gae_lambda) or not 0.0 <= args.gae_lambda <= 1.0:
        parser.error("--gae-lambda debe estar entre 0 y 1")
    if not math.isfinite(args.ent_coef) or args.ent_coef < 0:
        parser.error("--ent-coef no puede ser negativo")
    if args.action_repeat < 1 or args.frame_stack < 1:
        parser.error("--action-repeat y --frame-stack deben ser al menos 1")
    if args.max_episode_steps < 1:
        parser.error("--max-episode-steps debe ser mayor que cero")
    if args.checkpoint_freq < 1:
        parser.error("--checkpoint-freq debe ser mayor que cero")
    if not 1 <= args.port <= 65_535:
        parser.error("--port debe estar entre 1 y 65535")
    if any(dimension < 64 for dimension in args.image_size):
        parser.error("Cada dimensión de --image-size debe ser al menos 64")
    if args.resume_from is not None and not args.resume_from.is_file():
        parser.error(f"No existe el checkpoint indicado: {args.resume_from}")

    coordinates = (*args.spawn, args.yaw, *args.goal_min, *args.goal_max)
    if not all(math.isfinite(value) for value in coordinates):
        parser.error("Las coordenadas y la orientación deben ser valores finitos")
    try:
        GoalBounds(tuple(args.goal_min), tuple(args.goal_max))
    except ValueError as error:
        parser.error(str(error))


def create_environment(args: argparse.Namespace) -> NetherFortressEnv:
    """Create the real CraftGround Minecraft environment for this run."""
    return make_craftground_env(
        world_seed=args.world_seed,
        goal_bounds=GoalBounds(tuple(args.goal_min), tuple(args.goal_max)),
        spawn_points=[SpawnPoint(*args.spawn, yaw=args.yaw)],
        port=args.port,
        image_size=tuple(args.image_size),
        action_repeat=args.action_repeat,
        frame_stack=args.frame_stack,
        max_episode_steps=args.max_episode_steps,
        verbose_gradle=not args.quiet_gradle,
    )


def validate_craftground_toolchain() -> None:
    """Fail early with useful guidance instead of hanging at first startup."""
    java = shutil.which("java")
    if java is None:
        raise SystemExit(
            "No se encontró Java en PATH. Activa el entorno CraftGround con JDK 21."
        )

    java_result = subprocess.run(
        [java, "-version"], capture_output=True, text=True, check=False
    )
    java_output = java_result.stdout + java_result.stderr
    java_match = re.search(
        r"\b(?:java|openjdk)\s+(?:version\s+)?[\"']?(\d+)", java_output, re.I
    )
    if java_result.returncode != 0 or java_match is None:
        raise SystemExit(f"No se pudo comprobar Java ({java}):\n{java_output}")
    java_major = int(java_match.group(1))
    if java_major != 21:
        raise SystemExit(
            f"CraftGround para Minecraft 1.21 requiere JDK 21; se encontró Java "
            f"{java_major} ({java}). Activa el entorno craftground-mc121."
        )
    if shutil.which("javac") is None:
        raise SystemExit(
            "No se encontró javac en PATH. CraftGround necesita un JDK completo, "
            "no solo Java Runtime."
        )

    cmake = shutil.which("cmake")
    if cmake is None:
        raise SystemExit(
            "No se encontró CMake en PATH. Instálalo en el entorno CraftGround "
            "con `conda install -c conda-forge cmake`."
        )
    cmake_result = subprocess.run(
        [cmake, "--version"], capture_output=True, text=True, check=False
    )
    cmake_output = cmake_result.stdout + cmake_result.stderr
    cmake_match = re.search(r"cmake version\s+(\d+)\.(\d+)", cmake_output, re.I)
    if cmake_result.returncode != 0 or cmake_match is None:
        raise SystemExit(f"No se pudo comprobar CMake ({cmake}):\n{cmake_output}")
    cmake_version = (int(cmake_match.group(1)), int(cmake_match.group(2)))
    required_cmake = (4, 2) if os.name == "nt" else (3, 28)
    if cmake_version < required_cmake:
        required_text = "4.2 (Visual Studio 18)" if os.name == "nt" else "3.28"
        raise SystemExit(
            f"CraftGround necesita CMake {required_text} o superior; se encontró "
            f"{cmake_match.group(0)}. Actualízalo dentro del entorno CraftGround."
        )


_ICONIFY_WINDOW_BLOCK = (
    "        if (!hasMinimizedWindow) {\n"
    "            GLFW.glfwIconifyWindow(window.handle)\n"
    "            hasMinimizedWindow = true\n"
    "        }"
)
_VISIBLE_WINDOW_BLOCK = (
    "        if (!hasMinimizedWindow) {\n"
    '            if (System.getenv("CRAFTGROUND_KEEP_WINDOW_VISIBLE") != "1") {\n'
    "                GLFW.glfwIconifyWindow(window.handle)\n"
    "            }\n"
    "            hasMinimizedWindow = true\n"
    "        }"
)


def patch_craftground_window_source(source_path: Path) -> None:
    """Make CraftGround's initial minimize optional; safe to run repeatedly."""
    source = source_path.read_text(encoding="utf-8")
    if _VISIBLE_WINDOW_BLOCK in source:
        return
    if _ICONIFY_WINDOW_BLOCK not in source:
        raise RuntimeError(
            "CraftGround's EnvironmentInitializer.kt no longer has the expected "
            "window-minimize block; refusing to patch an unknown version."
        )
    source_path.write_text(
        source.replace(_ICONIFY_WINDOW_BLOCK, _VISIBLE_WINDOW_BLOCK, 1),
        encoding="utf-8",
    )


def configure_craftground_window(hide_window: bool) -> None:
    """Keep Minecraft visible by default; preserve CraftGround's headless mode."""
    if hide_window:
        os.environ.pop("CRAFTGROUND_KEEP_WINDOW_VISIBLE", None)
        return

    try:
        runtime_root = Path(
            distribution("craftground-runtime-mc121").locate_file(
                "craftground_runtime_mc121"
            )
        )
        source_path = (
            runtime_root
            / "src"
            / "main"
            / "java"
            / "com"
            / "kyhsgeekcode"
            / "minecraftenv"
            / "EnvironmentInitializer.kt"
        )
        patch_craftground_window_source(source_path)
    except Exception as error:
        raise SystemExit(
            "No pude habilitar la ventana visible de CraftGround. "
            "Comprueba que craftground-runtime-mc121 incluya sus fuentes."
        ) from error

    os.environ["CRAFTGROUND_KEEP_WINDOW_VISIBLE"] = "1"


def train(args: argparse.Namespace) -> Path:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import (
            CallbackList,
            CheckpointCallback,
        )
        from stable_baselines3.common.monitor import Monitor
    except ImportError as error:
        raise SystemExit(
            "Faltan dependencias de entrenamiento. Instálalas con "
            '`python -m pip install -e ".[minecraft,training]"`.'
        ) from error

    validate_craftground_toolchain()
    if not args.audio:
        # Audio is not an observation or action in this task. The silent
        # OpenAL backend also avoids crashes in some Windows audio drivers.
        os.environ["ALSOFT_DRIVERS"] = "null"
    configure_craftground_window(args.hide_window or args.preview)

    if args.run_dir is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        args.run_dir = Path("runs") / f"ppo_{run_id}"
    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(
        "Iniciando Minecraft/CraftGround "
        f"(seed del mundo: {args.world_seed}; puerto: {args.port})."
    )
    environment = create_environment(args)
    try:
        monitored_environment = Monitor(
            environment,
            filename=str(args.run_dir / "monitor"),
            info_keywords=("is_success",),
        )
        if args.resume_from is not None:
            model = PPO.load(
                str(args.resume_from),
                env=monitored_environment,
                device=args.device,
                tensorboard_log=str(args.run_dir / "tensorboard"),
            )
        else:
            # For Discrete(6), SB3's PPO samples from a categorical policy
            # during learn() (deterministic=False); it does not take argmax.
            model = PPO(
                "MlpPolicy",
                monitored_environment,
                learning_rate=args.learning_rate,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                ent_coef=args.ent_coef,
                policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
                tensorboard_log=str(args.run_dir / "tensorboard"),
                seed=args.seed,
                device=args.device,
                verbose=1,
            )
        checkpoint_callback = CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=str(checkpoint_dir),
            name_prefix="ppo_nether_fortress",
            verbose=1,
        )
        callbacks = [checkpoint_callback]
        if args.preview:
            callbacks.append(
                make_craftground_preview_callback(
                    environment,
                    "Minecraft 1.21 - Vista PPO (CraftGround)",
                )
            )
        callback = CallbackList(callbacks)

        config_path = args.run_dir / "config.json"
        config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        print(
            "Entrenamiento PPO en Minecraft/CraftGround "
            f"(seed del mundo: {args.world_seed}; acciones: {args.total_timesteps})."
        )
        try:
            model.learn(
                total_timesteps=args.total_timesteps,
                callback=callback,
                tb_log_name="ppo_nether_fortress",
                reset_num_timesteps=args.resume_from is None,
            )
        except KeyboardInterrupt:
            interrupted_path = args.run_dir / "ppo_nether_fortress_interrupted.zip"
            model.save(str(interrupted_path))
            print(f"Interrupción recibida; modelo guardado en: {interrupted_path}")
            raise
        model_path = args.run_dir / "ppo_nether_fortress_final.zip"
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
