"""Gymnasium wrapper for direct Nether Fortress navigation in CraftGround.

The wrapper deliberately keeps target coordinates out of the policy
observation. They are used only by the environment to produce the terminal
success reward and evaluation information.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import os
from typing import Any, Iterable, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def _capture_craftground_processes(craftground_env: gym.Env) -> list[Any]:
    """Capture this CraftGround launch and Java processes attached to its IPC.

    CraftGround normally terminates its launcher process group, but Gradle can
    detach the Minecraft client from that group. Its per-instance IPC socket
    gives us a narrow way to identify that client and its runtime ancestors.
    """
    try:
        import psutil
    except ImportError:
        return []

    captured: dict[int, Any] = {}

    def add(process: Any) -> None:
        try:
            captured[process.pid] = process
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return

    # Capture the process tree while CraftGround's Popen handle is still live.
    popen = getattr(craftground_env, "process", None)
    if popen is not None and popen.pid is not None:
        try:
            root = psutil.Process(popen.pid)
            add(root)
            for child in root.children(recursive=True):
                add(child)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # Gradle may daemonize its client. CraftGround's own orphan cleanup uses
    # this socket file to identify Java clients; restrict the lookup to this
    # environment's port instead of matching every CraftGround process.
    ipc = getattr(craftground_env, "ipc", None)
    port = getattr(ipc, "port", None)
    if port is not None:
        socket_path = os.path.normpath(f"/tmp/minecraftrl_{port}.sock")
        runtime_path = os.path.realpath(getattr(craftground_env, "env_path", ""))

        def capture_socket_owner(pid: int | None) -> None:
            if pid is None:
                return
            try:
                process = psutil.Process(pid)
                add(process)
                parent = process.parent()
                while parent is not None:
                    try:
                        if os.path.realpath(parent.cwd()) != runtime_path:
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        break
                    add(parent)
                    parent = parent.parent()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return

        # Unix-domain sockets do not consistently appear in open_files(); use
        # the socket table as the primary lookup and open_files as a fallback.
        try:
            for connection in psutil.net_connections(kind="unix"):
                if os.path.normpath(str(connection.laddr)) == socket_path:
                    capture_socket_owner(connection.pid)
        except (psutil.Error, OSError):
            pass

        for process in psutil.process_iter():
            try:
                if not any(
                    os.path.normpath(open_file.path) == socket_path
                    for open_file in process.open_files()
                ):
                    continue
                capture_socket_owner(process.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                continue

    return list(captured.values())


def _stop_craftground_processes(processes: Sequence[Any]) -> None:
    """Escalate shutdown only for process handles captured for this instance."""
    if not processes:
        return

    try:
        import psutil
    except ImportError:
        return

    _, alive = psutil.wait_procs(list(processes), timeout=2.0)
    for process in alive:
        try:
            process.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    _, alive = psutil.wait_procs(alive, timeout=3.0)
    for process in alive:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    psutil.wait_procs(alive, timeout=2.0)


@dataclass(frozen=True)
class GoalBounds:
    """Axis-aligned target region in Minecraft block coordinates."""

    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]

    def __post_init__(self) -> None:
        if any(low > high for low, high in zip(self.minimum, self.maximum)):
            raise ValueError("GoalBounds.minimum must be <= maximum on each axis")

    def contains(self, position: tuple[float, float, float]) -> bool:
        return all(
            low <= value <= high
            for value, low, high in zip(position, self.minimum, self.maximum)
        )


@dataclass(frozen=True)
class SpawnPoint:
    """A known fortress entrance and the direction the player should face."""

    x: float
    y: float
    z: float
    yaw: float = 0.0
    pitch: float = 0.0
    dimension: str = "minecraft:the_nether"

    def teleport_command(self) -> str:
        return (
            f"execute in {self.dimension} run tp @p "
            f"{self.x:g} {self.y:g} {self.z:g} {self.yaw:g} {self.pitch:g}"
        )


class NetherFortressEnv(gym.Env[np.ndarray, int]):
    """Turn a CraftGround instance into a compact, discrete Gymnasium task.

    Args:
        craftground_env: An already-created CraftGround Gymnasium environment.
        goal_bounds: Ground-truth AABB of the lava well room. Never included in
            the observation returned to the policy.
        spawn_points: Candidate entry points. One is sampled per episode.
        action_repeat: Number of CraftGround ticks per movement macro. Turns
            change the camera once, then issue no-op ticks for the remainder.
        frame_stack: Number of recent encoded observations concatenated.
        max_episode_steps: Maximum number of policy decisions per episode.
        reset_commands: Commands executed after each fast reset. Include
            ``tick rate 20`` and mob-disabling commands for the current setup.
        death_y: Optional Y threshold that marks a fall as terminal failure.
        yaw_tolerance: Maximum orientation error accepted after a spawn command.

    Notes:
        The CraftGround instance should be created with
        ``requires_surrounding_blocks=True`` and an initial command list that
        includes ``tick rate 20``. Minecraft's world seed is configured when
        creating the CraftGround instance; Gymnasium reset seeds control the
        wrapper's random spawn selection, not a live world's seed.
    """

    metadata = {"render_modes": []}

    ACTION_FORWARD = 0
    ACTION_BACKWARD = 1
    ACTION_STRAFE_LEFT = 2
    ACTION_STRAFE_RIGHT = 3
    ACTION_TURN_LEFT = 4
    ACTION_TURN_RIGHT = 5
    ACTION_NAMES = (
        "forward",
        "backward",
        "strafe_left",
        "strafe_right",
        "turn_left",
        "turn_right",
    )

    # unknown, air, nether brick, fence, lava, soul sand/soil, wart, spawner,
    # and other. The category ids are internal to the encoder, not room labels.
    BLOCK_CLASSES = (
        "unknown",
        "air",
        "nether_brick",
        "fence",
        "lava",
        "soul_sand",
        "nether_wart",
        "spawner",
        "other",
    )

    def __init__(
        self,
        craftground_env: gym.Env,
        goal_bounds: GoalBounds,
        spawn_points: Sequence[SpawnPoint] = (),
        *,
        action_repeat: int = 4,
        frame_stack: int = 4,
        max_episode_steps: int = 500,
        reset_commands: Iterable[str] = (
            "tick rate 20",
            "difficulty peaceful",
            "gamerule doMobSpawning false",
        ),
        spawn_settle_ticks: int = 40,
        spawn_tolerance: float = 1.0,
        yaw_tolerance: float = 10.0,
        turn_degrees: float = 90.0,
        death_y: float | None = None,
    ) -> None:
        super().__init__()
        if action_repeat < 1:
            raise ValueError("action_repeat must be >= 1")
        if frame_stack < 1:
            raise ValueError("frame_stack must be >= 1")
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be >= 1")
        if spawn_settle_ticks < 0:
            raise ValueError("spawn_settle_ticks must be >= 0")
        if spawn_tolerance < 0:
            raise ValueError("spawn_tolerance must be >= 0")
        if yaw_tolerance < 0:
            raise ValueError("yaw_tolerance must be >= 0")

        self.craftground_env = craftground_env
        self.goal_bounds = goal_bounds
        self.spawn_points = tuple(spawn_points)
        self.action_repeat = int(action_repeat)
        self.frame_stack = int(frame_stack)
        self.max_episode_steps = int(max_episode_steps)
        self.reset_commands = tuple(reset_commands)
        self.spawn_settle_ticks = int(spawn_settle_ticks)
        self.spawn_tolerance = float(spawn_tolerance)
        self.yaw_tolerance = float(yaw_tolerance)
        self.turn_degrees = float(turn_degrees)
        self.death_y = death_y

        self.action_space = spaces.Discrete(len(self.ACTION_NAMES))
        self._single_observation_size = 27 * len(self.BLOCK_CLASSES) + 5
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.frame_stack * self._single_observation_size,),
            dtype=np.float32,
        )

        self._frames: deque[np.ndarray] = deque(maxlen=self.frame_stack)
        self._episode_steps = 0
        self._has_reset = False
        self._closed = False
        self._spawn_index: int | None = None
        self._last_raw_observation: dict[str, Any] | None = None

    @staticmethod
    def _position(raw_observation: dict[str, Any]) -> tuple[float, float, float]:
        full = raw_observation["full"]
        return float(full.x), float(full.y), float(full.z)

    @classmethod
    def _block_class(cls, translation_key: str) -> int:
        key = translation_key.lower()
        if not key:
            return 0
        if "nether_brick_fence" in key or "fence" in key or "wall" in key:
            return 3
        if "lava" in key:
            return 4
        if "soul_sand" in key or "soul_soil" in key:
            return 5
        if "nether_wart" in key:
            return 6
        if "spawner" in key:
            return 7
        if "air" in key:
            return 1
        if "nether_brick" in key:
            return 2
        return 8

    @staticmethod
    def _world_offset_to_ego(
        dx: int,
        dz: int,
        yaw_degrees: float,
    ) -> tuple[int, int]:
        """Return (forward, right) offsets for Minecraft's yaw convention."""
        # Minecraft yaw 0 faces +Z; positive yaw rotates toward -X.
        yaw = math.radians(yaw_degrees)
        forward = -dx * math.sin(yaw) + dz * math.cos(yaw)
        right = -dx * math.cos(yaw) - dz * math.sin(yaw)
        return int(round(forward)), int(round(right))

    def _encode_observation(self, raw_observation: dict[str, Any]) -> np.ndarray:
        full = raw_observation["full"]
        player_x, player_y, player_z = self._position(raw_observation)
        origin_x, origin_y, origin_z = (
            math.floor(player_x),
            math.floor(player_y),
            math.floor(player_z),
        )
        yaw = float(full.yaw)

        voxels = np.zeros((3, 3, 3, len(self.BLOCK_CLASSES)), dtype=np.float32)
        # Missing/unreported cells are encoded as unknown.
        voxels[..., 0] = 1.0

        for block in getattr(full, "surrounding_blocks", ()):
            dx = int(block.x) - origin_x
            dy = int(block.y) - origin_y
            dz = int(block.z) - origin_z
            if abs(dx) > 1 or abs(dy) > 1 or abs(dz) > 1:
                continue

            forward, right = self._world_offset_to_ego(dx, dz, yaw)
            if abs(forward) > 1 or abs(right) > 1:
                continue

            # Axes: vertical, forward/back, left/right. Positive local forward
            # is index 2, so the agent's forward view has a consistent location.
            cell = (dy + 1, forward + 1, right + 1)
            voxels[cell] = 0.0
            voxels[cell][self._block_class(block.translation_key)] = 1.0

        yaw_radians = math.radians(yaw)
        extras = np.asarray(
            [
                math.sin(yaw_radians),
                math.cos(yaw_radians),
                float(bool(getattr(full, "is_on_ground", False))),
                float(bool(getattr(full, "is_in_lava", False))),
                float(np.clip(float(getattr(full, "health", 20.0)) / 20.0, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )
        return np.concatenate((voxels.reshape(-1), extras)).astype(np.float32)

    def _stack(self, encoded: np.ndarray) -> np.ndarray:
        self._frames.append(encoded)
        while len(self._frames) < self.frame_stack:
            self._frames.appendleft(encoded.copy())
        return np.concatenate(tuple(self._frames)).astype(np.float32)

    @staticmethod
    def _noop_action() -> dict[str, Any]:
        action: dict[str, Any] = {
            key: 0
            for key in (
                "attack",
                "back",
                "forward",
                "jump",
                "left",
                "right",
                "sneak",
                "sprint",
                "use",
                "drop",
                "inventory",
                *(f"hotbar.{index}" for index in range(1, 10)),
            )
        }
        action["camera"] = np.zeros(2, dtype=np.float32)
        return action

    def _craftground_action(self, action_id: int, first_tick: bool) -> dict[str, Any]:
        action = self._noop_action()
        if action_id == self.ACTION_FORWARD:
            action["forward"] = 1
        elif action_id == self.ACTION_BACKWARD:
            action["back"] = 1
        elif action_id == self.ACTION_STRAFE_LEFT:
            action["left"] = 1
        elif action_id == self.ACTION_STRAFE_RIGHT:
            action["right"] = 1
        elif action_id == self.ACTION_TURN_LEFT and first_tick:
            action["camera"][1] = -self.turn_degrees
        elif action_id == self.ACTION_TURN_RIGHT and first_tick:
            action["camera"][1] = self.turn_degrees
        return action

    def _select_spawn(self, options: dict[str, Any]) -> SpawnPoint | None:
        if not self.spawn_points:
            self._spawn_index = None
            return None
        if "spawn_index" in options:
            index = int(options["spawn_index"])
            if not 0 <= index < len(self.spawn_points):
                raise ValueError(f"spawn_index out of range: {index}")
        else:
            index = int(self.np_random.integers(len(self.spawn_points)))
        self._spawn_index = index
        return self.spawn_points[index]

    def _episode_commands(self, spawn: SpawnPoint | None) -> list[str]:
        commands = list(self.reset_commands)
        if spawn is not None:
            commands.append(spawn.teleport_command())
        return commands

    def _at_spawn(
        self, raw_observation: dict[str, Any], spawn: SpawnPoint
    ) -> bool:
        position = self._position(raw_observation)
        target = (spawn.x, spawn.y, spawn.z)
        close_to_target = all(
            abs(actual - expected) <= self.spawn_tolerance
            for actual, expected in zip(position, target)
        )
        full = raw_observation["full"]
        blocks = getattr(full, "surrounding_blocks", ())
        terrain_loaded = any(
            "void_air" not in str(getattr(block, "translation_key", ""))
            for block in blocks
        )
        yaw_delta = (float(full.yaw) - spawn.yaw + 180.0) % 360.0 - 180.0
        facing_spawn_direction = abs(yaw_delta) <= self.yaw_tolerance
        return (
            close_to_target
            and terrain_loaded
            and bool(getattr(full, "is_on_ground", False))
            and facing_spawn_direction
        )

    def _settle_at_spawn(
        self, raw_observation: dict[str, Any], spawn: SpawnPoint
    ) -> dict[str, Any]:
        """Wait until the teleport is reflected and destination terrain is loaded."""
        for _ in range(self.spawn_settle_ticks + 1):
            if self._at_spawn(raw_observation, spawn):
                return raw_observation
            raw_observation, _, terminated, truncated, _ = self.craftground_env.step(
                self._noop_action()
            )
            if terminated or truncated:
                break
        if not self._at_spawn(raw_observation, spawn):
            raise RuntimeError(
                "CraftGround did not reach the selected spawn point: "
                f"expected {(spawn.x, spawn.y, spawn.z)}, got "
                f"{self._position(raw_observation)}. Check the dimension, "
                "teleport command, and that the spawn block is safe."
            )
        return raw_observation

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        spawn = self._select_spawn(options)
        commands = self._episode_commands(spawn)
        was_alive = bool(getattr(self.craftground_env, "is_alive", False))

        if was_alive:
            raw_observation, _ = self.craftground_env.reset(
                seed=seed,
                options={"fast_reset": True, "extra_commands": commands},
            )
            if spawn is not None:
                raw_observation = self._settle_at_spawn(raw_observation, spawn)
        else:
            raw_observation, _ = self.craftground_env.reset(seed=seed)
            # CraftGround's initial config handles initial_extra_commands. A
            # spawn selected by this wrapper is sent on its first no-op tick.
            if spawn is not None:
                self.craftground_env.add_commands([spawn.teleport_command()])
                raw_observation, _, _, _, _ = self.craftground_env.step(
                    self._noop_action()
                )
                raw_observation = self._settle_at_spawn(raw_observation, spawn)

        self._last_raw_observation = raw_observation
        self._episode_steps = 0
        self._frames.clear()
        observation = self._stack(self._encode_observation(raw_observation))
        return observation, {
            "spawn_index": self._spawn_index,
            "position": self._position(raw_observation),
            "yaw": float(raw_observation["full"].yaw),
            "is_success": False,
        }

    def _is_success(self, raw_observation: dict[str, Any]) -> bool:
        return self.goal_bounds.contains(self._position(raw_observation))

    def _failure_reason(self, raw_observation: dict[str, Any]) -> str | None:
        full = raw_observation["full"]
        if bool(getattr(full, "is_dead", False)) or float(
            getattr(full, "health", 20.0)
        ) <= 0.0:
            return "death"
        if self.death_y is not None and float(full.y) <= self.death_y:
            return "fall"
        return None

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._last_raw_observation is None:
            raise RuntimeError("Call reset() before step()")
        action_id = int(action)
        if not self.action_space.contains(action_id):
            raise ValueError(f"Invalid action {action_id}; expected 0..5")

        raw_observation = self._last_raw_observation
        terminated = False
        truncated = False
        reward = 0.0
        failure_reason: str | None = None

        for tick in range(self.action_repeat):
            base_action = self._craftground_action(action_id, first_tick=(tick == 0))
            raw_observation, _, base_terminated, base_truncated, _ = (
                self.craftground_env.step(base_action)
            )
            self._last_raw_observation = raw_observation

            if self._is_success(raw_observation):
                reward = 1.0
                terminated = True
                break
            failure_reason = self._failure_reason(raw_observation)
            if failure_reason is not None or base_terminated:
                terminated = True
                break
            if base_truncated:
                truncated = True
                break

        self._episode_steps += 1
        if not terminated and not truncated and self._episode_steps >= self.max_episode_steps:
            truncated = True

        observation = self._stack(self._encode_observation(raw_observation))
        # CraftGround's base info is a copy of its full observation (including
        # the image). Do not forward it: keep info compact and do not leak it
        # into callbacks/logs accidentally.
        info = {
            "is_success": reward > 0.0,
            "failure_reason": failure_reason,
            "position": self._position(raw_observation),
            "yaw": float(raw_observation["full"].yaw),
            "episode_steps": self._episode_steps,
        }
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        owned_processes = _capture_craftground_processes(self.craftground_env)
        try:
            self.craftground_env.close()
        finally:
            # CraftGround's process-group shutdown can miss a Gradle-detached
            # client. Escalate only for processes tied to this env's Popen tree
            # or its unique IPC socket/port.
            _stop_craftground_processes(owned_processes)


def make_craftground_env(
    *,
    world_seed: int | str,
    goal_bounds: GoalBounds,
    spawn_points: Sequence[SpawnPoint],
    port: int = 8023,
    image_size: tuple[int, int] = (64, 64),
    action_repeat: int = 4,
    frame_stack: int = 4,
    max_episode_steps: int = 500,
) -> NetherFortressEnv:
    """Construct CraftGround 1.21.0 and wrap it for the fortress task.

    The wrapper expects CraftGround's V2 MineRL-style action interface. The
    ``tick rate 20`` command is retained from the successful CraftGround smoke
    test; adjust it only after benchmarking a stable setting on the target host.
    """
    import craftground

    initial_commands = [
        "tick rate 20",
        "difficulty peaceful",
        "gamerule doMobSpawning false",
    ]
    initial_config = craftground.InitialEnvironmentConfig(
        image_width=image_size[0],
        image_height=image_size[1],
        seed=str(world_seed),
        initial_extra_commands=initial_commands,
        requires_surrounding_blocks=True,
    )
    base_env = craftground.make(
        initial_env_config=initial_config,
        mc_version="1.21",
        port=port,
        action_space_version=craftground.ActionSpaceVersion.V2_MINERL_HUMAN,
    )
    return NetherFortressEnv(
        base_env,
        goal_bounds,
        spawn_points,
        action_repeat=action_repeat,
        frame_stack=frame_stack,
        max_episode_steps=max_episode_steps,
        reset_commands=initial_commands,
    )
