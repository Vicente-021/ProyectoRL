"""Live preview of the POV frames delivered by CraftGround."""

from __future__ import annotations

from typing import Any

import numpy as np


def make_craftground_preview_callback(environment: Any, title: str):
    """Create an optional Tk viewer callback without importing SB3 at module load."""
    from stable_baselines3.common.callbacks import BaseCallback

    class CraftGroundPreviewCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.environment = environment
            self.title = title
            self.root = None
            self.label = None
            self._tk = None
            self._image_tk = None
            self._Image = None
            self._photo = None

        def _on_training_start(self) -> None:
            import tkinter as tk
            from PIL import Image, ImageTk

            self._tk = tk
            self._image_tk = ImageTk
            self._Image = Image
            self.root = tk.Tk()
            self.root.title(self.title)
            self.root.configure(background="black")
            self.label = tk.Label(
                self.root,
                text="Esperando el primer frame de Minecraft...",
                background="black",
                foreground="white",
            )
            self.label.pack(fill="both", expand=True)
            self.root.protocol("WM_DELETE_WINDOW", self._close_preview)
            self.root.lift()
            self.root.focus_force()
            self._refresh()

        def _refresh(self) -> None:
            if self.root is None or self.label is None:
                return
            raw = self.environment._last_raw_observation
            if raw is None or raw.get("pov") is None:
                return
            frame = np.asarray(raw["pov"])
            if frame.ndim != 3:
                return
            if frame.shape[0] in (3, 4) and frame.shape[-1] not in (3, 4):
                frame = np.moveaxis(frame, 0, -1)
            if frame.shape[-1] == 4:
                frame = frame[..., :3]
            frame = np.clip(frame, 0, 255).astype(np.uint8, copy=False)
            self._photo = self._image_tk.PhotoImage(self._Image.fromarray(frame))
            try:
                self.label.configure(image=self._photo, text="")
                self.label.image = self._photo
                self.root.update_idletasks()
                self.root.update()
            except self._tk.TclError:
                self._close_preview()

        def _close_preview(self) -> None:
            if self.root is None:
                return
            root, self.root = self.root, None
            try:
                root.destroy()
            except self._tk.TclError:
                pass

        def _on_step(self) -> bool:
            self._refresh()
            return True

        def _on_training_end(self) -> None:
            self._close_preview()

    return CraftGroundPreviewCallback()
