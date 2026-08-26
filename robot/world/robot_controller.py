"""
robot_controller.py -- a mobile base and a pan/tilt camera on top of AI2-THOR.

Wraps `ai2thor.controller.Controller` with the small API everything else here
drives the robot through: move / rotate / pitch / height / teleport, plus the
frame geometry that turns a pixel into a world point and back.

This file once also held a behaviour layer -- generators for ORBIT_AROUND,
APPROACH_AND_PITCH and so on, dispatched by `execute()` from an ActionStruct
that `decision_tree.py` produced, with `check_success()` deciding when a search
had finished.  That was the interpreter for the height-conditioned object search
(Task 1), which was removed along with its results.  844 lines of it survived as
an unreachable island for months; the NVS experiments drive the robot by
teleporting to computed poses and never entered it.  It is gone.  If a policy
layer is wanted again, write it against the current API rather than reviving
that one -- it was built for a question nothing here now asks.

CAMERA HEIGHT LIMITATION, still true: the default iTHOR agent exposes only two
body poses (Stand / Crouch, roughly 1.55 m / 0.95 m eye level).  There is no
continuous height API for the acting agent, so `set_height_level` snaps.  A
third-party camera can sit anywhere -- `nvs_lemniscate.sweep` uses one -- but it
does NOT affect the `visible` flag in object metadata.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# The frame maths moved to `geometry.py`; re-exported here because `build/sgg/` and
# `archive/` ask this module for it.
# ---------------------------------------------------------------------------

from robot.geometry import (_xyz, _xz, horizon_towards,  # noqa: F401
                            point_in_box, unproject, wrap_deg, yaw_towards)




# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class RobotController:
    """Mobile base + pan/tilt camera on top of AI2-THOR."""

    # Legal cameraHorizon range for the default iTHOR agent.
    HORIZON_MIN = -30.0
    HORIZON_MAX = 60.0

    # Approximate eye heights of the two available body poses, metres above the
    # agent's foot position.  Used only to pick the nearest level.
    HEIGHT_LEVELS: Dict[str, float] = {"stand": 1.55, "crouch": 0.95}

    def __init__(
        self,
        scene: str = "FloorPlan1",
        width: int = 640,
        height: int = 480,
        field_of_view: float = 90.0,
        visibility_distance: float = 1.5,
        grid_size: float = 0.25,
        move_magnitude: float = 0.25,
        rotate_step: float = 30.0,
        seed: int = 0,
        headless: bool = False,
        # --- success thresholds (see module docstring for why not `visible`) --
        min_area_fraction: float = 0.002,   # >=0.2% of the image
        min_fill_ratio: float = 0.5,        # <=50% of the bbox occluded
        center_tolerance: float = 0.35,     # normalised, 0 = dead centre
        verbose: bool = True,
    ) -> None:
        from ai2thor.controller import Controller

        controller_kwargs: Dict[str, Any] = dict(
            scene=scene,
            width=width,
            height=height,
            fieldOfView=field_of_view,
            visibilityDistance=visibility_distance,
            gridSize=grid_size,
            snapToGrid=False,           # continuous poses -> Teleport works
            rotateStepDegrees=rotate_step,
            renderDepthImage=True,
            renderInstanceSegmentation=True,   # REQUIRED by check_success()
        )
        if headless:
            # For a GPU server with no X display.
            from ai2thor.platform import CloudRendering

            controller_kwargs["platform"] = CloudRendering

        self.controller = Controller(**controller_kwargs)
        self.event = self.controller.last_event

        self.width, self.height = width, height
        self._field_of_view = field_of_view   # VERTICAL, Unity convention
        self.move_magnitude = move_magnitude
        self.rotate_step = rotate_step
        self.visibility_distance = visibility_distance
        self.min_area_fraction = min_area_fraction
        self.min_fill_ratio = min_fill_ratio
        self.center_tolerance = center_tolerance
        self.verbose = verbose

        self.rng = random.Random(seed)
        self.step_count = 0
        self.path_length = 0.0        # metres actually travelled, for SPL
        # Steps split by actuator.  Counting base and camera moves equally
        # penalises camera-heavy strategies for no physical reason: panning takes
        # milliseconds, driving a metre takes seconds.  Measured, ~2/3 of
        # APPROACH_AND_PITCH's steps are pure camera adjustments.
        self.base_steps = 0
        self.camera_steps = 0
        self.height_level = "stand"
        self._reachable_cache: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------


    #: THOR actions that move only the camera, not the base.
    CAMERA_ACTIONS = frozenset({"LookUp", "LookDown", "Crouch", "Stand"})

    def _step(self, **kwargs) -> Any:
        """Single THOR step with step, path-length and actuator accounting."""
        action_name = kwargs.get("action", "")
        if action_name in self.CAMERA_ACTIONS:
            self.camera_steps += 1
        elif action_name not in ("GetReachablePositions", "GetShortestPathToPoint"):
            self.base_steps += 1
        before = _xz(self.agent_position) if self.event is not None else None
        self.event = self.controller.step(**kwargs)
        self.step_count += 1
        if before is not None:
            # Accumulate real displacement so SPL reflects the path taken,
            # including any Teleport recovery.
            self.path_length += float(
                np.linalg.norm(_xz(self.agent_position) - before)
            )
        return self.event


    def reset(self, scene: Optional[str] = None) -> None:
        """Reset the episode (and optionally switch scene)."""
        self.event = (
            self.controller.reset(scene=scene) if scene else self.controller.reset()
        )
        self.step_count = 0
        self.path_length = 0.0
        self.base_steps = 0
        self.camera_steps = 0
        self.height_level = "stand"
        self._reachable_cache = None

    def stop(self) -> None:
        self.controller.stop()

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    @property
    def fov_vertical(self) -> float:
        """
        Vertical field of view in degrees.

        THOR passes `fieldOfView` straight to Unity's Camera.fieldOfView, which
        is VERTICAL.  Anything reasoning about horizontal extent must convert.
        """
        return float(self.event.metadata.get("fov", self._field_of_view))

    @property
    def fov_horizontal(self) -> float:
        """Horizontal FOV derived from the vertical one and the aspect ratio."""
        half = math.radians(self.fov_vertical / 2.0)
        return 2.0 * math.degrees(
            math.atan(math.tan(half) * self.width / self.height)
        )

    @property
    def agent_position(self) -> Dict[str, float]:
        return self.event.metadata["agent"]["position"]

    @property
    def agent_yaw(self) -> float:
        return float(self.event.metadata["agent"]["rotation"]["y"]) % 360.0

    @property
    def camera_horizon(self) -> float:
        return float(self.event.metadata["agent"]["cameraHorizon"])

    @property
    def camera_xyz(self) -> np.ndarray:
        """World position of the camera (falls back to agent pos + eye height)."""
        cam = self.event.metadata.get("cameraPosition")
        if isinstance(cam, dict) and "y" in cam:
            return _xyz(cam)
        pos = _xyz(self.agent_position)
        pos[1] += self.HEIGHT_LEVELS.get(self.height_level, 1.55)
        return pos

    def get_reachable_positions(self, refresh: bool = False) -> np.ndarray:
        """Cached (N, 2) array of navigable XZ points."""
        if self._reachable_cache is None or refresh:
            event = self._step(action="GetReachablePositions")
            positions = event.metadata.get("actionReturn") or []
            self._reachable_cache = np.array(
                [[p["x"], p["z"]] for p in positions], dtype=float
            )
        return self._reachable_cache

    def nearest_reachable(self, point_xz: np.ndarray) -> Optional[np.ndarray]:
        """Snap an arbitrary XZ point to the nearest navigable position."""
        reachable = self.get_reachable_positions()
        if len(reachable) == 0:
            return None
        distances = np.linalg.norm(reachable - point_xz.reshape(1, 2), axis=1)
        return reachable[int(np.argmin(distances))]

    # ------------------------------------------------------------------
    # Object lookup
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Low-level primitives
    # ------------------------------------------------------------------

    def move_ahead(self, magnitude: Optional[float] = None) -> bool:
        self._step(
            action="MoveAhead",
            moveMagnitude=self.move_magnitude if magnitude is None else magnitude,
        )
        return bool(self.event.metadata["lastActionSuccess"])

    def move_back(self, magnitude: Optional[float] = None) -> bool:
        self._step(
            action="MoveBack",
            moveMagnitude=self.move_magnitude if magnitude is None else magnitude,
        )
        return bool(self.event.metadata["lastActionSuccess"])

    def rotate(self, degrees: float) -> bool:
        """Rotate in place; positive = right/clockwise."""
        degrees = wrap_deg(degrees)
        if abs(degrees) < 1e-2:
            return True
        action = "RotateRight" if degrees > 0 else "RotateLeft"
        self._step(action=action, degrees=abs(degrees))
        return bool(self.event.metadata["lastActionSuccess"])


    def look(self, delta_deg: float) -> bool:
        """Relative pitch; positive = look down."""
        return self.set_horizon(self.camera_horizon + delta_deg)

    def set_horizon(self, horizon_deg: float) -> bool:
        """
        Absolute camera pitch (positive = down), clamped to the legal range.

        Implemented with LookUp/LookDown so it behaves like a real pan/tilt unit
        rather than a state reset.
        """
        target = float(np.clip(horizon_deg, self.HORIZON_MIN, self.HORIZON_MAX))
        delta = target - self.camera_horizon
        if abs(delta) < 1e-2:
            return True
        action = "LookDown" if delta > 0 else "LookUp"
        self._step(action=action, degrees=abs(delta))
        if not self.event.metadata["lastActionSuccess"]:
            # Some builds only accept fixed increments -> fall back to Teleport.
            return self.teleport(horizon=target)
        return True

    def set_height_level(self, level: str) -> bool:
        """Switch body pose between the two available heights."""
        level = level.lower()
        if level not in self.HEIGHT_LEVELS:
            raise ValueError(f"height level must be one of {list(self.HEIGHT_LEVELS)}")
        if level == self.height_level:
            return True

        action = "Crouch" if level == "crouch" else "Stand"
        self._step(action=action)
        if self.event.metadata["lastActionSuccess"]:
            self.height_level = level
            return True

        # Older/newer builds route this through Teleport's `standing` flag.
        ok = self.teleport(standing=(level == "stand"))
        if ok:
            self.height_level = level
        return ok


    def teleport(
        self,
        position: Optional[Dict[str, float]] = None,
        yaw: Optional[float] = None,
        horizon: Optional[float] = None,
        standing: Optional[bool] = None,
    ) -> bool:
        """Direct pose set -- used as the recovery path when driving fails."""
        kwargs: Dict[str, Any] = {
            "action": "Teleport",
            "position": position if position is not None else self.agent_position,
            "rotation": {
                "x": 0.0,
                "y": self.agent_yaw if yaw is None else float(yaw) % 360.0,
                "z": 0.0,
            },
            "horizon": float(
                np.clip(
                    self.camera_horizon if horizon is None else horizon,
                    self.HORIZON_MIN,
                    self.HORIZON_MAX,
                )
            ),
            "standing": (self.height_level == "stand") if standing is None else standing,
        }
        self._step(**kwargs)
        return bool(self.event.metadata["lastActionSuccess"])

    # ------------------------------------------------------------------
    # Waypoint navigation
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Success criterion
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Behaviours -- one generator per ActionType, yielding after each viewpoint
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

