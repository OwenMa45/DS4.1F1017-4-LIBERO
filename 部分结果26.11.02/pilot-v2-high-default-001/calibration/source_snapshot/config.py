"""The first experiment's protocol. Changes create a different fingerprint."""

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .errors import ExperimentError


LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
SUITE_HORIZONS = {"libero_spatial": 220, "libero_object": 280,
                  "libero_goal": 300, "libero_10": 520}


def digest(value):
    data = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


@dataclass(frozen=True)
class Config:
    protocol_version: str = "ds-libero-direct-v1"
    suite: str = "libero_spatial"
    task_order_index: int = 0
    task_id: int = 0
    init_state_id: int = 0
    seed: int = 0
    settle_steps: int = 10
    max_steps: int = 220
    image_size: int = 256
    history_steps: int = 4
    control_freq: int = 20
    model: str = "deepseek-flash"
    expected_release: str = "DeepSeek-V4.1-Flash-1017"
    endpoint: str = "https://api.deepseek.com/chat/completions"
    reasoning_effort: str = "high"
    max_tokens: int | None = 8192
    timeout_seconds: int = 120
    max_retries: int = 2
    max_http_requests: int = 300
    action_horizon: int = 1
    visual_history: int = 0
    camera_geometry: bool = False
    evaluation_scope: str = "pilot"

    def to_dict(self):
        values = asdict(self)
        # Preserve the original v1 fingerprint and request contract.
        if self.protocol_version == "ds-libero-direct-v1":
            for key in ("action_horizon", "visual_history", "camera_geometry", "evaluation_scope"):
                values.pop(key)
        return values

    @property
    def fingerprint(self):
        return digest(self.to_dict())

    @classmethod
    def load(cls, path=None):
        values = {} if path is None else json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(values)

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict) or set(values) - {f.name for f in fields(cls)}:
            raise ExperimentError("Config must be a JSON object containing only documented fields")
        default = cls()
        for key, value in values.items():
            if key == "max_tokens" and value is None:
                continue
            if type(value) is not type(getattr(default, key)):
                raise ExperimentError(f"Wrong type for config field {key}")
        cfg = cls(**values)
        # v1 deliberately runs one fixed episode, not an accidentally expanded benchmark.
        if cfg.protocol_version not in ("ds-libero-direct-v1", "ds-libero-direct-v2"):
            raise ExperimentError("Unknown protocol_version")
        locked = ("task_order_index", "settle_steps", "image_size", "history_steps",
                  "control_freq", "model", "endpoint", "max_retries")
        if cfg.evaluation_scope not in ("pilot", "benchmark"):
            raise ExperimentError("evaluation_scope must be pilot or benchmark")
        if cfg.evaluation_scope == "pilot":
            locked += ("suite", "task_id", "init_state_id", "seed", "max_steps")
        elif (cfg.suite not in SUITE_HORIZONS or cfg.max_steps != SUITE_HORIZONS[cfg.suite]
              or not 0 <= cfg.task_id < 10 or not 0 <= cfg.init_state_id < 50
              or not 0 <= cfg.seed < 2 ** 32):
            raise ExperimentError("Benchmark config requires a supported suite/horizon, task 0..9, init 0..49 and uint32 seed")
        if cfg.protocol_version == "ds-libero-direct-v1":
            locked += ("reasoning_effort", "action_horizon", "visual_history", "camera_geometry", "evaluation_scope")
        for key in locked:
            if getattr(cfg, key) != getattr(default, key):
                raise ExperimentError(f"This pilot fixes {key}={getattr(default, key)!r}")
        if cfg.reasoning_effort not in ("low", "high"):
            raise ExperimentError("Use low or high; medium maps to high in the DeepSeek API")
        if not 1 <= cfg.action_horizon <= 8 or not 0 <= cfg.visual_history <= 2:
            raise ExperimentError("action_horizon must be 1..8; visual_history must be 0..2")
        for key in ("timeout_seconds", "max_http_requests"):
            if getattr(cfg, key) <= 0:
                raise ExperimentError(f"{key} must be positive")
        if cfg.max_tokens is not None and not 1 <= cfg.max_tokens <= 393216:
            raise ExperimentError("max_tokens must be null (omit API field) or 1..393216")
        return cfg
