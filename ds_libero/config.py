"""The first experiment's protocol. Changes create a different fingerprint."""

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .errors import ExperimentError


LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"


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
    max_tokens: int = 8192
    timeout_seconds: int = 120
    max_retries: int = 2
    max_http_requests: int = 300

    def to_dict(self):
        return asdict(self)

    @property
    def fingerprint(self):
        return digest(self.to_dict())

    @classmethod
    def load(cls, path=None):
        values = {} if path is None else json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict) or set(values) - {f.name for f in fields(cls)}:
            raise ExperimentError("Config must be a JSON object containing only documented fields")
        default = cls()
        for key, value in values.items():
            if type(value) is not type(getattr(default, key)):
                raise ExperimentError(f"Wrong type for config field {key}")
        cfg = cls(**values)
        # v1 deliberately runs one fixed episode, not an accidentally expanded benchmark.
        locked = ("protocol_version", "suite", "task_order_index", "task_id", "init_state_id",
                  "seed", "settle_steps", "max_steps", "image_size", "history_steps",
                  "control_freq", "model", "endpoint", "reasoning_effort", "max_retries")
        for key in locked:
            if getattr(cfg, key) != getattr(default, key):
                raise ExperimentError(f"v1 fixes {key}={getattr(default, key)!r}")
        for key in ("max_tokens", "timeout_seconds", "max_http_requests"):
            if getattr(cfg, key) <= 0:
                raise ExperimentError(f"{key} must be positive")
        return cfg
