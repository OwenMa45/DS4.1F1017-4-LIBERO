import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .config import digest
from .accounting import pricing_snapshot
from .protocol import system_prompt


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_files():
    return {p.name: p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py"))}


def source_fingerprint():
    return digest({name: hashlib.sha256(data).hexdigest() for name, data in source_files().items()})


class Store:
    def __init__(self, path, command, cfg):
        self.path = Path(path).resolve()
        # An existing experiment is never overwritten or silently resumed.
        self.path.mkdir(parents=True, exist_ok=False)
        for folder in ("frames", "responses", "requests", "source_snapshot"):
            (self.path / folder).mkdir()
        for name, data in source_files().items():
            (self.path / "source_snapshot" / name).write_bytes(data)
        prompt = system_prompt(cfg)
        (self.path / "prompt.txt").write_text(prompt, encoding="utf-8")
        self.manifest = {"schema_version": 1, "command": command, "created_utc": utc_now(),
                         "api_pricing": pricing_snapshot(),
                         "config": cfg.to_dict(), "config_sha256": cfg.fingerprint,
                         "source_sha256": source_fingerprint(),
                         "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                         "host_platform": platform.platform(), "model_version_pinned": False,
                         "model_identity_note": "Official API alias; expected release is a label, not an immutable revision."}
        try:
            result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent.parent,
                                    capture_output=True, text=True, timeout=5)
            self.manifest["experiment_git_commit"] = result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            self.manifest["experiment_git_commit"] = None
        write_json(self.path / "manifest.json", self.manifest)

    def environment(self, provenance):
        self.manifest["environment"] = provenance
        self.manifest["environment_sha256"] = digest(provenance)
        write_json(self.path / "manifest.json", self.manifest)

    def event(self, value):
        event = {"utc": utc_now(), **value}
        with (self.path / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()

    def response(self, attempt, body):
        path = self.path / "responses" / f"attempt_{attempt:04d}.txt"
        path.write_text(body, encoding="utf-8")
        return str(path.relative_to(self.path)).replace("\\", "/")

    def request(self, step, encoded):
        path = self.path / "requests" / f"step_{step:05d}.json"
        path.write_bytes(encoded)
        return str(path.relative_to(self.path)).replace("\\", "/")

    def observe(self, index, obs):
        label = f"{index:05d}" if isinstance(index, int) else str(index)
        refs = {}
        for camera, data in obs.images.items():
            path = self.path / "frames" / f"{label}_{camera}.png"
            path.write_bytes(data)
            refs[camera] = {"path": str(path.relative_to(self.path)).replace("\\", "/"),
                            "sha256": hashlib.sha256(data).hexdigest()}
        self.event({"event": "observation", "frame": index, "frames": refs,
                    "state": obs.state, "geometry": obs.geometry})
        return refs

    def finish(self, result):
        write_json(self.path / "result.json", {"finished_utc": utc_now(), **result})


def verify_gate(path, command, cfg, environment):
    from .errors import ExperimentError
    try:
        root = Path(path)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ExperimentError(f"Missing/invalid {command} artifacts at {path}") from exc
    expected = {"command": command, "config_sha256": cfg.fingerprint,
                "source_sha256": source_fingerprint(), "environment_sha256": digest(environment)}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ExperimentError(f"{command} gate mismatch: {key}; rerun it with this code/config/runtime")
    if result.get("status") != "pass":
        raise ExperimentError(f"{command} has not passed")
