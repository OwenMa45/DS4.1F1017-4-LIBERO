"""Create a SAM3 venv while retaining the node's vendor accelerator stack.

Run with the node Python, e.g. /usr/local/bin/python scripts/setup_sam3_env.py.
No Conda, driver installation, public torch wheel, or simulator changes.
"""

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback
import venv


def protected(name):
    name = re.sub(r"[-_.]+", "-", name.lower())
    return any(part in name for part in ("torch", "triton", "nvidia", "cuda", "ppu", "hggc", "flaggems", "flag-gems"))


def snapshot():
    result = {"python": sys.executable, "prefix": sys.prefix, "base_prefix": sys.base_prefix,
              "modules": {}, "errors": {}, "protected_packages": {}}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name", "")
        if protected(name):
            key = re.sub(r"[-_.]+", "-", name.lower())
            # venv distributions precede inherited ones, like module resolution.
            result["protected_packages"].setdefault(key, {
                "version": dist.version, "location": str(Path(dist.locate_file("")).resolve())})
    for name in ("torch", "torchvision", "triton"):
        try:
            if name == "triton" and importlib.util.find_spec(name) is None:
                continue
            module = importlib.import_module(name)
            result["modules"][name] = {"version": str(getattr(module, "__version__", "unknown")),
                                      "file": str(Path(module.__file__).resolve())}
            if name == "torch":
                result["torch_cuda_version"] = getattr(module.version, "cuda", None)
                result["cuda_available"] = bool(module.cuda.is_available())
                result["device_count"] = module.cuda.device_count()
                if result["cuda_available"]:
                    result["device_name"] = module.cuda.get_device_name(0)
        except Exception:
            result["errors"][name] = traceback.format_exc()
    return result


def approved_pins(report):
    """Review the complete resolver result before allowing any dependency install."""
    pins = []
    for entry in report["install"]:
        name, version = entry["metadata"]["name"], entry["metadata"]["version"]
        if protected(name):
            raise RuntimeError(f"Refusing to install/replace accelerator package {name}=={version}. "
                               "Keep the node's vendor stack; send the setup report for diagnosis.")
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name)
                or not re.fullmatch(r"[A-Za-z0-9.!+_-]+", version)):
            raise RuntimeError("Invalid resolved package name/version")
        pins.append(f"{name}=={version}")
    return pins


def verify_inherited(base, current):
    if current.get("errors") or not current.get("cuda_available"):
        raise RuntimeError("Accelerator imports or torch.cuda probe failed; see environment.json")
    for key in ("modules", "protected_packages"):
        if base[key] != current[key]:
            raise RuntimeError(f"The venv changed accelerator {key}; refusing to declare it ready")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def setup(args):
    target = Path(args.env).absolute()
    repo = Path(args.sam3_repo).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "incomplete", "environment": str(target), "sam3_repo": str(repo),
              "base_python": sys.executable, "stage": "preflight"}
    script = Path(__file__).resolve()
    child_env = os.environ.copy()
    child_env["PIP_USER"] = "false"
    # Avoid caller-supplied pip destinations escaping this new venv.
    for name in ("PIP_TARGET", "PIP_PREFIX", "PIP_ROOT"):
        child_env.pop(name, None)

    def run(stage, command):
        report["stage"] = stage
        log = out / f"{stage}.log"
        print(f"{stage}: see {log}", flush=True)
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.run([str(v) for v in command], stdout=handle, stderr=subprocess.STDOUT,
                                     env=child_env, timeout=1800)
        if process.returncode:
            print("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]), flush=True)
            raise RuntimeError(f"{stage} failed (exit {process.returncode}); see {log}")
        return log

    def probe(python, stage):
        log = run(stage, [python, script, "--probe"])
        for line in reversed(log.read_text(encoding="utf-8", errors="replace").splitlines()):
            if line.startswith("SAM3_ENV_JSON="):
                return json.loads(line[len("SAM3_ENV_JSON="):])
        raise RuntimeError(f"No environment probe result in {log}")

    try:
        if sys.version_info < (3, 12) or sys.platform != "linux":
            raise RuntimeError("Run this installer with the Linux node's Python 3.12+, not LIBERO's Python or Windows")
        if sys.prefix != sys.base_prefix:
            raise RuntimeError("Run with the base node interpreter, e.g. /usr/local/bin/python")
        if target.exists() or target.is_symlink():
            raise RuntimeError(f"Environment already exists: {target}. It was not modified; use --env with a new path.")
        import tomllib
        metadata = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))
        if metadata.get("project", {}).get("name", "").lower() != "sam3" or not (repo / "sam3/model_builder.py").is_file():
            raise RuntimeError("--sam3-repo must be the outer SAM3 checkout containing pyproject.toml and sam3/")
        base = report["base"] = probe(sys.executable, "base-probe")
        if base["errors"] or not base.get("cuda_available"):
            raise RuntimeError("Base torch/torchvision cannot access the PPU through torch.cuda. "
                               "Do not install a public CUDA wheel; send environment.json and base-probe.log.")
        print(f"Reusing {base['modules']['torch']['file']} on {base.get('device_name')}", flush=True)
        report["stage"] = "create-venv"
        venv.EnvBuilder(system_site_packages=True, with_pip=True).create(target)
        python = target / "bin/python"
        inherited = report["inherited"] = probe(python, "venv-probe")
        if inherited["prefix"] == inherited["base_prefix"] or Path(inherited["prefix"]).resolve() != target.resolve():
            raise RuntimeError("Target interpreter is not the requested virtual environment")
        verify_inherited(base, inherited)
        constraints = out / "accelerator-constraints.txt"
        constraints.write_text("".join(f"{name}==={item['version']}\n" for name, item in sorted(base["protected_packages"].items())), encoding="utf-8")
        requirements = out / "sam3-requirements.txt"
        dependencies = metadata["project"]["dependencies"]
        if not isinstance(dependencies, list) or any(not isinstance(dep, str) or "\n" in dep for dep in dependencies):
            raise RuntimeError("Invalid SAM3 dependency metadata")
        # Imports used by the image builder, including transitively imported modules.
        requirements.write_text("\n".join(dependencies + ["einops", "psutil", "pycocotools", "Pillow", "packaging",
                                                          "setuptools>=61,<81", "wheel"]) + "\n", encoding="utf-8")
        pip = [python, "-m", "pip", "--require-virtualenv", "install"]
        plan = out / "pip-plan.json"
        run("resolve-dependencies", pip + ["--dry-run", "--report", plan, "-c", constraints, "-r", requirements])
        pins = approved_pins(json.loads(plan.read_text(encoding="utf-8")))
        report["approved_installs"] = pins
        approved = out / "approved-dependencies.txt"
        approved.write_text("\n".join(pins) + "\n", encoding="utf-8")
        if pins:
            # No second dependency resolution: a transitive torch wheel can never
            # sneak into the real install after the dry-run guard.
            build_pins = [pin for pin in pins if pin.split("==")[0].lower() in ("setuptools", "wheel", "packaging")]
            if build_pins:
                run("install-build-tools", pip + ["--no-deps", "--no-build-isolation", *build_pins])
            run("install-dependencies", pip + ["--no-deps", "--no-build-isolation", "-r", approved])
        run("install-sam3", pip + ["--no-deps", "--no-build-isolation", "-e", repo])
        current = report["installed"] = probe(python, "installed-probe")
        verify_inherited(base, current)
        code = ("import sys, torch, sam3; from pathlib import Path; "
                "from sam3.model_builder import build_sam3_image_model; "
                "from sam3.model.sam3_image_processor import Sam3Processor; "
                "assert Path(sam3.__file__).resolve().parent == Path(sys.argv[1]).resolve() / 'sam3'; "
                "x=torch.ones((2,2), device='cuda'); y=(x @ x).cpu(); "
                "assert y.tolist()==[[2.,2.],[2.,2.]]; print('SAM3 imports and small accelerator matmul: OK')")
        run("sam3-import-check", [python, "-c", code, repo])
        report.update(status="ready", stage="complete", sam3_python=str(python),
                      note="Environment/import/2x2 matmul check only; SAM3 weights and full inference not yet tested.")
        print(f"READY. export SAM3_PYTHON={python}", flush=True)
        return 0
    except Exception as exc:
        report.update(status="error", error=f"{type(exc).__name__}: {exc}")
        print(report["error"], file=sys.stderr, flush=True)
        return 2
    finally:
        write_json(out / "environment.json", report)
        print(f"Environment report: {out / 'environment.json'}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="/mnt/cpfs/users/mrq/envs/sam3")
    parser.add_argument("--sam3-repo", default="/mnt/cpfs/users/mrq/package/sam3")
    parser.add_argument("--out", default="outputs/sam3-env-001")
    parser.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.probe:
        print("SAM3_ENV_JSON=" + json.dumps(snapshot(), ensure_ascii=False), flush=True)
        return 0
    try:
        return setup(args)
    except (OSError, ValueError) as exc:
        print(f"Cannot start setup: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
