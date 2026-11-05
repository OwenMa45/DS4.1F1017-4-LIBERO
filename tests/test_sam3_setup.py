"""Installer guards: no network, real pip install, or accelerator required."""

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("sam3_setup", Path(__file__).resolve().parents[1] / "scripts/setup_sam3_env.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


def environment():
    return {"cuda_available": True, "errors": {},
            "modules": {"torch": {"file": "/vendor/torch/__init__.py", "version": "2.7.1+ppu"},
                        "torchvision": {"file": "/vendor/torchvision/__init__.py", "version": "0.22.1+ppu"}},
            "protected_packages": {"torch": {"version": "2.7.1+ppu", "location": "/vendor"}}}


def plan(*names):
    return {"install": [{"metadata": {"name": name, "version": "1.0"}} for name in names]}


class SetupTests(unittest.TestCase):
    def test_complete_pip_plan_rejects_framework_and_transitive_accelerator_wheels(self):
        for name in ("torch", "TorchVision", "torchaudio", "triton", "pytorch_triton",
                     "nvidia-cudnn-cu12", "torch-ppu", "hggc", "flag_gems", "cuda-bindings"):
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "Refusing"):
                setup.approved_pins(plan("ftfy", "einops", name))

    def test_only_resolved_non_accelerator_pins_are_approved(self):
        self.assertEqual(setup.approved_pins(plan("ftfy", "einops", "numpy")), ["ftfy==1.0", "einops==1.0", "numpy==1.0"])
        self.assertEqual(setup.approved_pins(plan()), [])
        malicious = plan("ftfy\n--target=/usr/local")
        with self.assertRaisesRegex(RuntimeError, "Invalid"):
            setup.approved_pins(malicious)

    def test_same_version_shadow_of_vendor_torch_is_rejected(self):
        base = environment()
        current = copy.deepcopy(base)
        setup.verify_inherited(base, current)
        current["modules"]["torch"]["file"] = "/new/env/site-packages/torch/__init__.py"
        with self.assertRaisesRegex(RuntimeError, "changed accelerator"):
            setup.verify_inherited(base, current)

    def test_changed_vendor_distribution_and_missing_device_are_rejected(self):
        base = environment()
        current = copy.deepcopy(base)
        current["protected_packages"]["torch"]["version"] = "2.8.0"
        with self.assertRaisesRegex(RuntimeError, "changed accelerator"):
            setup.verify_inherited(base, current)
        current = {**base, "cuda_available": False}
        with self.assertRaisesRegex(RuntimeError, "probe failed"):
            setup.verify_inherited(base, current)

    def test_existing_env_is_not_modified_and_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            env = root / "existing"
            env.mkdir()
            sentinel = env / "keep.txt"
            sentinel.write_text("existing environment")
            with patch.object(setup.sys, "platform", "linux"), patch.object(setup.sys, "version_info", (3, 12)), \
                 patch.object(setup.sys, "prefix", "/base"), patch.object(setup.sys, "base_prefix", "/base"), \
                 patch.object(setup.venv.EnvBuilder, "create", side_effect=AssertionError("Do not recreate")), \
                 patch.object(setup.subprocess, "run", side_effect=AssertionError("No install")), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = setup.main(["--env", str(env), "--out", str(root / "diagnostics")])
            self.assertEqual(code, 2)
            self.assertEqual(sentinel.read_text(), "existing environment")
            report = json.loads((root / "diagnostics/environment.json").read_text())
            self.assertEqual(report["status"], "error")
            self.assertIn("already exists", report["error"])


if __name__ == "__main__":
    unittest.main()
