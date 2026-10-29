import contextlib
import getpass
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ds_libero.__main__ import main
from ds_libero.client import Client, resolve_api_key
from ds_libero.errors import ExperimentError
from ds_libero.replay import replay_request
from test_experiment import reply
import test_v2


class ReplayCredentialsTests(unittest.TestCase):
    def source(self, temp):
        root = Path(temp) / "source"
        test_v2.ReplayTests().legacy_episode(root)
        return root

    def test_missing_key_stops_before_artifact_creation_and_http(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {}, clear=True):
            root = self.source(temp)
            out = Path(temp) / "new"
            for missing in ("", "   "):
                with self.subTest(missing=missing), patch.dict(os.environ, {"DEEPSEEK_API_KEY": missing}):
                    with patch("ds_libero.replay.Client") as client, self.assertRaisesRegex(ExperimentError, "missing or empty"):
                        replay_request(root, 1, None, out)
                    client.assert_not_called()
                    self.assertFalse(out.exists())

    def test_cli_reports_actionable_error_without_a_fake_decision_report(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {}, clear=True):
            root = self.source(temp)
            out = Path(temp) / "new"
            error = io.StringIO()
            with contextlib.redirect_stderr(error):
                code = main(["replay-request", str(root), "--step", "1", "--out", str(out)])
            self.assertEqual(code, 2)
            self.assertIn("--prompt-api-key", error.getvalue())
            self.assertFalse(out.exists())

    def test_hidden_prompt_passes_key_only_in_memory_and_records_real_request(self):
        secret = "offline-test-hidden-key-7bc51d"
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {}, clear=True):
            root = self.source(temp)
            out = Path(temp) / "new"
            seen = []
            def send(endpoint, body, key, timeout):
                seen.append(key)
                return reply()
            def client(cfg, store, api_key):
                return Client(cfg, store, api_key=api_key, send=send)
            console = io.StringIO()
            with patch("ds_libero.client.sys.stdin.isatty", return_value=True), \
                    patch("ds_libero.client.getpass.getpass", return_value=secret) as prompt, \
                    patch("ds_libero.replay.Client", side_effect=client), contextlib.redirect_stdout(console):
                report = replay_request(root, 1, None, out, prompt_api_key=True)
            prompt.assert_called_once()
            self.assertEqual(seen, [secret])
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["decision_requests"], 1)
            self.assertEqual(report["http_requests"], 1)
            self.assertNotIn(secret, console.getvalue())
            self.assertNotIn("DEEPSEEK_API_KEY", os.environ)
            for path in out.rglob("*"):
                if path.is_file():
                    self.assertNotIn(secret.encode(), path.read_bytes(), str(path))

    def test_exported_key_is_reused_without_prompt(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "offline-existing"}), \
                patch("ds_libero.client.getpass.getpass") as prompt:
            self.assertEqual(resolve_api_key(prompt=True), "offline-existing")
            prompt.assert_not_called()

    def test_noninteractive_input_and_getpass_fallback_never_read_visible_input(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("ds_libero.client.sys.stdin.isatty", return_value=False), \
                    patch("ds_libero.client.getpass.getpass") as prompt:
                with self.assertRaisesRegex(ExperimentError, "interactive terminal"):
                    resolve_api_key(prompt=True)
                prompt.assert_not_called()
            for error in (getpass.GetPassWarning(), EOFError(), KeyboardInterrupt()):
                with patch("ds_libero.client.sys.stdin.isatty", return_value=True), \
                        patch("ds_libero.client.getpass.getpass", side_effect=error):
                    with self.assertRaisesRegex(ExperimentError, "cancelled or hidden input unavailable"):
                        resolve_api_key(prompt=True)

    def test_dry_run_does_not_require_or_prompt_for_key(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {}, clear=True):
            root = self.source(temp)
            with patch("ds_libero.replay.resolve_api_key") as resolve:
                report = replay_request(root, 1, None, Path(temp) / "new", dry_run=True, prompt_api_key=True)
            resolve.assert_not_called()
            self.assertEqual(report["status"], "prepared")
            self.assertEqual(report["decision_latency_seconds"]["count"], 0)

    def test_existing_output_does_not_prompt_or_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.source(temp)
            with patch("ds_libero.replay.resolve_api_key") as resolve, self.assertRaisesRegex(ExperimentError, "already exists"):
                replay_request(root, 1, None, root, prompt_api_key=True)
            resolve.assert_not_called()

    def test_client_setup_error_is_not_a_decision_latency_sample(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"DEEPSEEK_API_KEY": "offline-test"}):
            root = self.source(temp)
            with patch("ds_libero.replay.Client", side_effect=ExperimentError("setup failed")):
                report = replay_request(root, 1, None, Path(temp) / "new")
            self.assertEqual(report["status"], "configuration_error")
            self.assertEqual(report["decision_requests"], 0)
            self.assertEqual(report["decision_latency_seconds"]["count"], 0)
            self.assertEqual(report["http_requests"], 0)


if __name__ == "__main__":
    unittest.main()
