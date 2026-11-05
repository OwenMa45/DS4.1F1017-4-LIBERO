"""No SDK dependency; transport retries never execute simulator actions."""

import hashlib
import getpass
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path

from .artifacts import utc_now
from .errors import APIError, BudgetExhausted, ExperimentError, InvalidAction
from .protocol import parse_action, parse_actions, request_payload


@dataclass
class HTTPResponse:
    status: int
    body: bytes


def resolve_api_key(api_key=None, prompt=False, provider="deepseek"):
    if provider not in ("deepseek", "openai"):
        raise ExperimentError("Unknown API provider")
    env_name, filename, label = (("OPENAI_API_KEY", "gpt_API.txt", "OpenAI") if provider == "openai" else
                                 ("DEEPSEEK_API_KEY", "ds_API.txt", "DeepSeek"))
    key = api_key if api_key is not None else os.environ.get(env_name, "")
    if api_key is None and not key.strip():
        key_file = Path(__file__).resolve().parent.parent / filename
        try:
            key = key_file.read_text(encoding="utf-8-sig").strip()
        except FileNotFoundError:
            pass
        except OSError:
            raise ExperimentError(f"Cannot read project {filename}; check file permissions") from None
    if not key.strip() and prompt:
        if not sys.stdin.isatty():
            raise ExperimentError(f"Hidden key input requires an interactive terminal; export {env_name} before running")
        try:
            # Never allow getpass's fallback to echoed input.
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                key = getpass.getpass(f"{label} API key (hidden, this run only): ")
        except (EOFError, KeyboardInterrupt, getpass.GetPassWarning) as exc:
            raise ExperimentError("API key input cancelled or hidden input unavailable; no request was sent") from exc
    if not key.strip():
        raise ExperimentError(f"{env_name} is missing or empty in this process. Export it in the current shell, "
                              "or use replay-request --prompt-api-key in an interactive terminal. "
                              f"Alternatively put only the key in project {filename} (Git ignored). No API request was sent.")
    return key.strip()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward the API credential to another origin.


def transport(endpoint, payload, api_key, timeout):
    request = urllib.request.Request(endpoint, data=payload, method="POST", headers={
        "Authorization": "Bearer " + api_key, "Content-Type": "application/json"})
    opener = urllib.request.build_opener(NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            return HTTPResponse(response.status, response.read(16 * 1024 * 1024))
    except urllib.error.HTTPError as exc:
        return HTTPResponse(exc.code, exc.read(16 * 1024 * 1024))


class Client:
    def __init__(self, cfg, store, api_key=None, send=transport, sleep=time.sleep):
        self.cfg, self.store, self.send, self.sleep = cfg, store, send, sleep
        self.api_key = resolve_api_key(api_key, provider=cfg.api_provider)
        self.attempts = 0

    def complete(self, payload, step):
        encoded = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
        request_hash = hashlib.sha256(encoded).hexdigest()
        request_file = self.store.request(step, encoded) if hasattr(self.store, "request") else None
        for retry in range(self.cfg.max_retries + 1):
            if self.attempts >= self.cfg.max_http_requests:
                raise BudgetExhausted("HTTP request budget exhausted")
            self.attempts += 1
            started = time.perf_counter()
            event = {"event": "api_attempt", "step": step, "attempt": self.attempts,
                     "started_utc": utc_now(),
                     "retry": retry, "request_sha256": request_hash, "status_code": None,
                     "usage": None, "requested_model": self.cfg.model}
            event["request_file"] = request_file
            self.store.event({"event": "api_started", "step": step, "attempt": self.attempts,
                              "retry": retry, "requested_model": self.cfg.model, "request_file": request_file,
                              "request_sha256": request_hash, "started_utc": event["started_utc"]})
            print(f"API step={step} attempt={self.attempts} retry={retry} waiting (timeout={self.cfg.timeout_seconds}s)", flush=True)
            response = None
            network_error = None
            try:
                response = self.send(self.cfg.endpoint, encoded, self.api_key, self.cfg.timeout_seconds)
            except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError, OSError) as exc:
                network_error = str(exc).replace(self.api_key, "[REDACTED]")
                event["error"] = network_error
                event["error_type"] = type(exc).__name__
                cause = getattr(exc, "reason", None)
                if isinstance(cause, BaseException):
                    event["error_cause_type"] = type(cause).__name__
            except KeyboardInterrupt:
                event.update(error="interrupted", elapsed_seconds=time.perf_counter() - started,
                             finished_utc=utc_now())
                self.store.event(event)
                raise
            event["elapsed_seconds"] = time.perf_counter() - started
            event["finished_utc"] = utc_now()
            data = None
            if response is not None:
                event["status_code"] = response.status
                # Response bodies can contain third-party diagnostic text; redact the credential defensively.
                body = response.body.decode("utf-8", errors="replace").replace(self.api_key, "[REDACTED]")
                event["response_file"] = self.store.response(self.attempts, body)
                try:
                    data = json.loads(body)
                except ValueError:
                    pass
                if isinstance(data, dict):
                    if not 200 <= response.status < 300 and isinstance(data.get("error"), dict):
                        # Keep actionable server diagnostics in the result/summary, not just the raw body.
                        event["api_error"] = {}
                        for field in ("message", "type", "param", "code"):
                            value = data["error"].get(field)
                            if isinstance(value, str):
                                value = " ".join(value.replace(self.api_key, "[REDACTED]").split())[:1000]
                            if value is None or isinstance(value, str):
                                event["api_error"][field] = value
                    event["returned_model"] = data.get("model")
                    event["response_id"] = data.get("id")
                    event["response_created"] = data.get("created_at", data.get("created"))
                    event["usage"] = data.get("usage")
                    event["system_fingerprint"] = data.get("system_fingerprint")
                    event["service_tier"] = data.get("service_tier")
                    if self.cfg.api_provider == "openai":
                        event["finish_reason"] = response_finish(data)
                    choices = data.get("choices")
                    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                        event["finish_reason"] = choices[0].get("finish_reason")
            self.store.event(event)
            usage = event.get("usage") if isinstance(event.get("usage"), dict) else {}
            print(f"API attempt={self.attempts} elapsed={event['elapsed_seconds']:.3f}s "
                  f"status={event['status_code']} finish={event.get('finish_reason')} "
                  f"total_tokens={usage.get('total_tokens', 'unknown')}", flush=True)
            if response is not None and 200 <= response.status < 300:
                if not isinstance(data, dict):
                    raise InvalidAction("API returned non-JSON content on a successful HTTP response")
                return data
            retryable = network_error is not None or (response is not None and
                         (response.status == 429 or 500 <= response.status <= 599))
            if not retryable or retry == self.cfg.max_retries:
                detail = "transport error" if response is None else f"HTTP {response.status}"
                if network_error is not None:
                    detail += f" ({event['error_type']}): " + " ".join(network_error.split())[:1000]
                if event.get("api_error"):
                    detail += ": " + json.dumps(event["api_error"], ensure_ascii=False)
                raise APIError(f"{detail}; see redacted attempt log and response body")
            backoff_start = time.perf_counter()
            try:
                self.sleep(2 ** (retry + 1))
            finally:
                self.store.event({"event": "retry_backoff", "step": step, "attempt": self.attempts,
                                  "elapsed_seconds": time.perf_counter() - backoff_start})
        raise AssertionError("unreachable")


def response_finish(reply):
    status = reply.get("status")
    if status == "completed":
        return "stop"
    details = reply.get("incomplete_details") or {}
    if status == "incomplete" and isinstance(details, dict) and details.get("reason") == "max_output_tokens":
        return "length"
    return status or "missing_status"


def response_content(cfg, reply):
    if cfg.api_provider == "openai":
        finish = response_finish(reply)
        if reply.get("status") in ("failed", "cancelled", "queued", "in_progress"):
            raise APIError("Responses API did not complete; see the saved response")
        output = reply.get("output")
        if not isinstance(output, list):
            raise InvalidAction("Responses API has no output array")
        texts = []
        for item in output:
            if not isinstance(item, dict):
                raise InvalidAction("Malformed Responses output item")
            if item.get("type") == "reasoning":
                continue
            if item.get("type") != "message" or item.get("role") != "assistant":
                raise InvalidAction("Unexpected non-message output; this policy has no tools")
            if item.get("status") not in (None, "completed") and finish == "stop":
                raise InvalidAction("Incomplete assistant message")
            parts = item.get("content")
            if not isinstance(parts, list):
                raise InvalidAction("Malformed assistant message content")
            for part in parts:
                if not isinstance(part, dict) or part.get("type") != "output_text" or not isinstance(part.get("text"), str):
                    raise InvalidAction("Refused or non-text assistant output")
                texts.append(part["text"])
        return "".join(texts), finish
    try:
        choice = reply["choices"][0]
        return choice["message"]["content"], choice["finish_reason"]
    except (KeyError, IndexError, TypeError) as exc:
        raise InvalidAction("API response has no complete assistant choice") from exc


class ModelPolicy:
    def __init__(self, cfg, task, client, store):
        self.cfg, self.task, self.client, self.store = cfg, task, client, store
        self.visual_history = []
        self.previous_decision = None
        self.sam3 = None
        if cfg.sam3:
            from .sam3_tools import Sam3Tool
            self.sam3 = Sam3Tool(cfg.sam3, store)

    def act(self, obs, history, step, frame_refs):
        started = time.perf_counter()
        event = {"event": "decision", "step": step, "frames": frame_refs,
                 "proprioception": obs.state, "recent_history": history, "valid": False}
        try:
            feedback = None
            if self.cfg.grasp_feedback:
                from .feedback import decision_feedback
                feedback = decision_feedback(obs, step, self.previous_decision, self.cfg.control_freq)
                event["control_feedback"] = feedback
            event["visual_history_frames"] = [{"step": s, "frames": refs}
                                               for s, _, refs in self.visual_history]
            payload = request_payload(self.cfg, self.task, obs, history, step,
                [(s, o) for s, o, _ in self.visual_history], feedback)
            if self.sam3 is None:
                reply = self.client.complete(payload, step)
            else:
                from .sam3_tools import complete_with_sam3
                reply = complete_with_sam3(self.client, payload, obs, step, self.sam3)
            content, finish = response_content(self.cfg, reply)
            event.update(finish_reason=finish, content_characters=len(content) if isinstance(content, str) else None)
            v2 = self.cfg.protocol_version == "ds-libero-direct-v2"
            action = (parse_actions(content, finish, min(self.cfg.action_horizon, self.cfg.max_steps - step))
                      if v2 else parse_action(content, finish))
            event.update(valid=True, returned_model=reply.get("model"))
            event["actions" if v2 else "action"] = action
            if self.cfg.grasp_feedback:
                self.previous_decision = (step, obs, action)
            if self.cfg.visual_history:
                self.visual_history.append((step, obs, frame_refs))
                self.visual_history = self.visual_history[-self.cfg.visual_history:]
            return action
        except ExperimentError as exc:
            event.update(error=str(exc), error_status=exc.status)
            raise
        except KeyboardInterrupt:
            event.update(error="interrupted", error_status="interrupted")
            raise
        finally:
            event["elapsed_seconds"] = time.perf_counter() - started
            self.store.event(event)
