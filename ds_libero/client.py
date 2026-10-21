"""No SDK dependency; transport retries never execute simulator actions."""

import hashlib
import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from .errors import APIError, BudgetExhausted, ExperimentError, InvalidAction
from .protocol import parse_action, request_payload


@dataclass
class HTTPResponse:
    status: int
    body: bytes


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
        self.api_key = api_key if api_key is not None else os.environ.get("DEEPSEEK_API_KEY", "")
        if not self.api_key.strip():
            raise ExperimentError("Set DEEPSEEK_API_KEY in the remote shell (never in a file or command argument)")
        self.attempts = 0

    def complete(self, payload, step):
        encoded = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
        request_hash = hashlib.sha256(encoded).hexdigest()
        for retry in range(self.cfg.max_retries + 1):
            if self.attempts >= self.cfg.max_http_requests:
                raise BudgetExhausted("HTTP request budget exhausted")
            self.attempts += 1
            started = time.perf_counter()
            event = {"event": "api_attempt", "step": step, "attempt": self.attempts,
                     "retry": retry, "request_sha256": request_hash, "status_code": None,
                     "usage": None, "requested_model": self.cfg.model}
            response = None
            network_error = None
            try:
                response = self.send(self.cfg.endpoint, encoded, self.api_key, self.cfg.timeout_seconds)
            except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError, OSError) as exc:
                network_error = str(exc).replace(self.api_key, "[REDACTED]")
                event["error"] = network_error
            except KeyboardInterrupt:
                event.update(error="interrupted", elapsed_seconds=time.perf_counter() - started)
                self.store.event(event)
                raise
            event["elapsed_seconds"] = time.perf_counter() - started
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
                    event["returned_model"] = data.get("model")
                    event["response_id"] = data.get("id")
                    event["usage"] = data.get("usage")
            self.store.event(event)
            if response is not None and 200 <= response.status < 300:
                if not isinstance(data, dict):
                    raise InvalidAction("API returned non-JSON content on a successful HTTP response")
                return data
            retryable = network_error is not None or (response is not None and
                         (response.status == 429 or 500 <= response.status <= 599))
            if not retryable or retry == self.cfg.max_retries:
                detail = "transport error" if response is None else f"HTTP {response.status}"
                raise APIError(f"{detail}; see redacted attempt log and response body")
            self.sleep(2 ** (retry + 1))
        raise AssertionError("unreachable")


class ModelPolicy:
    def __init__(self, cfg, task, client, store):
        self.cfg, self.task, self.client, self.store = cfg, task, client, store

    def act(self, obs, history, step, frame_refs):
        started = time.perf_counter()
        event = {"event": "decision", "step": step, "frames": frame_refs,
                 "proprioception": obs.state, "recent_history": history, "valid": False}
        try:
            reply = self.client.complete(request_payload(self.cfg, self.task, obs, history, step), step)
            try:
                choice = reply["choices"][0]
                content = choice["message"]["content"]
                finish = choice["finish_reason"]
            except (KeyError, IndexError, TypeError) as exc:
                raise InvalidAction("API response has no complete assistant choice") from exc
            action = parse_action(content, finish)
            event.update(valid=True, action=action, returned_model=reply.get("model"))
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
