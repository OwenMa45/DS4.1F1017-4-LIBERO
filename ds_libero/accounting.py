"""Auditable token and monetary accounting; never infer zero cost from missing usage."""

import copy
from decimal import Decimal


TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens",
                "prompt_cache_hit_tokens", "prompt_cache_miss_tokens")
PRICE_SNAPSHOT = {
    "schema_version": 1, "model": "deepseek-flash", "currency": "CNY",
    "verified_date": "2026-10-29",
    "source": "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/",
    "per_million_tokens": {
        "off_peak": {"prompt_cache_hit_tokens": "0.02", "prompt_cache_miss_tokens": "1", "completion_tokens": "4"},
        "peak": {"prompt_cache_hit_tokens": "0.04", "prompt_cache_miss_tokens": "2", "completion_tokens": "8"}},
    "method": "Range at published off-peak/peak rates, not an invoice; no holiday or billing-time guess.",
    "note": "Reasoning is included in completion_tokens and is never billed twice. Prices may change."}


def pricing_snapshot():
    return copy.deepcopy(PRICE_SNAPSHOT)


def token_usage(raw):
    raw = raw if isinstance(raw, dict) else {}
    details = raw.get("completion_tokens_details")
    details = details if isinstance(details, dict) else {}
    values = {}
    for key in TOKEN_FIELDS:
        value = details.get(key, raw.get(key)) if key == "reasoning_tokens" else raw.get(key)
        values[key] = value if type(value) is int and value >= 0 else None
    return values


def estimate_cost(tokens, pricing, returned_model):
    # Only price the exact documented alias; unexpected models stay unpriced.
    if not pricing or returned_model != pricing.get("model"):
        return None
    hit, miss, prompt, output = (tokens[k] for k in
        ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "prompt_tokens", "completion_tokens"))
    if None in (hit, miss, prompt, output) or hit + miss != prompt:
        return None
    total = tokens["total_tokens"]
    if total is not None and total != prompt + output:
        return None
    return [float(sum(Decimal(tokens[k]) * Decimal(rate) for k, rate in rates.items()) / Decimal(1000000))
            for rates in (pricing["per_million_tokens"]["off_peak"], pricing["per_million_tokens"]["peak"])]


def aggregate_cost(costs, pricing_available=True):
    unknown = sum(c["unknown_cost_attempts"] for c in costs)
    known = sum(c["priced_attempts"] for c in costs)
    subtotal = [float(sum(Decimal(str(c["known_cost_range_cny"][i])) for c in costs)) for i in (0, 1)]
    return {"currency": "CNY", "priced_attempts": known, "unknown_cost_attempts": unknown,
            "incomplete": unknown > 0, "pricing_available": pricing_available,
            "known_cost_range_cny": subtotal,
            "total_cost_range_cny": subtotal if unknown == 0 else None,
            "note": "Published off-peak/peak estimate range, not actual billing; unknown requests are excluded from known subtotal."}


def account_attempts(attempts, unfinished, pricing):
    rows = []
    for event, complete in [(e, True) for e in attempts] + [(e, False) for e in unfinished]:
        tokens = token_usage(event.get("usage"))
        known = all(tokens[k] is not None for k in TOKEN_FIELDS[:3])
        if known:
            known = tokens["prompt_tokens"] + tokens["completion_tokens"] == tokens["total_tokens"]
        cost = estimate_cost(tokens, pricing, event.get("returned_model"))
        rows.append({**{k: event.get(k) for k in ("step", "attempt", "retry", "started_utc",
                        "finished_utc", "response_created", "response_id", "status_code", "finish_reason",
                        "returned_model", "system_fingerprint", "request_sha256", "request_file", "error")},
                     "completed": complete, "http_latency_seconds": event.get("elapsed_seconds"),
                     **tokens, "usage_complete": known, "estimated_cost_range_cny": cost})
    totals = {k: sum(r[k] for r in rows if r[k] is not None) for k in TOKEN_FIELDS}
    availability = {k: sum(r[k] is not None for r in rows) for k in TOKEN_FIELDS}
    known_usage = sum(r["usage_complete"] for r in rows)
    costs = [{"priced_attempts": int(r["estimated_cost_range_cny"] is not None),
              "unknown_cost_attempts": int(r["estimated_cost_range_cny"] is None),
              "known_cost_range_cny": r["estimated_cost_range_cny"] or [0., 0.]} for r in rows]
    return rows, {"reported_usage": totals, "usage_available_attempts": known_usage,
                  "usage_field_available_attempts": availability,
                  "usage_incomplete": known_usage != len(rows),
                  "api_cost": aggregate_cost(costs, pricing_available=bool(pricing))}
