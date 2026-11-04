"""Auditable token and monetary accounting; never infer zero cost from missing usage."""

import copy
from decimal import Decimal

from .errors import ExperimentError


TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens",
                "prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "prompt_cache_write_tokens",
                "prompt_cache_write_30m_tokens", "prompt_cache_write_12h_tokens")
PRICE_SNAPSHOT = {
    "schema_version": 1, "model": "deepseek-flash", "currency": "CNY",
    "verified_date": "2026-10-29",
    "source": "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/",
    "per_million_tokens": {
        "off_peak": {"prompt_cache_hit_tokens": "0.02", "prompt_cache_miss_tokens": "1", "completion_tokens": "4"},
        "peak": {"prompt_cache_hit_tokens": "0.04", "prompt_cache_miss_tokens": "2", "completion_tokens": "8"}},
    "method": "Range at published off-peak/peak rates, not an invoice; no holiday or billing-time guess.",
    "note": "Reasoning is included in completion_tokens and is never billed twice. Prices may change."}


OPENAI_PRICE_SNAPSHOT = {
    "schema_version": 2, "model": "gpt-6.1-sol", "currency": "USD",
    "verified_date": "2026-11-02", "service_tier": "default", "cache_ttl": "30m",
    "source": "https://developers.openai.com/api/docs/models/gpt-6.1-sol",
    "cache_source": "https://developers.openai.com/api/docs/guides/prompt-caching",
    "per_million_tokens": {"standard": {"prompt_cache_hit_tokens": "0.10",
        "prompt_cache_miss_tokens": "2", "prompt_cache_write_30m_tokens": "2.50",
        "prompt_cache_write_12h_tokens": "5", "completion_tokens": "10"}},
    "long_context_threshold": 272000, "long_context_input_multiplier": "2",
    "long_context_output_multiplier": "1.5",
    "note": "Standard USD estimate, not an invoice. Reasoning is included in output; no currency conversion."}


def pricing_snapshot(model="deepseek-flash"):
    snapshot = {"deepseek-flash": PRICE_SNAPSHOT, "gpt-6.1-sol": OPENAI_PRICE_SNAPSHOT}.get(model)
    return copy.deepcopy(snapshot)


def token_usage(raw):
    raw = raw if isinstance(raw, dict) else {}
    if "input_tokens" in raw or "output_tokens" in raw:
        inputs = raw.get("input_tokens_details") or {}
        outputs = raw.get("output_tokens_details") or {}
        inputs = inputs if isinstance(inputs, dict) else {}
        outputs = outputs if isinstance(outputs, dict) else {}
        writes = inputs.get("cache_write_tokens_details") or {}
        writes = writes if isinstance(writes, dict) else {}
        raw = {"prompt_tokens": raw.get("input_tokens"), "completion_tokens": raw.get("output_tokens"),
               "total_tokens": raw.get("total_tokens"), "reasoning_tokens": outputs.get("reasoning_tokens"),
               "prompt_cache_hit_tokens": inputs.get("cached_tokens"),
               "prompt_cache_write_tokens": inputs.get("cache_write_tokens"),
               "prompt_cache_write_30m_tokens": writes.get("ttl_30m_tokens"),
               "prompt_cache_write_12h_tokens": writes.get("ttl_12h_tokens")}
        prompt, hit, written = (raw[k] for k in ("prompt_tokens", "prompt_cache_hit_tokens", "prompt_cache_write_tokens"))
        if all(type(v) is int and v >= 0 for v in (prompt, hit, written)) and hit + written <= prompt:
            raw["prompt_cache_miss_tokens"] = prompt - hit - written
        if type(written) is int and written == 0 and not writes:
            raw.update(prompt_cache_write_30m_tokens=0, prompt_cache_write_12h_tokens=0)
    details = raw.get("completion_tokens_details")
    details = details if isinstance(details, dict) else {}
    values = {}
    for key in TOKEN_FIELDS:
        value = details.get(key, raw.get(key)) if key == "reasoning_tokens" else raw.get(key)
        values[key] = value if type(value) is int and value >= 0 else None
    return values


def estimate_cost(tokens, pricing, returned_model, service_tier=None):
    # Only price the exact documented alias; unexpected models stay unpriced.
    if not pricing or returned_model != pricing.get("model"):
        return None
    if pricing.get("currency") == "USD":
        if service_tier not in (None, "default"):
            return None  # Never silently bill Fast/Flex usage at Standard prices.
        prompt, output, total = (tokens[k] for k in TOKEN_FIELDS[:3])
        hit, miss, written = (tokens[k] for k in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "prompt_cache_write_tokens"))
        short, long = (tokens[k] for k in ("prompt_cache_write_30m_tokens", "prompt_cache_write_12h_tokens"))
        # Requests explicitly use 30m caching. If the provider omits the lifetime breakdown,
        # use that frozen request setting, but never invent the total cache-write count.
        if short is None and long is None and written is not None and pricing.get("cache_ttl") == "30m":
            short, long = written, 0
        if (None in (prompt, output, total, hit, miss, written, short, long)
                or prompt + output != total or hit + miss + written != prompt or short + long != written):
            return None
        counted = {**tokens, "prompt_cache_write_30m_tokens": short, "prompt_cache_write_12h_tokens": long}
        long_context = prompt > pricing["long_context_threshold"]
        cost = Decimal(0)
        for key, rate in pricing["per_million_tokens"]["standard"].items():
            multiplier = (pricing["long_context_output_multiplier"] if key == "completion_tokens" else
                          pricing["long_context_input_multiplier"]) if long_context else "1"
            cost += Decimal(counted[key]) * Decimal(rate) * Decimal(multiplier)
        value = float(cost / Decimal(1000000))
        return [value, value]
    hit, miss, prompt, output = (tokens[k] for k in
        ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "prompt_tokens", "completion_tokens"))
    if None in (hit, miss, prompt, output) or hit + miss != prompt:
        return None
    total = tokens["total_tokens"]
    if total is not None and total != prompt + output:
        return None
    return [float(sum(Decimal(tokens[k]) * Decimal(rate) for k, rate in rates.items()) / Decimal(1000000))
            for rates in (pricing["per_million_tokens"]["off_peak"], pricing["per_million_tokens"]["peak"])]


def aggregate_cost(costs, pricing_available=True, currency=None):
    currencies = {c.get("currency", "CNY") for c in costs}
    if currency is not None:
        currencies.add(currency)
    if len(currencies) > 1:
        raise ExperimentError("Cannot add costs in different currencies")
    currency = next(iter(currencies), "CNY")
    unknown = sum(c["unknown_cost_attempts"] for c in costs)
    known = sum(c["priced_attempts"] for c in costs)
    subtotal = [float(sum(Decimal(str(c.get("known_cost_range", c.get("known_cost_range_cny"))[i])) for c in costs)) for i in (0, 1)]
    total = subtotal if unknown == 0 else None
    return {"currency": currency, "priced_attempts": known, "unknown_cost_attempts": unknown,
            "incomplete": unknown > 0, "pricing_available": pricing_available,
            "known_cost_range": subtotal, "total_cost_range": total,
            f"known_cost_range_{currency.lower()}": subtotal, f"total_cost_range_{currency.lower()}": total,
            "note": "Published price estimate, not actual billing; unknown requests are excluded from known subtotal. No currency conversion."}


def account_attempts(attempts, unfinished, pricing):
    rows = []
    currency = pricing.get("currency", "CNY") if pricing else "CNY"
    for event, complete in [(e, True) for e in attempts] + [(e, False) for e in unfinished]:
        tokens = token_usage(event.get("usage"))
        known = all(tokens[k] is not None for k in TOKEN_FIELDS[:3])
        if known:
            known = tokens["prompt_tokens"] + tokens["completion_tokens"] == tokens["total_tokens"]
        cost = estimate_cost(tokens, pricing, event.get("returned_model"), event.get("service_tier"))
        rows.append({**{k: event.get(k) for k in ("step", "attempt", "retry", "started_utc",
                        "finished_utc", "response_created", "response_id", "status_code", "finish_reason",
                        "returned_model", "service_tier", "system_fingerprint", "request_sha256", "request_file", "error")},
                     "completed": complete, "http_latency_seconds": event.get("elapsed_seconds"),
                     **tokens, "usage_complete": known, "currency": currency,
                     "estimated_cost_range": cost, f"estimated_cost_range_{currency.lower()}": cost})
    totals = {k: sum(r[k] for r in rows if r[k] is not None) for k in TOKEN_FIELDS}
    availability = {k: sum(r[k] is not None for r in rows) for k in TOKEN_FIELDS}
    known_usage = sum(r["usage_complete"] for r in rows)
    costs = [{"priced_attempts": int(r["estimated_cost_range"] is not None),
              "unknown_cost_attempts": int(r["estimated_cost_range"] is None), "currency": currency,
              "known_cost_range": r["estimated_cost_range"] or [0., 0.]} for r in rows]
    return rows, {"reported_usage": totals, "usage_available_attempts": known_usage,
                  "usage_field_available_attempts": availability,
                  "usage_incomplete": known_usage != len(rows),
                  "api_cost": aggregate_cost(costs, pricing_available=bool(pricing), currency=currency)}
