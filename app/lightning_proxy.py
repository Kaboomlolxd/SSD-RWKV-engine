"""Optional forwarder to a rwkv_lightning OpenAI-compatible server."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


def forward_openai_request(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    *,
    timeout: float = 300.0,
) -> tuple[int, dict[str, Any] | str, dict[str, str]]:
    url = base_url.rstrip("/") + path
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            headers = dict(resp.headers.items())
            try:
                return resp.status, json.loads(body), headers
            except json.JSONDecodeError:
                return resp.status, body, headers
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(body), dict(exc.headers.items())
        except json.JSONDecodeError:
            return exc.code, {"error": body}, dict(exc.headers.items())
