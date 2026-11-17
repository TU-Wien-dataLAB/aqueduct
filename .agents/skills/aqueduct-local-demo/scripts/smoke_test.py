#!/usr/bin/env python3
"""Smoke-test the local demo's OIDC login and token-create page."""

import os
import re
import sys

import httpx

WEB_URL = os.environ.get("DEMO_WEB_URL", "http://localhost:8000")
DEX_URL = os.environ.get("DEMO_DEX_URL", "http://localhost:5556")
EMAIL = os.environ.get("DEMO_EMAIL", "you@example.com")
PASSWORD = os.environ.get("DEMO_PASSWORD", "1234")


def _capture(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text)
    if not match:
        raise RuntimeError(f"{label} not found")
    return match.group(1)


def main() -> int:
    with httpx.Client(follow_redirects=True, timeout=30) as client:
        login_page = client.get(f"{WEB_URL}/oidc/authenticate/")
        login_page.raise_for_status()

        login_action = _capture(r'action="([^"]+)"', login_page.text, "Dex login form").replace(
            "&amp;", "&"
        )
        login_response = client.post(
            f"{DEX_URL}{login_action}", data={"login": EMAIL, "password": PASSWORD}
        )
        login_response.raise_for_status()

        approval_req = re.search(r'name="req" value="([^"]+)"', login_response.text)
        if approval_req:
            approval_response = client.post(
                str(login_response.url), data={"req": approval_req.group(1), "approval": "approve"}
            )
            approval_response.raise_for_status()

        token_page = client.get(f"{WEB_URL}/aqueduct/management/tokens/create/")
        token_page.raise_for_status()
        if "Create New API Key" not in token_page.text:
            raise RuntimeError("token-create page did not render")

    print("Smoke test passed: OIDC login and token-create page work.")  # noqa: T201
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Smoke test failed: {exc}", file=sys.stderr)  # noqa: T201
        raise SystemExit(1) from exc
