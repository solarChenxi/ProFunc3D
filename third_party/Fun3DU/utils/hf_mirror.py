"""Adaptive Hugging Face endpoint setup (official vs hf-mirror).

Call ``setup_hf_endpoint()`` before importing ``transformers`` / ``huggingface_hub``,
or at the very start of entry scripts that download models.
"""

from __future__ import annotations

import os
import socket
import urllib.error
import urllib.request
from typing import Optional
from urllib.parse import urlparse

HF_OFFICIAL = "https://huggingface.co"
HF_MIRROR = "https://hf-mirror.com"


def _endpoint_reachable(url: str, timeout: float) -> bool:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= getattr(resp, "status", 200) < 500
    except Exception:
        # Some mirrors reject HEAD; fall back to a short GET of the root.
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return 200 <= getattr(resp, "status", 200) < 500
        except Exception:
            return False


def _host_resolves(url: str) -> bool:
    host = urlparse(url).hostname
    if not host:
        return False
    try:
        socket.getaddrinfo(host, 443)
        return True
    except OSError:
        return False


def setup_hf_endpoint(
    timeout: float = 3.0,
    force_mirror: Optional[bool] = None,
    verbose: bool = True,
) -> str:
    """Choose and set ``HF_ENDPOINT`` for Hugging Face Hub downloads.

    Priority:
    1. Explicit ``force_mirror``
    2. Existing ``HF_ENDPOINT`` / ``FUN3DU_HF_MIRROR=1``
    3. Probe official endpoint; fall back to ``https://hf-mirror.com`` if unreachable
    """
    if force_mirror is True or os.environ.get("FUN3DU_HF_MIRROR", "").strip() in {
        "1",
        "true",
        "True",
        "yes",
    }:
        os.environ["HF_ENDPOINT"] = HF_MIRROR
        if verbose:
            print(f"[hf_mirror] Using mirror (forced): {HF_MIRROR}")
        return HF_MIRROR

    if force_mirror is False:
        os.environ.pop("HF_ENDPOINT", None)
        if verbose:
            print(f"[hf_mirror] Using official (forced): {HF_OFFICIAL}")
        return HF_OFFICIAL

    existing = os.environ.get("HF_ENDPOINT", "").strip()
    if existing:
        if verbose:
            print(f"[hf_mirror] Keeping existing HF_ENDPOINT={existing}")
        return existing

    if _host_resolves(HF_OFFICIAL) and _endpoint_reachable(HF_OFFICIAL, timeout):
        # Leave unset so huggingface_hub uses the default official host.
        if verbose:
            print(f"[hf_mirror] Official endpoint reachable: {HF_OFFICIAL}")
        return HF_OFFICIAL

    os.environ["HF_ENDPOINT"] = HF_MIRROR
    if verbose:
        print(
            f"[hf_mirror] Official endpoint unreachable; "
            f"falling back to {HF_MIRROR}"
        )
    return HF_MIRROR
