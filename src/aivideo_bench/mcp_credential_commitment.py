"""Secret-free commitments to the actual MCP authentication material."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from .standard import canonical_bytes


CREDENTIAL_HEADER_NAMES = frozenset(
    {"authorization", "api-key", "x-api-key", "x-aivideo-api-key"}
)


def mcp_credential_material_sha256(
    headers: Mapping[str, str] | Sequence[tuple[str, str]],
    *,
    required: bool = True,
) -> str | None:
    """Commit the credential used on the wire without publishing its value."""

    raw_items: list[Any] = (
        list(headers.items()) if isinstance(headers, Mapping) else list(headers)
    )
    normalized: list[tuple[str, str]] = []
    for item in raw_items:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("MCP credential headers are malformed")
        name = str(item[0]).strip().casefold()
        value = str(item[1]).strip()
        if name in CREDENTIAL_HEADER_NAMES:
            if not value or "\n" in value or "\r" in value:
                raise ValueError("MCP credential material is malformed")
            normalized.append((name, value))
    if not normalized and not required:
        return None
    if len(normalized) != 1:
        raise ValueError(
            "MCP execution requires exactly one recognized credential header"
        )
    return hashlib.sha256(
        canonical_bytes(
            {
                "domain": "aivideo-bench-task-mcp-credential-material-v1",
                "credential_header": normalized[0][0],
                "credential_material": normalized[0][1],
            }
        )
    ).hexdigest()
