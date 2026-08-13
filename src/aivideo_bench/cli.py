"""Small offline-first command line for AIVideo Bench."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from .mcp_client import StreamableHTTPConfig, StreamableHTTPMCPClient
from .private_pack import validate_private_pack
from .results import render_markdown, summarize_results
from .standard import coverage_summary, standard, validate_standard


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write(value: str, output: Path | None) -> None:
    if output is None:
        sys.stdout.write(value)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(value)


def _write_json(value: Any, output: Path | None) -> None:
    _write(json.dumps(value, indent=2, sort_keys=True) + "\n", output)


def _result_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text().strip()
    if not text:
        raise ValueError("result file is empty")
    if text.startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError("result JSON must be a list")
        return value
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("every JSONL result row must be an object")
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aivideo-bench",
        description="Inspect and operate the AIVideo media-agent benchmark.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    standard_command = commands.add_parser(
        "standard", help="print the public 100-task standard"
    )
    standard_command.add_argument("--output", type=Path)

    validate_command = commands.add_parser(
        "validate", help="validate the public standard without network calls"
    )
    validate_command.add_argument("--output", type=Path)

    coverage_command = commands.add_parser(
        "coverage", help="summarize domain and difficulty coverage"
    )
    coverage_command.add_argument("--output", type=Path)

    pack_command = commands.add_parser(
        "pack-validate", help="validate an owner-only sealed task pack"
    )
    pack_command.add_argument("--pack", type=Path, required=True)
    pack_command.add_argument(
        "--signing-key-env", default="AIVIDEO_BENCH_PACK_SIGNING_KEY"
    )
    pack_command.add_argument("--output", type=Path)

    report_command = commands.add_parser(
        "report", help="render one complete 300-cell result matrix"
    )
    report_command.add_argument("--rows", type=Path, required=True)
    report_command.add_argument("--model", required=True)
    report_command.add_argument("--provider", required=True)
    report_command.add_argument("--format", choices=("markdown", "json"), default="markdown")
    report_command.add_argument("--output", type=Path)

    snapshot_command = commands.add_parser(
        "mcp-snapshot",
        help="authenticate and snapshot the real MCP tool surface without tool calls",
    )
    snapshot_command.add_argument("--endpoint", required=True)
    snapshot_command.add_argument("--credential-id", required=True)
    snapshot_command.add_argument("--token-env", default="AIVIDEO_MCP_TOKEN")
    snapshot_command.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "standard":
            _write_json(standard(), args.output)
        elif args.command == "validate":
            errors = validate_standard(standard())
            result = {"valid": not errors, "errors": errors}
            _write_json(result, args.output)
            return int(bool(errors))
        elif args.command == "coverage":
            _write_json(coverage_summary(), args.output)
        elif args.command == "pack-validate":
            raw_key = os.environ.get(args.signing_key_env)
            if raw_key is None:
                raise ValueError(
                    "signing key environment variable is unset: "
                    f"{args.signing_key_env}"
                )
            errors = validate_private_pack(
                _read_json(args.pack), signing_key=raw_key.encode()
            )
            result = {"valid": not errors, "errors": errors}
            _write_json(result, args.output)
            return int(bool(errors))
        elif args.command == "report":
            summary = summarize_results(
                _result_rows(args.rows), model=args.model, provider=args.provider
            )
            if args.format == "json":
                _write_json(summary, args.output)
            else:
                _write(render_markdown(summary), args.output)
        elif args.command == "mcp-snapshot":
            token = os.environ.get(args.token_env)
            if token is None:
                raise ValueError(f"MCP token environment variable is unset: {args.token_env}")
            config = StreamableHTTPConfig(
                endpoint=args.endpoint,
                credential_id=args.credential_id,
                headers=(("Authorization", f"Bearer {token}"),),
            )
            client = StreamableHTTPMCPClient(config)
            try:
                surface = client.initialize()
                client.assert_tool_surface_unchanged()
            finally:
                client.close()
            if any(row.get("method") == "tools/call" for row in client.transcript):
                raise RuntimeError("MCP snapshot unexpectedly invoked a tool")
            _write_json(surface, args.output)
        else:
            raise AssertionError(f"unhandled command: {args.command}")
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0
