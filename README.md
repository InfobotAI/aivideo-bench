# AIVideo Bench

AIVideo Bench is a 100-point standard for media agents operating the real
AIVideo MCP tool surface.

> Status: release candidate. The public standard and execution framework are
> implemented; no Luna, Terra, Sol, or OpenRouter result is official until the
> private pack, authenticated MCP preflight, and complete 300-cell run pass.

## The score

The benchmark contains exactly 100 task families across ten domains. Every task
is worth one continuous point in `[0, 1]`. An official task score is the mean of
three canonical trials. The benchmark score is the sum of the 100 task means.

A score of `23.4/100` means 23.4 quality-adjusted points—not that precisely
23.4% of tasks were completed. Reports therefore show four separate concepts:

- quality score: continuous partial credit;
- operational success: usable output above the quality and critical thresholds;
- exact success: every scored component is perfect;
- invalid execution: the run cannot claim an official score.

Quality is additive: 60% critical outcomes, 30% task criteria, and 10%
robustness. Constraints, project isolation, and catastrophic failures are
explicit gates rather than hidden score multipliers.

The acceptance target is deliberately demanding: experts must score at least
`90/100`, while the median of a multi-family frontier-model panel must remain at
or below `50/100`. No-op, render-only, and scripted-partial baselines must stay
below `5`, `20`, and `40` points.

## What candidates can use

Candidates receive only the exact authenticated AIVideo MCP tool schemas sealed
for the run. The benchmark does not invent simulator tools. Captions, for
example, use text options exposed by `place_clips`; `add_captions` is rejected as
a synthetic tool.

Candidate cost is limited to LLM inference. Every allowed MCP tool must be
positively classified as zero-media-cost in the signed task pack. Local frame
decoding, audio measurement, hashing, scoring, and reporting are evaluator-side
work and consume zero paid media credits.

## Repository boundary

This public repository contains:

- the 100-task capability registry and scoring policy;
- the strict Streamable HTTP MCP client;
- the bounded OpenRouter candidate loop;
- private-pack sealing and validation code;
- result aggregation and human-readable reporting;
- synthetic protocol and policy tests.

It never contains private prompts, project IDs, fixtures, references, answer
keys, tolerances, signing keys, credentials, or raw model runs. Private packs
and execution artifacts belong in separate access-controlled storage.

## Install and inspect offline

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

aivideo-bench validate
aivideo-bench coverage
python -m pytest
```

Expected coverage is 100 tasks, 100 points, ten tasks per domain, and a
20/30/30/20 foundation/integration/adversarial/frontier distribution.

## Snapshot the real MCP surface

This performs authenticated initialization and two `tools/list` reads. It does
not invoke an MCP tool, a model, or media processing.

```bash
export AIVIDEO_MCP_TOKEN="..."
aivideo-bench mcp-snapshot \
  --endpoint "https://example.aivideo.com/mcp" \
  --credential-id "benchmark-candidate" \
  --output /secure/path/mcp-surface.json
```

The bearer token is committed by hash inside secret-free runtime identity; its
value is never written into the snapshot or transcript.

## Validate a private pack

```bash
export AIVIDEO_BENCH_PACK_SIGNING_KEY="..."
aivideo-bench pack-validate --pack /secure/path/private-pack.json
```

The validator requires all 100 public task IDs, exact blueprint hashes, the
sealed MCP surface, canonical task/tool ordering, a positive zero-media-cost
tool classification, and a valid HMAC.

## Report a complete run

```bash
aivideo-bench report \
  --rows /secure/path/result-rows.jsonl \
  --model "Luna" \
  --provider "Codex" \
  --output /secure/path/report.md
```

Exactly 300 canonical rows are required. The report separates score, success,
invalid executions, domain performance, inference cost, paid-media credits,
and cost-cap compliance.

## Architecture

```text
public standard ──> sealed private pack ──> authenticated MCP snapshot
                                              │
                                              v
                                      bounded candidate loop
                                              │
                                              v
independent verifier evidence ──> continuous task scores ──> 300-row report
```

The system under test and the benchmark are separate repositories. StreamsAPI
owns the AIVideo MCP implementation; this repository owns evaluation policy,
execution contracts, and reproducibility.
