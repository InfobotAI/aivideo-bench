# Company scorecard

The company scorecard answers five different decisions without collapsing them
into a number nobody can audit:

1. **Can the model do the work?** The existing 100-point benchmark measures
   verified quality, operational success, exact success, and invalid runs.
2. **How fast is the experience?** It reports end-to-end p50/p95, first model
   response, model inference time, and AIVideo MCP time separately.
3. **What does it cost?** It reports total inference cost, cost per trial, cost
   per operational success, inference calls, and decomposed token usage.
4. **How does it use tools?** It reports per-tool dispatch success, invalid
   arguments, redundant calls, latency, and recovery after errors.
5. **Does it help the business?** It accepts denominator-backed production,
   billing, support, sales, audit, and experiment observations with an explicit
   claim level.

There is deliberately no grand composite. A model can be excellent but too
slow for a live sales demo, cheap but destructive, or reliable in the benchmark
without yet having causal business evidence. Those are different decisions.

## Evidence planes

```text
sealed 300-cell benchmark ──> capability quality and usable success
execution telemetry ────────> speed, inference economics, tools, reliability
production cohort joins ────> adoption, support, revenue, retention, refunds
controlled comparison ─────> causal business impact
```

Missing business data is rendered as `not measured`, never zero. Model failures
and external dependency failures remain visible as separate rates.

## Generate the metric catalog

```bash
aivideo-bench company-metrics --output /secure/path/company-metrics.json
```

The catalog is data-driven. Adding a business KPI is one registry entry with a
stable ID, decision question, unit, direction, and expected source.

## Candidate input

Each private candidate JSON uses
`aivideo-bench-company-candidate-v1` and contains:

- `model` and `provider`;
- the complete 300-row verifier result matrix accepted by `aivideo-bench report`;
- a complete 300-row runtime matrix for latency, token, tool, and incident data;
- zero or more aggregate business observations.

Business observations contain only aggregates:

```json
{
  "metric_id": "task_to_usable_project_rate",
  "value": 0.72,
  "numerator": 72,
  "denominator": 100,
  "window": "2026-07-01/2026-07-31",
  "claim_level": "observed",
  "source": "production_telemetry"
}
```

Do not place raw user events, Discord messages, customer identifiers, project
IDs, prompts, or raw model runs in this public repository.

## Compare candidates

```bash
aivideo-bench company-report \
  --candidate /secure/runs/luna.json \
  --candidate /secure/runs/terra.json \
  --candidate /secure/runs/sol.json \
  --output /secure/reports/company-scorecard.md
```

The executive table shows quality, operational and exact success, p50/p95
latency, total inference cost, cost per usable success, tool success, and human
intervention. It then expands reliability incidents, per-tool performance, and
every registered business question with its denominator and claim level.

## Claim levels

- `observed`: a defined production cohort and time window were measured.
- `directional`: a comparable cohort or before/after reference exists, but the
  result may still be confounded.
- `causal`: a randomized experiment or an equivalent defensible design supports
  attribution.

The report never upgrades these claims automatically. A good benchmark score
does not prove revenue or retention impact.

## Required production join

To move beyond capability scores, production telemetry needs durable,
privacy-reviewed joins among:

- agent task and model-configuration IDs;
- committed project mutations and successful exports;
- inference usage and cost receipts;
- AIVideo tool calls and terminal outcomes;
- aggregate support/refund events and billing outcomes;
- cohort windows for return, retention, and cancellation.

Store hashed or access-controlled user/project identifiers outside this public
repository. Preserve model ID, prompt/config version, MCP surface hash, and
benchmark version so changes are attributable instead of becoming blended
historical averages.

## Minimum comparison set

For a model decision, compare:

1. the proposed model with the sealed current AIVideo MCP surface;
2. the current production model/configuration on the same cells;
3. the same proposed model with a constrained or absent tool surface when
   measuring MCP contribution;
4. expert and scripted baselines used by benchmark calibration.

This distinguishes raw model ability from the value added by AIVideo tools and
from changes to prompts, providers, or tool schemas.

## Recommended operating cadence

- **Every model or prompt change:** run the sealed 300-cell benchmark.
- **Weekly:** inspect latency, cost per usable success, tool regressions, false
  completions, loops, and intervention rates.
- **Monthly:** join model/configuration IDs to aggregate adoption, support,
  refund, revenue, retention, and rework cohorts.
- **Before replacing a production model:** compare at least quality, p95
  latency, cost per operational success, catastrophic incidents, and one
  product-specific business outcome.
