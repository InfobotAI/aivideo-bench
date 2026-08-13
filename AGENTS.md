# AIVideo Bench repository rules

- Never commit private prompts, fixtures, references, project IDs, answer keys,
  verifier parameters, signing keys, credentials, or raw model runs.
- CI and ordinary tests must make zero model-inference, MCP tool, paid-media, or
  paid external calls.
- Official candidates use only an authenticated, sealed AIVideo MCP tool
  surface. Do not add synthetic simulator tools.
- Every candidate-allowed tool must be positively classified as
  zero-media-cost in the sealed private pack.
- A scoring-policy change requires a contract-version change and focused tests
  explaining its effect on partial, operational, exact, and invalid outcomes.
- Never publish a model score unless all 300 canonical cells, execution
  validity, independent verifier evidence, cost receipts, and calibration gates
  pass.
