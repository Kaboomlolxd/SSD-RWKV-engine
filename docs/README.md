# Documentation index

[`PROJECT_STATUS.md`](PROJECT_STATUS.md) is the current source of truth for
recommendations, measured CPU results, capability boundaries, and the work
queue. [`REPOSITORY_AUDIT.md`](REPOSITORY_AUDIT.md) is the current source of
truth for the architecture/path audit and production-readiness decision. The
full remediation record is [`ENGINE_EVALUATION_REPORT.md`](ENGINE_EVALUATION_REPORT.md).

## Read by need

| Question | Document |
|---|---|
 | What should I use now? | [`PROJECT_STATUS.md`](PROJECT_STATUS.md) |
| How do I import and prepare a model? | [`MODEL_IMPORT.md`](MODEL_IMPORT.md) |
| What is implemented, tested, and production-fit by path? | [`REPOSITORY_AUDIT.md`](REPOSITORY_AUDIT.md) |
| Which backend is supported? | [`BACKENDS.md`](BACKENDS.md) |
| Which RAM/speed tier should I choose? | [`PRESETS.md`](PRESETS.md) |
| How does per-layer streaming work? | [`V1_STREAMING.md`](V1_STREAMING.md) and [`SSD_STREAMING_FRONTIER.md`](SSD_STREAMING_FRONTIER.md) |
| How do I run a benchmark? | [`../bench/README.md`](../bench/README.md) |
| How should results be reported? | [`../bench/RESULTS.md`](../bench/RESULTS.md) |
| Is a GPU/accelerator supported? | [`ACCELERATORS.md`](ACCELERATORS.md) |
| What bugs were fixed? | [`BASELINE_BUGS.md`](BASELINE_BUGS.md) and [`../CHANGELOG.md`](../CHANGELOG.md) |
| What remains open? | [`GOALS_AND_MILESTONES.md`](GOALS_AND_MILESTONES.md) and [`IDEAS.md`](IDEAS.md) |
| What SSD deployment ideas are active? | [`SSD_EXPLOITATION.md`](SSD_EXPLOITATION.md), [`SSD_HEALTH.md`](SSD_HEALTH.md), and [`OPPORTUNITY_ATLAS.md`](OPPORTUNITY_ATLAS.md) |
| What architecture claims are validated? | [`RESEARCH_AND_ARCHITECTURE.md`](RESEARCH_AND_ARCHITECTURE.md) |

## Document roles

- `PROJECT_STATUS.md` — current release posture and evidence.
- `REPOSITORY_AUDIT.md` — architecture map, path progress, evidence limits,
  repository hygiene, and production-readiness audit.
- `ENGINE_EVALUATION_REPORT.md` — full production-readiness evaluation,
  remediation record, benchmarks, and acceptance gates.
- `BACKENDS.md` — backend contracts, build prerequisites, and readiness.
- `PRESETS.md` — one-place reference for residency tiers and environment
  variables.
- `V1_STREAMING.md` / `SSD_STREAMING_FRONTIER.md` — implementation and
  instrumentation details for the storage path.
- `BENCH.md` — detailed historical benchmark notes; use `bench/README.md` for
  the active command catalog.
- `BASELINE_BUGS.md` — historical F-tier audit and fixes.
- `THROUGHPUT_PLAN.md`, `GOALS_AND_MILESTONES.md`, and `IDEAS.md` — roadmap and
  backlog, not current support claims.
- `ACCELERATORS.md` — explicitly hardware-gated work.

## Navigation rules

Keep one current status table in `PROJECT_STATUS.md`. Put new measurements in
`bench/results/` with enough metadata to reproduce them, then update the status
document if the recommendation changes. Keep superseded documents in
[`../archive/`](../archive/) with their date and scope rather than mixing old
numbers into the current release guidance.

Root entry points are [`../README.md`](../README.md),
[`../ENGINE.md`](../ENGINE.md), and [`../REPO_LAYOUT.md`](../REPO_LAYOUT.md).
