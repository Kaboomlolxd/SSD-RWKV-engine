# Local HTTP serving

`app.serve` exposes OpenAI-compatible text and chat completions for local
tools and service-boundary deployments. It supports both the legacy single
engine mode and a bounded spawned-process pool. The default is one worker:

This is an API server, not a browser chat UI. Opening `/` or `/favicon.ico`
returns `404` by design; use `/health`, `/v1/models`, or an OpenAI-compatible
frontend pointed at `http://127.0.0.1:8080/v1`.

```powershell
python -m app.serve --model C:\prepared\model.pack `
  --backend rwkvcpp --mode resident --workers 1 --max-queue 32
```

Use `--workers N` or `RWKV_SSD_WORKERS=N` to request more workers. Each worker
owns an independent `InferenceEngine`, native backend instance, allocator, and
model state. The parent owns the listener, authentication, CORS, request
parsing, global admission queue, aggregate metrics, and worker lifecycle.
`--max-queue` is global; it bounds requests waiting for a worker and the
request timeout covers both queue wait and generation time.

Worker counts are checked against `RWKV_SSD_MAX_WORKERS` and explicit per-worker
RAM/cache budgets before startup. Memory-heavy native models should be
qualified with the real checkpoint and observed per-worker RSS before raising
the count. A partially healthy pool reports `degraded` health while the
remaining workers continue serving.

## Routing and sessions

Requests without a `session_id` are distributed round-robin. A session is
assigned a stable worker so recurrent state remains local between turns. When
state must leave a worker, the parent parks a versioned backend-specific
envelope containing:

- backend kind;
- model/pack fingerprint;
- tokenizer fingerprint;
- serialized state payload;
- last token ID; and
- serialization version.

The parent rejects an envelope whose backend, model, tokenizer, or serialization
version does not match the current pool. State is never restored into a
different model path as a generic byte blob.

Streaming workers send token, completion, metrics, failure, and cancellation
events over IPC. A client disconnect or deadline sends cancellation to the
assigned worker. The parent keeps the worker slot occupied until a terminal
event arrives; an unresponsive worker is terminated and restarted only within
the configured bounded restart policy.

## Request controls and observability

`temperature`, `top_p`, `seed`, and `greedy` are accepted on both text and chat
completion requests. `top_p` must be in `(0, 1]`, temperature must be finite
and non-negative, and seed must be an integer. Sampling is deterministic for a
fixed seed within a backend; cross-backend sampled token identity is not a
parity requirement.

The endpoints are:

- `POST /v1/chat/completions`;
- `POST /v1/completions`;
- `GET /v1/models`;
- `GET /health`; and
- `GET /metrics`.

`/health` reports worker count, healthy capacity, active generations, worker
PIDs/RSS, and degraded state. `/metrics` exposes Prometheus-style counters and
gauges for queue wait, active generations, time-to-first-token, p50/p95
generation latency, cancellations, failures, worker health, and per-worker
RSS. Generation responses also include structured engine metrics when the
backend produces them.

Bind the service to localhost or a protected private network. TLS, external
rate limiting, and multi-host deployment are intentionally outside this local
serving surface.
