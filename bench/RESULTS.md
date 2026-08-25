# Benchmark results policy

`bench/` contains runnable measurements. `bench/results/` contains checked-in
machine-readable outputs when a result is intended to be reproducible. Large
or application-specific reports may live outside this repository, but the
canonical project-status page must link to or summarize them.

## Naming

Use:

```text
<model>_<backend>_<scope>_<date-or-revision>.json
```

Do not overwrite a prior result to make a number look current. Add a new dated
result and update `docs/PROJECT_STATUS.md` when it becomes authoritative.

## Required metadata

Every real-model tok/s result should record checkpoint and pack paths, backend,
device, strategy, CPU thread count, I/O backend, warm-up count, measured token
count, sample count, prompt label, decode-only tok/s, end-to-end wall tok/s,
load time, `z`/provider/native bytes, layer steps, stage timings, and a
correctness or quality result (or explicit `not measured`).

## Comparison rules

- Keep checkpoint, prompt, token count, warm-up, and sampling policy fixed.
- Never compare decode-only rate with reply wall time without labeling it.
- Use at least 16 generated tokens and two samples for a headline claim when
  the model can complete that run. Short smokes are reported separately.
- Do not stack independent optimizations into one claimed speedup.
- A timeout is a result: report the timeout bound instead of inventing tok/s.
