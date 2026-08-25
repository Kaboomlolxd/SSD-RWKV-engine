from pathlib import Path

from bench.bench_streaming_matrix import _percentile, run_scenario, write_results_csv


def test_streaming_matrix_reports_cache_categories(synthetic_pack: Path) -> None:
    row = run_scenario(
        synthetic_pack,
        backend="synthetic",
        checkpoint=None,
        cache_format="prepared",
        io_cap_mbps=0,
        max_tokens=2,
        prompt="matrix",
        device="cpu",
        strategy="cpu bf16",
        packed_cache_bytes=0,
        prepared_cache_bytes=1024 * 1024,
    )
    assert row["cache_format"] == "prepared"
    assert row["prepared_cache_bytes"] > 0
    assert row["tok_s"] > 0


def test_streaming_matrix_reports_repeated_sample_schema(synthetic_pack: Path) -> None:
    row = run_scenario(
        synthetic_pack,
        backend="synthetic",
        checkpoint=None,
        cache_format="none",
        io_cap_mbps=0,
        max_tokens=2,
        prompt="matrix schema",
        device="cpu",
        strategy="cpu bf16",
        packed_cache_bytes=0,
        prepared_cache_bytes=0,
        warmup_tokens=1,
        samples=2,
    )
    assert row["schema_version"] == 2
    assert row["sample_count"] == 2
    assert row["tok_s_median"] == row["tok_s"]
    assert row["tok_s_p95"] >= 0
    assert row["read_ms_per_token"] >= 0
    assert row["staging_ms_per_token"] >= 0
    assert row["compute_ms_per_token"] >= 0
    assert "cache_hits" in row
    assert "cache_evictions" in row
    assert len(row["samples"]) == 2


def test_streaming_matrix_percentile_and_csv_schema(tmp_path: Path) -> None:
    assert abs(_percentile([1.0, 2.0, 3.0, 4.0], 95.0) - 3.85) < 1e-9
    output = {
        "model": "small",
        "backend": "synthetic",
        "device": "cpu",
        "rows": [{"schema_version": 2, "tok_s": 1.5, "samples": []}],
    }
    csv_path = tmp_path / "matrix.csv"
    write_results_csv(output, csv_path)
    text = csv_path.read_text(encoding="utf-8")
    assert "model" in text
    assert "tok_s" in text
    assert "1.5" in text
