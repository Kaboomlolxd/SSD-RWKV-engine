from bench.bench_weight_stationary import run_weight_stationary_benchmark


def test_weight_stationary_benchmark_reports_parity_and_amortization(
    synthetic_pack,
) -> None:
    result = run_weight_stationary_benchmark(
        synthetic_pack,
        ["a", "longer", "xyz"],
        max_tokens=2,
        samples=1,
    )
    assert result["outputs_match"] is True
    assert result["batch_size"] == 3
    assert result["tokens_generated"] == 6
    assert result["batch_weight_sweeps"] == 8
    assert result["independent_weight_sweeps"] == 16
    assert result["layer_load_amortization"] == 2.0
