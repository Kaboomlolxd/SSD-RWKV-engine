from pathlib import Path

import torch
import pytest

from rwkv_ssd.tools.quant_quality import (
    compare_logits,
    compare_packs,
    compare_state_sequence,
    compare_tensors,
    make_pass_fail_summary,
    main,
)


def test_compare_tensors_reports_error_and_cosine() -> None:
    result = compare_tensors(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 3.0]))
    assert result["elements"] == 2
    assert result["rmse"] > 0
    assert 0 < result["cosine"] < 1


def test_compare_logits_reports_topk_overlap_and_kl() -> None:
    ref = torch.tensor([[4.0, 2.0, 0.0]])
    candidate = torch.tensor([[3.9, 2.1, 0.0]])
    result = compare_logits(ref, candidate, top_k=2)
    assert result["top_k_overlap"] == 1.0
    assert result["min_top_k_overlap"] == 1.0
    assert result["kl_candidate_to_reference"] >= 0
    assert result["max_kl_candidate_to_reference"] >= 0


def test_compare_pack_to_itself_is_exact(synthetic_pack: Path) -> None:
    result = compare_packs(synthetic_pack, synthetic_pack, max_elements=128)
    assert result["aggregate"]["compared"] > 0
    assert result["aggregate"]["complete"] is True
    assert result["aggregate"]["weighted_rmse"] == 0.0


def test_compare_tensors_handles_empty_inputs() -> None:
    result = compare_tensors(torch.empty(0), torch.empty(0))
    assert result == {
        "elements": 0,
        "rmse": 0.0,
        "mean_abs": 0.0,
        "max_abs": 0.0,
        "relative_l2": 0.0,
        "cosine": 1.0,
    }


def test_compare_tensors_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        compare_tensors(torch.zeros(2), torch.zeros(3))


def test_compare_logits_handles_empty_batch_and_threshold_values() -> None:
    result = compare_logits(torch.empty((0, 4)), torch.empty((0, 4)), top_k=2)
    assert result["top_k"] == 2
    assert result["top_k_overlap"] == 1.0
    assert result["kl_candidate_to_reference"] == 0.0

    changed = compare_logits(
        torch.tensor([[8.0, 1.0, 0.0]]),
        torch.tensor([[0.0, 1.0, 8.0]]),
        top_k=1,
    )
    assert changed["top_k_overlap"] == 0.0
    assert changed["kl_candidate_to_reference"] > 0.0

    sequence = compare_logits(
        torch.tensor([[8.0, 1.0, 0.0], [8.0, 1.0, 0.0]]),
        torch.tensor([[8.0, 1.0, 0.0], [0.0, 1.0, 8.0]]),
        top_k=1,
    )
    assert sequence["top_k_overlap"] == 0.5
    assert sequence["min_top_k_overlap"] == 0.0
    assert sequence["max_kl_candidate_to_reference"] >= sequence["kl_candidate_to_reference"]


def test_compare_state_sequence_reports_drift_and_empty_sequence() -> None:
    reference = torch.zeros((3, 2))
    candidate = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    result = compare_state_sequence(reference, candidate)
    assert result["steps"] == 3
    assert result["max_relative_l2"] > 0.0
    assert result["final_relative_l2"] == result["max_relative_l2"]

    empty = compare_state_sequence(torch.empty((0, 2)), torch.empty((0, 2)))
    assert empty["steps"] == 0
    assert empty["max_relative_l2"] == 0.0


def test_compare_packs_reports_missing_shape_and_per_layer_metrics(
    synthetic_pack: Path, tmp_path: Path
) -> None:
    import json
    import shutil

    candidate = tmp_path / "candidate"
    shutil.copytree(synthetic_pack, candidate)
    manifest_path = candidate / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    removed = raw["tensors"].pop()
    raw["tensors"][0]["shape"] = [999]
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    result = compare_packs(
        synthetic_pack,
        candidate,
        max_elements=32,
        per_layer=True,
    )
    aggregate = result["aggregate"]
    assert aggregate["complete"] is False
    assert removed["name"] in aggregate["missing_candidate"]
    assert aggregate["errors"] >= 1
    assert result["per_layer"]


def test_machine_readable_summary_marks_failed_strict_gates() -> None:
    result = {
        "packs": {"aggregate": {"complete": False, "weighted_rmse": 0.2, "max_rmse": 0.3}},
        "logits": {"top_k_overlap": 0.5, "kl_candidate_to_reference": 0.4},
        "state": {"max_relative_l2": 0.7},
    }
    gates = {
        "pack_complete": False,
        "weighted_rmse": False,
        "max_rmse": False,
        "top_k_overlap": False,
        "kl": False,
        "state_relative_l2": False,
    }
    summary = make_pass_fail_summary(result, gates)
    assert summary["pack_completeness"] is False
    assert summary["max_rmse"] == 0.3
    assert summary["weighted_rmse"] == 0.2
    assert summary["min_top_k_overlap"] == 0.5
    assert summary["max_kl"] == 0.4
    assert summary["max_recurrent_state_drift"] == 0.7
    assert summary["passed"] is False


def test_strict_cli_exits_nonzero_when_logit_gates_fail(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    reference = tmp_path / "reference-logits.pt"
    candidate = tmp_path / "candidate-logits.pt"
    torch.save(torch.tensor([[8.0, 1.0, 0.0]]), reference)
    torch.save(torch.tensor([[0.0, 1.0, 8.0]]), candidate)
    monkeypatch.setattr(
        "sys.argv",
        [
            "quant_quality",
            "--reference-logits",
            str(reference),
            "--candidate-logits",
            str(candidate),
            "--top-k",
            "1",
            "--min-top-k-overlap",
            "1.0",
            "--max-kl",
            "0.01",
            "--strict",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    payload = capsys.readouterr().out
    assert '"passed": false' in payload
