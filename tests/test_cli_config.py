"""CLI and config smoke tests (M4)."""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.tools.make_synthetic_pack import create_synthetic_pack
from app.engine_args import add_engine_args, build_engine_config


def test_config_yaml_load(tmp_path: Path) -> None:
    pack = create_synthetic_pack(tmp_path / "pack")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.dump(
            {
                "pack_dir": str(pack),
                "backend": "synthetic",
                "mode": "streaming",
                "max_tokens": 4,
                "device": "cpu",
            }
        ),
        encoding="utf-8",
    )
    cfg = EngineConfig.from_file(cfg_path)
    assert cfg.mode == "streaming"
    assert cfg.max_tokens == 4


def test_config_yaml_cache_budget_and_power(tmp_path: Path) -> None:
    pack = create_synthetic_pack(tmp_path / "pack")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.dump(
            {
                "pack_dir": str(pack),
                "backend": "synthetic",
                "mode": "streaming",
                "cache_budget_gb": 0.25,
                "power_percent": 75,
            }
        ),
        encoding="utf-8",
    )
    cfg = EngineConfig.from_file(cfg_path)
    assert cfg.cache_budget_gb == 0.25
    assert cfg.power_percent == 75


def test_config_yaml_sampling_defaults(tmp_path: Path) -> None:
    pack = create_synthetic_pack(tmp_path / "pack")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "pack_dir": str(pack),
                "backend": "synthetic",
                "temperature": 0.65,
                "top_p": 0.82,
                "seed": 123,
                "greedy": False,
            }
        ),
        encoding="utf-8",
    )

    cfg = EngineConfig.from_file(cfg_path)
    assert cfg.temperature == 0.65
    assert cfg.top_p == 0.82
    assert cfg.seed == 123
    assert cfg.greedy is False


def test_sampling_environment_overrides(monkeypatch, tmp_path: Path) -> None:
    pack = create_synthetic_pack(tmp_path / "pack")
    cfg = EngineConfig.from_mapping(
        {
            "pack_dir": str(pack),
            "temperature": 0.9,
            "top_p": 1.0,
            "seed": None,
        }
    )
    monkeypatch.setenv("RWKV_SSD_TOP_P", "0.73")
    monkeypatch.setenv("RWKV_SSD_SEED", "987")

    merged = EngineConfig.merge_env(cfg)
    assert merged.top_p == 0.73
    assert merged.seed == 987


def test_config_relative_paths_and_string_booleans(tmp_path: Path) -> None:
    pack = create_synthetic_pack(tmp_path / "pack")
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    cfg_path = config_dir / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "pack_dir": "../pack",
                "greedy": "false",
                "prefetch_enabled": "off",
                "stream_layer_cache": "true",
            }
        ),
        encoding="utf-8",
    )

    cfg = EngineConfig.from_file(cfg_path)
    assert cfg.pack_dir == pack.resolve()
    assert cfg.greedy is False
    assert cfg.prefetch_enabled is False
    assert cfg.stream_layer_cache is True


def test_config_packed_residency_axes(tmp_path: Path) -> None:
    pack = create_synthetic_pack(tmp_path / "pack")
    cfg = EngineConfig.from_mapping(
        {
            "pack_dir": str(pack),
            "mode": "streaming",
            "cache_format": "packed",
            "packed_cache_bytes": 123456,
            "prepared_cache_bytes": 654321,
            "residency_policy": "auto",
        }
    )
    assert cfg.cache_format == "packed"
    assert cfg.packed_cache_bytes == 123456
    assert cfg.prepared_cache_bytes == 654321
    assert cfg.residency_policy == "auto"


def test_config_adaptive_residency_controls(tmp_path: Path) -> None:
    pack = create_synthetic_pack(tmp_path / "pack")
    cfg = EngineConfig.from_mapping(
        {
            "pack_dir": str(pack),
            "adaptive_residency": True,
            "adaptive_residency_window": 5,
            "adaptive_residency_min_dwell_tokens": 12,
            "adaptive_residency_hysteresis": 0.25,
            "adaptive_residency_max_changes": 3,
        }
    )
    assert cfg.adaptive_residency is True
    assert cfg.adaptive_residency_window == 5
    assert cfg.adaptive_residency_min_dwell_tokens == 12
    assert cfg.adaptive_residency_hysteresis == 0.25
    assert cfg.adaptive_residency_max_changes == 3


def test_cli_packed_residency_axes(tmp_path: Path) -> None:
    pack = create_synthetic_pack(tmp_path / "pack")
    parser = argparse.ArgumentParser()
    add_engine_args(parser)
    args = parser.parse_args(
        [
            "--model",
            str(pack),
            "--mode",
            "streaming",
            "--cache-format",
            "none",
            "--packed-cache-bytes",
            "4096",
            "--prepared-cache-bytes",
            "8192",
            "--residency-policy",
            "auto",
        ]
    )
    cfg = build_engine_config(args)
    assert cfg.cache_format == "none"
    assert cfg.packed_cache_bytes == 4096
    assert cfg.prepared_cache_bytes == 8192
    assert cfg.residency_policy == "auto"


def test_cli_sampling_overrides(tmp_path: Path) -> None:
    pack = create_synthetic_pack(tmp_path / "pack")
    parser = argparse.ArgumentParser()
    add_engine_args(parser)
    args = parser.parse_args(
        [
            "--model",
            str(pack),
            "--temperature",
            "0.7",
            "--top-p",
            "0.81",
            "--seed",
            "41",
        ]
    )

    cfg = build_engine_config(args)
    assert cfg.temperature == 0.7
    assert cfg.top_p == 0.81
    assert cfg.seed == 41
    # A positive temperature opts a request into sampling unless --greedy was
    # explicitly supplied.
    assert cfg.greedy is False


def test_cli_adaptive_residency_controls(tmp_path: Path) -> None:
    pack = create_synthetic_pack(tmp_path / "pack")
    parser = argparse.ArgumentParser()
    add_engine_args(parser)
    args = parser.parse_args(
        [
            "--model", str(pack),
            "--adaptive-residency",
            "--adaptive-residency-window", "4",
            "--adaptive-residency-min-dwell-tokens", "9",
            "--adaptive-residency-hysteresis", "0.2",
            "--adaptive-residency-max-changes", "1",
        ]
    )
    cfg = build_engine_config(args)
    assert cfg.adaptive_residency is True
    assert cfg.adaptive_residency_window == 4
    assert cfg.adaptive_residency_min_dwell_tokens == 9
    assert cfg.adaptive_residency_hysteresis == 0.2
    assert cfg.adaptive_residency_max_changes == 1


def test_config_and_cli_session_promotion_controls(tmp_path: Path) -> None:
    pack = create_synthetic_pack(tmp_path / "pack")
    mapped = EngineConfig.from_mapping(
        {
            "pack_dir": str(pack),
            "session_promotion": True,
            "session_expected_tokens": 128,
            "session_promotion_bytes": 1048576,
            "session_promotion_policy": "highest_stall",
        }
    )
    assert mapped.session_promotion is True
    assert mapped.session_expected_tokens == 128
    assert mapped.session_promotion_bytes == 1048576
    assert mapped.session_promotion_policy == "highest_stall"

    parser = argparse.ArgumentParser()
    add_engine_args(parser)
    args = parser.parse_args(
        [
            "--model", str(pack),
            "--session-promotion",
            "--session-expected-tokens", "256",
            "--session-promotion-bytes", "2097152",
            "--session-promotion-policy", "benefit_per_byte",
        ]
    )
    cfg = build_engine_config(args)
    assert cfg.session_promotion is True
    assert cfg.session_expected_tokens == 256
    assert cfg.session_promotion_bytes == 2097152
    assert cfg.session_promotion_policy == "benefit_per_byte"


def test_serve_parser_does_not_override_config_defaults(tmp_path: Path) -> None:
    pack = create_synthetic_pack(tmp_path / "pack")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "pack_dir": str(pack),
                "backend": "synthetic",
                "mode": "resident",
                "max_tokens": 7,
                "device": "cpu",
            }
        ),
        encoding="utf-8",
    )
    parser = argparse.ArgumentParser()
    add_engine_args(parser, for_serve=True)
    args = parser.parse_args(["--model", str(pack), "--config", str(cfg_path)])
    cfg = build_engine_config(args)
    assert cfg.backend == "synthetic"
    assert cfg.mode == "resident"
    assert cfg.max_tokens == 7
    assert cfg.device == "cpu"


def test_cli_smoke(synthetic_pack: Path) -> None:
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.cli",
            "--model",
            str(synthetic_pack),
            "--backend",
            "synthetic",
            "--mode",
            "resident",
            "--max-tokens",
            "3",
            "--prompt",
            "x",
            "--log-level",
            "WARNING",
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert r.returncode == 0, r.stderr


def test_default_backend_is_rwkvcpp(tmp_path: Path) -> None:
    pack = create_synthetic_pack(tmp_path / "pack")

    cfg = EngineConfig.from_mapping({"pack_dir": str(pack)})
    assert cfg.backend == "rwkvcpp"

    parser = argparse.ArgumentParser()
    add_engine_args(parser)
    args = parser.parse_args(["--model", str(pack)])
    assert build_engine_config(args).backend == "rwkvcpp"
