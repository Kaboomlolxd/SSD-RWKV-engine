"""RAM vs tok/s frontier scenarios — Pareto-best configs at each ``z`` budget."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class FrontierScenario:
    """One point on the RAM→speed frontier (best known preset at this ``z`` tier)."""

    id: str
    label: str
    z_target_mb: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    pack_key: str = "lut2"  # lut2 | shadow_sel | tiered_hot3
    frontier: bool = True  # Pareto candidate; False = diagnostic only


def pack_lut2() -> Path:
    return ROOT / "test_model/trinity_eval/trinity_grouped_0.1b"


def pack_shadow_sel() -> Path:
    return ROOT / "test_model/trinity_eval/trinity_safe_0.1b"


def pack_tiered_hot3() -> Path:
    return ROOT / "test_model/trinity_eval/trinity_tiered_hot3_0.1b"


def resolve_pack(key: str) -> Path | None:
    p = {
        "lut2": pack_lut2(),
        "shadow_sel": pack_shadow_sel(),
        "tiered_hot3": pack_tiered_hot3(),
    }.get(key, pack_lut2())
    return p if p.is_dir() else None


def frontier_scenarios() -> list[FrontierScenario]:
    """
      Pareto-best presets ordered by increasing RAM.

      Each tier is the **best implementation** at that budget — not every env combination.
    Subsumed / worse paths live in ``archive/bench/io_ceiling_legacy_scenarios.py``.
    """
    base_mmap = {
        "io_backend": "mmap",
        "mmap_dontneed": False,
        "decode_disk_cache": "auto",
    }
    out: list[FrontierScenario] = [
        FrontierScenario(
            id="F1",
            label="F1 ssd-tier-min",
            z_target_mb="~201",
            kwargs={
                **base_mmap,
                "stream_layer_cache": False,
                "fused_gemm": True,
                "apply_ssd_tier": True,
            },
        ),
        FrontierScenario(
            id="F2",
            label="F2 bounded-fused",
            z_target_mb="~203",
            kwargs={
                **base_mmap,
                "stream_layer_cache": True,
                "fused_gemm": True,
                "apply_bounded_fused": True,
            },
        ),
        FrontierScenario(
            id="F3",
            label="F3 partial-hot3",
            z_target_mb="~262",
            kwargs={
                **base_mmap,
                "stream_layer_cache": False,
                "fused_gemm": True,
                "apply_partial_ssd_tier": True,
            },
        ),
        FrontierScenario(
            id="F4",
            label="F4 partial-hot4",
            z_target_mb="~280",
            kwargs={
                **base_mmap,
                "stream_layer_cache": False,
                "fused_gemm": True,
                "apply_partial_hot4": True,
            },
        ),
        FrontierScenario(
            id="F5",
            label="F5 promote-max",
            z_target_mb="~382",
            kwargs={
                **base_mmap,
                "stream_layer_cache": True,
                "fused_gemm": True,
                "apply_promote_max": True,
            },
        ),
        FrontierScenario(
            id="F6",
            label="F6 resident-all-ram",
            z_target_mb="~full",
            frontier=False,
            kwargs={
                "mode": "resident",
                "skeleton_load": False,
            },
        ),
    ]
    if pack_tiered_hot3().is_dir():
        out.insert(
            3,
            FrontierScenario(
                id="F3t",
                label="F3t tiered-hot3-partial",
                z_target_mb="~262",
                pack_key="tiered_hot3",
                kwargs={
                    **base_mmap,
                    "stream_layer_cache": False,
                    "fused_gemm": True,
                    "apply_partial_ssd_tier": True,
                },
            ),
        )
    if pack_shadow_sel().is_dir():
        out.append(
            FrontierScenario(
                id="F5s",
                label="F5s promote-shadow",
                z_target_mb="~382",
                pack_key="shadow_sel",
                kwargs={
                    **base_mmap,
                    "stream_layer_cache": True,
                    "fused_gemm": True,
                    "apply_stacked": True,
                },
            ),
        )
    out.append(
        FrontierScenario(
            id="Fb",
            label="Fb auto-tier (ram_budget=0.5 GB)",
            z_target_mb="~382",
            pack_key="lut2",
            kwargs={
                **base_mmap,
                "stream_layer_cache": True,
                "fused_gemm": True,
                "ram_budget_gb": 0.5,
            },
        ),
    )
    return out


def diagnostic_scenarios() -> list[FrontierScenario]:
    """Non-frontier baselines (--diagnostics). Not Pareto candidates."""
    return [
        FrontierScenario(
            id="D0",
            label="D0 strict-no-opt",
            z_target_mb="~201",
            frontier=False,
            kwargs={
                "io_backend": "mmap",
                "stream_layer_cache": False,
                "fused_gemm": False,
                "mmap_dontneed": False,
                "decode_disk_cache": "0",
            },
        ),
        FrontierScenario(
            id="D1",
            label="D1 cold-strict",
            z_target_mb="~201",
            frontier=False,
            kwargs={
                "io_backend": "cold",
                "stream_layer_cache": False,
                "fused_gemm": False,
                "mmap_dontneed": False,
                "decode_disk_cache": "0",
            },
        ),
    ]


def _extrapolate_ram_mb(
    ref_mb: float,
    ref_scale_b: float,
    target_scale_b: float,
    model: str,
) -> float | None:
    """Extrapolate RAM from reference model size using a scaling hypothesis."""
    if ref_mb <= 0 or ref_scale_b <= 0 or target_scale_b <= 0:
        return None
    ratio = target_scale_b / ref_scale_b
    if model == "linear":
        return ref_mb * ratio
    if model == "sqrt":
        return ref_mb * (ratio**0.5)
    if model == "log2":
        # +log2 growth above reference (sub-linear)
        import math

        return ref_mb * (1.0 + math.log2(ratio)) if ratio >= 1 else ref_mb * ratio
    if model == "half":
        # Each 2x params -> +50% RAM (between constant and linear)
        import math

        return ref_mb * (ratio**0.5)  # same as sqrt — label separately in docs
    return None


def ram_extrapolation_table(
    ref_scale_b: float,
    resident_z_mb: float | None,
    skeleton_z_mb: float | None,
    stream_full_z_mb: float | None,
    *,
    targets_b: tuple[float, ...] = (0.01, 0.1, 0.5, 1.0, 3.0, 7.0),
) -> dict:
    """Project RAM at other model sizes under several scaling hypotheses."""
    models = ("linear", "sqrt", "log2")
    out: dict = {"ref_scale_b": ref_scale_b, "targets_b": list(targets_b), "models": {}}
    if resident_z_mb is None:
        return out

    for model in models:
        row: dict[str, float | None] = {}
        for b in targets_b:
            row[f"{b}B"] = round(
                _extrapolate_ram_mb(resident_z_mb, ref_scale_b, b, model) or 0, 1
            )
        out["models"][f"resident_{model}"] = row

    if skeleton_z_mb is not None:
        skel: dict[str, float | None] = {}
        for b in targets_b:
            v = _extrapolate_ram_mb(skeleton_z_mb, ref_scale_b, b, "linear")
            skel[f"{b}B"] = round(v or 0, 1)
        out["models"]["stream_min_linear"] = skel

    if stream_full_z_mb is not None:
        full: dict[str, float | None] = {}
        for b in targets_b:
            v = _extrapolate_ram_mb(stream_full_z_mb, ref_scale_b, b, "linear")
            full[f"{b}B"] = round(v or 0, 1)
        out["models"]["stream_full_linear"] = full

    out["best_fit_note"] = (
        "bf16 resident RAM ~ linear in params for same architecture family; "
        "sqrt/log2 shown as alternatives. Measure at 0.01B/0.5B to validate."
    )
    return out


def model_scale_stats(
    pack: Path,
    ckpt: Path | None,
    rows: list[dict],
) -> dict:
    """Pack geometry + measured ``z_mb`` scaling from frontier rows."""
    from rwkv_ssd.runtime.manifest import Manifest

    manifest = Manifest.load(pack)
    meta = manifest.meta
    n_layer = int(meta.get("n_layer", 0) or 0)
    n_embd = int(meta.get("n_embd", 0) or 0)
    vocab = int(meta.get("vocab_size", 0) or 0)

    weights_path = manifest.weights_path
    pack_weights_mb = (
        round(weights_path.stat().st_size / 1e6, 2) if weights_path.is_file() else None
    )
    ckpt_mb = round(ckpt.stat().st_size / 1e6, 2) if ckpt and ckpt.is_file() else None
    bf16_model_mb = ckpt_mb

    def _row(id_prefix: str) -> dict | None:
        for r in rows:
            scen = str(r.get("scenario", ""))
            if scen.startswith(id_prefix) or r.get("frontier_id") == id_prefix:
                return r
        return None

    f1 = _row("F1")
    f5 = _row("F5")
    f6 = _row("F6")
    skeleton_z = float(f1["z_mb"]) if f1 and f1.get("z_mb") is not None else None
    stream_full_z = float(f5["z_mb"]) if f5 and f5.get("z_mb") is not None else None
    resident_z = float(f6["z_mb"]) if f6 and f6.get("z_mb") is not None else None
    n_block = max(n_layer - 1, 0) if n_layer > 0 else 0

    per_block_layer_z_mb = None
    if skeleton_z is not None and stream_full_z is not None and n_block > 0:
        per_block_layer_z_mb = round((stream_full_z - skeleton_z) / n_block, 3)

    resident_tok_s = float(f6["tok_s"]) if f6 and f6.get("tok_s") else None
    stream_best_tok_s = None
    gap_pct = None
    if f5 and f5.get("tok_s"):
        stream_best_tok_s = float(f5["tok_s"])
        if resident_tok_s and resident_tok_s > 0:
            gap_pct = round(100.0 * stream_best_tok_s / resident_tok_s, 1)

    model_scale_b = 0.1
    extrapolated_z_mb_per_1b = (
        round(resident_z / model_scale_b, 1) if resident_z is not None else None
    )

    pack_vs_bf16_ratio = None
    if pack_weights_mb and bf16_model_mb and bf16_model_mb > 0:
        pack_vs_bf16_ratio = round(pack_weights_mb / bf16_model_mb, 3)

    extrapolation = ram_extrapolation_table(
        model_scale_b,
        resident_z,
        skeleton_z,
        stream_full_z,
    )

    f5_vs_f6_pct = gap_pct
    f5_beats_f6 = (
        stream_best_tok_s is not None
        and resident_tok_s is not None
        and stream_best_tok_s >= resident_tok_s
    )

    return {
        "n_layer": n_layer,
        "n_embd": n_embd,
        "vocab_size": vocab,
        "model_scale_b": model_scale_b,
        "pack_codec": meta.get("codec") or meta.get("pack_codec"),
        "pack_weights_mb": pack_weights_mb,
        "checkpoint_mb": ckpt_mb,
        "bf16_model_mb_est": bf16_model_mb,
        "pack_vs_bf16_storage_ratio": pack_vs_bf16_ratio,
        "trinity_storage_win": (
            f"{pack_weights_mb} MB pack / {bf16_model_mb} MB ckpt "
            f"({round(1.0 / pack_vs_bf16_ratio, 1)}x smaller on SSD)"
            if pack_vs_bf16_ratio and pack_vs_bf16_ratio > 0
            else None
        ),
        "skeleton_z_mb": skeleton_z,
        "stream_full_z_mb": stream_full_z,
        "resident_z_mb": resident_z,
        "per_block_layer_z_mb": per_block_layer_z_mb,
        "resident_z_mb_at_scale": resident_z,
        "extrapolated_z_mb_per_1b_params": extrapolated_z_mb_per_1b,
        "ram_extrapolation": extrapolation,
        "resident_tok_s": resident_tok_s,
        "stream_best_tok_s": stream_best_tok_s,
        "f5_tok_s": stream_best_tok_s,
        "f6_tok_s": resident_tok_s,
        "f5_vs_f6_pct": f5_vs_f6_pct,
        "f5_beats_f6": f5_beats_f6,
        "gap_reference": (
            "Use F5 as streaming ceiling at full z; F6 as classic resident .pth load. "
            "When F5>=F6, tok/s gap is at low-RAM tiers (F1-F4), not max RAM."
        ),
        "scaling_note": (
            "z_mb ~ skeleton + per_block_layer_z_mb * block_layers_in_z; "
            "resident_z_mb ~ full bf16 weights (linear in params)."
        ),
    }


def annotate_gap_to_resident(rows: list[dict]) -> None:
    """Add vs_F5 / vs_F6 ratios and tok/s gaps."""
    f5 = next((r for r in rows if r.get("frontier_id") == "F5"), None)
    f6 = next((r for r in rows if r.get("frontier_id") == "F6"), None)
    ref_f5_tok = float(f5["tok_s"]) if f5 and f5.get("tok_s") else None
    ref_f6_tok = float(f6["tok_s"]) if f6 and f6.get("tok_s") else None
    ref_f5_z = f5.get("z_mb") if f5 else None
    ref_f6_z = f6.get("z_mb") if f6 else None

    for r in rows:
        tok = float(r.get("tok_s", 0))
        if ref_f5_tok and ref_f5_tok > 0:
            r["vs_f5"] = round(tok / ref_f5_tok, 3)
            r["gap_to_f5_tok_s"] = round(ref_f5_tok - tok, 2)
        if ref_f6_tok and ref_f6_tok > 0:
            r["vs_f6"] = round(tok / ref_f6_tok, 3)
            r["gap_to_f6_tok_s"] = round(ref_f6_tok - tok, 2)
        if ref_f5_z is not None and r.get("z_mb") is not None:
            r["delta_z_vs_f5_mb"] = round(float(r["z_mb"]) - float(ref_f5_z), 2)
        if ref_f6_z is not None and r.get("z_mb") is not None:
            r["delta_z_vs_f6_mb"] = round(float(r["z_mb"]) - float(ref_f6_z), 2)


def pareto_frontier(rows: list[dict]) -> list[dict]:
    """Return non-dominated streaming rows (excludes F6 resident reference)."""
    valid = [
        r
        for r in rows
        if r.get("z_mb") is not None
        and r.get("tok_s") is not None
        and r.get("frontier_candidate")
        and not str(r.get("scenario", "")).startswith("F6")
    ]
    if not valid:
        return []
    valid.sort(key=lambda r: (r["z_mb"], -r["tok_s"]))
    frontier: list[dict] = []
    best_tok = -1.0
    for r in valid:
        tok = float(r["tok_s"])
        if tok > best_tok:
            frontier.append(r)
            best_tok = tok
    return frontier


def frontier_slopes(frontier_rows: list[dict]) -> list[dict]:
    """Δtok/s per MB between consecutive frontier points."""
    slopes: list[dict] = []
    for i in range(1, len(frontier_rows)):
        a, b = frontier_rows[i - 1], frontier_rows[i]
        dz = float(b["z_mb"]) - float(a["z_mb"])
        dt = float(b["tok_s"]) - float(a["tok_s"])
        slopes.append(
            {
                "from": a.get("scenario", a.get("label")),
                "to": b.get("scenario", b.get("label")),
                "dz_mb": round(dz, 2),
                "dtok_s": round(dt, 3),
                "tok_per_mb": round(dt / dz, 4) if dz > 0 else None,
            }
        )
    return slopes


def format_frontier_table(
    rows: list[dict],
    raw_gbs: float | None = None,
    model_scale: dict | None = None,
) -> str:
    lines: list[str] = []
    lines.append("\n=== RAM -> tok/s frontier (Pareto-best per tier) ===\n")
    if raw_gbs is not None:
        lines.append(f"Raw pack read (sequential mmap): ~{raw_gbs} GB/s\n")
    if model_scale:
        lines.append(
            f"Model: {model_scale.get('model_scale_b')}B class, "
            f"n_layer={model_scale.get('n_layer')}, n_embd={model_scale.get('n_embd')}"
        )
        if model_scale.get("trinity_storage_win"):
            lines.append(f"Trinity SSD: {model_scale.get('trinity_storage_win')}")
        if model_scale.get("per_block_layer_z_mb") is not None:
            lines.append(
                f"Measured 0.1B: skeleton(F1)={model_scale.get('skeleton_z_mb')} MB, "
                f"+{model_scale.get('per_block_layer_z_mb')} MB/block-layer, "
                f"F5={model_scale.get('stream_full_z_mb')} MB, F6={model_scale.get('resident_z_mb')} MB"
            )
        if model_scale.get("f5_beats_f6"):
            lines.append(
                "F5 (Trinity stream+promote) >= F6 (resident .pth) tok/s — "
                "compare F1-F4 to F5 for the streaming gap."
            )
        elif model_scale.get("f5_vs_f6_pct") is not None:
            lines.append(
                f"F5={model_scale.get('f5_tok_s')} vs F6={model_scale.get('f6_tok_s')} tok/s "
                f"({model_scale.get('f5_vs_f6_pct')}% of F6)"
            )
        extrap = model_scale.get("ram_extrapolation") or {}
        models = extrap.get("models") or {}
        if models.get("resident_linear"):
            lines.append("RAM extrap (resident, linear in params):")
            for k, v in models["resident_linear"].items():
                lines.append(f"  {k}: {v} MB")
        if models.get("resident_sqrt"):
            lines.append("RAM extrap (resident, sqrt params):")
            for k, v in models["resident_sqrt"].items():
                lines.append(f"  {k}: {v} MB")
        if models.get("resident_log2"):
            lines.append("RAM extrap (resident, log2 params):")
            for k, v in models["resident_log2"].items():
                lines.append(f"  {k}: {v} MB")
        if models.get("stream_min_linear"):
            lines.append("RAM extrap (F1 min-RAM skeleton, linear):")
            for k, v in models["stream_min_linear"].items():
                lines.append(f"  {k}: {v} MB")
        lines.append("")
    frontier_rows = pareto_frontier([r for r in rows if r.get("on_frontier")])
    dominated = [
        r for r in rows if not r.get("on_frontier") and r.get("frontier_candidate")
    ]
    lines.append(
        f"{'Scenario':<26} {'tok/s':>7} {'z_mb':>7} {'vs_F5':>6} "
        f"{'vs_F6':>6} {'layers':>3} {'pareto':>6}"
    )
    for r in sorted(rows, key=lambda x: (x.get("z_mb") or 0, -x.get("tok_s", 0))):
        pareto = (
            "yes"
            if r.get("on_frontier")
            else ("dom" if r.get("frontier_candidate") else "ref")
        )
        z_mb = r.get("z_mb")
        z_s = f"{z_mb:.1f}" if z_mb is not None else "—"
        vs5 = r.get("vs_f5")
        vs5_s = f"{vs5:.0%}" if vs5 is not None else "—"
        vs6 = r.get("vs_f6")
        vs6_s = f"{vs6:.0%}" if vs6 is not None else "—"
        lines.append(
            f"{r['scenario']:<26} {r['tok_s']:7.2f} {z_s:>7} {vs5_s:>6} "
            f"{vs6_s:>6} {r.get('layers_in_z', 0):3d} {pareto:>6}"
        )
    if frontier_rows:
        lines.append("\nFrontier slope (dtok/s per MB):")
        for s in frontier_slopes(frontier_rows):
            tpm = s.get("tok_per_mb")
            tpm_s = f"{tpm:.4f}" if tpm is not None else "—"
            lines.append(
                f"  {s['from']} -> {s['to']}: +{s['dtok_s']:.2f} tok/s "
                f"for +{s['dz_mb']:.1f} MB ({tpm_s} tok/s/MB)"
            )
    if dominated:
        lines.append(
            "\nDominated frontier candidates (more RAM but not faster - do not deploy):"
        )
        for r in dominated:
            lines.append(
                f"  - {r['scenario']}: {r['tok_s']:.2f} tok/s @ {r['z_mb']:.1f} MB"
            )
    lines.append(
        "\nExpect monotonic tok/s vs z_mb along the frontier. "
        "Off-frontier points are archived or diagnostic."
    )
    lines.append("")
    return "\n".join(lines)
