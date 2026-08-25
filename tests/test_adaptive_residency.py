from rwkv_ssd.runtime.adaptive_residency import AdaptiveResidencyController


def _observe_pair(controller, current_cost, candidate_cost, *, tokens=1) -> None:
    controller.observe("packed", read_ms=current_cost, tokens=tokens)
    controller.observe("prepared", read_ms=candidate_cost, tokens=0)


def test_adaptive_residency_waits_for_minimum_dwell() -> None:
    controller = AdaptiveResidencyController(
        initial_format="packed", min_dwell_tokens=4, hysteresis=0.1
    )
    for _ in range(3):
        _observe_pair(controller, 10, 5)
    assert controller.maybe_retier() is None
    _observe_pair(controller, 10, 5)
    assert controller.maybe_retier().new_format == "prepared"


def test_adaptive_residency_hysteresis_rejects_small_difference() -> None:
    controller = AdaptiveResidencyController(
        initial_format="packed", min_dwell_tokens=1, hysteresis=0.15
    )
    _observe_pair(controller, 10, 9)
    assert controller.maybe_retier() is None


def test_adaptive_residency_switches_on_sustained_cost_ratio() -> None:
    controller = AdaptiveResidencyController(
        initial_format="packed", window=3, min_dwell_tokens=3, hysteresis=0.2
    )
    for _ in range(3):
        _observe_pair(controller, 12, 6)
    decision = controller.maybe_retier()
    assert decision is not None
    assert (decision.old_format, decision.new_format) == ("packed", "prepared")
    assert decision.observed_costs == {"packed": 12.0, "prepared": 6.0}


def test_adaptive_residency_honors_change_limit_and_explicit_format() -> None:
    controller = AdaptiveResidencyController(
        initial_format="packed", min_dwell_tokens=1, max_changes=1
    )
    _observe_pair(controller, 10, 2)
    assert controller.maybe_retier().new_format == "prepared"
    controller.observe("prepared", read_ms=10, tokens=1)
    controller.observe("none", read_ms=1, tokens=0)
    assert controller.maybe_retier(current_format="prepared") is None

    disabled = AdaptiveResidencyController(initial_format="packed", min_dwell_tokens=0)
    _observe_pair(disabled, 10, 1, tokens=0)
    assert disabled.maybe_retier(explicit_cache_format="packed") is None


def test_adaptive_residency_can_probe_adjacent_tier_from_live_components() -> None:
    controller = AdaptiveResidencyController(
        initial_format="packed", min_dwell_tokens=2, hysteresis=0.2
    )
    controller.observe("packed", read_ms=8, compute_ms=2, tokens=1)
    assert controller.maybe_retier() is None
    controller.observe("packed", read_ms=9, compute_ms=2, tokens=1)
    decision = controller.maybe_retier()
    assert decision is not None
    assert decision.new_format == "prepared"
