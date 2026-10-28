from quantguard.validation.simulate import required_sample_size, simulate


def _rates(n, drop, threshold=2.0):
    return {r.policy: r for r in simulate(n=n, drop_pp=drop, threshold_pp=threshold, trials=2000, seed=1)}


def test_fixed_threshold_false_alarms_at_small_n_and_paired_tolerance_cuts_them() -> None:
    rates = _rates(100, 0.0)
    assert rates["fixed-threshold"].fail_rate > 0.10
    assert rates["stderr-paired"].fail_rate < rates["fixed-threshold"].fail_rate / 3


def test_paired_policy_still_detects_a_large_drop() -> None:
    assert _rates(500, 5.0)["stderr-paired"].fail_rate > 0.9


def test_independent_policy_is_conservative() -> None:
    rates = _rates(500, 4.0)
    assert rates["stderr-independent"].fail_rate < rates["stderr-paired"].fail_rate


def test_rates_sum_to_one_and_are_reproducible() -> None:
    for item in simulate(n=200, drop_pp=2.0, threshold_pp=1.0, trials=500, seed=3):
        assert abs(item.fail_rate + item.warn_rate + item.pass_rate - 1.0) < 1e-9
    a = simulate(n=200, drop_pp=2.0, threshold_pp=1.0, trials=500, seed=3)
    b = simulate(n=200, drop_pp=2.0, threshold_pp=1.0, trials=500, seed=3)
    assert a == b


def test_required_sample_size_grows_as_threshold_shrinks() -> None:
    assert required_sample_size(1.0) > required_sample_size(2.0) > required_sample_size(4.0)
