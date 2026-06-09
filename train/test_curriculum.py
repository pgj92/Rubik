"""
Tests for the Curriculum class. Run with: python -m train.test_curriculum
"""
import numpy as np
from train.curriculum import Curriculum


def test_initial_depth():
    c = Curriculum(initial_depth=1)
    assert c.depth == 1


def test_sample_depth_1_always_returns_1():
    c = Curriculum(initial_depth=1, rng=np.random.default_rng(0))
    for _ in range(100):
        assert c.sample_depth() == 1


def test_sample_depth_higher_is_mostly_current():
    """With mix_current=0.8, ~80% of samples should be at current depth."""
    c = Curriculum(initial_depth=5, mix_current=0.8, rng=np.random.default_rng(0))
    samples = [c.sample_depth() for _ in range(10000)]
    frac_current = np.mean([s == 5 for s in samples])
    assert 0.75 <= frac_current <= 0.85, f"current fraction {frac_current}, expected ~0.80"
    # Lower depths should all appear
    lower = [s for s in samples if s != 5]
    for d in [1, 2, 3, 4]:
        assert d in lower, f"depth {d} never sampled as lower"


def test_record_and_success_rate():
    c = Curriculum(initial_depth=1, window=10)
    # Not enough data
    assert c.current_success_rate() is None
    for _ in range(5):
        c.record(1, True)
    assert c.current_success_rate() is None    # still < window samples
    for _ in range(5):
        c.record(1, False)
    rate = c.current_success_rate()
    assert rate is not None
    assert abs(rate - 0.5) < 1e-9, f"expected 0.5, got {rate}"


def test_record_window_forgets_old():
    """After window fills, only the most recent `window` samples count."""
    c = Curriculum(initial_depth=1, window=10)
    for _ in range(10):
        c.record(1, False)
    assert c.current_success_rate() == 0.0
    for _ in range(10):
        c.record(1, True)
    assert c.current_success_rate() == 1.0


def test_promote_when_threshold_met():
    c = Curriculum(initial_depth=1, window=10, promote_threshold=0.8)
    for _ in range(8):
        c.record(1, True)
    for _ in range(2):
        c.record(1, False)
    assert c.current_success_rate() == 0.8
    promoted = c.maybe_promote()
    assert promoted
    assert c.depth == 2


def test_no_promote_below_threshold():
    c = Curriculum(initial_depth=1, window=10, promote_threshold=0.8)
    for _ in range(7):
        c.record(1, True)
    for _ in range(3):
        c.record(1, False)
    assert not c.maybe_promote()
    assert c.depth == 1


def test_no_promote_without_full_window():
    c = Curriculum(initial_depth=1, window=10, promote_threshold=0.8)
    for _ in range(9):
        c.record(1, True)        # 100% but only 9 samples
    assert not c.maybe_promote()
    assert c.depth == 1


def test_promote_caps_at_max_depth():
    c = Curriculum(initial_depth=3, max_depth=3, window=10, promote_threshold=0.8)
    for _ in range(10):
        c.record(3, True)
    assert not c.maybe_promote()
    assert c.depth == 3


def test_history_at_new_depth_starts_empty():
    """After promotion, the new depth shouldn't auto-inherit old history."""
    c = Curriculum(initial_depth=1, window=10, promote_threshold=0.8)
    for _ in range(10):
        c.record(1, True)
    c.maybe_promote()
    assert c.depth == 2
    assert c.current_success_rate() is None  # depth-2 has no episodes recorded


ALL_TESTS = [
    test_initial_depth,
    test_sample_depth_1_always_returns_1,
    test_sample_depth_higher_is_mostly_current,
    test_record_and_success_rate,
    test_record_window_forgets_old,
    test_promote_when_threshold_met,
    test_no_promote_below_threshold,
    test_no_promote_without_full_window,
    test_promote_caps_at_max_depth,
    test_history_at_new_depth_starts_empty,
]

if __name__ == "__main__":
    passed, failed = 0, 0
    for t in ALL_TESTS:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except NotImplementedError as e:
            print(f"  SKIP  {t.__name__}  (TODO not done: {e})")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}  {e}")
            failed += 1
        except Exception as e:
            print(f"  ERR   {t.__name__}  {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
