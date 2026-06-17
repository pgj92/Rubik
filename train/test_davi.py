"""
Tests for DAVI's vectorized cube ops and value-iteration target.

Run: python -m train.test_davi
"""
import numpy as np
import torch

from cube.cube import apply_move, solved_state, N_MOVES, inverse_move
from train.davi import (
    ValueNet,
    random_scramble_batch,
    expand_children,
    compute_targets,
)


def test_scramble_batch_shape_and_colors():
    rng = np.random.default_rng(0)
    states = random_scramble_batch(64, max_depth=12, rng=rng)
    assert states.shape == (64, 54)
    # A cube is a permutation of stickers: every color appears exactly 9 times.
    for s in states:
        counts = np.bincount(s, minlength=6)
        assert (counts == 9).all(), f"color counts not preserved: {counts}"


def test_expand_children_matches_apply_move():
    rng = np.random.default_rng(1)
    states = random_scramble_batch(8, max_depth=6, rng=rng)
    children = expand_children(states)            # (8, 12, 54)
    assert children.shape == (8, 12, 54)
    for b in range(8):
        for m in range(N_MOVES):
            expected = apply_move(states[b], m)
            assert np.array_equal(children[b, m], expected)


def test_target_one_move_from_solved_is_one():
    # A state one move from solved has exactly one child (the inverse move)
    # equal to solved, so its value-iteration target must be exactly 1.0,
    # regardless of the (random) target network.
    device = torch.device("cpu")
    net = ValueNet(hidden_size=32, n_layers=2)
    solved = solved_state()
    for m in range(N_MOVES):
        s = apply_move(solved, m)[None, :]        # (1, 54)
        target = compute_targets(s, net, device)
        assert abs(float(target[0]) - 1.0) < 1e-5, \
            f"move {m}: expected target 1.0, got {float(target[0])}"


def test_target_solved_is_zero():
    device = torch.device("cpu")
    net = ValueNet(hidden_size=32, n_layers=2)
    s = solved_state()[None, :]
    target = compute_targets(s, net, device)
    assert abs(float(target[0])) < 1e-5, f"solved target should be 0, got {float(target[0])}"


def test_value_net_can_overfit_tiny_set():
    # Sanity: the net + loss can drive predictions toward fixed targets.
    torch.manual_seed(0)
    device = torch.device("cpu")
    net = ValueNet(hidden_size=64, n_layers=2)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    rng = np.random.default_rng(2)
    states = torch.as_tensor(random_scramble_batch(16, max_depth=8, rng=rng))
    targets = torch.arange(16, dtype=torch.float32) % 5 + 1.0
    first = None
    for _ in range(300):
        pred = net(states)
        loss = torch.nn.functional.mse_loss(pred, targets)
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.1, f"net failed to fit: {first:.3f} -> {loss.item():.3f}"


ALL_TESTS = [
    test_scramble_batch_shape_and_colors,
    test_expand_children_matches_apply_move,
    test_target_one_move_from_solved_is_one,
    test_target_solved_is_zero,
    test_value_net_can_overfit_tiny_set,
]

if __name__ == "__main__":
    passed, failed = 0, 0
    for t in ALL_TESTS:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}  {e}")
            failed += 1
        except Exception as e:
            print(f"  ERR   {t.__name__}  {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
