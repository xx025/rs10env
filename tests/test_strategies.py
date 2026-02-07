"""策略与 create_strategy 测试。"""
import pytest

from rs10env import create_strategy, STRATEGY_NAMES, RS10Env


def test_strategy_names_non_empty():
    assert len(STRATEGY_NAMES) >= 1
    assert "random" in STRATEGY_NAMES
    assert "greedy" in STRATEGY_NAMES


def test_create_strategy_random(env):
    s = create_strategy("random", device="cpu")
    assert s.name == "Random"
    mask = env.get_valid_actions_mask()
    if mask.any():
        action = s.get_action(env, mask)
        assert mask[action].item() or action in mask.nonzero(as_tuple=True)[0]


def test_create_strategy_greedy(env):
    s = create_strategy("greedy", device="cpu")
    assert s.name == "Greedy"
    env.reset(seed=7)
    mask = env.get_valid_actions_mask()
    if mask.any():
        action = s.get_action(env, mask)
        assert mask[action].item()


def test_create_strategy_unknown_raises():
    with pytest.raises(ValueError, match="Unknown strategy"):
        create_strategy("unknown_xyz", device="cpu")


def test_create_strategy_case_insensitive(env):
    s = create_strategy("RANDOM", device="cpu")
    assert s.name == "Random"


def test_all_strategy_names_create():
    for name in STRATEGY_NAMES:
        s = create_strategy(name, device="cpu")
        assert s is not None
        assert hasattr(s, "get_action")
