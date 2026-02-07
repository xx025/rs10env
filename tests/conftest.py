"""Pytest 共享 fixture：统一用 CPU 便于 CI。"""
import pytest
import numpy as np

from rs10env import RS10Env, create_strategy, STRATEGY_NAMES


@pytest.fixture
def device():
    return "cpu"


@pytest.fixture
def env(device):
    return RS10Env(device=device, H=16, W=10, target_sum=10)


@pytest.fixture
def env_small(device):
    """小棋盘，便于快速跑完一局。"""
    return RS10Env(device=device, H=6, W=4, target_sum=10)
