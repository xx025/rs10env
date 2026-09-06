import pytest
import torch

from rs10env.env import RS10Env
from rs10env.learning import PolicyNet, play, train_episode


def test_legal_logits_and_prefix_gradients():
    torch.manual_seed(7)
    board = torch.tensor([[5, 5], [5, 5]])
    rects = torch.tensor([[0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, 0]])
    mask = torch.tensor([True, False, True])
    policy = PolicyNet(channels=4, hidden=8)
    logits, indices, value = policy(board, rects, mask)
    assert indices.tolist() == [0, 2]
    assert logits.shape == (2,)
    assert value.shape == ()
    # Pool channels alone must carry CNN gradients through prefix indexing.
    with torch.no_grad():
        policy.encoder[0].weight.fill_(0.01)
        policy.encoder[0].bias.fill_(0.1)
        policy.encoder[2].weight.fill_(0.01)
        policy.encoder[2].bias.fill_(0.1)
        policy.scorer[0].weight.zero_()
        policy.scorer[0].weight[:, :4].fill_(1)
        policy.scorer[0].bias.fill_(1)
        policy.scorer[2].weight.fill_(1)
    policy(board, rects, mask)[0].sum().backward()
    gradient = policy.encoder[0].weight.grad
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
    logits, indices, value = policy(board, rects, torch.zeros_like(mask))
    assert logits.numel() == indices.numel() == 0
    assert torch.isfinite(value)


def test_episode_inference_training_and_checkpoint(tmp_path):
    torch.manual_seed(11)
    env = RS10Env(H=2, W=2, device='cpu')
    # Exercise play's normal reset path with a known legal, two-step board.
    original_reset = env.reset
    env.reset = lambda **kwargs: original_reset(board=torch.full((2, 2), 5))
    policy = PolicyNet(channels=4, hidden=8)
    episode, metrics = play(env, policy, seed=1)
    assert metrics['cleared'] == 4
    assert metrics['remaining'] == 0
    assert [step['return'] for step in episode] == [1.0, 0.5]
    for step in episode:
        original_reset(board=step['board'])
        assert env.get_valid_actions_mask_prefix()[step['action']]
        assert step['mask'][step['action']]
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    old = policy.encoder[0].weight.detach().clone()
    for imitation in (True, False):
        loss = train_episode(policy, env.all_rects, episode, optimizer, imitation=imitation)
        assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(old, policy.encoder[0].weight)
    assert train_episode(policy, env.all_rects, [], optimizer) == 0
    path = tmp_path / 'policy.pt'
    torch.save(dict(model_state=policy.state_dict(), model_config=policy.config), path)
    saved = torch.load(path, weights_only=True)
    restored = PolicyNet(**saved['model_config'])
    restored.load_state_dict(saved['model_state'])
    inputs = (episode[0]['board'], env.all_rects, episode[0]['mask'])
    with torch.no_grad():
        assert torch.equal(policy(*inputs)[0], restored(*inputs)[0])
    assert play(env, restored, seed=1)[1] == metrics
    torch.manual_seed(23)
    sampled, _ = play(env, restored, seed=1, greedy=False)
    torch.manual_seed(23)
    repeated, _ = play(env, restored, seed=1, greedy=False)
    assert [step['action'] for step in sampled] == [step['action'] for step in repeated]


def test_single_step_reinforce():
    env = RS10Env(H=1, W=2, device='cpu')
    original_reset = env.reset
    env.reset = lambda **kwargs: original_reset(board=torch.tensor([[5, 5]]))
    policy = PolicyNet(channels=4, hidden=8)
    episode, metrics = play(env, policy, seed=1, greedy=False)
    assert metrics['steps'] == 1
    assert episode[0]['return'] == 1.0
    loss = train_episode(policy, env.all_rects, episode,
                         torch.optim.Adam(policy.parameters(), lr=0.001))
    assert torch.isfinite(torch.tensor(loss))


def test_invalid_teacher_never_steps():
    env = RS10Env(H=1, W=2, device='cpu')
    original_reset = env.reset
    env.reset = lambda **kwargs: original_reset(board=torch.tensor([[5, 5]]))

    class InvalidTeacher:
        def get_action(self, env, mask):
            return -1

    with pytest.raises(ValueError, match='invalid action'):
        play(env, seed=1, teacher=InvalidTeacher())
    assert env.step_count == 0


def test_no_legal_actions():
    env = RS10Env(H=1, W=2, device='cpu')
    original_reset = env.reset
    env.reset = lambda **kwargs: original_reset(board=torch.tensor([[1, 1]]))
    episode, metrics = play(env, PolicyNet(channels=4, hidden=8), seed=1)
    assert episode == []
    assert metrics['cleared'] == metrics['steps'] == 0
    assert metrics['remaining'] == 2
