"""Exercise the real diffusion sampler on a tiny CPU model; no ROS/hardware."""
import numpy as np
import pytest
import torch
from diffusers import DDIMScheduler, DDPMScheduler

from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.real_world.p2p_franka_deployment import (
    predict_relative_actions, validate_inference_seed,
)
from eval_real_p2p_common import parse_args


class ObservationEncoder(torch.nn.Module):
    def output_shape(self):
        return (3,)

    def forward(self, obs):
        return obs['state']


@pytest.mark.parametrize('action_dim', [9, 10])
@pytest.mark.parametrize('scheduler_type', [DDIMScheduler, DDPMScheduler])
def test_seed_restarts_noise_each_inference(action_dim, scheduler_type):
    policy = DiffusionUnetImagePolicy(
        shape_meta={'action': {'shape': [action_dim]}},
        noise_scheduler=scheduler_type(num_train_timesteps=10),
        obs_encoder=ObservationEncoder(), horizon=8, n_action_steps=6,
        n_obs_steps=2, num_inference_steps=2, diffusion_step_embed_dim=16,
        down_dims=(16, 32), n_groups=8,
    ).eval()
    for name in ['state', 'action']:
        policy.normalizer[name] = SingleFieldLinearNormalizer.create_identity()
    observation = {'state': torch.zeros((1, 2, 3))}
    initial_noises = []

    def capture_noise(module, args):
        initial_noises.append(args[0].detach().clone())

    handle = policy.model.register_forward_pre_hook(capture_noise)
    before = torch.random.get_rng_state().clone()
    try:
        a = predict_relative_actions(policy, observation, 0)
        # Unrelated global draws must not affect a supplied local seed.
        assert torch.equal(before, torch.random.get_rng_state())
        torch.randn(17)
        b = predict_relative_actions(policy, observation, 0)
        c = predict_relative_actions(policy, observation, 1)
        predict_relative_actions(policy, {'state': torch.ones((1, 2, 3))}, 0)
        predict_relative_actions(policy, observation)
        predict_relative_actions(policy, observation)
    finally:
        handle.remove()
    first_steps = initial_noises[::policy.num_inference_steps]
    assert torch.equal(first_steps[0], first_steps[1])
    assert torch.equal(first_steps[0], first_steps[3])  # Changed observation, same noise.
    assert not torch.equal(first_steps[0], first_steps[2])
    assert not torch.equal(first_steps[4], first_steps[5])  # Unseeded remains random.
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize('seed', [-1, 2**63, True, 1.5, '42'])
def test_invalid_seed_rejected(seed):
    with pytest.raises(ValueError, match='inference_seed'):
        validate_inference_seed(seed)


@pytest.mark.parametrize('seed', [None, 0, 42, 2**63 - 1])
def test_valid_seed(seed):
    assert validate_inference_seed(seed) == seed


@pytest.mark.parametrize('task', ['peg', 'valve'])
def test_seed_cli(task, monkeypatch):
    monkeypatch.setattr('sys.argv', [f'eval_real_{task}.py', '--inference-seed', '42'])
    assert parse_args(task).inference_seed == 42
