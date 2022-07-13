import acme
from absl import flags
from acme.agents.jax import ppo
from absl import app
from acme.jax import experiments
from acme.utils import lp_utils
import launchpad as lp

import src.helpers as helpers
from src.util import environment_setup 
from src.agent import *


import dataclasses
from typing import Any, Callable, Optional, Sequence

from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import utils


import haiku as hk
from acme.specs import EnvironmentSpec

FLAGS = flags.FLAGS

flags.DEFINE_bool(
    'run_distributed', True, 'Should an agent be executed in a '
    'distributed way (the default is a single-threaded agent)')
flags.DEFINE_string('env_name', 'gym:HalfCheetah-v2', 'What environment to run')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('num_steps', 1_000, 'Number of env steps to run.')
flags.DEFINE_integer('eval_every', 100, 'How often to run evaluation.')
flags.DEFINE_integer('evaluation_episodes', 10, 'Evaluation episodes.')


env = environment_setup(test=True)
state =env.reset()
environment_spec= EnvironmentSpec(
    observations=env.observation_spec(),
    actions=env.action_spec(),
    rewards=env.reward_spec(),
    discounts=env.discount_spec()
)
print(environment_spec)
#exit()







print('actions:\n', environment_spec.actions, '\n')
print('observations:\n', environment_spec.observations.shape, '\n')
print('states:\n', state.observation.shape, state.observation.flatten().shape, '\n')
print('rewards:\n', environment_spec.rewards, '\n')
print('discounts:\n', environment_spec.discounts, '\n')

def my_net_factory(environment_spec):
    # Transform into pure functions.
    forward_fn = hk.without_apply_rng(hk.transform(my_model))
    dummy_obs = utils.zeros_like(environment_spec.observations)
    dummy_obs = utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.
    network = networks_lib.FeedForwardNetwork(
      lambda rng: forward_fn.init(rng, dummy_obs), forward_fn.apply)
    # Create PPONetworks to add functionality required by the agent.
    rng = jax.random.PRNGKey(0)
    network = ppo.make_ppo_networks(network)
    return network

def build_experiment_config(env_spec):
  """Builds PPO experiment config which can be executed in different ways."""
  # Create an environment, grab the spec, and use it to create networks.
  config = ppo.PPOConfig(entropy_cost=0, learning_rate=1e-4)
  ppo_builder = ppo.PPOBuilder(config)

  return experiments.ExperimentConfig(
      builder=ppo_builder,
      environment_spec=env_spec,
      environment_factory=lambda seed: environment_setup(test=True),
      network_factory=lambda spec: my_net_factory(spec),
      seed=FLAGS.seed,
      max_num_actor_steps=FLAGS.num_steps)


def main(_):
    env = environment_setup(test=True) 
    environment_spec= EnvironmentSpec(
        observations=env.observation_spec(),
        actions=env.action_spec(),
        rewards=env.reward_spec(),
        discounts=env.discount_spec()
    )
    env.close()
    #
    config = build_experiment_config(environment_spec)
    if FLAGS.run_distributed:
        program = experiments.make_distributed_experiment(
            experiment=config, num_actors=4)
        lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program))
    else:
        experiments.run_experiment(
            experiment=config,
            eval_every=FLAGS.eval_every,
            num_eval_episodes=FLAGS.evaluation_episodes)


if __name__ == '__main__':
  app.run(main)
