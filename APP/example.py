from absl import flags
from acme.agents.jax import ppo
from absl import app
from acme.jax import experiments
from acme.utils import lp_utils
import launchpad as lp

import src.helpers as helpers
from src.util import environment_setup 
FLAGS = flags.FLAGS

flags.DEFINE_bool(
    'run_distributed', False, 'Should an agent be executed in a '
    'distributed way (the default is a single-threaded agent)')
flags.DEFINE_string('env_name', 'gym:HalfCheetah-v2', 'What environment to run')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('num_steps', 1_000_000, 'Number of env steps to run.')
flags.DEFINE_integer('eval_every', 50_000, 'How often to run evaluation.')
flags.DEFINE_integer('evaluation_episodes', 10, 'Evaluation episodes.')


def build_experiment_config():
  """Builds PPO experiment config which can be executed in different ways."""
  # Create an environment, grab the spec, and use it to create networks.
  config = ppo.PPOConfig(entropy_cost=0, learning_rate=1e-4)
  ppo_builder = ppo.PPOBuilder(config)

  layer_sizes = (256, 256, 256)
  return experiments.ExperimentConfig(
      builder=ppo_builder,
      environment_factory=lambda seed: environment_setup(test=True),
      network_factory=lambda spec: ppo.make_networks(spec, layer_sizes),
      seed=FLAGS.seed,
      max_num_actor_steps=FLAGS.num_steps)


def main(_):
  config = build_experiment_config()
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
