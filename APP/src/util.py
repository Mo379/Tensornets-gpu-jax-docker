#system
import os
import pickle
from functools import partial
from typing import Any, Tuple
from datetime import timedelta,datetime
from time import time
from pathlib import Path
#ML
import haiku as hk
import numpy as np
import optax 
import jax 
import jax.numpy as jnp
import acme
from acme import specs
from acme import wrappers as acmewrappers
#Env
from pettingzoo.butterfly import pistonball_v6
import supersuit as ss
from gym.spaces import Box, Discrete
#Log
import wandb
import imageio
#local

#environment setup
def environment_setup():
    #setting up the testing and live cases
    n_pistons = 20
    env = pistonball_v6.parallel_env(
        n_pistons=n_pistons,
        time_penalty=-0.1,
        continuous=True,
        random_drop=True,
        random_rotate=True,
        ball_mass=0.75,
        ball_friction=0.3,
        ball_elasticity=1.5,
        max_cycles=125
    )
    env = ss.color_reduction_v0(env, mode='B')
    env = ss.resize_v1(env, x_size=84,y_size=84)
    env = ss.frame_stack_v1(env, stack_size=3)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.dtype_v0(env, np.float32)
    #
    obs_space = Box(0.0,255.0,(n_pistons,84,84,3),np.float32)
    action_space = Box(-1.0,1.0,(n_pistons,1),np.float32)
    reward_spec = specs.Array(shape=(20,), dtype=np.float32,name='reward')
    discount_spec = specs.BoundedArray(shape=(20,), dtype=np.float32,name='discount',minimum=0.0,maximum=1.0)
    #
    env.observation_space = obs_space
    env.action_space = action_space
    #
    env = acmewrappers.GymWrapper(env)
    return env
def play_enviromnet_setup():
    env = pistonball_v6.env(n_pistons=20)
    env = ss.color_reduction_v0(env,mode='B')
    env = ss.resize_v1(env, x_size=84,y_size=84)
    env = ss.frame_stack_v1(env, 3)
    return env



