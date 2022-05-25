import collections
import logging
import os
import pathlib
import re
import sys
import numpy as np
from unittest import runner
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel('ERROR')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

sys.path.append("/home/mslee/crafter_dataset/dreamerv2/dreamerv2")
sys.path.append("/home/mslee/crafter_dataset/dreamerv2")

import dreamerv2.dreamerv2.agent as agent
import dreamerv2.dreamerv2.common as common
from dreamerv2.dreamerv2.common.envs import OneHotAction
import gym
import dreamerv2.api as dv2
import crafter

config = dv2.defaults.update({
    'logdir': '/home/mslee/crafter_dataset/logdir/makewoodpickaxe_dreamerv2',
    'log_every': 1e2,
    'train_every': 10,
    'prefill': 5e4,
    'actor_ent': 3e-3,
    'loss_scales.kl': 1.0,
    'discount': 0.99,
}).parse_flags()

env = crafter.Env(
    level=1,
    area=(64, 64),
    view=(9, 9),
    length=1000,
    seed=None,
    subtask="make_wood_pickaxe"
)

def sample_action():
  actions = 17
  index = np.random.randint(0, actions)
  reference = np.zeros(actions, dtype=np.float32)
  reference[index] = 1.0
  return {'action': reference}

def eval(env, config, outputs=None):
  logdir = pathlib.Path(config.logdir).expanduser()
  replay_logdir = pathlib.Path('/home/mslee/crafter_dataset/logdir/crafter_dummy/train_episodes').expanduser()
  logdir.mkdir(parents=True, exist_ok=True)
  config.save(logdir / 'config.yaml')
  print(config, '\n')
  print('Logdir', logdir)

  replay = common.Replay(replay_logdir, **config.replay)
  step = common.Counter(replay.stats['total_steps'])
  
  def per_episode(ep):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    print(f'Episode has {length} steps and return {score:.1f}.')
    
  env = crafter.Recorder(env, "/home/mslee/crafter_dataset/logdir/recorded")
  env = common.GymWrapper(env)
  env = common.ResizeImage(env)
  if hasattr(env.act_space['action'], 'n'):
    env = common.OneHotAction(env)
  else:
    env = common.NormalizeAction(env)
  env = common.TimeLimit(env, config.time_limit)

  driver = common.Driver([env])
  driver.on_episode(per_episode)
  driver.on_step(lambda tran, worker: step.increment())
  driver.on_step(replay.add_step)
  driver.on_reset(replay.add_step)

  # prefill = max(0, config.prefill - replay.stats['total_steps'])
  prefill = 100
  if prefill:
    print(f'Prefill dataset ({prefill} steps).')
    random_agent = common.RandomAgent(env.act_space)
    driver(random_agent, steps=prefill, episodes=1)
    driver.reset()

  print('Create agent.')
  agnt = agent.Agent(config, env.obs_space, env.act_space, step)
  dataset = iter(replay.dataset(**config.dataset))
  train_agent = common.CarryOverState(agnt.train)
  train_agent(next(dataset))
  if (logdir / 'variables.pkl').exists():
    agnt.load(logdir / 'variables.pkl')
    print("load success")
  
  policy = lambda *args: agnt.policy(
      *args, mode='eval')
  driver(policy, steps=20000)

eval(env, config)