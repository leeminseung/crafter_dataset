import argparse
import pathlib

import chunkedfile
import crafter
import stable_baselines3

chunkedfile.patch_pathlib_append(3600)

parser = argparse.ArgumentParser()
boolean = lambda x: bool(['False', 'True'].index(x))
parser.add_argument('--logdir', type=str, default='logdir')
parser.add_argument('--steps', type=float, default=5e6)
args = parser.parse_args()

env = crafter.Env(
    level=1,
    area=(64, 64),
    view=(9, 9),
    length=1000,
    seed=None,
    subtask="make_iron_pickaxe"
)

env = crafter.Recorder(
    env, pathlib.Path(args.logdir) / 'make_iron_pickaxe_ppo',
    save_stats=True,
    save_video=False,
    save_episode=False,
)

model = stable_baselines3.PPO('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=args.steps)