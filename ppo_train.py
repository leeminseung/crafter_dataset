import argparse
import pathlib
import os
import chunkedfile
import crafter
import stable_baselines3

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

os.environ["CUDA_VISIBLE_DEVICES"]= "0"

class EvalCallback(BaseCallback):
    def __init__(self, eval_env, check_freq: int, log_dir: str, verbose=1):
        super(EvalCallback, self).__init__(verbose)
        self.eval_env = Monitor(eval_env)
        self.check_freq = check_freq
        self.log_dir = log_dir

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            evaluate_policy(self.model, self.eval_env, n_eval_episodes=1)
            self.model.save(self.log_dir + "_agent_1")
        return True

chunkedfile.patch_pathlib_append(3600)

parser = argparse.ArgumentParser()
boolean = lambda x: bool(['False', 'True'].index(x))
parser.add_argument('--logdir', type=str, default="logdir/ppo_test")
parser.add_argument('--subtask', type=str, default="collect_wood")
parser.add_argument('--steps', type=float, default=1e7)
parser.add_argument("--expl", type=bool, default=True)
parser.add_argument("--beta", type=float, default=0.1)
    
args = parser.parse_args()

env = crafter.Env(
    level=1,
    area=(64, 64),
    view=(9, 9),
    length=1000,
    seed=None,
    subtask=args.subtask,
    expl_mode=args.expl,
    beta=args.beta,
)

eval_env = crafter.Recorder(
    env, args.logdir,
    save_stats=True,
    save_video=True,
    save_episode=False,
)

# Use deterministic actions for evaluation
eval_callback = EvalCallback(eval_env = eval_env, check_freq=1e5, log_dir=args.logdir)
model = stable_baselines3.PPO('CnnPolicy', env, verbose=1, tensorboard_log=args.logdir)
model.learn(total_timesteps=args.steps, callback=eval_callback)