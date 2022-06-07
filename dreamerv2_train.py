import gym
import dreamerv2.api as dv2
import crafter

config = dv2.defaults.update({
    'logdir': '/home/mslee/crafter_dataset/logdir/collect_wood_dreamerv2',
    'log_every': 1e3,
    'train_every': 10,
    'prefill': 1e5,
    'actor_ent': 3e-3,
    'loss_scales.kl': 1.0,
    'discount': 0.99,
    'eval_every': 1e3,
}).parse_flags()

env = crafter.Env(
    level=1,
    area=(64, 64),
    view=(9, 9),
    length=1000,
    seed=None,
    subtask="collect_wood"
)

dv2.train(env, config)