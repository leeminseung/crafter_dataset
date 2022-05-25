# crafter_dataset

## Crafter

```sh
cd crafter
pip install -e .
```

## Dreamer-v2
### Install
```sh
cd dreamverv2
pip install -e .
```

### train
```sh
python dreamerv2_train.py
```

### Evaluate trained agent and record as videos.
```sh
python evaluator.py
```


## PPO
### Install
```sh
pip install stable-baselines3
```
and need to install other trivial dependency. (I did not track these...)

### train
```sh
python ppo_train.py
```
### Evaluate trained agent and record as videos
not implemented yet.