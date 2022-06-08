import collections

import numpy as np
import random 

from . import constants
from . import engine
from . import objects
from . import worldgen


# Gym is an optional dependency.
try:
    import gym

    DiscreteSpace = gym.spaces.Discrete
    BoxSpace = gym.spaces.Box
    DictSpace = gym.spaces.Dict
    BaseClass = gym.Env
except ImportError:
    DiscreteSpace = collections.namedtuple("DiscreteSpace", "n")
    BoxSpace = collections.namedtuple("BoxSpace", "low, high, shape, dtype")
    DictSpace = collections.namedtuple("DictSpace", "spaces")
    BaseClass = object


class Env(BaseClass):
    def __init__(
        self,
        level=3,
        area=(64, 64),
        view=(9, 9),
        size=(64, 64),
        reward=True,
        length=10000,
        seed=None,
        subtask=None,
        expl_mode=False,
        expl_method=None,
        beta=0.1,
        mdp="partial",
    ):
        view = np.array(view if hasattr(view, "__len__") else (view, view))
        size = np.array(size if hasattr(size, "__len__") else (size, size))
        seed = np.random.randint(0, 2 ** 31 - 1) if seed is None else seed
        assert mdp in ["partial", "full"]
        if expl_method:
            assert expl_mode, "Specifying exploration method is only allowed when explr_mode is True"
            assert expl_method in ["binary_discovery", "count_based"]
        if subtask:
            assert subtask in ["exploration", "collect_stone", "collect_coal", "collect_iron", "collect_diamond", "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe"], "not supported subtask."
        assert level in [1, 2, 3, 4], "only support level in [1, 2, 3, 4]."
        """
        level:
        - 1: no health, no die, no enemy
        - 2: health, no die, no enemy
        - 3: health, die, no enemy
        - 4: health, die, enemy (original)
        """
        self.level = level
        self._area = area
        self._view = view
        self._size = size
        self._reward = reward
        self._length = length
        self._seed = seed
        self._subtask = subtask
        self._episode = 0
        self._world = engine.World(level, area, constants.materials, (12, 12))
        self._textures = engine.Textures(constants.root / "assets")
        item_rows = int(np.ceil(len(constants.items) / view[0]))
        self._local_view = engine.LocalView(
            self._world, self._textures, [view[0], view[1] - item_rows]
        )
        if mdp == "full":
            self._global_view = engine.GlobalView(
            self._world, self._textures, [view[0], view[1] - item_rows]
        )
        self._item_view = engine.ItemView(self._textures, [view[0], item_rows])
        self._sem_view = engine.SemanticView(
            self._world,
            [
                objects.Player,
                objects.Cow,
                objects.Zombie,
                objects.Skeleton,
                objects.Arrow,
                objects.Plant,
            ],
        )
        self._step = None
        self._player = None
        self._last_health = None
        self._unlocked = None
        # Some libraries expect these attributes to be set.
        self.reward_range = None
        self.metadata = None
        self.expl_mode = expl_mode
        self._expl_method = expl_method
        self._beta = beta
        self._mdp = mdp


    @property
    def observation_space(self):
        if self.expl_mode:
            return BoxSpace(0, 255, tuple(self._size) + (5,), np.uint8)
        else:
            return BoxSpace(0, 255, tuple(self._size) + (3,), np.uint8)

    @property
    def action_space(self):
        return DiscreteSpace(len(constants.actions))

    @property
    def action_names(self):
        return constants.actions

    def reset(self):
        if self._expl_method == "binary_discovery":
            self._visit_map = np.ones(list(self._size) + [1], dtype=np.uint8)
        elif self._expl_method == "count_based": # count based intrisic exploration
            self._visit_map = np.zeros(list(self._size) + [1], dtype=np.uint8)
        self._location_map = np.zeros(list(self._size) + [1], dtype=np.uint8)

        random_init_task = ["exploration", "collect_stone", "collect_coal", "collect_iron", "collect_diamond", "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe"]
        center = (self._world.area[0] // 2, self._world.area[1] // 2)
        # if self._subtask in random_init_task:
        #     init_pos = (np.random.randint(self._world.area[0]), np.random.randint(self._world.area[1]))
        # else:
        #     init_pos = center

        self._episode += 1
        self._step = 0
        self._world.reset(seed=hash((self._seed, self._episode)) % (2 ** 31 - 1))
        self._update_time()
        self._unlocked = set()

        # hard coded for level-1
        worldgen.generate_world(self._world, self._player, subtask=self._subtask, player_init_pose=center)
        path_x, path_y = np.where(self._world._mat_map == 4)
        self._player = objects.Player(self._world, center)
        if self._subtask:
            if self._subtask.startswith("collect_"):
                path_ind = random.choice(range(len(path_x)))
                self._player.pos = np.array([path_x[path_ind], path_y[path_ind]])
        # worldgen.generate_world(self._world, self._player, center)
        
        # self._player = objects.Player(self._world, center)
        # if self._subtask in random_init_task:
        #     while True:
        #         reset_pos = np.array((np.random.randint(self._world.area[0]), np.random.randint(self._world.area[1])))
        #         if self._player.is_free(reset_pos):
        #             self._player.pos = reset_pos
        #             break

        self._last_health = self._player.health
        self._world.add(self._player)

        if self._subtask == "collect_coal":
            self._player.inventory['wood_pickaxe'] = 1

        elif self._subtask == "collect_stone":
            self._player.inventory['wood_pickaxe'] = 1

        elif self._subtask == "collect_iron":
            self._player.inventory['wood_pickaxe'] = 1
            self._player.inventory['stone_pickaxe'] = 1
            self._player.inventory['wood'] = np.random.randint(3, 10)
            self._player.inventory['stone'] = np.random.randint(4, 10)
            self._player.inventory['coal'] = np.random.randint(1, 4)
        
        elif self._subtask == "collect_diamond":
            self._player.inventory['wood_pickaxe'] = 1
            self._player.inventory['stone_pickaxe'] = 1
            self._player.inventory['iron_pickaxe'] = 1
            self._player.inventory['wood'] = np.random.randint(3, 8)
            self._player.inventory['stone'] = np.random.randint(4, 8)
            self._player.inventory['coal'] = np.random.randint(0, 3)
            self._player.inventory['iron'] = np.random.randint(0, 3)

        elif self._subtask == "make_wood_pickaxe":
            self._player.inventory['wood'] = np.random.randint(3, 10)

        elif self._subtask == "make_stone_pickaxe":
            self._player.inventory['wood_pickaxe'] = 1
            self._player.inventory['wood'] = np.random.randint(3, 10)
            self._player.inventory['stone'] = np.random.randint(1, 10)
        
        elif self._subtask == "make_iron_pickaxe":
            self._player.inventory['wood_pickaxe'] = 1
            self._player.inventory['stone_pickaxe'] = 1
            self._player.inventory['wood'] = np.random.randint(3, 10)
            self._player.inventory['stone'] = np.random.randint(4, 10)
            self._player.inventory['coal'] = np.random.randint(1, 8)
            self._player.inventory['iron'] = np.random.randint(1, 5)
        
        elif self._subtask == "exploration":
            self._player.inventory['wood_pickaxe'] = 1
            self._player.inventory['stone_pickaxe'] = 1
            self._player.inventory['iron_pickaxe'] = 1
            self._player.inventory['stone'] = 10

        self._last_inventory = self._player.inventory.copy()

        mask_left_boundary = max(self._player.pos[0] - 3, 0)
        mask_right_boundary = min(self._player.pos[0] + 3, 63)
        mask_top_boundary = max(self._player.pos[1] - 4, 0)
        mask_bottom_boundary = min(self._player.pos[1] + 4, 63)
        
        if self._expl_method == "binary_discovery":
            self._visit_map[mask_top_boundary:mask_bottom_boundary, mask_left_boundary:mask_right_boundary] = 0
        elif self._expl_method == "count_based": # count based intrisic exploration
            # self._visit_map[self._player.pos[1], self._player.pos[0]] += 1
            self._visit_map[mask_top_boundary:mask_bottom_boundary, mask_left_boundary:mask_right_boundary] += 1

        self._location_map[self._player.pos[1], self._player.pos[0]] = 1
        self._prev_loc = self._player.pos
        if self.expl_mode:
            return np.concatenate((self._obs(), self._visit_map, self._location_map), axis=-1)
        else:
            return self._obs()
        
    def step(self, action):
        self._step += 1
        # self._update_time()
        self._player.action = constants.actions[action]
        for obj in self._world.objects:
            if self._player.distance(obj) < 2 * max(self._view):
                obj.update()
        # if self._step % 10 == 0:
        #     for chunk, objs in self._world.chunks.items():
        #         # xmin, xmax, ymin, ymax = chunk
        #         # center = (xmax - xmin) // 2, (ymax - ymin) // 2
        #         # if self._player.distance(center) < 4 * max(self._view):
        #         self._balance_chunk(chunk, objs)
        obs = self._obs()
        if self.level in [1]:
            reward = 0
        else:
            reward = (self._player.health - self._last_health) / 10
        self._last_health = self._player.health
        unlocked = {
            name
            for name, count in self._player.achievements.items()
            if count > 0 and name not in self._unlocked
        }
        if self._subtask == "collect_wood":
            if self._player.inventory['wood'] - self._last_inventory['wood'] > 0:
                reward += 1.0
                
        elif self._subtask == "collect_stone":
            if self._player.inventory['stone'] - self._last_inventory['stone'] > 0:
                reward += 1.0

        elif self._subtask == "collect_coal":
            if self._player.inventory['coal'] - self._last_inventory['coal'] > 0:
                reward += 1.0

        elif self._subtask == "collect_iron":
            if self._player.inventory['iron'] - self._last_inventory['iron'] > 0:
                reward += 1.0
        
        elif self._subtask == "collect_diamond":
            if self._player.inventory['diamond'] - self._last_inventory['diamond'] > 0:
                reward += 1.0

        elif self._subtask == "make_wood_pickaxe":
            if self._player.inventory['wood_pickaxe'] - self._last_inventory['wood_pickaxe'] > 0:
                reward += 1.0

        elif self._subtask == "make_stone_pickaxe":
            if self._player.inventory['stone_pickaxe'] - self._last_inventory['stone_pickaxe'] > 0:
                reward += 1.0

        elif self._subtask == "make_iron_pickaxe":
            if self._player.inventory['iron_pickaxe'] - self._last_inventory['iron_pickaxe'] > 0:
                reward += 1.0
        
        elif self._subtask == "exploration":
            pass

        elif unlocked:
            self._unlocked |= unlocked
            reward += 1.0

        self._last_inventory = self._player.inventory.copy()

        dead = self._player.health <= 0
        over = self._length and self._step >= self._length
        if self._subtask == "make_wood_pickaxe":
            if self._player.inventory["wood_pickaxe"]:
                over = True
        
        elif self._subtask == "make_stone_pickaxe":
            if self._player.inventory["stone_pickaxe"]:
                over = True

        elif self._subtask == "make_iron_pickaxe":
            if self._player.inventory["iron_pickaxe"]:
                over = True
        
        elif not self._subtask:
            if self._player.inventory["diamond"]:
                over = True

        if self.level in [1, 2]:
            done = over
        else:
            done = dead or over
        info = {
            "inventory": self._player.inventory.copy(),
            "achievements": self._player.achievements.copy(),
            "discount": 1 - float(dead),
            "semantic": self._sem_view(),
            "player_pos": self._player.pos,
            "reward": reward,
        }
        if not self._reward:
            reward = 0.0

        if self.expl_mode or self._subtask == "exploration":
            mask_left_boundary = max(self._player.pos[0] - 3, 0)
            mask_right_boundary = min(self._player.pos[0] + 3, 63)
            mask_top_boundary = max(self._player.pos[1] - 4, 0)
            mask_bottom_boundary = min(self._player.pos[1] + 4, 63)

            self._location_map[self._prev_loc[1], self._prev_loc[0]] = 0
            self._location_map[self._player.pos[1], self._player.pos[0]] = 1
            self._prev_loc = self._player.pos

            if self._expl_method == "binary_discovery":
                reward += self._beta * np.sum(self._visit_map[mask_top_boundary:mask_bottom_boundary, mask_left_boundary:mask_right_boundary])
                self._visit_map[mask_top_boundary:mask_bottom_boundary, mask_left_boundary:mask_right_boundary] = 0
                
            elif self._expl_method == "count_based":
                self._visit_map[self._player.pos[1], self._player.pos[0]] += 1
                reward += self._beta * 1/self._visit_map[self._player.pos[1], self._player.pos[0]]

                # self._visit_map[mask_top_boundary:mask_bottom_boundary, mask_left_boundary:mask_right_boundary] += 1
                # reward += self._beta * 1/np.mean(self._visit_map[mask_top_boundary:mask_bottom_boundary, mask_left_boundary:mask_right_boundary])
                # reward += self._beta * 1/np.sqrt(self._visit_map[self._player.pos[1], self._player.pos[0]])
                # reward += self._beta * 1/np.square(np.mean(self._visit_map[mask_top_boundary:mask_bottom_boundary, mask_left_boundary:mask_right_boundary]))
            obs = np.concatenate((obs, self._visit_map, self._location_map), axis=-1)
        
        # if reward:
        #     print(reward, self._player.inventory)
        return obs, reward, done, info

    def render(self, size=None):
        size = size or self._size
        unit = size // self._view
        canvas = np.zeros(tuple(size) + (3,), np.uint8)
        if self._mdp == "partial":
            local_view = self._local_view(self._player, unit)
        else:
            local_view = self._global_view(self._player, unit)
        item_view = self._item_view(self._player.inventory, unit)
        view = np.concatenate([local_view, item_view], 1)
        border = (size - (size // self._view) * self._view) // 2
        (x, y), (w, h) = border, view.shape[:2]
        canvas[x : x + w, y : y + h] = view
        return canvas.transpose((1, 0, 2))

    def _obs(self):
        return self.render()

    def _update_time(self):
        # https://www.desmos.com/calculator/grfbc6rs3h
        progress = (self._step / 300) % 1 + 0.3
        daylight = 1 - np.abs(np.cos(np.pi * progress)) ** 3
        self._world.daylight = daylight

    def _balance_chunk(self, chunk, objs):
        light = self._world.daylight
        if self.level in [4]:
            self._balance_object(
                chunk,
                objs,
                objects.Zombie,
                "grass",
                6,
                0,
                0.3,
                0.4,
                lambda pos: objects.Zombie(self._world, pos, self._player),
                lambda num, space: (
                    0 if space < 50 else 3.5 - 3 * light,
                    3.5 - 3 * light,
                ),
            )
        if self.level in [4]:
            self._balance_object(
                chunk,
                objs,
                objects.Skeleton,
                "path",
                7,
                7,
                0.1,
                0.1,
                lambda pos: objects.Skeleton(self._world, pos, self._player),
                lambda num, space: (0 if space < 6 else 1, 2),
            )
        self._balance_object(
            chunk,
            objs,
            objects.Cow,
            "grass",
            5,
            5,
            0.01,
            0.1,
            lambda pos: objects.Cow(self._world, pos),
            lambda num, space: (0 if space < 30 else 1, 1.5 + light),
        )

    def _balance_object(
        self,
        chunk,
        objs,
        cls,
        material,
        span_dist,
        despan_dist,
        spawn_prob,
        despawn_prob,
        ctor,
        target_fn,
    ):
        xmin, xmax, ymin, ymax = chunk
        random = self._world.random
        creatures = [obj for obj in objs if isinstance(obj, cls)]
        mask = self._world.mask(*chunk, material)
        target_min, target_max = target_fn(len(creatures), mask.sum())
        if len(creatures) < int(target_min) and random.uniform() < spawn_prob:
            xs = np.tile(np.arange(xmin, xmax)[:, None], [1, ymax - ymin])
            ys = np.tile(np.arange(ymin, ymax)[None, :], [xmax - xmin, 1])
            xs, ys = xs[mask], ys[mask]
            i = random.randint(0, len(xs))
            pos = np.array((xs[i], ys[i]))
            empty = self._world[pos][1] is None
            away = self._player.distance(pos) >= span_dist
            if empty and away:
                self._world.add(ctor(pos))
        elif len(creatures) > int(target_max) and random.uniform() < despawn_prob:
            obj = creatures[random.randint(0, len(creatures))]
            away = self._player.distance(obj.pos) >= despan_dist
            if away:
                self._world.remove(obj)

