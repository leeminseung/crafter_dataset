import functools

import numpy as np
import opensimplex

from . import constants
from . import objects


def generate_world(world, player, subtask, player_init_pose=None):
    simplex = opensimplex.OpenSimplex(seed=world.random.randint(0, 2 ** 31 - 1))
    tunnels = np.zeros(world.area, np.bool)
    if player_init_pose:
        player_pos = player_init_pose
    else:
        player_pos = player.pos
    for x in range(world.area[0]):
        for y in range(world.area[1]):
            _set_material(world, (x, y), player_pos, tunnels, simplex, subtask)
    for x in range(world.area[0]):
        for y in range(world.area[1]):
            _set_object(world, (x, y), player_pos, tunnels)


def _set_material(world, pos, player_pos, tunnels, simplex, subtask):
    x, y = pos
    simplex = functools.partial(_simplex, simplex)
    uniform = world.random.uniform
    start = 4/8 - np.sqrt((x - player_pos[0]) ** 2 + (y - player_pos[1]) ** 2)
    start += 2 * simplex(x, y, 8, 3/4)
    start = 1 / (1 + np.exp(-start))
    if uniform() > 0.5:
        water = simplex(2*x, y/2, 3, {15/4: 1, 5/4: 0.15}, False) + 0.1
    else:
        water = simplex(x/2, 2*y, 3, {15/4: 1, 5/4: 0.15}, False) + 0.1
    water -= 2 * start
    mountain = simplex(x, y, 0, {15/4: 1, 5/4: 0.3})
    mountain -= 4 * start + 0.3 * water
    # original setting
    coal_uni_thr = 0.5
    iron_uni_thr = 0.5
    iron_simplex_thr = 0.3
    diamond_uni_thr = 0.85
    
    if subtask == "collect_diamond":
        if start > 0.2:
            world[x, y] = "grass"
        elif mountain > 0.0:
            if simplex(x, y, 6, 7/4) > 0.15 and mountain > 0.3:  # cave
                world[x, y] = "path"
            elif simplex(2 * x, y / 5, 7, 3/4) > 0.4:  # horizonal tunnle
                world[x, y] = "path"
                tunnels[x, y] = True
            elif simplex(x / 5, 2 * y, 7, 3/4) > 0.4:  # vertical tunnle
                world[x, y] = "path"
                tunnels[x, y] = True
            elif simplex(x, y, 1, 8/4) > 0 and uniform() >coal_uni_thr:
                if uniform() > 0.6:
                    world[x, y] = "coal"
                else:
                    world[x, y] = "path"
                    tunnels[x, y] = True
            elif simplex(x, y, 2, 6/4) > iron_simplex_thr and uniform() > iron_uni_thr:
                if uniform() > 0.6:
                    world[x, y] = "iron"
                else:
                    world[x, y] = "path"
                    tunnels[x, y] = True
            elif mountain > 0.18 and uniform() > diamond_uni_thr:
                world[x, y] = "diamond"
            else:
                if uniform() > 0.5:
                    world[x, y] = "stone"
                else:
                    world[x, y] = "path"
                    tunnels[x, y] = True
        # elif 0.65 < water <= 0.75 and simplex(x, y, 4, 9) > -0.2:
        #     world[x, y] = "sand"
        elif 0.3 < water:
            world[x, y] = "water"
        else:  # grassland
            if simplex(x, y, 5, 7/4) > -0.2 and uniform() > 0.5:
                if uniform() > 0.5:
                    world[x, y] = "tree"
                else:
                    world[x, y] = "path"
                    tunnels[x, y] = True
            else:
                world[x, y] = "grass"

    elif subtask == "collect_iron":
        if start > 0.2:
            world[x, y] = "grass"
        elif mountain > 0.0:
            if simplex(x, y, 6, 7/4) > 0.15 and mountain > 0.3:  # cave
                world[x, y] = "path"
            elif simplex(2 * x, y / 5, 7, 3/4) > 0.4:  # horizonal tunnle
                world[x, y] = "path"
                tunnels[x, y] = True
            elif simplex(x / 5, 2 * y, 7, 3/4) > 0.4:  # vertical tunnle
                world[x, y] = "path"
                tunnels[x, y] = True
            elif simplex(x, y, 1, 8/4) > 0 and uniform() >coal_uni_thr:
                if uniform() > 0.6:
                    world[x, y] = "coal"
                else:
                    world[x, y] = "path"
                    tunnels[x, y] = True
            elif simplex(x, y, 2, 6/4) > iron_simplex_thr and uniform() > iron_uni_thr:
                world[x, y] = "iron"
            elif mountain > 0.18 and uniform() > diamond_uni_thr:
                world[x, y] = "diamond"
            else:
                if uniform() > 0.5:
                    world[x, y] = "stone"
                else:
                    world[x, y] = "path"
                    tunnels[x, y] = True
        # elif 0.65 < water <= 0.75 and simplex(x, y, 4, 9) > -0.2:
        #     world[x, y] = "sand"
        elif 0.3 < water:
            world[x, y] = "water"
        else:  # grassland
            if simplex(x, y, 5, 7/4) > -0.2 and uniform() > 0.5:
                if uniform() > 0.5:
                    world[x, y] = "tree"
                else:
                    world[x, y] = "path"
                    tunnels[x, y] = True
            else:
                world[x, y] = "grass"
    
    else:
        if start > 0.2:
            world[x, y] = "grass"
        elif mountain > 0.0:
            if simplex(x, y, 6, 7/4) > 0.15 and mountain > 0.3:  # cave
                world[x, y] = "path"
            elif simplex(2 * x, y / 5, 7, 3/4) > 0.4:  # horizonal tunnle
                world[x, y] = "path"
                tunnels[x, y] = True
            elif simplex(x / 5, 2 * y, 7, 3/4) > 0.4:  # vertical tunnle
                world[x, y] = "path"
                tunnels[x, y] = True
            elif simplex(x, y, 1, 8/4) > 0 and uniform() > coal_uni_thr:
                world[x, y] = "coal"
            elif simplex(x, y, 2, 6/4) > iron_simplex_thr and uniform() > iron_uni_thr:
                world[x, y] = "iron"
            elif mountain > 0.18 and uniform() > diamond_uni_thr:
                world[x, y] = "diamond"
            else:
                world[x, y] = "stone"

        elif 0.3 < water:
            world[x, y] = "water"
        else:  # grassland
            if simplex(x, y, 5, 7/4) > -0.2 and uniform() > 0.5:
                world[x, y] = "tree"
            else:
                world[x, y] = "grass"


def _set_object(world, pos, player_pos, tunnels):
    x, y = pos
    uniform = world.random.uniform
    dist = np.sqrt((x - player_pos[0]) ** 2 + (y - player_pos[1]) ** 2)
    material, _ = world[x, y]
    if material not in constants.walkable:
        pass
    # elif dist > 3 and material == "grass" and uniform() > 0.985:
    #     world.add(objects.Cow(world, (x, y)))
    # elif dist > 10 and uniform() > 0.993:
    #     if world.level in [4]:
    #         world.add(objects.Zombie(world, (x, y), player))
    # elif material == "path" and tunnels[x, y] and uniform() > 0.95:
    #     if world.level in [4]:
    #         world.add(objects.Skeleton(world, (x, y), player))


def _simplex(simplex, x, y, z, sizes, normalize=True):
    if not isinstance(sizes, dict):
        sizes = {sizes: 1}
    value = 0
    for size, weight in sizes.items():
        if hasattr(simplex, "noise3d"):
            noise = simplex.noise3d(x / size, y / size, z)
        else:
            noise = simplex.noise3(x / size, y / size, z)
        value += weight * noise
    if normalize:
        value /= sum(sizes.values())
    return value