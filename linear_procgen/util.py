import logging
from typing import Literal, Optional, Union, overload

import numpy as np
from gym3 import ExtractDictObWrapper  # type: ignore
from gym3.env import Env  # type: ignore
from gym3.wrapper import Wrapper  # type: ignore
from procgen.env import ProcgenGym3Env

from linear_procgen import Maze, Miner
from linear_procgen.feature_envs import FeatureEnv

ENV_NAMES = Literal["maze", "miner"]


def make_env(
    name: ENV_NAMES,
    num: int,
    reward: Optional[Union[float, np.ndarray]] = None,
    extract_rgb: bool = True,
    **kwargs
) -> FeatureEnv:
    if name == "maze":
        assert reward is not None
        if not isinstance(reward, np.ndarray):
            reward = np.full(shape=2, fill_value=reward)
        env = Maze(reward, num, **kwargs)
    elif name == "miner":
        assert reward is not None
        if not isinstance(reward, np.ndarray):
            reward = np.full(shape=4, fill_value=reward)
        env = Miner(reward, num, **kwargs)
    else:
        env = ProcgenGym3Env(num=num, env_name=name)

    if extract_rgb:
        env = ExtractDictObWrapper(env, "rgb")
    return env


def get_root_env(env: Wrapper, max_layers: int = 100) -> Env:
    root_env = env
    layer = 0
    while isinstance(root_env, Wrapper) and layer < max_layers:
        root_env = root_env.env
        layer += 1
    if layer == max_layers:
        raise RuntimeError("Infinite loop looking for root_env")
    return root_env
