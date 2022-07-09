import logging
from typing import Literal, Optional, Union, overload

import numpy as np
from gym3 import ExtractDictObWrapper  # type: ignore
from gym3.env import Env  # type: ignore
from gym3.wrapper import Wrapper  # type: ignore
from procgen.env import ProcgenGym3Env

from linear_procgen import Maze, Miner
from linear_procgen.feature_envs import FeatureEnv
from linear_procgen.logged_env.writer import SequentialWriter

ENV_NAMES = Literal["maze", "miner"]


def make_env(
    name: ENV_NAMES,
    num: int,
    reward: Optional[Union[float, np.ndarray]] = None,
    extract_rgb: bool = True,
    log_writer: Optional[SequentialWriter] = None,
    **kwargs,
) -> FeatureEnv:
    """Builds a FeatureEnv by name.

    Args:
        name (ENV_NAMES): Name of a feature environment.
        num (int): How many parallel environments to build.
        reward (Optional[Union[float, np.ndarray]], optional): Either the reward weight vector for the environment, or a scalar which will be taken as the weight on all values. Defaults to None but must be specified.
        extract_rgb (bool, optional): Whether the environment should observe just the pixels, or the raw gym3 output. Defaults to True.

    Returns:
        FeatureEnv: _description_
    """
    if name == "maze":
        assert reward is not None
        if not isinstance(reward, np.ndarray):
            reward = np.full(shape=2, fill_value=reward)
        env = Maze(reward, num, **kwargs)
    elif name == "miner":
        assert reward is not None
        if not isinstance(reward, np.ndarray):
            reward = np.full(shape=6, fill_value=reward)
        env = Miner(reward, num, **kwargs)
        if log_writer is not None:
            # Delay import because it assumes pytorch.
            from linear_procgen.logged_env.logged_miner import LoggedMiner

            env = LoggedMiner(env, log_writer)
    else:
        raise ValueError(f"Unknown env name: {name}. Supported names are {ENV_NAMES}")

    if extract_rgb:
        env = ExtractDictObWrapper(env, "rgb")
    return env


def get_root_env(env: Wrapper, max_layers: int = 100) -> Env:
    """Returns the base environment of a wrapped gym3 environment.

    Args:
        env (Wrapper): Wrapped environment to get the base environment of.
        max_layers (int, optional): Maximum number of wrappers to look through. Defaults to 100.

    Raises:
        RuntimeError: If there are more than max_layers wrappers on the env.

    Returns:
        Env: Base gym3 environment.
    """
    root_env = env
    layer = 0
    while isinstance(root_env, Wrapper) and layer < max_layers:
        root_env = root_env.env
        layer += 1
    if layer == max_layers:
        raise RuntimeError("Infinite loop looking for root_env")
    return root_env
