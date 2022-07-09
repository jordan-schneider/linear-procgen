from abc import ABC
from typing import Generic, List, Tuple, TypeVar

import numpy as np
from procgen.env import ProcgenGym3Env


class StateInterface(ABC):
    grid: np.ndarray
    agent_pos: Tuple[int, int]


S = TypeVar("S", bound=StateInterface)


class FeatureEnv(ProcgenGym3Env, Generic[S]):
    """Procgen Gym3 Environment that has a linear reward function, and exposes the features of that reward.

    Formally, the reward function of the environment takes the form r(s) = \vec{w}^T \phi(s), where \phi(s) is the reward feature vector of the current state.
    """

    _reward_weights: np.ndarray
    features: np.ndarray

    def get_features(self) -> np.ndarray:
        """Returns a 1d array of the values of the linear reward features at the current state."""
        raise NotImplementedError()

    def make_latent_states(self) -> List[S]:
        """Returns the latent state of the environment.

        This will consist of at least a grid from the underlying grid-world and the position of the agent within that grid.
        """
        raise NotImplementedError()

    @property
    def n_features(self) -> int:
        """Returns the number of features in the linear reward function."""
        raise NotImplementedError()
