from ctypes import c_int
from typing import Any, Dict, Final, List, Tuple, Union, cast

import numpy as np

from linear_procgen.feature_envs import FeatureEnv, StateInterface
from linear_procgen.gym3_util import recover_grid

# Array of L_1 distances along a grid of size at most (34, 34).
__DIST_ARRAY = np.array(
    [[np.abs(x) + np.abs(y) for x in range(-34, 35)] for y in range(-34, 35)]
)
DIAMOND_PERCENT = 12 / 400.0  # from miner.cpp in procgen


def get_dist_array(agent_x: int, agent_y: int, width: int, height: int) -> np.ndarray:
    """Returns a (width, height) array containing the distance between the agent and each cell in the grid."""
    return __DIST_ARRAY[
        34 - agent_x : 34 - agent_x + width, 34 - agent_y : 34 - agent_y + height
    ]


class MinerState(StateInterface):
    GRID_ITEM_NAMES = {
        1: "boulder",
        2: "diamond",
        3: "moving_boulder",
        4: "moving_diamond",
        5: "enemy",
        6: "exit",
        9: "dirt",
        10: "oob_wall",
        11: "mud",
        12: "dead_player",
        100: "space",
    }

    def __init__(
        self,
        grid_size: Tuple[int, int],
        grid: np.ndarray,
        agent_pos: Tuple[int, int],
        exit_pos: Tuple[int, int],
    ) -> None:
        self.grid = recover_grid(grid, grid_size)
        assert len(self.grid.shape) == 2
        self.agent_pos = agent_pos
        self.exit_pos = exit_pos

    @staticmethod
    def grid_item_codes() -> Dict[str, int]:
        return {name: code for code, name in MinerState.GRID_ITEM_NAMES.items()}

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MinerState):
            return False
        if not np.array_equal(self.grid, other.grid):
            return False
        if not self.agent_pos == other.agent_pos:
            return False
        if not self.exit_pos == other.exit_pos:
            return False
        return True

    def to_bytes(self) -> bytes:
        c_grid = self.grid.transpose().flatten()
        c_grid.resize(35 * 35)
        return (
            bytes(c_int(self.grid.shape[0]))
            + bytes(c_int(self.grid.shape[1]))
            + bytes(c_int(35 * 35))
            + bytes(np.ctypeslib.as_ctypes(c_grid))
            + bytes(c_int(self.agent_pos[0]))
            + bytes(c_int(self.agent_pos[1]))
            + bytes(c_int(self.exit_pos[0]))
            + bytes(c_int(self.exit_pos[1]))
        )


class Miner(FeatureEnv[MinerState]):
    ACTION_DICT: Final = {
        "up": 5,
        "down": 3,
        "left": 1,
        "right": 7,
        "stay": 4,
    }

    class State(MinerState):
        # Allows the typing for Miner.State to work.
        pass

    def __init__(
        self,
        reward_weights: np.ndarray,
        num: int,
        center_agent: bool = True,
        use_backgrounds: bool = True,
        use_monochrome_assets: bool = False,
        restrict_themes: bool = False,
        use_generated_assets: bool = False,
        paint_vel_info: bool = False,
        distribution_mode: str = "hard",
        normalize_features: bool = False,
        **kwargs,
    ) -> None:
        if reward_weights.shape[0] != 6:
            raise ValueError(f"Must supply 6 reward weights, {reward_weights=}")

        self.stale_features = True
        self._reward_weights = reward_weights
        self._n_features = reward_weights.shape[0]
        self.use_normalized_features = normalize_features
        super().__init__(
            num=num,
            env_name="miner",
            center_agent=center_agent,
            use_backgrounds=use_backgrounds,
            use_monochrome_assets=use_monochrome_assets,
            restrict_themes=restrict_themes,
            use_generated_assets=use_generated_assets,
            paint_vel_info=paint_vel_info,
            distribution_mode=distribution_mode,
            **kwargs,
        )
        self.states = self.make_latent_states()
        self.last_diamonds = np.ones(num, dtype=int) * -1
        self.last_muds = np.ones(num, dtype=int) * -1
        self.diamonds = np.array(
            [Miner.diamonds_remaining(state) for state in self.states], dtype=int
        )
        self.muds = np.array(
            [Miner.muds_remaining(state) for state in self.states], dtype=int
        )
        self.firsts = [True] * num

        self.get_features()

    def act(self, action: np.ndarray) -> None:
        super().act(action)
        self.last_diamonds = self.diamonds
        self.last_muds = self.muds
        self.states = self.make_latent_states()
        self.diamonds = np.array(
            [Miner.diamonds_remaining(state) for state in self.states], dtype=np.float32
        )
        self.muds = np.array(
            [Miner.muds_remaining(state) for state in self.states], dtype=np.float32
        )
        self.stale_features = True

    def observe(self) -> Tuple[np.ndarray, Any, Any]:
        _, observations, self.firsts = super().observe()

        rewards = self.get_features() @ self._reward_weights

        return rewards, observations, self.firsts

    def make_latent_states(self) -> List[MinerState]:
        infos = self.get_info()
        return [self.make_latent_state(info) for info in infos]

    @staticmethod
    def make_latent_state(info: Dict[str, Any]) -> MinerState:
        agent_pos = cast(Tuple[int, int], tuple(info["agent_pos"]))
        exit_pos = cast(Tuple[int, int], tuple(info["exit_pos"]))
        return Miner.State(info["grid_size"], info["grid"], agent_pos, exit_pos)

    def get_features(self) -> np.ndarray:
        if not self.stale_features:
            return self.features
        dangers = np.array([self.in_danger(state) for state in self.states])
        dists = np.array(
            [
                Miner.dist_to_diamond(state, diamonds_remaining)
                for state, diamonds_remaining in zip(self.states, self.diamonds)
            ],
            dtype=np.float32,
        )
        pickup = np.array(
            [
                Miner.got_diamond(n_diamonds, last_n_diamonds, first)
                for n_diamonds, last_n_diamonds, first in zip(
                    self.diamonds, self.last_diamonds, self.firsts
                )
            ]
        )
        step_in_mud = np.array(
            [
                Miner.got_mud(n_mud, last_n_mud, first)
                for n_mud, last_n_mud, first in zip(
                    self.muds, self.last_muds, self.firsts
                )
            ]
        )

        diamonds = np.array(self.diamonds, dtype=np.float32)
        exits = np.array(
            [
                Miner.reached_exit(state, n_diamonds)
                for state, n_diamonds in zip(self.states, self.diamonds)
            ]
        )

        assert len(pickup) == self.num
        assert len(dangers) == self.num
        assert len(dists) == self.num
        assert len(diamonds) == self.num
        assert len(step_in_mud) == self.num
        assert len(exits) == self.num

        if self.use_normalized_features:
            max_dist = float(self.states[0].grid.shape[0] * 2 - 1)
            dists /= max_dist

            max_diamonds = DIAMOND_PERCENT * self.states[0].grid.size
            diamonds /= max_diamonds

        features = np.array(
            [pickup, step_in_mud, dangers, dists, diamonds, exits], dtype=np.float32
        ).T
        assert features.shape == (self.num, self._n_features)

        self.features = features
        self.stale_features = False
        return features

    def set_miner_state(self, states: List[MinerState]) -> None:
        assert len(states) == self.num
        for i in range(self.num):
            c_state = states[i].to_bytes()
            self.call_c_func("set_miner_state", i, c_state, len(c_state))

    @property
    def n_features(self) -> int:
        return self._n_features

    @staticmethod
    def in_danger(
        state: MinerState, return_time_to_die: bool = False, debug: bool = False
    ) -> Union[bool, Tuple[bool, int]]:
        agent_x, agent_y = state.agent_pos
        codes = MinerState.grid_item_codes()
        MOVING_OBJECTS = {codes["moving_diamond"], codes["moving_boulder"]}
        SOLID_OBJECTS = {
            codes["dirt"],
            codes["mud"],
            codes["boulder"],
            codes["diamond"],
            codes["oob_wall"],
        }
        DANGEROUS_OBJECTS = {
            codes["boulder"],
            codes["diamond"],
            codes["moving_diamond"],
            codes["moving_boulder"],
        }
        STATIONARY_OBJECTS = {codes["dirt"], codes["mud"], codes["oob_wall"]}
        # You can't be in danger if there's nothing above you
        if agent_y + 1 >= state.grid.shape[1]:
            return (False, -1) if return_time_to_die else False

        # You are only in danger if the thing directly above you is moving
        above = state.grid[agent_x, agent_y + 1]
        if above in MOVING_OBJECTS:
            return (True, 1) if return_time_to_die else True
        elif above in SOLID_OBJECTS:
            return (False, -1) if return_time_to_die else False

        for y in range(agent_y + 2, state.grid.shape[1]):
            if state.grid[agent_x, y] in DANGEROUS_OBJECTS:
                t = y - agent_y
                return (True, t) if return_time_to_die else True
            elif state.grid[agent_x, y] in STATIONARY_OBJECTS:
                return (False, -1) if return_time_to_die else False

        return (False, -1) if return_time_to_die else False

    @staticmethod
    def dist_to_diamond(
        state: MinerState, diamonds_remaining: int, return_pos: bool = False
    ) -> Union[int, Tuple[int, Tuple[int, int]]]:
        """Determines the distance between the agent and the nearest diamond, or 0 if there are no diamonds remaining.

        Args:
            state (MinerState): Current state of the miner environment.
            diamonds_remaining (int): Number of diamonds remaining in the environment. We could recompute this from the state but we already have it.
            return_pos (bool, optional): Additionally return the position of that nearest diamond. Defaults to False.

        Returns:
            Union[int, Tuple[int, Tuple[int, int]]]: Either the distance to the nearest diamond, or the distance and position of the nearest diamond.
        """
        if diamonds_remaining == 0:
            if return_pos:
                return 0, (-1, -1)
            else:
                return 0

        agent_x, agent_y = state.agent_pos
        width, height = state.grid.shape

        codes = MinerState.grid_item_codes()
        diamonds = cast(
            np.ndarray,
            np.logical_or(
                state.grid == codes["diamond"], state.grid == codes["moving_diamond"]
            ),
        )

        # Because our arrays are small and our cpu is good at array ops, it's faster to find the distance of the nearest
        # diamond by precomputing an array of L1 distances, finding the distances to diamonds by masking, and then
        # finding the minimum. Searching via floodfill/bfs turned out to be slower.
        dists = get_dist_array(agent_x, agent_y, width, height)
        diamond_dists = np.ma.array(dists, mask=np.logical_not(diamonds))
        pos_closest_diamond = cast(
            Tuple[int, int],
            np.unravel_index(diamond_dists.argmin(), diamond_dists.shape),
        )
        min_dist = diamond_dists[pos_closest_diamond]

        if return_pos:
            return min_dist, pos_closest_diamond
        else:
            return min_dist

    @staticmethod
    def diamonds_remaining(state: MinerState) -> int:
        """Returns how many diamonds are left in the environment."""
        codes = MinerState.grid_item_codes()
        return np.sum(
            (state.grid == codes["diamond"]) | (state.grid == codes["moving_diamond"])
        )

    @staticmethod
    def muds_remaining(state: MinerState) -> int:
        """Returns how many tiles of mud there are in the environment."""
        return np.sum((state.grid == MinerState.grid_item_codes()["mud"]))

    @staticmethod
    def got_diamond(n_diamonds: int, last_n_diamonds: int, first: bool) -> bool:
        """Returns if the agent has just picked up a diamond this timestep."""
        if first:
            return False

        if n_diamonds > last_n_diamonds:
            raise Exception(
                f"Diamonds increased from {n_diamonds} to {last_n_diamonds} and first={first}."
            )
        return n_diamonds != last_n_diamonds

    @staticmethod
    def got_mud(n_mud: int, last_n_mud: int, first: bool) -> bool:
        """Returns if the agent has just dug through some mud this timestep."""
        if first:
            return False

        if n_mud > last_n_mud:
            raise Exception(
                f"Mud tiles increased from {last_n_mud} to {n_mud} and first={first}."
            )
        return n_mud != last_n_mud

    @staticmethod
    def reached_exit(state: MinerState, n_diamonds: int) -> bool:
        """Returns if the agent has successfully mined all the diamonds and left the mine."""
        return n_diamonds == 0 and state.agent_pos == state.exit_pos
