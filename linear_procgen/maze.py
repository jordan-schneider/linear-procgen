import logging
from typing import Any, Dict, Final, List, Optional, Sequence, Set, Tuple, cast

import numpy as np

from linear_procgen.feature_envs import FeatureEnv, StateInterface
from linear_procgen.gym3_util import recover_grid


class MazeState(StateInterface):
    """Latent state of the maze procgen environment. Consists only of a grid of objects and the agent's position."""

    def __init__(
        self,
        grid_size: Tuple[int, int],
        grid: np.ndarray,
        agent_pos: Tuple[int, int],
    ) -> None:
        self.grid = recover_grid(grid, grid_size)
        assert len(self.grid.shape) == 2
        self.agent_pos = agent_pos

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MazeState):
            return False
        if not np.array_equal(self.grid, other.grid):
            return False
        if not self.agent_pos == other.agent_pos:
            return False
        return True


class Maze(FeatureEnv[MazeState]):
    """Procgen's Maze environment with a linear reward function.

    The reward for each state is linear in two features:
    1. The shortest path distance to the goal.
    2. If the agent just took a step closer to the goal.
    """

    ACTION_DICT: Final = {
        "up": 5,
        "down": 3,
        "left": 1,
        "right": 7,
        "stay": 4,
    }
    STATE_ID_MOVABLE: Final = (2, 100)
    N_FEATURES = 2

    class State(MazeState):
        # Allows the typing for Maze.State to work
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
        **kwargs,
    ) -> None:
        """Constructs a procgen 3 maze environment using the provided reward weights. All undocument arguments are as in procgen.envs.maze.Maze.

        Args:
            reward_weights (np.ndarray): An array of shape (2,) containing the weights on each reward feature. The reward at each timestep is exactly reward_weights.T @ features.
        """
        super().__init__(
            num=num,
            env_name="maze",
            center_agent=center_agent,
            use_backgrounds=use_backgrounds,
            use_monochrome_assets=use_monochrome_assets,
            restrict_themes=restrict_themes,
            use_generated_assets=use_generated_assets,
            paint_vel_info=paint_vel_info,
            distribution_mode=distribution_mode,
            **kwargs,
        )

        self._reward_weights = reward_weights

        # Is this the first state of an episode? for each of num environments
        self.firsts = np.ones(num, dtype=bool)
        self.make_latent_states()
        self.shortest_paths = self.get_shortest_paths()
        self.last_dists = self.get_dists(self.shortest_paths)
        self.features = np.stack((self.last_dists, np.zeros(num, dtype=int)), axis=1)
        self.features_computed = True

    def act(self, action: np.ndarray) -> None:
        super().act(action)
        self.last_actions = action
        self.features_computed = False
        self.states_computed = False

    def observe(self) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        _, observations, firsts = super().observe()
        self.firsts = firsts
        self.features = self.get_features()
        rewards = self.features @ self._reward_weights

        return rewards, observations, firsts

    def get_features(self) -> np.ndarray:
        if self.features_computed:
            return self.features
        self.features_computed = True

        self.make_latent_states()

        self.shortest_paths = self.get_shortest_paths()
        dists = self.get_dists(self.shortest_paths)
        dist_decrease = self.last_dists - dists
        dist_decrease[self.firsts] = 0

        assert np.all(np.abs(dist_decrease) <= 1)

        self.features = np.stack((dists, dist_decrease), axis=1)

        self.last_dists = dists
        return self.features

    def get_shortest_paths(self) -> List[List[Tuple[int, int]]]:
        """Get the shortest path to the exit of the maze from the current position for each environment.

        This function only recomputes a path from scratch if the environment has been reset.

        Returns:
            List[List[Tuple[int, int]]]: For each of self.num environments, a list of (x,y) pairs representing the
            sequence of states on the shortest path to the exit.
        """
        for i in range(self.num):
            state = self.make_latent_states()[i]
            path = self.shortest_paths[i]
            if self.firsts[i]:
                self.shortest_paths[i] = self.find_shortest_path(state)
            else:
                if state.agent_pos == (*path[1],):
                    # If the agent took a step along the shortest path, you know the rest of that path is still the shortest from the new position.
                    self.shortest_paths[i] = path[1:]
                elif state.agent_pos != (*path[0],):
                    # If the agent took a step in any other direction, then the shortest path is just going back to
                    # the place you were just at, and then following the shortest path from there.
                    # This assumes there's exactly one path out.
                    # Mazegen uses Kruskal's algorithm, this is true of minimum spanning trees.
                    path.insert(0, state.agent_pos)
                    self.shortest_paths[i] = path

                else:
                    # Current and previous postion are the same, you bumped into a wall, do nothing
                    pass
            assert state.agent_pos == self.shortest_paths[i][0]
            assert self.get_goal_state(state) == self.shortest_paths[i][-1]
        return self.shortest_paths

    def find_shortest_path(self, state: MazeState) -> List[Tuple[int, int]]:
        """Djikstra's algorithm to find the shortest path to the exit of the maze from the current position.

        This is a somewhat memory inefficient implementation, but our maze size is bounded by 31x31, so it should be fine.

        Returns:
            List[Tuple[int, int]]: A list of (x,y) pairs representing the sequence of states on the shortest path to the exit."""
        Position = Tuple[int, int]

        def get_neighbors(x: int, y: int) -> List[Position]:
            neighbors: List[Position] = []
            if x > 0 and state.grid[x - 1, y] in self.STATE_ID_MOVABLE:
                neighbors.append((x - 1, y))
            if (
                x < state.grid.shape[0] - 1
                and state.grid[x + 1, y] in self.STATE_ID_MOVABLE
            ):
                neighbors.append((x + 1, y))
            if y > 0 and state.grid[x, y - 1] in self.STATE_ID_MOVABLE:
                neighbors.append((x, y - 1))
            if (
                y < state.grid.shape[1] - 1
                and state.grid[x, y + 1] in self.STATE_ID_MOVABLE
            ):
                neighbors.append((x, y + 1))
            return neighbors

        current_state = state.agent_pos
        goal_state = self.get_goal_state(state)

        previous: Dict[Position, Optional[Position]] = {current_state: None}
        visited: Set[Position] = {current_state}
        queue: List[Tuple[Position, Position]] = []

        first_neighbors = get_neighbors(*current_state)
        queue.extend([(current_state, neighbor) for neighbor in first_neighbors])

        while len(queue) > 0 and goal_state not in visited:
            prev_state, current_state = queue.pop(0)
            visited.add(current_state)
            previous[current_state] = prev_state
            queue.extend(
                [
                    (current_state, neighbor)
                    for neighbor in get_neighbors(*current_state)
                    if neighbor not in visited
                ]
            )

        if goal_state not in visited:
            raise ValueError(
                f"No path from current {current_state} to goal {goal_state} in\n {state.grid}"
            )

        trace_state: Optional[Position] = goal_state
        path: List[Position] = []
        while trace_state is not None:
            path.append(trace_state)
            trace_state = previous[trace_state]
        path = list(reversed(path))
        return path

    def get_goal_state(self, state: MazeState) -> Tuple[int, int]:
        """Returns the (x,y) position of the exit of the maze described in the given MazeState."""
        GOAL_OBJ_ID = 2
        goal_state = np.where(state.grid == GOAL_OBJ_ID)
        return (int(goal_state[0][0]), int(goal_state[1][0]))

    def get_dists(self, paths: Sequence[Sequence[Tuple[int, int]]]) -> np.ndarray:
        """Returns a (self.num,) array of shortest-path distances from the agent to the exit for each environment."""
        return np.array([len(path) - 1 for path in paths])

    def make_latent_states(self) -> List[MazeState]:
        infos = self.get_info()
        self.states = [self.make_latent_state(info) for info in infos]
        self.states_computed = True
        return self.states

    @staticmethod
    def make_latent_state(info: Dict[str, Any]) -> MazeState:
        agent_pos = cast(Tuple[int, int], tuple(info["agent_pos"]))
        return MazeState(info["grid_size"], info["grid"], agent_pos)

    @property
    def n_features(self) -> int:
        return self.N_FEATURES
