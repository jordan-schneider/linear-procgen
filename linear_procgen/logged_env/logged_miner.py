from typing import List

import numpy as np
from gym3.wrapper import Wrapper  # type: ignore
from linear_procgen.logged_env.writer import SequentialWriter
from linear_procgen.miner import Miner


class LoggedMiner(Wrapper):
    env: Miner

    def __init__(
        self, env: Miner, writer: SequentialWriter, ob_space=None, ac_space=None
    ):
        super().__init__(env, ob_space, ac_space)
        self.writer = writer
        # List of features within each episode
        self.features: List[List[np.ndarray]] = [[]] * self.env.num

    def act(self, ob: np.ndarray) -> None:
        self.env.act(ob)

        _, _, firsts = self.env.observe()
        for i in np.where(firsts)[0]:
            mean_features = np.mean(self.features[i], axis=0)
            self.writer.add_scalar("features/pickup_diamond", mean_features[0])
            self.writer.add_scalar("features/step_in_mud", mean_features[1])
            self.writer.add_scalar("features/danger", mean_features[2])
            self.writer.add_scalar("features/dist_to_diamond", mean_features[3])
            self.writer.add_scalar("features/diamonds_left", mean_features[4])
            self.writer.add_scalar("features/exit", mean_features[5])
            self.features[i] = []

        current_features = self.env.get_features()
        for i in range(self.env.num):
            self.features[i].append(current_features[i])
