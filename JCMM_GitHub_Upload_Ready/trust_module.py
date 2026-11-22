import numpy as np

class TrustModule:
    def __init__(self):
        self.trust = {}

    def init(self, veh_id):
        if veh_id not in self.trust:
            self.trust[veh_id] = 1.0

    def update(self, veh_id, speed, wait_time):
        self.init(veh_id)

        efficiency = speed / 10
        penalty = wait_time / 50

        score = efficiency - penalty

        self.trust[veh_id] = 0.9 * self.trust[veh_id] + 0.1 * score
        self.trust[veh_id] = np.clip(self.trust[veh_id], 0.1, 2.0)

    def get(self, veh_id):
        self.init(veh_id)
        return self.trust[veh_id]