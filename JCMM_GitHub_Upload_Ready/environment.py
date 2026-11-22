import traci
import random

DRIVER_DISTRIBUTION = {
    "normal": 0.70,
    "aggressive": 0.10,
    "conservative": 0.10,
    "non_compliant": 0.10
}
class SumoEnvironment:

    def __init__(self, config_file):
        self.config_file = config_file

        # Fixed driver behaviour for each vehicle
        self.driver_profiles = {}

    def reset(self):
        traci.start(["sumo-gui", "-c", self.config_file])
        self.driver_profiles = {}

    def step(self):
        traci.simulationStep()

    def get_vehicles(self):
        return traci.vehicle.getIDList()

    def get_state(self, veh_id):
        pos = traci.vehicle.getPosition(veh_id)
        speed = traci.vehicle.getSpeed(veh_id)
        wait = traci.vehicle.getWaitingTime(veh_id)
        lane = traci.vehicle.getLaneIndex(veh_id)

        return pos, speed, wait, lane

    def get_route(self, veh_id):
        return traci.vehicle.getRouteID(veh_id)

    def apply_speed(self, veh_id, commanded_speed):

        traci.vehicle.setSpeedMode(veh_id, 31)

        # -------------------------------------------------------
        # Assign ONE driver profile for the whole simulation
        # -------------------------------------------------------
        if veh_id not in self.driver_profiles:

            r = random.random()

            cumulative=0.0
            for profile, probability in DRIVER_DISTRIBUTION.items():
               cumulative+=probability
               if r < cumulative:
                  self.driver_profiles[veh_id]=profile
                  break
        profile = self.driver_profiles[veh_id]

        # -------------------------------------------------------
        # Driver executes the commanded speed differently
        # -------------------------------------------------------

        if profile == "normal":

            actual_speed = commanded_speed

        elif profile == "aggressive":

            # Drives faster than instructed
            actual_speed = min(commanded_speed * 1.20, 8.0)

        elif profile == "conservative":

            # Drives slower than instructed
            actual_speed = commanded_speed * 0.80

        else:   # non_compliant

            # Sometimes obeys, sometimes not
            actual_speed = commanded_speed * random.uniform(0.7, 1.3)

            # Respect vehicle speed limit
            actual_speed = max(0.0, min(actual_speed, 8.0))

        traci.vehicle.setSpeed(veh_id, actual_speed)

    def close(self):
        traci.close()