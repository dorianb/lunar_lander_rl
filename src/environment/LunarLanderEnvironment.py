import gym
from gym import spaces
import numpy as np

from simulator.LunarLanderSimulator import LunarLanderSimulator


class LunarLanderEnvironment(gym.Env):

    def __init__(self, continuous, env_config):
        self.simulator = LunarLanderSimulator(continuous)

        self.prev_reward = None

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)

        if self.simulator.continuous:
            # Action is two floats [main engine, left-right engines].
            # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
            self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
        else:
            # Nop, fire left engine, main engine, right engine
            self.action_space = spaces.Discrete(4)

        self.reset()

    def render(self, mode='human'):
        return self.simulator.render(mode=mode)

    def reset(self):
        self.simulator.reset()
        self.prev_shaping = None
        return self.step(np.array([0, 0]) if self.simulator.continuous else 0)[0]

    def step(self, action):

        if self.simulator.continuous:
            action = np.clip(action, -1, +1).astype(np.float32)
        else:
            assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))

        self.simulator.step(action)

        is_awake = self.simulator.is_lander_awake()
        pos = self.simulator.get_lander_position()
        vel = self.simulator.get_lander_velocity()
        angle = self.simulator.get_lander_angle()
        angular_vel = self.simulator.get_lander_angular_velocity()
        leg_contact = self.simulator.get_lander_leg_contact()
        m_power = self.simulator.get_lander_main_power()
        s_power = self.simulator.get_lander_side_power()

        state = [
            pos["x"],
            pos["y"],
            vel["x"],
            vel["y"],
            angle,
            angular_vel,
            1.0 if leg_contact[0] else 0.0,
            1.0 if leg_contact[1] else 0.0
        ]
        assert len(state) == 8

        reward = 0
        shaping = \
            - 100 * np.sqrt(state[0] * state[0] + state[1] * state[1]) \
            - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3]) \
            - 100 * abs(state[4]) + 10 * state[6] + 10 * state[7]  # And ten points for legs contact, the idea is if you
        # lose contact again after landing, you get negative reward
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= m_power * 0.30  # less fuel spent is better, about -30 for heurisic landing
        reward -= s_power * 0.03

        done = False
        if self.simulator.game_over or abs(state[0]) >= 1.0:
            done = True
            reward = -100
        if not is_awake:
            done = True
            reward = +100
        return np.array(state, dtype=np.float32), reward, done, {}

    def close(self):
        self.simulator.close()