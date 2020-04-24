import gym


class LunarLanderEnvironment(gym.Env):

    def __init__(self, env_config):
        self.env = gym.make("LunarLander-v2")
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)