from ray.rllib.agents import dqn, ppo


class Agent:

    def __new__(cls, config={}):

        name = config.pop('agent', None)
        if name == "DQN":
            return dqn.DQNTrainer(config=config)
        elif name == "PPO":
            return ppo.APPOTrainer(config=config)
        else:
            raise Exception("{} agent is not supported".format(name))