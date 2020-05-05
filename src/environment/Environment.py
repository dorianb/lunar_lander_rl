from environment.LunarLanderEnvironment import LunarLanderEnvironment


class Environment:

    def __new__(cls, config={}):

        name = config["env"]
        if name == "LunarLander":
            config["env"] = LunarLanderEnvironment
            return LunarLanderEnvironment(config["env_config"])
        else:
            raise Exception("{} environment is not supported".format(name))
