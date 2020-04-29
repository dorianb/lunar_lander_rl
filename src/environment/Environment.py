from environment.LunarLanderEnvironment import LunarLanderEnvironment


class Environment:

    def __new__(cls, name, env_config={}):

        if name == "LunarLander":
            return LunarLanderEnvironment(False, env_config)
        else:
            raise Exception("{} environment is not supported".format(name))
