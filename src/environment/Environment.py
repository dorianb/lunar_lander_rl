from environment.LunarLanderEnvironment import LunarLanderEnvironment


class Environment:

    def __new__(cls, env_config={}):

        name = env_config["env"]
        if name == "LunarLander":
            env_config["env"] = LunarLanderEnvironment
            return LunarLanderEnvironment(env_config)
        else:
            raise Exception("{} environment is not supported".format(name))
