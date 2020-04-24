import ray
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.ppo import PPOTrainer

from environment import LunarLanderEnvironment


ray.init()

tune.run(
    DQNTrainer,
    stop={"episode_reward_mean": 200},
    config={
        #"env": LunarLanderEnvironment,
        "env": "LunarLander-v2",
        "num_gpus": 0,
        "num_workers": 1,
        #"lr": tune.grid_search([0.01, 0.001, 0.0001]),
        "eager": False,
        "monitor": True
    },
    local_dir="~/workspace/lunar_lander_rl/results"
)