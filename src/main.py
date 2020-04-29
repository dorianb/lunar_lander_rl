import argparse
import os
import time
import ray
from ray import tune
from ray.rllib.agents import dqn
from ray.rllib.agents import ppo

from environment.LunarLanderEnvironment import LunarLanderEnvironment


parser = argparse.ArgumentParser(description='Lunar lander RL')
parser.add_argument('--environment', type=str, help="Environment", default="LunarLander-v2")
parser.add_argument('--checkpoint-path', type=str, help="Path to checkpoint", default="")
parser.add_argument('--mode', type=str, help='Mode', default="train",
                    choices=["train", "infer"])
args = parser.parse_args()

ray.init()

if args.mode == "train":

    tune.run(
        dqn.DQNTrainer,
        name="DQN",
        stop={"episode_reward_mean": 200},
        config={
            "env": args.environment,
            "num_gpus": 0,
            "num_workers": 1,
            #"lr": tune.grid_search([0.01, 0.001, 0.0001]),
            "eager": False,
            "monitor": True
        },
        local_dir="~/workspace/lunar_lander_rl/results",
        trial_name_creator=lambda x: "trial",
        checkpoint_freq=10,
        checkpoint_at_end=True
    )

elif args.mode == "infer":

    env = LunarLanderEnvironment({})

    config = dqn.DEFAULT_CONFIG.copy()
    config["explore"] = False
    agent = dqn.DQNTrainer(env=LunarLanderEnvironment, config=config)
    assert os.path.exists(args.checkpoint_path), "Checkpoint path {} is invalid".format(args.checkpoint_path)
    print("Restoring from checkpoint path", args.checkpoint_path)
    agent.restore(args.checkpoint_path)

    obs = env.reset()
    rewards = 0
    action = None
    reward = None
    done = False

    while not done:
        action = agent.compute_action(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.05)
        rewards += reward

    print("Reward = {}".format(rewards))

