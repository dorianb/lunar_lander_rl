import argparse
import os
import json
import time
import ray
from ray import tune

from environment.Environment import Environment
from agent.Agent import Agent


parser = argparse.ArgumentParser(description='RL toolkit')
parser.add_argument('--agent-config-path', type=str, help="Path to the agent configuration", default="")
parser.add_argument('--environment-config-path', type=str, help="Path to the environment configuration", default="")
parser.add_argument('--checkpoint-path', type=str, help="Path to checkpoint", default="")
parser.add_argument('--mode', type=str, help='Mode', default="train", choices=["train", "infer"])
args = parser.parse_args()

ray.init()

config = {}

with open(args.agent_config_path) as config_file:
    config.update(json.load(config_file))

with open(args.environment_config_path) as config_file:
    config.update(json.load(config_file))

env = Environment(config)
agent = Agent(config)

if args.mode == "train":

    config.update({

        "num_gpus": 0,
        "num_workers": 1,

        "eager": False,
        "monitor": False
    })

    tune.run(
        agent.__class__,
        name=env.__class__.__name__+"_"+agent.__class__.__name__,
        stop={"episode_reward_mean": 200},
        config=config,
        local_dir="~/workspace/RL_toolkit/results",
        trial_name_creator=lambda x: "trial",
        checkpoint_freq=10,
        checkpoint_at_end=True
    )

elif args.mode == "infer":

    config["explore"] = False

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

