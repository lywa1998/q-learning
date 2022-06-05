import argparse
import time

import gym
import numpy as np

from agents import QLearning, Sarsa


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="qlearning", type=str)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--episode", default=500, type=int)
    parser.add_argument("--path", default="./save", type=str)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    args = vars( args )
    return args


def train(env, agent, args):
    render = args["render"]

    rewards = np.zeros((args["episode"],))
    s = time.time()
    for episode in range(args["episode"]):
        ep_reward = 0
        obs = env.reset()  # start new episode
        done = False
        action = agent.sample(obs)
        while not done:
            next_obs, reward, done, _ = env.step(action)

            if args["agent"] == "qlearning":
                agent.learn(obs, action, reward, next_obs, done)
                next_action = agent.sample(next_obs)
            else:
                next_action = agent.sample(next_obs)
                agent.learn(obs, action, reward, next_obs, next_action, done)

            action = next_action
            ep_reward += reward
            if episode % 20 == 0 and render:
                env.render()
        rewards[episode] += ep_reward
    print(time.time() - s)
    np.save(f"{args['path']}/{args['agent']}", rewards)


def test(env, agents, args):
    pass


if __name__ == "__main__":
    args = get_args()
    # args['agent'] = "sarsa"
    # np.random.seed(args['seed'])
    env = gym.make("CliffWalking-v0")
    obs_dim = env.observation_space.n
    action_dim = env.action_space.n
    if args["agent"] == "qlearning":
        agent = QLearning(obs_dim, action_dim)
    elif args["agent"] == "sarsa":
        agent = Sarsa(obs_dim, action_dim)
    else:
        print("agent definition error")
        exit(-1)

    if args["train"]:
        train(env, agent, args)
    else:
        test(env, agent, args)