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
    run_num = 20  # 训练次数
    rewards = np.zeros((args["episode"],))
    for nn in range(run_num):
        agent.Q_table = np.zeros_like(agent.Q_table)
        s = time.time()
        for episode in range(args["episode"]):
            ep_reward = 0  # reward in an episode
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
                obs = next_obs
                action = next_action
                ep_reward += reward
                if reward == -100:
                    done = True
                if episode % 20 == 0 and render:
                    env.render()
            rewards[episode] += ep_reward
        print(f"Run number {nn} duration {time.time() - s} seconds")
    rewards /= run_num
    avg_rewards = []
    for i in range(9):
        avg_rewards.append(np.mean(rewards[:i + 1]))
    for i in range(10, len(rewards) + 1):
        avg_rewards.append(np.mean(rewards[i - 10:i]))
    np.save(f"{args['path']}/reward-{args['agent']}-test.npy", avg_rewards)  # save reward
    agent.save_model(f"{args['path']}/model-{args['agent']}-test.npy")  # save trained model


def test(env, agents, args):
    obs_dim = env.observation_space.n
    assert  obs_dim == 12 * 4, "classic environment' observation space should be 48."
    agent.load_model(f"{args['path']}/model-{args['agent']}.npy")
    run_num = 5
    for nn in range(run_num):
        print(f"========== Evaluate {nn} =========")
        map = np.full(obs_dim, -1).reshape(4, 12) # generate map
        obs = env.reset()
        done = False
        while not done:
            action = agent.sample(obs)
            map[obs // 12, obs % 12] = action
            next_obs, reward, done, _ = env.step(action)
            obs = next_obs
        for ii in range(map.shape[0]):
            for jj in range(map.shape[1]):
                if ii == map.shape[0] - 1 and jj == map.shape[1] - 1:  # arrive end point
                    print("G", end="")
                    continue
                action = map[ii][jj]
                if action == -1:
                    print("0 ", end="")
                elif action == 0:
                    print("↑ ", end="")
                elif action == 1:
                    print("→ ", end="")
                elif action == 2:
                    print("↓ ", end="")
                elif action == 3:
                    print("← ", end="")
            print("")
    pass


if __name__ == "__main__":
    args = get_args()
    np.random.seed(args['seed'])
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
