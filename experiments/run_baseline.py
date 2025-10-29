# experiments/run_baseline.py
import numpy as np
from simulator.env import FederatedClientSelectionEnv
import json, os

def random_policy(obs, k):
    n = obs.shape[0]
    idx = np.random.choice(n, size=k, replace=False)
    action = np.zeros(n, dtype=int)
    action[idx] = 1
    return action

def greedy_battery_policy(obs, k):
    # pick top-k by battery (obs[:,0])
    idx = np.argsort(-obs[:,0])[:k]
    action = np.zeros(obs.shape[0], dtype=int)
    action[idx] = 1
    return action

def run(env, policy_fn, n_episodes=5, out_dir="results/baselines", name="random"):
    os.makedirs(out_dir, exist_ok=True)
    logs = []
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_log = {"rewards":[], "acc_gain":[], "energy": []}
        while not done:
            action = policy_fn(obs, env.k)
            obs, r, done, info = env.step(action)
            ep_log["rewards"].append(r)
            ep_log["acc_gain"].append(info.get("acc_gain",0))
            ep_log["energy"].append(info.get("energy_cost",0))
        logs.append(ep_log)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(logs, f, indent=2)
    print(f"saved {name} logs to {out_dir}")

if __name__ == "__main__":
    env = FederatedClientSelectionEnv(n_clients=12, k=3, max_rounds=50, seed=0)
    run(env, random_policy, n_episodes=10, name="random")
    env = FederatedClientSelectionEnv(n_clients=12, k=3, max_rounds=50, seed=1)
    run(env, greedy_battery_policy, n_episodes=10, name="greedy_battery")
