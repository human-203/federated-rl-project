# experiments/eval_policies.py
import json, os, numpy as np
import matplotlib.pyplot as plt

def load_and_plot(log_json):
    with open(log_json) as f:
        logs = json.load(f)
    avg_rewards = [sum(ep["rewards"]) for ep in logs]
    print("Avg reward:", np.mean(avg_rewards), "std:", np.std(avg_rewards))
    plt.figure()
    plt.plot([np.mean(ep["rewards"]) for ep in logs])
    plt.title(os.path.basename(log_json))
    plt.savefig(log_json.replace(".json", ".png"))

if __name__ == "__main__":
    for p in ["results/baselines/random.json", "results/baselines/greedy_battery.json"]:
        if os.path.exists(p):
            load_and_plot(p)
