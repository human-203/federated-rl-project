# agents/train_dqn.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from simulator.env import FederatedClientSelectionEnv
from collections import deque
import random
import os
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNNet(nn.Module):
    def __init__(self, n_clients, n_feats=5, hidden=128):
        super().__init__()
        self.input_dim = n_clients * n_feats
        self.n_clients = n_clients
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_clients)  # Q value per client
        )
    def forward(self, obs):
        # obs: n_clients x n_feats -> flatten
        x = obs.flatten().astype(np.float32)
        x = torch.tensor(x, device=device).unsqueeze(0)
        return self.fc(x)  # shape (1, n_clients)

def select_action(q_values, k):
    # q_values: numpy 1D array len n_clients
    idx = np.argsort(-q_values)[:k]
    action = np.zeros_like(q_values, dtype=int)
    action[idx] = 1
    return action

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buf = deque(maxlen=capacity)
    def push(self, s, a, r, s2, done):
        self.buf.append((s,a,r,s2,done))
    def sample(self, bs):
        return random.sample(self.buf, bs)
    def __len__(self):
        return len(self.buf)

def train(env, n_episodes=200, k=3, out_dir="results/dqn"):
    os.makedirs(out_dir, exist_ok=True)
    n_clients = env.n
    net = DQNNet(n_clients).to(device)
    target = DQNNet(n_clients).to(device)
    target.load_state_dict(net.state_dict())
    opt = optim.Adam(net.parameters(), lr=1e-3)
    buf = ReplayBuffer(20000)
    gamma = 0.99
    batch_size = 32
    sync_every = 50
    eps = 1.0
    eps_min = 0.05
    eps_decay = 0.995

    for ep in range(n_episodes):
        s = env.reset()
        done = False
        total_r = 0.0
        while not done:
            q = net(s).detach().cpu().numpy().squeeze()
            if random.random() < eps:
                # random action (top-k random)
                a = np.zeros(n_clients, dtype=int)
                chosen = np.random.choice(n_clients, k, replace=False)
                a[chosen] = 1
            else:
                a = select_action(q, k)
            s2, r, done, info = env.step(a)
            buf.push(s, a, r, s2, done)
            total_r += r

            # train
            if len(buf) > batch_size:
                batch = buf.sample(batch_size)
                states = np.stack([b[0].flatten() for b in batch])
                actions = np.stack([b[1] for b in batch])
                rewards = np.array([b[2] for b in batch], dtype=np.float32)
                next_states = np.stack([b[3].flatten() for b in batch])
                dones = np.array([b[4] for b in batch], dtype=np.float32)

                states_t = torch.tensor(states, device=device)
                next_states_t = torch.tensor(next_states, device=device)
                rewards_t = torch.tensor(rewards, device=device)
                dones_t = torch.tensor(dones, device=device)

                q_values = net(states_t)  # (bs, n_clients)
                # estimate Q for selected clients by summing Qs of selected clients
                q_sa = (q_values * torch.tensor(actions, device=device, dtype=torch.float32)).sum(dim=1)

                with torch.no_grad():
                    next_q = target(next_states_t)
                    # greedy sum of top-k
                    topk_vals, _ = torch.topk(next_q, k, dim=1)
                    next_q_sa = topk_vals.sum(dim=1)

                target_vals = rewards_t + gamma * (1.0 - dones_t) * next_q_sa
                loss = nn.functional.mse_loss(q_sa, target_vals)

                opt.zero_grad()
                loss.backward()
                opt.step()

            s = s2

        eps = max(eps*eps_decay, eps_min)
        if ep % sync_every == 0:
            target.load_state_dict(net.state_dict())

        print(f"EP {ep} total_r={total_r:.3f} eps={eps:.3f}")

        # optional: save model periodically
        if ep % 50 == 0:
            torch.save(net.state_dict(), os.path.join(out_dir, f"dqn_ep{ep}.pth"))

    torch.save(net.state_dict(), os.path.join(out_dir, f"dqn_final.pth"))

if __name__ == "__main__":
    env = FederatedClientSelectionEnv(n_clients=12, k=3, max_rounds=50, seed=42)
    train(env, n_episodes=300, k=3)
