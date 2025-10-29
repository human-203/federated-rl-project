# simulator/env.py
import gym
from gym import spaces
import numpy as np

class FederatedClientSelectionEnv(gym.Env):
    """
    Observations: for each client -> [battery_level, data_quantity, data_quality, past_contribution, uplink_speed]
    Action: select K clients (we encode as a binary vector of length N or as top-k indices)
    Reward: +delta_val_acc - energy_cost - deadline_penalty
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, n_clients=10, k=3, max_rounds=100, seed=None):
        super().__init__()
        self.n = n_clients
        self.k = k
        self.max_rounds = max_rounds
        self.rng = np.random.RandomState(seed)
        # observation: per-client 5 features
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.n,5), dtype=np.float32)
        # action: choose K clients -> we return indices in wrapper/agent. For RL you may use MultiBinary(n) with constraint
        self.action_space = spaces.MultiBinary(self.n)
        self.round = 0
        self._gen_clients()

    def _gen_clients(self):
        # initialize client state
        # battery (0-1), data quantity (0-1), data quality (0-1), past_contribution (0-1), uplink_speed (0-1)
        self.clients = self.rng.rand(self.n,5).astype(np.float32)
        # scale some fields if you like
        self.global_val_acc = 0.0

    def step(self, action):
        assert action.shape == (self.n,)
        selected = np.where(action==1)[0]
        if len(selected) == 0:
            # penalize empty select
            reward = -0.1
            done = False
            self.round += 1
            return self._get_obs(), reward, done, {}
        # cap to k by selecting top k indices (if more were selected)
        if len(selected) > self.k:
            # simple rule: take first k
            selected = selected[:self.k]

        # compute energy costs and expected accuracy gain
        energy_cost = 0.0
        acc_gain = 0.0
        deadline_miss = 0.0
        for i in selected:
            battery, data_q, data_qual, past, uplink = self.clients[i]
            # energy consumed proportional to data quantity and poor uplink
            cost = 0.2 * data_q * (1.0 + (1.0 - uplink))
            energy_cost += cost
            # accuracy contribution: data_quality * log(1+data_quantity)
            acc_gain += float(data_qual * np.log1p(data_q*10) * (1.0 - past*0.2))
            # deadline: if uplink too small, penalty
            if uplink < 0.2:
                deadline_miss += 0.1

            # update client battery
            self.clients[i,0] = max(0.0, battery - cost)

            # increment past contribution to discourage repeated selection
            self.clients[i,3] = min(1.0, self.clients[i,3] + 0.1)

        # reward = acc_gain - energy_cost - deadline_penalty
        reward = acc_gain - energy_cost - deadline_miss

        # update global validation accuracy approximation
        self.global_val_acc += 0.01 * acc_gain
        self.round += 1
        done = self.round >= self.max_rounds

        # optionally add small stochastic dynamics (clients recharge a bit)
        recharge = self.rng.rand(self.n) * 0.01
        self.clients[:,0] = np.clip(self.clients[:,0] + recharge, 0.0, 1.0)

        return self._get_obs(), float(reward), done, {"acc_gain": acc_gain, "energy_cost": energy_cost}

    def reset(self):
        self.round = 0
        self._gen_clients()
        return self._get_obs()

    def _get_obs(self):
        return self.clients.copy()

    def render(self, mode="human"):
        print(f"Round {self.round}: global_val_acc={self.global_val_acc:.3f}")
