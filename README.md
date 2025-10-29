# Federated RL for Client Selection on Battery-Limited IoT ⚡

This project explores **Reinforcement Learning (RL)** for optimized **client selection** in **Federated Learning** environments, where IoT devices are constrained by limited battery and network bandwidth.

---

## 🚀 Objective
Develop and compare strategies for selecting which IoT clients participate in each federated learning round to balance:
- Model convergence speed,
- Energy efficiency,
- Communication fairness.

---

## 🧠 Approach Overview
We model the client selection as a **Reinforcement Learning problem**:
- **State (S):** Client battery, data size/quality, contribution history, uplink speed.  
- **Action (A):** Select K clients to train in the current round.  
- **Reward (R):** Improvement in global validation accuracy − energy cost − penalty for deadline misses.

**RL Agents:**  
Compare baseline strategies (Random, K-Greedy, FedCS) vs. RL-based policies such as **DQN**, **PPO**, and **Combinatorial UCB**.

---

## 🧩 Repository Structure
  federated-rl-project/
├─ simulator/
│ ├─ client.py # Client device logic
│ ├─ server.py # Federated aggregation logic
│ ├─ env.py # RL environment (Gym-compatible)
├─ agents/
│ ├─ train_dqn.py # DQN training loop
│ ├─ train_ppo.py # PPO training loop
├─ experiments/
│ ├─ run_baseline.py # Random & Greedy baseline runs
│ ├─ eval_policies.py # Evaluation across seeds
├─ notebooks/
│ ├─ Week2_Baselines.ipynb # Colab: baseline training
│ ├─ Week3_RL_Training.ipynb# Colab: RL policy training
│ ├─ Week4_Evaluation.ipynb # Colab: metrics + plots
├─ results/ # Logs, metrics, plots, models
├─ docs/
│ ├─ report.md / slides.pdf # Final report and presentation
├─ requirements.txt
└─ README.md


---

## ⚙️ Installation
To set up locally:
git clone https://github.com/human-203/federated-rl-project.git
cd federated-rl-project
python3 -m venv venv
source venv/bin/activate  # on Mac/Linux
pip install -r requirements.txt

## To run a simple simulation:
python experiments/run_baseline.py
