# ==========================================
# LunarLander_PPO_Train.py
# Huấn luyện mô hình PPO trên môi trường LunarLander-v3
# ==========================================
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import os


# ==========================================
# 1. Mô hình Actor–Critic
# ==========================================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.actor(x), self.critic(x)

    def act(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value


# ==========================================
# 2. Hàm tính GAE (Generalized Advantage Estimation)
# ==========================================
def compute_gae(rewards, dones, values, gamma=0.99, lam=0.95):
    advantages, gae = [], 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * (1 - dones[i]) - values[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        advantages.insert(0, gae)
    returns = np.array(advantages) + values[:-1]
    return np.array(advantages, dtype=np.float32), np.array(returns, dtype=np.float32)


# ==========================================
# 3. Cập nhật PPO
# ==========================================
def ppo_update(
    model,
    optimizer,
    states,
    actions,
    log_probs_old,
    returns,
    advantages,
    clip_eps=0.2,
    epochs=4,
    batch_size=64,
):
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    dataset_size = len(states)

    for _ in range(epochs):
        idxs = np.arange(dataset_size)
        np.random.shuffle(idxs)

        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            batch_idx = idxs[start:end]

            s_batch = states[batch_idx]
            a_batch = actions[batch_idx]
            logp_old_batch = log_probs_old[batch_idx]
            ret_batch = returns[batch_idx]
            adv_batch = advantages[batch_idx]

            logits, values = model(s_batch)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(a_batch)
            ratio = torch.exp(logp - logp_old_batch)

            surr1 = ratio * adv_batch
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_batch
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values.squeeze(), ret_batch)
            entropy = dist.entropy().mean()

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# ==========================================
# 4. Hàm huấn luyện chính
# ==========================================
def train_ppo(
    env_name="LunarLander-v3", max_episodes=2000, lr=3e-4, gamma=0.99, lam=0.95
):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ActorCritic(state_dim, action_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs("models", exist_ok=True)
    reward_history = []
    avg_reward_history = deque(maxlen=100)

    for episode in range(max_episodes):
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
        obs, _ = env.reset(seed=np.random.randint(0, 10000))
        total_reward = 0

        for _ in range(2048):
            s = torch.tensor(obs, dtype=torch.float32)
            action, logp, value = model.act(s)
            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            states.append(s)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(logp)
            values.append(value.item())

            obs = next_obs
            total_reward += reward

            if done:
                obs, _ = env.reset()

        # Giá trị cuối cùng
        _, _, next_value = model.act(torch.tensor(obs, dtype=torch.float32))
        values.append(next_value.item())

        advs, rets = compute_gae(rewards, dones, values, gamma, lam)
        ppo_update(
            model,
            optimizer,
            torch.stack(states),
            torch.stack(actions),
            torch.stack(log_probs),
            torch.tensor(rets),
            torch.tensor(advs),
        )

        reward_history.append(total_reward)
        avg_reward_history.append(total_reward)
        avg_r = np.mean(avg_reward_history)

        print(
            f"Tập {episode + 1:4d} | Phần thưởng: {total_reward:7.2f} | Trung bình 100 tập: {avg_r:7.2f}"
        )

        # Lưu mô hình khi đạt >200 điểm trung bình
        if avg_r >= 200:
            path = f"models/ppo_lander_{episode + 1}_s_{int(avg_r)}.pth"
            torch.save(model.state_dict(), path)
            print(f"✅ Đã lưu mô hình tại: {path}")
            print("🚀 Tàu đã học cách hạ cánh an toàn!")
            break

    env.close()

    # ==========================================
    # 5. Hiển thị đồ thị kết quả
    # ==========================================
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, label="Reward mỗi tập", alpha=0.6)
    plt.plot(
        [
            np.mean(reward_history[max(0, i - 100) : i + 1])
            for i in range(len(reward_history))
        ],
        label="Trung bình 100 tập",
        color="orange",
    )
    plt.xlabel("Tập")
    plt.ylabel("Phần thưởng")
    plt.title("Quá trình học của PPO trên LunarLander-v3")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_ppo()
