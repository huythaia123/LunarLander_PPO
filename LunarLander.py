# LunarLander_PPO.py
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 1 Mô hình Actor-Critic
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
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value
        # return action.item(), dist.log_prob(action)


# 2 Hàm tính Advantage (GAE)
def compute_gae(rewards, dones, values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * (1 - dones[i]) - values[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        advantages.insert(0, gae)
    returns = np.array(advantages) + values[:-1]
    return np.array(advantages), returns


# 3 Hàm cập nhật PPO
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
    dataset_size = len(states)
    for _ in range(epochs):
        idxs = np.arange(dataset_size)
        np.random.shuffle(idxs)

        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            batch_idx = idxs[start:end]

            s_batch = states[batch_idx]
            a_batch = actions[batch_idx]
            logp_old_batch = log_probs_old[batch_idx].detach()
            ret_batch = returns[batch_idx].detach()
            adv_batch = advantages[batch_idx].detach()

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


# 4 Vòng huấn luyện chính
env = gym.make("LunarLander-v3")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model = ActorCritic(state_dim, action_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

max_episodes = 2000
steps_per_update = 2048
gamma, lam = 0.99, 0.95

reward_history = []

for episode in range(max_episodes):
    states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
    state, _ = env.reset()
    # obs, info = env.reset(seed=42)
    total_reward = 0

    for step in range(steps_per_update):
        s = torch.tensor(state, dtype=torch.float32)
        action, logp, value = model.act(s)
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        # next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        states.append(s)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        log_probs.append(logp)
        values.append(value.item())

        state = next_state
        total_reward += reward
        if done:
            state, _ = env.reset()

    # Giá trị cuối cùng để tính GAE
    _, _, next_value = model.act(torch.tensor(state, dtype=torch.float32))
    values.append(next_value.item())

    advs, rets = compute_gae(rewards, dones, values, gamma, lam)
    advs = (advs - np.mean(advs)) / (np.std(advs) + 1e-8)

    ppo_update(
        model,
        optimizer,
        torch.stack(states),
        torch.stack(actions),
        torch.stack(log_probs),
        torch.tensor(rets, dtype=torch.float32),
        torch.tensor(advs, dtype=torch.float32),
    )

    reward_history.append(total_reward)
    avg_reward = np.mean(reward_history[-10:])
    print(
        f"Tập {episode + 1}, phần thưởng: {total_reward:.1f}, trung bình 10 tập: {avg_reward:.1f}"
    )

    if avg_reward > 200:
        torch.save(
            model.state_dict(), f"models/ppo_lander_{episode + 1}_s_{avg_reward}.pth"
        )
        print(f"✅ Đã lưu mô hình tại tập {episode + 1}")
        print("🚀 Tàu đã học cách hạ cánh an toàn!")
        # break


env.close()
