# Demo.py
# Mô phỏng và ghi lại video của tác nhân PPO trên môi trường LunarLander-v3
import gymnasium as gym
import torch
import torch.nn.functional as F
import imageio


# =============================
# 1. Định nghĩa lại ActorCritic
# =============================
class ActorCritic(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.actor = torch.nn.Linear(128, action_dim)
        self.critic = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.actor(x), self.critic(x)

    def act(self, state):
        logits, _ = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = torch.argmax(dist.probs).item()
        return action


# =============================
# 2. Thiết lập môi trường
# =============================
env = gym.make("LunarLander-v3", render_mode="rgb_array")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model = ActorCritic(state_dim, action_dim)
model.load_state_dict(
    torch.load("models/ppo_lander_174_s_496.19811964933405.pth", map_location="cpu")
)
model.eval()

# =============================
# 3. Mô phỏng & ghi video
# =============================
frames = []
obs, info = env.reset(seed=42)
total_reward = 0

for t in range(1000):
    state = torch.tensor(obs, dtype=torch.float32)
    action = model.act(state)
    obs, reward, terminated, truncated, info = env.step(action)
    frame = env.render()
    frames.append(frame)
    total_reward += reward
    if terminated or truncated:
        print(f"🏁 Hạ cánh xong! Tổng thưởng: {total_reward:.2f}")
        break

env.close()

# =============================
# 4. Lưu video mô phỏng
# =============================
output_path = "gifs/LunarLander_PPO_Demo.gif"
imageio.mimsave(output_path, frames, fps=30)
print(f"🎬 Đã lưu video mô phỏng tại: {output_path}")
