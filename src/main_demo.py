# main_demo.py
import os
import time
import torch
import gymnasium as gym
import imageio.v2 as imageio
from ppo_agent import PPO, device
from utils import load_model


def demo(
    env_id="LunarLander-v3",
    model_path="ppo_lunarlander_best.pth",
    episodes=5,
    deterministic=True,
):
    env = gym.make(env_id, render_mode="rgb_array")
    # env = gym.make(env_id, render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPO(state_dim, action_dim)
    if not load_model(agent, model_path):
        fallback = "ppo_lunarlander.pth"
        if not load_model(agent, fallback):
            print("[ERROR] No saved model. Train first.")
            return

    # tạo thư mục lưu GIF
    os.makedirs("gifs", exist_ok=True)

    for ep in range(episodes):
        frames = []
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        trunc = False
        while not (done or trunc):
            frame = env.render()
            frames.append(frame)

            if deterministic:
                # choose action with highest probability under current policy
                s = torch.FloatTensor(state).to(device).unsqueeze(0)
                probs = agent.policy.actor(s).detach().cpu().numpy().squeeze()
                action = int(probs.argmax())
            else:
                action = agent.select_action(state, deterministic=False)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated
            trunc = truncated
            time.sleep(0.01)

        # Xuất GIF mỗi episode
        gif_path = f"gifs/episode_{ep + 1}.gif"
        imageio.mimsave(gif_path, frames, fps=30)

        print(f"[Demo] Episode {ep + 1} Reward: {total_reward:.2f}")
        time.sleep(0.5)

    env.close()


if __name__ == "__main__":
    demo()
