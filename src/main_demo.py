import gymnasium as gym
from ppo_agent import PPO
from utils import load_model
import time


def demo():
    env = gym.make("LunarLander-v3", render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(state_dim, action_dim)

    test_load_model = load_model(agent, "ppo_lunarlander_best.pth")
    if not test_load_model:
        load_model(agent, "ppo_lunarlander.pth")

    for ep in range(5):
        state, _ = env.reset()
        total_reward = 0
        for t in range(1500):
            action = agent.select_action(state)
            state, reward, done, trunc, _ = env.step(action)
            total_reward += reward
            if done or trunc:
                break
        print(f"Episode {ep + 1}: Reward = {total_reward:.2f}")
        time.sleep(1)

    env.close()


if __name__ == "__main__":
    demo()
