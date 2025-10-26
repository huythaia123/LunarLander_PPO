import gymnasium as gym
import matplotlib.pyplot as plt
from ppo_agent import PPO
from utils import save_model


def train_ppo():
    env = gym.make("LunarLander-v3")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPO(state_dim, action_dim)
    max_episodes = 2000
    max_timesteps = 1500
    update_timestep = 4000
    timestep = 0

    running_reward = 0
    avg_length = 0
    reward_history = []  # lưu reward trung bình 50 episode gần nhất

    for ep in range(1, max_episodes + 1):
        state, _ = env.reset()
        ep_reward = 0

        for t in range(max_timesteps):
            timestep += 1
            action = agent.select_action(state)
            state, reward, done, trunc, _ = env.step(action)

            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done or trunc)
            ep_reward += reward

            if done or trunc:
                break

            # cập nhật PPO sau mỗi update_timestep bước
            if timestep % update_timestep == 0:
                agent.update()
                timestep = 0

        running_reward += ep_reward
        avg_length += t

        # lưu reward trung bình mỗi 50 episode
        if ep % 50 == 0:
            avg_reward = running_reward / 50
            reward_history.append(avg_reward)
            print(f"Episode {ep}\tAverage Reward: {avg_reward:.2f}")
            running_reward = 0
            avg_length = 0

        # Lưu mô hình tốt nhất khi reward cao
        if ep_reward > 250:
            save_model(agent, "ppo_lunarlander_best.pth")
            print(f"🎉 Saved best model at episode {ep} with reward {ep_reward:.2f}")

    env.close()
    save_model(agent)

    # ----- Vẽ đồ thị reward -----
    plt.figure(figsize=(10, 5))
    plt.plot(
        [i * 50 for i in range(1, len(reward_history) + 1)],
        reward_history,
        marker="o",
        color="blue",
    )
    plt.title("PPO Training Progress on LunarLander-v3")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward (per 50 episodes)")
    plt.grid(True)
    plt.savefig("ppo_training_curve.png")
    plt.show()
    print("📈 Training curve saved as ppo_training_curve.png")


if __name__ == "__main__":
    train_ppo()
