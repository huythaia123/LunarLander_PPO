# main_train.py
import os
import time
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from ppo_agent import PPO
from utils import save_model, load_model


def train_ppo(
    env_id="LunarLander-v3",
    save_dir=".",
    seed=42,
    max_episodes=4000,
    max_timesteps=1000,
    update_timestep=4000,
    resume=False,
):
    env = gym.make(env_id)
    env.reset(seed=seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPO(
        state_dim,
        action_dim,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        K_epochs=10,  # N_e as in pseudocode
        eps_clip=0.2,
        entropy_coef=0.01,
        vf_coef=0.5,
        batch_size=64,
    )

    last_checkpoint = os.path.join(save_dir, "ppo_lunarlander.pth")
    best_checkpoint = os.path.join(save_dir, "ppo_lunarlander_best.pth")

    if resume and os.path.exists(last_checkpoint):
        print("[INFO] Resuming from last checkpoint...")
        agent.load(last_checkpoint)

    timestep = 0
    running_reward = 0.0
    reward_history = []
    best_avg = -1e9
    recent_rewards = []

    start = time.time()
    for ep in range(1, max_episodes + 1):
        state, _ = env.reset()
        ep_reward = 0.0

        for t in range(max_timesteps):
            timestep += 1
            # sample from behavior policy π_beta (policy_old)
            action = agent.select_action(state, deterministic=False)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # store immediate reward and terminal flag (following pseudocode)
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)

            ep_reward += reward
            state = next_state

            # If enough timesteps collected, update
            if timestep % update_timestep == 0:
                agent.update()
                timestep = 0

            if done:
                break

        recent_rewards.append(ep_reward)
        running_reward += ep_reward

        # logging and saving average every 50 episodes
        if ep % 50 == 0:
            avg50 = sum(recent_rewards[-50:]) / 50.0
            reward_history.append(avg50)
            print(f"[EP {ep}] Avg50: {avg50:.2f} | EpReward: {ep_reward:.2f}")
            if avg50 > best_avg:
                best_avg = avg50
                save_model(agent, best_checkpoint)
                print(f"\t- New best avg50 {best_avg:.2f} saved.")
            # save periodic checkpoint
            save_model(agent, last_checkpoint)

        # save exceptionally good single episodes
        if ep_reward >= 250.0:
            save_model(agent, best_checkpoint)
            print(f"\t- Exceptional episode saved (ep {ep}, reward {ep_reward:.2f})")

    # final saves & plotting
    save_model(agent, last_checkpoint)
    env.close()

    # plot training curve
    if reward_history:
        xs = [i * 50 for i in range(1, len(reward_history) + 1)]
        plt.figure(figsize=(10, 5))
        plt.plot(xs, reward_history, marker="o")
        plt.title("PPO Training Curve (avg per 50 episodes)")
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.grid(True)
        plot_path = os.path.join(save_dir, "ppo_training_curve.png")
        plt.savefig(plot_path)
        # plt.show()
        print(f"[OK] Training curve saved: {plot_path}")

    print(f"[DONE] Training finished. Time elapsed {time.time() - start:.1f}s")
    print(
        f"Best avg50: {best_avg:.2f} (file: {best_checkpoint if os.path.exists(best_checkpoint) else last_checkpoint})"
    )


if __name__ == "__main__":
    # set resume=True to continue from last checkpoint if exists
    train_ppo(resume=True, max_episodes=3000)
