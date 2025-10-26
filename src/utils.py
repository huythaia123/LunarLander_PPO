import torch
import os


def save_model(agent, filename="ppo_lunarlander.pth"):
    torch.save(agent.policy.state_dict(), filename)
    print(f"[OK] Model saved to {filename}")


def load_model(agent, filename="ppo_lunarlander.pth"):
    if os.path.exists(filename):
        agent.policy.load_state_dict(torch.load(filename))
        agent.policy_old.load_state_dict(agent.policy.state_dict())
        print(f"[OK] Model loaded from {filename}")
    else:
        print("[ERROR] No model found to load.")
