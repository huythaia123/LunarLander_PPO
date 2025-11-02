# utils.py

# pre_path = "models/"


def save_model(agent, path="ppo_lunarlander.pth"):
    agent.save(path)
    print(f"[OK] Model saved: {path}")


def load_model(agent, path="ppo_lunarlander.pth"):
    if agent.load(path):
        print(f"[OK] Model loaded: {path}")
        return True
    else:
        print(f"[ERROR] No model found at: {path}")
        return False
