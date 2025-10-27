# utils.py

pre_path = "models/"


def save_model(agent, path="ppo_lunarlander.pth"):
    agent.save(pre_path + path)
    print(f"[OK] Model saved: {pre_path + path}")


def load_model(agent, path="ppo_lunarlander.pth"):
    if agent.load(pre_path + path):
        print(f"[OK] Model loaded: {pre_path + path}")
        return True
    else:
        print(f"[ERROR] No model found at: {pre_path + path}")
        return False
