# memory_buffer.py
"""
Simple rollout buffer storing one batch (one or many episodes worth)
Used by the algorithm to store states, actions, rewards, log_probs (under policy beta),
and values V(s) from the critic of the policy_old.
"""

from typing import List


class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states: List = []
        self.actions: List[int] = []
        self.log_probs: List = []  # log prob under policy_beta (policy_old)
        self.rewards: List[float] = []
        self.is_terminals: List[bool] = []
        self.values: List[float] = []  # V(s) computed when sampling

    def __len__(self):
        return len(self.rewards)
