# ppo_agent.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from memory_buffer import RolloutBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(128, 128)):
        super().__init__()
        actor_layers = []
        in_dim = state_dim
        for h in hidden_sizes:
            actor_layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        actor_layers += [nn.Linear(in_dim, action_dim), nn.Softmax(dim=-1)]
        self.actor = nn.Sequential(*actor_layers)

        critic_layers = []
        in_dim = state_dim
        for h in hidden_sizes:
            critic_layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        critic_layers += [nn.Linear(in_dim, 1)]
        self.critic = nn.Sequential(*critic_layers)

    def forward(self):
        raise NotImplementedError

    def get_action_and_value(self, state, deterministic=False):
        # state: numpy array
        s = torch.FloatTensor(state).to(device)
        if s.dim() == 1:
            s = s.unsqueeze(0)
        probs = self.actor(s)
        dist = Categorical(probs)
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(s).squeeze(-1)
        return (
            int(action.item()),
            log_prob.squeeze(0).detach(),
            value.squeeze(0).detach(),
        )

    def evaluate(self, states, actions):
        """
        states: tensor (N, state_dim)
        actions: tensor (N,)
        returns: log_probs (N,), state_values (N,), dist_entropy (N,)
        """
        probs = self.actor(states)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic(states).squeeze(-1)
        return log_probs, values, entropy


class PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        K_epochs=4,  # N_e in pseudocode
        eps_clip=0.2,
        entropy_coef=0.01,
        vf_coef=0.5,
        batch_size=None,
    ):
        self.buffer = RolloutBuffer()
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # separate learning rates for actor & critic inside same optimizer
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": lr_actor},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
            ]
        )

        self.MSE = nn.MSELoss()
        self.batch_size = batch_size  # if None, use full-batch updates

    def select_action(self, state, deterministic=False):
        # sample under policy_old (the behavior policy π_β)
        action, log_prob, value = self.policy_old.get_action_and_value(
            state, deterministic=deterministic
        )
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.log_probs.append(log_prob)  # tensor scalar
        # store V(s) computed under policy_old critic
        self.buffer.values.append(float(value.cpu().numpy()))
        return action

    def compute_advantages(self):
        """
        Compute per-step one-step TD advantage as in Algorithm 15:
        If s_{t+1} terminal:
            Adv = r_t - V(s_t)
            y_t = r_t
        else:
            Adv = r_t + gamma * V(s_{t+1}) - V(s_t)
            y_t = r_t + gamma * V(s_{t+1})
        Return advantages (tensor) and targets y (tensor) aligned with buffer length.
        """
        rewards = self.buffer.rewards
        is_terminals = self.buffer.is_terminals
        values = self.buffer.values
        n = len(rewards)
        advantages = []
        targets = []
        for t in range(n):
            v_s = values[t]
            if t + 1 < n:
                v_next = values[t + 1]
            else:
                v_next = 0.0
            if is_terminals[t]:
                adv = rewards[t] - v_s
                y = rewards[t]
            else:
                adv = rewards[t] + self.gamma * v_next - v_s
                y = rewards[t] + self.gamma * v_next
            advantages.append(adv)
            targets.append(y)
        adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
        # normalize advantages (helps training)
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)
        targets_tensor = torch.tensor(targets, dtype=torch.float32, device=device)
        return adv_tensor, targets_tensor

    def update(self):
        """
        Implements the inner loop of Algorithm 15:
        - compute advantages and critic targets y_t
        - for epoch = 1..N_e: compute rho = pi(a|s;phi)/pi_beta(a|s) using logprobs
          and minimize actor loss: - min(rho * Adv, clip(rho,1-eps,1+eps)*Adv)
        - minimize critic loss (y_t - V(s))^2
        """
        if len(self.buffer) == 0:
            return

        # get tensors for stored batch
        states = torch.FloatTensor(np.array(self.buffer.states)).to(device)
        actions = torch.LongTensor(np.array(self.buffer.actions)).to(device)
        old_log_probs = torch.stack(self.buffer.log_probs).to(device)  # shape (N,)

        advantages, targets = self.compute_advantages()  # tensors on device

        dataset_size = states.size(0)
        batch_size = self.batch_size or dataset_size

        for epoch in range(self.K_epochs):
            # iterate mini-batches
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                mb_idx = indices[start : start + batch_size]
                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logprobs = old_log_probs[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_targets = targets[mb_idx]

                # evaluate current policy π (with current parameters φ)
                logprobs, state_values, dist_entropy = self.policy.evaluate(
                    mb_states, mb_actions
                )

                # importance sampling ratio ρ = π(a|s;φ) / π_beta(a|s) = exp(logπ - logπ_beta)
                ratios = torch.exp(logprobs - mb_old_logprobs.detach())

                # surrogate (clipped)
                surr1 = ratios * mb_adv
                surr2 = (
                    torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip)
                    * mb_adv
                )
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = self.MSE(state_values, mb_targets)

                entropy_loss = dist_entropy.mean()

                loss = (
                    actor_loss
                    + self.vf_coef * critic_loss
                    - self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # after updates copy weights to policy_old (π_β <- π)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer for next batch
        self.buffer.clear()

    def save(self, path="ppo_lunarlander.pth"):
        torch.save(self.policy.state_dict(), path)

    def load(self, path="ppo_lunarlander.pth"):
        if os.path.exists(path):
            self.policy.load_state_dict(torch.load(path, map_location=device))
            self.policy_old.load_state_dict(self.policy.state_dict())
            return True
        return False
