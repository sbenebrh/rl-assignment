import numpy as np
import torch
import torch.nn as nn
from src.network import QNetwork
from src.replay_buffer import ReplayBuffer


class DQNAgent:

    def __init__(self, n_actions, config):
        self.n_actions = n_actions
        self.config = config
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Two networks: online (trained) and target (frozen, synced every C steps)
        self.online_net = QNetwork(n_actions).to(self.device)
        self.target_net = QNetwork(n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=config["lr"])
        self.buffer = ReplayBuffer(config["buffer_size"])

        # Counts every env.step() call (used for eps decay)
        self.total_steps = 0

    # ── Action selection ────────────────────────────────────────────────────

    def select_action(self, state, eval_mode=False):
        eps = self.config["eval_eps"] if eval_mode else self._current_eps()
        if np.random.random() < eps:
            return np.random.randint(self.n_actions)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.online_net(state_t).argmax().item()

    def _current_eps(self):
        # Linear decay from eps_start → eps_end over eps_decay_steps
        progress = min(self.total_steps / self.config["eps_decay_steps"], 1.0)
        return self.config["eps_start"] + progress * (
            self.config["eps_end"] - self.config["eps_start"]
        )

    # ── Experience storage ──────────────────────────────────────────────────

    def store(self, state, action, reward, next_state, done):
        self.buffer.store(state, action, reward, next_state, done)
        self.total_steps += 1

    # ── Learning ────────────────────────────────────────────────────────────

    def learn(self):
        if len(self.buffer) < self.config["learning_starts"]:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.config["batch_size"]
        )

        states      = torch.tensor(states).to(self.device)
        actions     = torch.tensor(actions).to(self.device)
        rewards     = torch.tensor(rewards).to(self.device)
        next_states = torch.tensor(next_states).to(self.device)
        dones       = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Q(s, a) from the online network
        q_pred = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Bellman target: r + γ · max_a' Q_target(s', a') · (1 - done)
        # torch.no_grad(): target is treated as a constant — no gradient flows through it
        with torch.no_grad():
            q_next = self.target_net(next_states).max(1)[0]
            q_target = rewards + self.config["gamma"] * q_next * (1.0 - dones)

        # Huber loss (SmoothL1) — more stable than MSE for large errors
        loss = nn.functional.smooth_l1_loss(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping prevents exploding gradients
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()

    # ── Target network sync ──────────────────────────────────────────────────

    def sync_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ── Checkpointing ────────────────────────────────────────────────────────

    def save(self, path):
        torch.save({
            "online_net":  self.online_net.state_dict(),
            "target_net":  self.target_net.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt["total_steps"]
