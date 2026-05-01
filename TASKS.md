# DQN Assignment — Task List

Deadline: **2026-05-13** | Team of 2 | Game: TBD

---

## Phase 0 — Foundation
- [ ] Read the 2013 DQN paper (Mnih et al.) — focus on §4 (Preprocessing) and Algorithm 1
- [ ] Pick a game from the spreadsheet, write student IDs in the row
- [ ] Set up Python environment: `pip install -r requirements.txt` + `AutoROM --accept-license`

## Phase 1 — Environment & Preprocessing (`src/preprocessing.py`)
- [ ] Grayscale conversion
- [ ] Frame resize to 84×84
- [ ] Max-pooling over last 2 raw frames (flicker fix)
- [ ] Frame-skip of k=4 (repeat action 4 frames, accumulate rewards)
- [ ] Frame-stack of 4 processed frames → state shape: `(4, 84, 84)`
- [ ] Unit tests in `tests/test_preprocessing.py` (verify shapes, dtypes, values)

## Phase 2 — Replay Buffer (`src/replay_buffer.py`)
- [ ] Fixed-size circular buffer (target: 500k–1M transitions)
- [ ] Store single frames as `uint8`, reconstruct stacks at sample time
- [ ] Random batch sampling → returns float tensors on the correct device
- [ ] Unit test: fill, sample, check shapes and types

## Phase 3 — Q-Network (`src/network.py`)
- [ ] CNN architecture from the paper (3 conv layers → 2 FC layers)
- [ ] Input: `(batch, 4, 84, 84)` float32 / 255
- [ ] Output: `(batch, n_actions)` Q-values (no activation on final layer)
- [ ] Confirm output shape with a dummy forward pass

## Phase 4 — DQN Agent (`src/agent.py`)
- [ ] Online network + target network (same architecture)
- [ ] ε-greedy action selection with linear ε decay
- [ ] Bellman target: `r + γ · max_a' Q_target(s', a') · (1 - done)`
- [ ] Huber loss (SmoothL1) + optimizer step on online network
- [ ] Target network hard sync every C steps

## Phase 5 — Training Loop (`src/train.py`)
- [ ] Warm-up phase: fill buffer with ~50k random-policy transitions before learning starts
- [ ] Main loop: `env.step()` → store → (every 4 steps) sample & update
- [ ] **Evaluate every 10,000 steps** — pause training, run N eval episodes with fixed ε, log mean reward
- [ ] Clip rewards to [-1, 1] during training; log **unclipped** rewards during eval
- [ ] Save checkpoints periodically + on best eval score
- [ ] Write eval rewards to `logs/run{N}_eval.csv` (columns: step, mean_reward)

## Phase 6 — Runs & Results
- [ ] Run 1 — full training, save logs + final checkpoint
- [ ] Run 2 — same config, different random seed
- [ ] Run 3 — same config, different random seed
- [ ] Generate 3 reward-vs-steps plots (`plots/`) — one per run, one point per 10k steps
- [ ] Compute final score per run (mean over last eval or over N episodes with best checkpoint)
- [ ] Compute average across 3 runs

## Phase 7 — Report (PDF)
- [ ] Game name + reason for choice
- [ ] Preprocessing description (match exactly what the code does)
- [ ] Hyperparameter table (every value)
- [ ] What worked (with evidence)
- [ ] What did NOT work (failed runs, bad hyperparams)
- [ ] 3 learning-curve plots (reward vs training steps)
- [ ] Number of eval episodes (N) stated clearly
- [ ] Total training steps per run
- [ ] Final reward: 3 scores + scalar mean

---

## Responsibility split (suggestion for a team of 2)

| Person A | Person B |
|----------|----------|
| Phase 1 (preprocessing) + Phase 2 (replay buffer) | Phase 3 (network) + Phase 4 (agent) |
| Phase 6 (running experiments + plots) | Phase 5 (training loop) |
| Report sections: preprocessing, results | Report sections: architecture, hyperparams, analysis |

Review each other's code before integration.

---

## Notes
- One `env.step()` = one training step (assignment's definition)
- Eval ε: use ~0.05 (not greedy) during evaluation
- Reward clipping: only during training, not evaluation
- Paper preprocessing uses `NoFrameskip-v4` — frame-skip is manual, not gym's built-in
