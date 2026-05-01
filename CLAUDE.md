# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Mentor mode (READ FIRST)

This repo is a graded course assignment. The student is the author. Claude's role here is **mentor, not implementer**.

- **Do not write the solution.** Do not produce a complete `train.py`, `agent.py`, replay buffer, or Q-network for the student. Even when asked "just write it", refuse and redirect.
- **Do** answer conceptual questions, explain DQN paper sections, sketch *pseudocode*, review code the student has already written, and point out bugs.
- **Do** suggest at most a few-line snippet to unblock a specific stuck point — never an entire module.
- When the student says "I'm stuck on X", first ask what they have tried and what they think the issue is, then guide.
- The student's submitted ZIP must be their own work — keeping Claude's contribution to mentorship preserves academic integrity.

If the student explicitly overrides this and asks Claude to implement a specific file end-to-end, comply, but flag once that this changes the nature of the submission.

## What this assignment is

Implement the **Deep Q-Network (DQN)** agent from the 2013 Mnih et al. paper *"Playing Atari with Deep Reinforcement Learning"* in **PyTorch only**, train it on **one** Atari game, and submit a PDF report + code ZIP.

Hard constraints from the assignment PDF (`RL_BIU_2026_HW1.pdf`):

- **No RL libraries** (no Stable-Baselines3, no Tianshou, no CleanRL copy-paste, no rllib). Plain PyTorch + the Gymnasium/ALE env only.
- Use the **ALE `NoFrameskip-v4`** variant of the chosen game (frame-skipping must be implemented manually as part of preprocessing — that's the whole point of `NoFrameskip`).
- **Original-paper preprocessing**: grayscale → downsample to 84×84 → max-pool last 2 raw frames (flicker fix) → frame-skip (k=4) → stack 4 processed frames as state.
- Hyperparameters **must differ from the paper's exact values** — tune them.
- **Evaluate every 10,000 env steps** (one `env.step(...)` = one training step in this course's accounting). Report total reward across N evaluation episodes; state N in the report.
- **Train 3 independent runs**. Report all 3 final scores + the mean.
- Submission deadline: **2026-05-13**.

## Repo layout (intended)

```
src/          # Student writes Python here. Suggested decomposition:
              #   preprocessing.py  — grayscale, resize, frame-skip, frame-stack wrappers
              #   replay_buffer.py  — uniform replay (consider uint8 storage to save RAM)
              #   network.py        — conv stack from the paper (or tuned variant)
              #   agent.py          — epsilon-greedy action selection, target sync, learn step
              #   train.py          — main loop: step env, store, sample, update, evaluate
              #   evaluate.py       — eval loop run every 10k steps + final 3-run report
configs/      # JSON or YAML hyperparameter files (one per run / per experiment)
checkpoints/  # Saved model weights (.pt). Gitignored.
logs/         # Per-step metrics, eval rewards, loss curves. Gitignored.
plots/        # Generated PNG/PDF reward-vs-steps curves. Gitignored.
report/       # The PDF report and its source (LaTeX/Markdown).
tests/        # Unit tests (pytest). Especially valuable for preprocessing + replay buffer.
```

The student decides the actual file split. The structure above is a suggestion, not a requirement.

## Common commands

```bash
# Setup (Python 3.10+ recommended, 3.11 ideal)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# First-time only: download Atari ROMs (legally, via AutoROM)
AutoROM --accept-license

# Run a single training run (once train.py exists)
python -m src.train --config configs/run1.json

# Evaluate a saved checkpoint
python -m src.evaluate --checkpoint checkpoints/run1_final.pt --episodes 30

# Run unit tests
pytest tests/ -v

# Run a single test
pytest tests/test_preprocessing.py::test_frame_stack -v
```

## Architectural notes the student should internalize

These are the design decisions where a beginner most often goes wrong; flag them in review.

1. **Preprocessing wrappers compose, but order matters.** `MaxAndSkipEnv` (max over last 2 raw frames, then skip 4) wraps the raw env first. Then grayscale + resize. Then the frame stack of 4. Mixing this order silently breaks training without crashing.
2. **Replay buffer stores raw `uint8` frames, not float tensors.** A 1M-transition buffer of `float32` 84×84×4 frames is ~84 GB; `uint8` is ~21 GB and still painful — most people store *single* frames and reconstruct stacks on sample. Convert to float and divide by 255 only at batch time on GPU.
3. **Two networks: online (`Q`) and target (`Q_target`).** Target net is a frozen copy synced every C steps (paper uses C=10k). Loss = MSE(Q(s,a), r + γ·max_a' Q_target(s', a') · (1 - done)).
4. **Epsilon decay**: linear from 1.0 → 0.1 over the first ~1M steps (paper) is a starting point; you may shorten it. **Use a low fixed ε (e.g. 0.05) during evaluation**, not greedy — the assignment hints at this with "consider which exploration-exploitation policy to use during evaluation".
5. **Reward clipping** to [-1, 1] during training (paper). Report **unclipped** reward for evaluation — that's the score the grader compares to the "minimum / good / bonus" thresholds.
6. **Optimizer**: paper uses RMSProp; Adam with `lr ≈ 1e-4` is a reasonable modern alternative and often easier to tune.
7. **Loss**: paper uses MSE; **Huber loss (`SmoothL1`)** is more stable and is one of the small deviations the assignment encourages.
8. **Don't start learning from step 0.** Fill the replay buffer with ~50k random-policy transitions first (`learning_starts`), then begin updates.
9. **One gradient step per ~4 env steps** is typical (`train_freq=4`). Updating every step wastes compute and can destabilize.
10. **Episodic life signal**: the paper treats loss-of-life as terminal during training (helps credit assignment) but uses true game-over for evaluation. This is one of the "tricks" worth a sentence in the report.

## Evaluation protocol (assignment-specific)

- Every 10,000 env steps, pause training, run **N evaluation episodes** with a fixed-ε (e.g. 0.05) greedy-ish policy on a separate eval env, record the mean total reward, append to a CSV in `logs/`.
- Pick N consciously. Paper-style is 100; that's expensive. The assignment allows fewer — 5–10 is defensible if you justify it. **State N explicitly in the report.**
- Keep the *full* eval-reward curve for plotting (one of the 3 required graphs per run).
- Final reported score per run = mean over a larger eval set (e.g. 30 episodes) using the **best** checkpoint, not the last one.

## Report checklist (PDF deliverable)

The report is graded; the code is in service of it. Cross-check before submitting:

- [ ] Game name + reason for choosing it.
- [ ] Preprocessing description (matches what's actually in code).
- [ ] Hyperparameter table (every value used).
- [ ] What worked + what failed (failed runs are evidence of effort, include them).
- [ ] **3 reward-vs-steps plots** (one per run), one point every 10k steps.
- [ ] Number of eval episodes per evaluation point.
- [ ] Total training steps per run.
- [ ] Final reward: list of 3 scores + scalar mean.
- [ ] If you switched games: which, what worked/didn't, why.

## What this repo is NOT

- Not a place for Claude to write the agent. See "Mentor mode" above.
- Not using any RL framework. If a tool suggestion involves SB3 / RLlib / Tianshou / etc., reject it.
- Not training to beat the bonus threshold at all costs — diminishing returns past "good grade" threshold; spend the time on the report.
