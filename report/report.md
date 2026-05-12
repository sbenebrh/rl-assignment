# Deep Q-Network (DQN) on Atari Seaquest
## Reinforcement Learning - Homework Assignment

---

## 1. Game Choice

**Game:** SeaquestNoFrameskip-v4

**Reason:** Seaquest was selected from the available games in the assignment spreadsheet for the following reasons:
- It was one of the remaining games with available slots when we made our selection
- The game provides a good balance of challenge: sparse rewards require careful hyperparameter tuning, but the task is learnable within reasonable training time

---

## 2. Preprocessing

The preprocessing pipeline follows the original DQN 2013 paper specification:

### Wrapper Order
```
Raw ALE Environment → NoopReset → MaxAndSkip → EpisodicLife → WarpFrame → FrameStack
```

### Individual Preprocessing Steps

1. **NoopResetEnv**: At the start of each episode, execute 1-30 random NOOP actions to introduce stochasticity in the initial state. (This is a standard technique from later implementations to prevent the agent from memorizing fixed action sequences from deterministic start states.)

2. **MaxAndSkipEnv (Frame-skip = 4)**: 
   - Repeat each action for 4 consecutive frames
   - Accumulate rewards over the skipped frames

3. **EpisodicLifeEnv**: Treat loss of life as episode termination during training. This improves credit assignment by providing more frequent terminal signals. During evaluation, only true game-over ends the episode.

4. **WarpFrame**: 
   - Convert RGB frames to grayscale
   - Resize from 210×160 to 84×84 pixels using area interpolation

5. **FrameStack (k=4)**: Stack the 4 most recent frames to provide temporal information (velocity, direction). Final observation shape: (4, 84, 84).

6. **Pixel Normalization**: Pixels are stored as uint8 [0-255] in the replay buffer and normalized to float32 [0-1] at batch sampling time (memory efficient).

---

## 3. Hyperparameters

| Parameter | Value | Paper 2013 | Notes |
|-----------|-------|------------|-------|
| `buffer_size` | 500,000 | 1,000,000 | Reduced for memory |
| `batch_size` | 64 | 32 | Increased to capture more reward signals (Seaquest is sparse) |
| `gamma` | 0.99 | Not specified | Discount factor |
| `learning_rate` | 0.0001 | Not specified | Tuned for stability without target network |
| `eps_start` | 1.0 | 1.0 | Initial exploration rate |
| `eps_end` | 0.1 | 0.1 | Final exploration rate |
| `eps_decay_steps` | 250,000 steps | 1,000,000 frames | Paper uses frames; with frame_skip=4, 1M frames ≈ 250k steps |
| `eval_eps` | 0.05 | 0.05 | Paper uses ε=0.05 for evaluation |
| `learning_starts` | 50,000 | Not specified | Warm-up period before learning begins |
| `train_freq` | 1 | 1 | Paper: "weights are updated after every time-step" |
| `eval_freq` | 10,000 | - | Evaluation every 10k steps (assignment requirement) |
| `eval_episodes` | 10 | Not specified | Reduced for speed; 30 episodes for final evaluation |
| `max_steps` | 3,000,000 | 10,000,000 frames | Reduced training budget |
| `frame_skip` | 4 | 4 | "k = 4 for all games except Space Invaders" |

### Network Architecture (2013 Paper)
| Layer | Configuration |
|-------|--------------|
| Conv1 | 16 filters, 8×8 kernel, stride 4, ReLU |
| Conv2 | 32 filters, 4×4 kernel, stride 2, ReLU |
| FC1 | 256 units, ReLU |
| Output | 18 actions (Seaquest) |

### Optimizer
- **Type:** RMSprop (as in paper)
- **Parameters:** lr=0.0001, alpha=0.95, eps=0.01, momentum=0.0

### Other Settings
- **Loss function:** MSE (Mean Squared Error) as in 2013 paper
- **Gradient clipping:** max_norm=1.0 (added for stability)
- **Reward clipping:** [-1, 1] during training (unclipped for evaluation)
- **Target network:** None (2013 paper uses single network)

---

## 4. What Worked

### Key Improvements That Led to Success

1. **Learning Rate Reduction (0.00025 → 0.0001)**
   - Without a target network, Q-values can explode
   - Lower learning rate prevents the "chasing its own tail" instability
   - The loss went from exploding (billions) to stable (~0.01)

2. **Gradient Clipping (max_norm=1.0)**
   - Prevents gradient explosion during early training
   - Essential for stability with MSE loss and no target network

3. **Larger Batch Size (32 → 64)**
   - Seaquest has very sparse rewards (~1.8% of transitions have non-zero reward)
   - Larger batches increase the probability of sampling meaningful transitions
   - With batch_size=32, most batches had 0 reward signals

4. **MSE Loss (as in 2013 paper)**
   - Huber loss (SmoothL1) was initially used but caused issues
   - MSE provides stronger gradients for learning from sparse rewards

5. **Buffer Size Tuning**
   - Tested multiple buffer sizes (250k,500k,750k) to find optimal balance
   - With limited training budget (3M steps), smaller buffers retain more recent, high-quality experiences
   - 500k was the sweet spot for stability and performance

---

## 5. What Did Not Work

### Failed Approaches and Bugs Encountered

1. **Target Network (2015 approach) - Removed**
   - Initially implemented target network with sync every 1000 steps
   - Had to remove it to comply with 2013 paper requirement
   - This made training less stable but was necessary

2. **Huber Loss (SmoothL1)**
   - Tried Huber loss for stability, but it clipped gradients too aggressively
   - With sparse rewards, the network wasn't learning anything (loss ~0.0001)
   - Switching to MSE fixed this

3. **batch_size=256**
   - Tested larger batch for more reward signal
   - too slow to train and didn't improve performance significantly
   - Reverted to batch_size=64 which was more stable

4. **batch_size=32 (original paper)**
   - Too small for Seaquest's sparse rewards
   - Most batches had 0/32 transitions with rewards
   - Loss was near-zero, agent wasn't learning

5. **No Gradient Clipping**
   - Without gradient clipping, training was unstable
   - Q-values would occasionally explode

6. **train_freq=4**
   - Initially set to 4 (as in 2015 paper)
   - 2013 paper updates every step (train_freq=1)
   - Changed for paper compliance

7. **Normalization Mismatch (Critical Bug)**
   - Most debugging time was spent on this issue
   - Symptoms: Loss collapsed to 0.0001, Q-values identical for all actions (~0.9)
   - Agent performed randomly despite low loss
   - Root cause: Network saw different pixel distributions in training vs inference

8. **Large Buffer with Small Training Budget**
   - buffer_size=750k was tested but gave worse results
   - With 3M steps, old random-policy transitions dilute the good experiences
   - buffer_size=500k was the sweet spot

---

## 6. Learning Progress Graphs

### Run 1 (seed=42, buffer_size=500k)
![Run 1 Learning Curve](plots/run1_learning_curve.png)

### Run 2 (seed=84, buffer_size=500k)
![Run 2 Learning Curve](plots/run2_learning_curve.png)

### Run 3 (seed=126, buffer_size=500k)
![Run 3 Learning Curve](plots/run3_learning_curve.png)

**Observations:**
- All runs show clear learning progress from ~100-200 (random policy) to 800-900
- High variance in evaluation rewards is expected (mentioned in 2013 paper as "quite noisy")
- Peak performance typically reached around 2-2.7M steps
- Some runs show regression after peak (instability without target network)

---

## 7. Evaluation Episodes

| Training Phase | Evaluation Episodes |
|----------------|---------------------|
| During training (every 10k steps) | 10 episodes |
| Final evaluation (best checkpoint) | 30 episodes |

**Justification:** 
- 10 episodes during training provides a reasonable balance between measurement precision and training speed
- 30 episodes for final evaluation gives more reliable scores (lower variance)

---

## 8. Final Results

### Best Evaluation Rewards (during training, 10 episodes)

| Run | Seed | Buffer Size | Best Reward | Step Achieved |
|-----|------|-------------|-------------|---------------|
| Run 1 | 42 | 500,000     | 878.0 | 3,000,000 |
| Run 2 | 84 | 500,000     | 900.0 | 2,750,000 |
| Run 3 | 126 | 500,000     | 896.0 | 2,380,000 |

### Final Evaluation (best checkpoint, 30 episodes)

| Run | Mean | Median | Std | Min | Max |
|-----|------|--------|-----|-----|-----|
| Run 1 | 862.7 | 880.0 | 37.5 | 700 | 880 |
| Run 2 | 887.3 | 900.0 | 29.4 | 760 | 920 |
| Run 3 | 882.0 | 880.0 | 27.0 | 780 | 920 |

### Summary

| Metric | Value |
|--------|-------|
| **Run 1 Final Score** | 862.7 |
| **Run 2 Final Score** | 887.3 |
| **Run 3 Final Score** | 882.0 |
| **Average (Mean)** | **877.3** |

### Comparison to Thresholds

| Threshold | Score | Our Result | Status               |
|-------|-------|------------|----------------------|
| Minimum (random agent) | 68.4 | 877.3 | **PASSED** (12.8x)   |
| Good Grade | 664.8 | 877.3 | **PASSED** (1.32x)   |
| Bonus | 5286 | 877.3 | 16.6% of bonus threshold |
| DQN 2013 paper | 1705 | 877.3 | 51.4% of paper score |

---

## Conclusion

We successfully implemented the DQN algorithm from the 2013 paper and achieved scores well above the "Good Grade" threshold. The main challenges were:

1. **Stability without target network:** Required careful tuning of learning rate and gradient clipping
2. **Sparse rewards:** Seaquest's low reward frequency (~1.8%) made batch_size and buffer_size critical
3. **Normalization bugs:** Ensuring consistent pixel preprocessing throughout the pipeline

The average score of **877.3** demonstrates that the agent learned meaningful behavior, achieving 51.4% of the 2013 paper's reported performance (1705). However, we only reached 16.6% of the bonus threshold (5286). The gap to the bonus threshold is likely due to:
- **Resource and time constraints:** Training to 3M steps already takes significant time on available hardware. Going to 10M+ steps (as in the paper) would require substantially more computational resources and time than we had available
- **Lack of target network:** The 2013 paper uses a single network, which is inherently less stable than later DQN variants with target networks. Achieving the bonus score (which exceeds even the 2013 paper result) appears very difficult without this stabilization mechanism
- Potential differences in evaluation methodology

Given these constraints, we focused on achieving a reliable implementation that surpasses the "Good Grade" threshold rather than pursuing diminishing returns toward the bonus score.

All code was implemented from scratch in PyTorch, using only the ALE/Gymnasium environment interface.
