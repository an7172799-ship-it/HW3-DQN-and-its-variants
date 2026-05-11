# HW3: DQN and its Variants

> **Course**: Deep Reinforcement Learning  
> **Environment**: GridWorld (4×4) from [DRL in Action](https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction)  
> **Demo Site**: [GitHub Pages](https://an7172799-ship-it.github.io/HW3-DQN-and-its-variants/)

---

## 📁 File Structure

```
├── GridBoard.py              # Board engine (from DRL in Action repo)
├── Gridworld.py              # GridWorld environment (3 modes)
├── hw3_1_naive_dqn.py        # HW3-1: Naive DQN + Experience Replay (static)
├── hw3_2_double_dueling.py   # HW3-2: Double DQN & Dueling DQN (player)
├── hw3_3_lightning.py        # HW3-3: PyTorch Lightning DQN (random)
├── hw3_4_rainbow.py          # HW3-4: Rainbow DQN (random) [BONUS]
├── requirements.txt
└── docs/
    └── index.html            # Interactive demo site
```

---

## 🌍 Environment: GridWorld

A 4×4 grid with four objects — Player `P`, Goal `+`, Pit `-`, Wall `W`.

```
static mode layout:
+  -  .  P
.  W  .  .
.  .  .  .
.  .  .  .
```

| Object | Symbol | Effect |
|--------|--------|--------|
| Goal   | `+`    | +10 reward, episode ends (win) |
| Pit    | `-`    | −10 reward, episode ends (lose) |
| Wall   | `W`    | Impassable; staying in place |
| Player | `P`    | Controlled by agent |

### Three Modes

| Mode | Player | Goal/Pit/Wall | State Dim | Difficulty |
|------|--------|---------------|-----------|------------|
| `static` | Fixed at (0,3) | Fixed | 64 | Easy |
| `player` | Random | Fixed | 64 | Medium |
| `random` | Random | All Random | 64 | Hard |

**State representation**: `game.board.render_np().reshape(64)`  
→ 4 binary channels (one per object) × 4×4 grid, flattened to 64 dims

---

## HW3-1: Naive DQN — Static Mode [30%]

### What is DQN?

Deep Q-Network (DQN) approximates the Q-function Q(s, a) with a neural network, replacing the tabular Q-table. It adds two key stabilization tricks:

1. **Experience Replay** — store transitions `(s, a, r, s', done)` in a buffer; train on random mini-batches to break temporal correlations
2. **Target Network** — a frozen copy of the online network used to compute Bellman targets; updated every N steps

### Architecture

```
State (64) → Linear(150) → ReLU → Linear(100) → ReLU → Linear(4)
                                                         ↑
                                              Q-values for each action
```

### Bellman Update

$$Q(s,a) \leftarrow r + \gamma \cdot \max_{a'} Q_{\text{target}}(s', a') \cdot (1 - \text{done})$$

### Training Details

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam, lr=1e-3 |
| γ (discount) | 0.9 |
| ε decay | 1.0 → 0.1 over 2000 epochs |
| Replay size | 1000 |
| Batch size | 200 |
| Target sync | every 200 steps |
| Reward | Goal=+10, Pit=−10, Step=−1 |

### Run

```bash
python hw3_1_naive_dqn.py
```

---

## HW3-2: Double DQN & Dueling DQN — Player Mode [40%]

### Problem with Vanilla DQN: Overestimation

Vanilla DQN target:
$$Y = r + \gamma \max_{a'} Q_{\text{target}}(s', a')$$

`max` over a noisy Q-network → **systematically overestimates** future rewards → unstable learning.

---

### Double DQN

**Idea**: Decouple action *selection* from action *evaluation*.

$$Y^{\text{Double}} = r + \gamma \cdot Q_{\text{target}}\!\left(s',\; \arg\max_{a'} Q_{\text{online}}(s', a')\right)$$

- Online network → **selects** which action looks best  
- Target network → **evaluates** how good that action actually is  
- Eliminates the self-reinforcing overestimation loop

```python
# Standard DQN
Q2_val = target(s2).max(dim=1).values

# Double DQN  ← key change
a_star = online(s2).argmax(dim=1, keepdim=True)   # select with online
Q2_val = target(s2).gather(1, a_star).squeeze()    # evaluate with target
```

---

### Dueling DQN

**Idea**: Decompose Q into **state value** V(s) and **action advantage** A(s,a).

$$Q(s, a) = V(s) + \left[ A(s,a) - \frac{1}{|A|} \sum_{a'} A(s, a') \right]$$

- **V(s)**: How good is this state in general? (independent of action)  
- **A(s,a)**: How much better is action `a` vs. the average action?  
- Mean subtraction ensures identifiability (V and A are uniquely determined)

```
State (64) → Linear(150) → ReLU → Linear(100) → ReLU
                                                    ├─ Value head     → Linear(1)   → V(s)
                                                    └─ Advantage head → Linear(4)   → A(s,·)
                                                                              ↓
                                              Q = V + A − mean(A)
```

**Why it helps for `player` mode**: Even without visiting a specific (player_pos, action) pair, the agent can learn that certain *states* have high value, making generalization faster.

---

### Comparison

| Model | Overestimation Fix | Faster State Eval | Best For |
|-------|:-----------------:|:-----------------:|----------|
| Vanilla DQN | ✗ | ✗ | static |
| Double DQN | ✅ | ✗ | player |
| Dueling DQN | ✅ (with Double) | ✅ | player / random |

### Run

```bash
python hw3_2_double_dueling.py
# Trains all 3 variants, saves comparison plot to hw3_2_results.png
```

---

## HW3-3: PyTorch Lightning — Random Mode [30%]

### Why Lightning?

PyTorch Lightning separates *research code* (the model + loss) from *engineering code* (training loops, logging, checkpointing). It also natively supports training techniques via `Trainer` flags.

### Key Conversion

```python
# Raw PyTorch                     # PyTorch Lightning
for epoch in range(EPOCHS):   →   class DQNLightning(pl.LightningModule):
    for step in ...:                   def training_step(self, batch, idx):
        loss = compute_loss()              return compute_loss(batch)
        optimizer.zero_grad()
        loss.backward()            def configure_optimizers(self):
        optimizer.step()               return {"optimizer": ...,
                                               "lr_scheduler": ...}
```

### Training Techniques Integrated

| Technique | Implementation | Purpose |
|-----------|---------------|---------|
| **Gradient Clipping** | `Trainer(gradient_clip_val=1.0)` | Prevent exploding gradients |
| **LR Scheduling** | `StepLR(step_size=1000, gamma=0.5)` | Fine-tune as training stabilizes |
| **Huber Loss** | `SmoothL1Loss` instead of MSE | Robust to outlier TD errors |
| **Double DQN** | Online selects, target evaluates | Reduce overestimation |
| **Dueling Network** | V(s) + A(s,a) architecture | Better value estimation |
| **Target Network** | Sync every 300 steps | Stable training targets |

### Run

```bash
pip install pytorch-lightning
python hw3_3_lightning.py
```

---

## HW3-4 (Bonus): Rainbow DQN — Random Mode

### What is Rainbow?

Rainbow (Hessel et al., 2017) combines 6 independent DQN improvements into one agent. We implement 4:

| # | Component | What It Does |
|---|-----------|-------------|
| 1 | **Double DQN** | Removes overestimation bias |
| 2 | **Dueling Network** | Separates V(s) and A(s,a) |
| 3 | **Prioritized Experience Replay (PER)** | Trains more on surprising transitions |
| 4 | **N-step Returns** | Bootstraps over n=3 steps for faster credit assignment |

### Prioritized Experience Replay (PER)

Standard replay samples uniformly. PER samples proportional to **TD error**:

$$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}, \quad p_i = |\delta_i| + \epsilon$$

Where δᵢ is the TD error. High-error transitions = more surprising = more training value.

**Implementation**: SumTree data structure for O(log n) priority sampling.

**Importance-Sampling correction** to remove bias introduced by non-uniform sampling:

$$w_i = \left(\frac{1}{N \cdot P(i)}\right)^\beta, \quad \beta: 0.4 \to 1.0$$

### N-step Returns

Instead of 1-step Bellman:
$$Y_1 = r_t + \gamma \max Q(s_{t+1})$$

Use n-step return:
$$Y_n = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n \max Q(s_{t+n})$$

With n=3: the agent sees 3 steps of real reward before bootstrapping → less reliance on potentially inaccurate Q estimates.

### Run

```bash
python hw3_4_rainbow.py
```

---

## 🔧 Setup

```bash
pip install -r requirements.txt
```

Run all experiments:
```bash
python hw3_1_naive_dqn.py        # static mode
python hw3_2_double_dueling.py   # player mode (3 variants)
python hw3_3_lightning.py        # random mode (Lightning)
python hw3_4_rainbow.py          # random mode (Rainbow, bonus)
```

---

## 📊 Expected Results

| Model | Mode | Expected Win Rate |
|-------|------|:-----------------:|
| Naive DQN | static | ~95%+ |
| Vanilla DQN | player | ~60-70% |
| Double DQN | player | ~75-85% |
| Dueling DQN | player | ~80-90% |
| Lightning DQN | random | ~50-70% |
| Rainbow DQN | random | ~65-80% |

> Results vary by random seed and training duration.

---

## 📚 References

- Mnih et al. (2015). *Human-level control through deep reinforcement learning*. Nature.
- van Hasselt et al. (2016). *Deep Reinforcement Learning with Double Q-learning*. AAAI.
- Wang et al. (2016). *Dueling Network Architectures for Deep Reinforcement Learning*. ICML.
- Schaul et al. (2016). *Prioritized Experience Replay*. ICLR.
- Hessel et al. (2018). *Rainbow: Combining Improvements in Deep Reinforcement Learning*. AAAI.
- Zai & Brown (2020). *Deep Reinforcement Learning in Action*. Manning.
