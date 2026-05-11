"""
HW3-1: Naive DQN with Experience Replay — Static Mode [30%]

Based on DRL in Action Ch.3 (Listings 3.5 / 3.7).
Architecture : 64 → 150 → 100 → 4  (same as book)
Mode         : static  (fixed layout, easiest)
"""

import numpy as np
import torch
import torch.nn as nn
from collections import deque
import random
import copy
import matplotlib.pyplot as plt

from Gridworld import Gridworld

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── Hyper-parameters ─────────────────────────────────────────────────────────
MODE        = 'static'
L1, L2, L3, L4 = 64, 150, 100, 4
LR          = 1e-3
GAMMA       = 0.9
EPSILON     = 1.0
EPSILON_MIN = 0.1
EPOCHS      = 2000
MEM_SIZE    = 1000
BATCH_SIZE  = 200
MAX_MOVES   = 50
SYNC_FREQ   = 200          # copy online → target every N steps
ACTION_SET  = {0:'u', 1:'d', 2:'l', 3:'r'}

# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    return nn.Sequential(
        nn.Linear(L1, L2), nn.ReLU(),
        nn.Linear(L2, L3), nn.ReLU(),
        nn.Linear(L3, L4),
    )

model  = build_model()
target = copy.deepcopy(model)
target.load_state_dict(model.state_dict())

loss_fn   = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
replay    = deque(maxlen=MEM_SIZE)
epsilon   = EPSILON

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_state(game):
    """Add small noise to break symmetry (same as book)."""
    s = game.board.render_np().reshape(1, 64).astype(np.float32)
    s += np.random.rand(1, 64).astype(np.float32) / 100.0
    return torch.from_numpy(s)

def get_reward_done(game):
    """Richer reward signal: goal=+10, pit=-10, step=-1."""
    r = game.reward()        # +1 / -1 / 0  (original)
    if r == 1:
        return 10.0, True
    if r == -1:
        return -10.0, True
    return -1.0, False

# ── Training Loop ─────────────────────────────────────────────────────────────
losses    = []
win_hist  = []
step_cnt  = 0

for epoch in range(EPOCHS):
    game   = Gridworld(size=4, mode=MODE)
    state1 = get_state(game)
    status = 1
    mov    = 0

    while status == 1:
        mov      += 1
        step_cnt += 1

        # ε-greedy action selection
        qval = model(state1)
        if random.random() < epsilon:
            action_ = np.random.randint(0, 4)
        else:
            action_ = int(torch.argmax(qval).item())

        game.makeMove(ACTION_SET[action_])
        state2        = get_state(game)
        reward, done  = get_reward_done(game)

        replay.append((state1, action_, reward, state2, done))
        state1 = state2

        # ── Mini-batch update ──────────────────────────────────────────────
        if len(replay) >= BATCH_SIZE:
            batch = random.sample(replay, BATCH_SIZE)
            s1b   = torch.cat([e[0] for e in batch])
            ab    = torch.tensor([e[1] for e in batch], dtype=torch.long)
            rb    = torch.tensor([e[2] for e in batch], dtype=torch.float32)
            s2b   = torch.cat([e[3] for e in batch])
            db    = torch.tensor([e[4] for e in batch], dtype=torch.float32)

            Q1 = model(s1b)
            with torch.no_grad():
                Q2 = target(s2b)

            # Bellman target
            Y = rb + GAMMA * (1 - db) * Q2.max(dim=1).values
            X = Q1.gather(1, ab.unsqueeze(1)).squeeze()

            loss = loss_fn(X, Y.detach())
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

        # Sync target network
        if step_cnt % SYNC_FREQ == 0:
            target.load_state_dict(model.state_dict())

        if done or mov >= MAX_MOVES:
            win_hist.append(1 if done and reward > 0 else 0)
            status = 0

    # Decay epsilon
    if epsilon > EPSILON_MIN:
        epsilon -= (EPSILON - EPSILON_MIN) / EPOCHS

    if (epoch + 1) % 200 == 0:
        recent_wr = np.mean(win_hist[-200:]) * 100
        print(f"Epoch {epoch+1:>4d} | ε={epsilon:.3f} | "
              f"Win rate (last 200): {recent_wr:.1f}%")

# ── Evaluation ────────────────────────────────────────────────────────────────
def test_model(model, mode='static', n=100, display=False):
    wins = 0
    for _ in range(n):
        game   = Gridworld(size=4, mode=mode)
        state  = get_state(game)
        for _ in range(MAX_MOVES):
            with torch.no_grad():
                action_ = int(torch.argmax(model(state)).item())
            game.makeMove(ACTION_SET[action_])
            state        = get_state(game)
            reward, done = get_reward_done(game)
            if display:
                print(game.display())
            if done:
                wins += (reward > 0)
                break
    return wins / n

wr = test_model(model, mode=MODE, n=200)
print(f"\nFinal win rate over 200 episodes: {wr*100:.1f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(losses);  axes[0].set_title('Training Loss'); axes[0].set_xlabel('Step')
axes[1].plot(np.convolve(win_hist, np.ones(50)/50, mode='valid'))
axes[1].set_title('Win Rate (50-ep moving avg)'); axes[1].set_xlabel('Episode')
plt.tight_layout()
plt.savefig('hw3_1_results.png', dpi=120)
plt.show()
print("Saved hw3_1_results.png")
