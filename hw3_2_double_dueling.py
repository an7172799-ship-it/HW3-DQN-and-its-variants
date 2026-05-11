"""
HW3-2: Double DQN  +  Dueling DQN — Player Mode [40%]

Key ideas
─────────
Double DQN   : use online net to SELECT action, target net to EVALUATE it
               → removes overestimation bias from Vanilla DQN
Dueling DQN  : split Q into Value V(s) + Advantage A(s,a)
               → better credit-assignment, faster policy evaluation
"""

import numpy as np
import torch
import torch.nn as nn
from collections import deque
import random
import copy
import matplotlib.pyplot as plt

from Gridworld import Gridworld

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

MODE       = 'player'
LR         = 1e-3
GAMMA      = 0.9
EPSILON    = 1.0
EPS_MIN    = 0.05
EPOCHS     = 3000
MEM_SIZE   = 2000
BATCH_SIZE = 256
MAX_MOVES  = 50
SYNC_FREQ  = 300
ACTION_SET = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}


# ── Dueling DQN architecture ──────────────────────────────────────────────────
class DuelingDQN(nn.Module):
    """
    Q(s,a) = V(s) + [ A(s,a) - mean_a A(s,a) ]

    Shared feature extractor → two heads:
      Value head     : scalar V(s)
      Advantage head : vector A(s, ·)
    """
    def __init__(self, state_dim=64, action_dim=4, hidden=150):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 100),       nn.ReLU(),
        )
        self.value     = nn.Linear(100, 1)
        self.advantage = nn.Linear(100, action_dim)

    def forward(self, x):
        feat = self.feature(x)
        V    = self.value(feat)
        A    = self.advantage(feat)
        # mean-subtracted advantage for identifiability
        Q    = V + A - A.mean(dim=1, keepdim=True)
        return Q


# ── Vanilla DQN (same book architecture, for comparison) ─────────────────────
def build_vanilla():
    return nn.Sequential(
        nn.Linear(64, 150), nn.ReLU(),
        nn.Linear(150, 100), nn.ReLU(),
        nn.Linear(100, 4),
    )


# ── Shared helpers ────────────────────────────────────────────────────────────
def get_state(game):
    s = game.board.render_np().reshape(1, 64).astype(np.float32)
    s += np.random.rand(1, 64).astype(np.float32) / 100.0
    return torch.from_numpy(s)

def get_reward_done(game):
    r = game.reward()
    if r == 1:
        return 10.0, True
    if r == -1:
        return -10.0, True
    return -1.0, False


# ── Generic training function ─────────────────────────────────────────────────
def train(model_name, online, target, use_double=False):
    """
    model_name  : label for printing
    online      : online (learned) network
    target      : target network (frozen copy)
    use_double  : if True → Double DQN update rule
    """
    optimizer = torch.optim.Adam(online.parameters(), lr=LR)
    loss_fn   = nn.MSELoss()
    replay    = deque(maxlen=MEM_SIZE)
    epsilon   = EPSILON
    losses, win_hist = [], []
    step_cnt  = 0

    for epoch in range(EPOCHS):
        game   = Gridworld(size=4, mode=MODE)
        state1 = get_state(game)
        mov    = 0

        while True:
            mov += 1; step_cnt += 1

            qval = online(state1)
            if random.random() < epsilon:
                action_ = np.random.randint(0, 4)
            else:
                action_ = int(torch.argmax(qval).item())

            game.makeMove(ACTION_SET[action_])
            state2        = get_state(game)
            reward, done  = get_reward_done(game)

            replay.append((state1, action_, reward, state2, done))
            state1 = state2

            if len(replay) >= BATCH_SIZE:
                batch = random.sample(replay, BATCH_SIZE)
                s1b = torch.cat([e[0] for e in batch])
                ab  = torch.tensor([e[1] for e in batch], dtype=torch.long)
                rb  = torch.tensor([e[2] for e in batch], dtype=torch.float32)
                s2b = torch.cat([e[3] for e in batch])
                db  = torch.tensor([e[4] for e in batch], dtype=torch.float32)

                Q1 = online(s1b)

                with torch.no_grad():
                    if use_double:
                        # Double DQN: select with online, evaluate with target
                        a_star = online(s2b).argmax(dim=1, keepdim=True)
                        Q2_val = target(s2b).gather(1, a_star).squeeze()
                    else:
                        Q2_val = target(s2b).max(dim=1).values

                Y = rb + GAMMA * (1 - db) * Q2_val
                X = Q1.gather(1, ab.unsqueeze(1)).squeeze()

                loss = loss_fn(X, Y.detach())
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()

            if step_cnt % SYNC_FREQ == 0:
                target.load_state_dict(online.state_dict())

            if done or mov >= MAX_MOVES:
                win_hist.append(1 if (done and reward > 0) else 0)
                break

        if epsilon > EPS_MIN:
            epsilon -= (EPSILON - EPS_MIN) / EPOCHS

        if (epoch + 1) % 500 == 0:
            wr = np.mean(win_hist[-300:]) * 100
            print(f"[{model_name}] Epoch {epoch+1:>4d} | ε={epsilon:.3f} | "
                  f"WinRate(300): {wr:.1f}%")

    return losses, win_hist


# ── Run all three variants ────────────────────────────────────────────────────
print("=" * 60)
print("Training Vanilla DQN (baseline, player mode)")
print("=" * 60)
van_online = build_vanilla()
van_target = copy.deepcopy(van_online)
van_target.load_state_dict(van_online.state_dict())
van_losses, van_wins = train("Vanilla", van_online, van_target, use_double=False)

print("\n" + "=" * 60)
print("Training Double DQN (player mode)")
print("=" * 60)
dbl_online = build_vanilla()
dbl_target = copy.deepcopy(dbl_online)
dbl_target.load_state_dict(dbl_online.state_dict())
dbl_losses, dbl_wins = train("DoubleDQN", dbl_online, dbl_target, use_double=True)

print("\n" + "=" * 60)
print("Training Dueling DQN (player mode)")
print("=" * 60)
due_online = DuelingDQN()
due_target = copy.deepcopy(due_online)
due_target.load_state_dict(due_online.state_dict())
due_losses, due_wins = train("DuelingDQN", due_online, due_target, use_double=True)


# ── Final evaluation ──────────────────────────────────────────────────────────
def evaluate(model, mode, n=300):
    wins = 0
    for _ in range(n):
        game  = Gridworld(size=4, mode=mode)
        state = get_state(game)
        for _ in range(MAX_MOVES):
            with torch.no_grad():
                action_ = int(torch.argmax(model(state)).item())
            game.makeMove(ACTION_SET[action_])
            state        = get_state(game)
            reward, done = get_reward_done(game)
            if done:
                wins += (reward > 0)
                break
    return wins / n

print("\n── Final evaluation (300 episodes) ──")
for name, mdl in [("Vanilla  ", van_online),
                  ("Double   ", dbl_online),
                  ("Dueling  ", due_online)]:
    print(f"  {name}: {evaluate(mdl, MODE)*100:.1f}% win rate")


# ── Comparison plot ───────────────────────────────────────────────────────────
def smooth(x, w=100):
    return np.convolve(x, np.ones(w) / w, mode='valid')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Win rate curves
axes[0].plot(smooth(van_wins), label='Vanilla DQN')
axes[0].plot(smooth(dbl_wins), label='Double DQN')
axes[0].plot(smooth(due_wins), label='Dueling DQN')
axes[0].set_title(f'Win Rate (100-ep avg) — {MODE} mode')
axes[0].set_xlabel('Episode'); axes[0].legend()

# Loss curves
win_w = 500
axes[1].plot(np.convolve(van_losses, np.ones(win_w)/win_w, mode='valid'), label='Vanilla DQN')
axes[1].plot(np.convolve(dbl_losses, np.ones(win_w)/win_w, mode='valid'), label='Double DQN')
axes[1].plot(np.convolve(due_losses, np.ones(win_w)/win_w, mode='valid'), label='Dueling DQN')
axes[1].set_title('Training Loss (500-step avg)')
axes[1].set_xlabel('Step'); axes[1].legend()

plt.tight_layout()
plt.savefig('hw3_2_results.png', dpi=120)
plt.show()
print("Saved hw3_2_results.png")
