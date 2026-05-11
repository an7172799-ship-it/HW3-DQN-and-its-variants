"""
HW3-4 (Bonus): Rainbow DQN — Random Mode

Rainbow combines 4 key improvements over vanilla DQN:
  1. Double DQN          — remove overestimation bias
  2. Dueling Network     — separate V(s) and A(s,a)
  3. Prioritized Replay  — sample important transitions more often
  4. N-step Returns      — bootstrap over multiple steps (reduces variance)

(Distributional RL and Noisy Nets are omitted to keep code readable.)
"""

import numpy as np
import torch
import torch.nn as nn
import random
import copy
import matplotlib.pyplot as plt

from Gridworld import Gridworld

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── Hyper-parameters ─────────────────────────────────────────────────────────
MODE       = 'random'
LR         = 5e-4
GAMMA      = 0.99
EPSILON    = 1.0
EPS_MIN    = 0.02
EPOCHS     = 8000
MEM_SIZE   = 10000
BATCH_SIZE = 256
MAX_MOVES  = 60
SYNC_FREQ  = 400
N_STEP     = 3             # multi-step return horizon
ALPHA      = 0.6           # PER: prioritization exponent
BETA_START = 0.4           # PER: importance-sampling start
BETA_END   = 1.0           # PER: importance-sampling end (anneal to 1)
ACTION_SET = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}


# ── 1. Dueling DQN ────────────────────────────────────────────────────────────
class RainbowNet(nn.Module):
    def __init__(self, state_dim=64, action_dim=4):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),       nn.ReLU(),
        )
        self.value     = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self.advantage = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, action_dim))

    def forward(self, x):
        feat = self.feature(x)
        V    = self.value(feat)
        A    = self.advantage(feat)
        return V + A - A.mean(dim=1, keepdim=True)


# ── 2. Prioritized Experience Replay (PER) ────────────────────────────────────
class SumTree:
    """
    Binary sum-tree for O(log n) priority sampling.
    Leaves hold per-transition priorities; internal nodes hold sums.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity)   # tree nodes
        self.data     = [None] * capacity        # actual transitions
        self.write    = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left  = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    @property
    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        change         = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx  = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplay:
    def __init__(self, capacity, alpha=ALPHA):
        self.tree    = SumTree(capacity)
        self.alpha   = alpha
        self.max_p   = 1.0          # initial priority for new transitions
        self.epsilon = 1e-6         # small constant to avoid zero priority

    def add(self, transition):
        self.tree.add(self.max_p, transition)

    def sample(self, n, beta):
        batch, indices, weights = [], [], []
        segment = self.tree.total / n

        for i in range(n):
            lo, hi = segment * i, segment * (i + 1)
            s      = random.uniform(lo, hi)
            idx, priority, data = self.tree.get(s)
            if data is None:
                continue
            prob = priority / self.tree.total
            w    = (self.tree.n_entries * prob) ** (-beta)
            batch.append(data)
            indices.append(idx)
            weights.append(w)

        weights = np.array(weights, dtype=np.float32)
        weights /= weights.max()          # normalize
        return batch, indices, weights

    def update_priorities(self, indices, errors):
        for idx, err in zip(indices, errors):
            p = (abs(err) + self.epsilon) ** self.alpha
            self.max_p = max(self.max_p, p)
            self.tree.update(idx, p)

    def __len__(self):
        return self.tree.n_entries


# ── 3. N-step Return Buffer ───────────────────────────────────────────────────
class NStepBuffer:
    """Accumulates n consecutive transitions and computes n-step return."""
    def __init__(self, n, gamma):
        self.n     = n
        self.gamma = gamma
        self.buf   = []

    def add(self, transition):
        self.buf.append(transition)

    def get(self):
        """Return n-step (s0, a0, R_n, s_n, done_n) if buffer full."""
        if len(self.buf) < self.n:
            return None
        s0, a0, _, _, _ = self.buf[0]
        R = 0.0
        done_n = False
        for i, (_, _, r, _, d) in enumerate(self.buf):
            R += (self.gamma ** i) * r
            if d:
                done_n = True
                break
        sn, _, _, _, _ = self.buf[-1]
        self.buf.pop(0)
        return (s0, a0, R, sn, done_n)

    def reset(self):
        self.buf.clear()


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_state(game):
    s = game.board.render_np().reshape(1, 64).astype(np.float32)
    s += np.random.rand(1, 64).astype(np.float32) / 100.0
    return torch.from_numpy(s)

def get_reward_done(game):
    r = game.reward()
    if r == 1:  return 10.0, True
    if r == -1: return -10.0, True
    return -1.0, False


# ── Training ──────────────────────────────────────────────────────────────────
online   = RainbowNet()
target   = copy.deepcopy(online)
target.load_state_dict(online.state_dict())

optimizer = torch.optim.Adam(online.parameters(), lr=LR)
per       = PrioritizedReplay(MEM_SIZE)
epsilon   = EPSILON
losses, win_hist = [], []
step_cnt  = 0

for epoch in range(EPOCHS):
    game     = Gridworld(size=4, mode=MODE)
    state1   = get_state(game)
    n_buf    = NStepBuffer(N_STEP, GAMMA)
    beta     = BETA_START + (BETA_END - BETA_START) * epoch / EPOCHS

    for mov in range(MAX_MOVES):
        step_cnt += 1

        qval = online(state1)
        if random.random() < epsilon:
            action_ = np.random.randint(0, 4)
        else:
            action_ = int(torch.argmax(qval).item())

        game.makeMove(ACTION_SET[action_])
        state2        = get_state(game)
        reward, done  = get_reward_done(game)

        n_buf.add((state1, action_, reward, state2, done))
        nstep_trans = n_buf.get()
        if nstep_trans is not None:
            per.add(nstep_trans)

        state1 = state2

        # ── PER mini-batch update ──────────────────────────────────────────
        if len(per) >= BATCH_SIZE:
            batch, indices, weights = per.sample(BATCH_SIZE, beta)
            w_tensor = torch.tensor(weights, dtype=torch.float32)

            s1b = torch.cat([e[0] for e in batch])
            ab  = torch.tensor([e[1] for e in batch], dtype=torch.long)
            rb  = torch.tensor([e[2] for e in batch], dtype=torch.float32)
            s2b = torch.cat([e[3] for e in batch])
            db  = torch.tensor([e[4] for e in batch], dtype=torch.float32)

            Q1 = online(s1b)
            with torch.no_grad():
                # Double DQN selection + target evaluation
                a_star = online(s2b).argmax(dim=1, keepdim=True)
                Q2_val = target(s2b).gather(1, a_star).squeeze()

            gamma_n = GAMMA ** N_STEP
            Y  = rb + gamma_n * (1 - db) * Q2_val
            X  = Q1.gather(1, ab.unsqueeze(1)).squeeze()

            td_errors = (X - Y.detach()).abs().detach().cpu().numpy()
            per.update_priorities(indices, td_errors)

            # Importance-weighted loss
            loss = (w_tensor * nn.functional.smooth_l1_loss(X, Y.detach(),
                                                             reduction='none')).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), 10.0)  # gradient clip
            optimizer.step()
            losses.append(loss.item())

        if step_cnt % SYNC_FREQ == 0:
            target.load_state_dict(online.state_dict())

        if done:
            break

    n_buf.reset()
    win_hist.append(1 if (done and reward > 0) else 0)

    if epsilon > EPS_MIN:
        epsilon -= (EPSILON - EPS_MIN) / EPOCHS

    if (epoch + 1) % 500 == 0:
        wr = np.mean(win_hist[-300:]) * 100
        print(f"[Rainbow] Epoch {epoch+1:>5d} | ε={epsilon:.3f} | β={beta:.3f} | "
              f"WinRate(300): {wr:.1f}%")


# ── Final evaluation ──────────────────────────────────────────────────────────
online.eval()
wins = 0
N_EVAL = 500
for _ in range(N_EVAL):
    game  = Gridworld(size=4, mode=MODE)
    state = get_state(game)
    for _ in range(MAX_MOVES):
        with torch.no_grad():
            action_ = int(torch.argmax(online(state)).item())
        game.makeMove(ACTION_SET[action_])
        state        = get_state(game)
        reward, done = get_reward_done(game)
        if done:
            wins += (reward > 0)
            break

print(f"\nRainbow DQN final win rate ({N_EVAL} eps): {wins/N_EVAL*100:.1f}%")
print("Components used:")
print("  1. Double DQN     — online net selects, target net evaluates")
print("  2. Dueling Net    — V(s) + A(s,a) architecture")
print("  3. PER            — SumTree priority sampling (α=0.6, β→1.0)")
print(f"  4. N-step return  — horizon={N_STEP}, γ={GAMMA}")
print("  5. Gradient clip  — max_norm=10.0")
print("  6. Huber loss     — SmoothL1 weighted by IS weights")

# ── Plot ──────────────────────────────────────────────────────────────────────
def smooth(x, w=200):
    return np.convolve(x, np.ones(w)/w, mode='valid')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(smooth(win_hist)); axes[0].set_title('Win Rate (200-ep avg) — Rainbow DQN (random mode)')
axes[0].set_xlabel('Episode'); axes[0].set_ylabel('Win Rate')
axes[1].plot(np.convolve(losses, np.ones(1000)/1000, mode='valid'))
axes[1].set_title('Training Loss (1000-step avg)'); axes[1].set_xlabel('Step')
plt.tight_layout()
plt.savefig('hw3_4_results.png', dpi=120)
plt.show()
print("Saved hw3_4_results.png")
