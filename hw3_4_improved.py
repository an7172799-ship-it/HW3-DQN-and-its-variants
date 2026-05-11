"""
HW3-4 Improved: Rainbow DQN — 三項關鍵升級

原版 Rainbow 問題：
  1. state = 64-dim one-hot  →  網路要自己學空間關係，效率低
  2. 沒存最佳 checkpoint    →  peak 67.7% 訓練完卻剩 23%
  3. EPS_MIN 太低、訓練末期探索不足

改進方案：
  A. 相對位置特徵 (8-dim)   — Player 到每個物件的 (Δrow, Δcol)，直接告訴網路方向
  B. Best-model checkpoint  — 追蹤滾動 win rate，自動保存最好的模型
  C. Curriculum learning    — player mode 暖機 1500 epoch → random mode 正式訓練

保留 Rainbow 四大組件：
  1. Double DQN      4. N-step returns (n=2)
  2. Dueling Net     + Gradient clip + Huber loss
  3. PER (SumTree)
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

# ── Hyper-parameters ──────────────────────────────────────────────────────────
STATE_DIM   = 8            # 相對位置特徵維度
LR          = 5e-4
GAMMA       = 0.95
EPSILON     = 1.0
EPS_MIN     = 0.05
EPOCHS_WARM = 1500         # player mode 暖機
EPOCHS_MAIN = 5000         # random mode 主訓練
MEM_SIZE    = 10000
BATCH_SIZE  = 256
MAX_MOVES   = 50
SYNC_FREQ   = 200
N_STEP      = 3
ALPHA       = 0.6
BETA_START  = 0.4
BETA_END    = 1.0
ACTION_SET  = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ── 改進 A：相對位置特徵 ──────────────────────────────────────────────────────
def get_state(game):
    """
    原版：64-dim one-hot（網路要自己學距離概念）
    改進：8-dim 相對向量（直接給距離和方向）

    特徵：[P_row, P_col, ΔG_row, ΔG_col, ΔPit_row, ΔPit_col, ΔW_row, ΔW_col]
    全部除以 3（最大距離 = grid_size - 1），縮放到 [-1, 1]
    """
    b    = game.board
    p    = b.components['Player'].pos   # (row, col)
    g    = b.components['Goal'].pos
    pit  = b.components['Pit'].pos
    w    = b.components['Wall'].pos
    size = b.size - 1                   # 3 for 4x4 grid

    feat = np.array([
        p[0]/size - 0.5, p[1]/size - 0.5,          # player 絕對位置 (centered)
        (g[0]   - p[0]) / size,                     # goal 方向
        (g[1]   - p[1]) / size,
        (pit[0] - p[0]) / size,                     # pit 方向
        (pit[1] - p[1]) / size,
        (w[0]   - p[0]) / size,                     # wall 方向
        (w[1]   - p[1]) / size,
    ], dtype=np.float32)

    return torch.from_numpy(feat.reshape(1, STATE_DIM))

def get_reward_done(game):
    r = game.reward()
    if r == 1:  return 10.0, True
    if r == -1: return -10.0, True
    return -1.0, False


# ── Dueling DQN（輸入改為 8-dim）────────────────────────────────────────────
class RainbowNet(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=4):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU(),
        )
        self.value     = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.advantage = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, action_dim))

    def forward(self, x):
        feat = self.feature(x)
        V    = self.value(feat)
        A    = self.advantage(feat)
        return V + A - A.mean(dim=1, keepdim=True)


# ── SumTree（1-indexed，避免越界）────────────────────────────────────────────
class SumTree:
    def __init__(self, capacity):
        self.capacity  = capacity
        self.tree      = np.zeros(2 * capacity)
        self.data      = [None] * capacity
        self.write     = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = idx >> 1
        self.tree[parent] += change
        if parent > 1:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left  = idx << 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    @property
    def total(self):
        return self.tree[1]

    def add(self, priority, data):
        idx = self.write + self.capacity
        self.data[self.write] = data
        self.update(idx, priority)
        self.write     = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        change         = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx      = self._retrieve(1, s)
        data_idx = idx - self.capacity
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplay:
    def __init__(self, capacity, alpha=ALPHA):
        self.tree    = SumTree(capacity)
        self.alpha   = alpha
        self.max_p   = 1.0
        self.epsilon = 1e-6

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
                rand_i = random.randint(0, self.tree.n_entries - 1)
                leaf   = rand_i + self.tree.capacity
                priority = self.tree.tree[leaf]
                data   = self.tree.data[rand_i]
                idx    = leaf
            prob = max(priority, 1e-8) / self.tree.total
            w    = (self.tree.n_entries * prob) ** (-beta)
            batch.append(data); indices.append(idx); weights.append(w)
        weights = np.array(weights, dtype=np.float32)
        weights /= weights.max()
        return batch, indices, weights

    def update_priorities(self, indices, errors):
        for idx, err in zip(indices, errors):
            p = (abs(err) + self.epsilon) ** self.alpha
            self.max_p = max(self.max_p, p)
            self.tree.update(idx, p)

    def __len__(self):
        return self.tree.n_entries


class NStepBuffer:
    def __init__(self, n, gamma):
        self.n = n; self.gamma = gamma; self.buf = []

    def add(self, transition):
        self.buf.append(transition)

    def get(self):
        if len(self.buf) < self.n:
            return None
        s0, a0, _, _, _ = self.buf[0]
        R = 0.0; done_n = False
        for i, (_, _, r, _, d) in enumerate(self.buf):
            R += (self.gamma ** i) * r
            if d: done_n = True; break
        sn = self.buf[-1][3]
        self.buf.pop(0)
        return (s0, a0, R, sn, done_n)

    def reset(self):
        self.buf.clear()


# ── 訓練函式（可切換 mode）────────────────────────────────────────────────────
def train_phase(online, target, optimizer, per, mode, epochs,
                epsilon_start, epsilon_end, beta_start, beta_end,
                step_cnt_init=0, win_hist=None, losses=None):
    """
    通用訓練函式，回傳 (losses, win_hist, epsilon, step_cnt)
    """
    if win_hist is None: win_hist = []
    if losses   is None: losses   = []

    epsilon   = epsilon_start
    step_cnt  = step_cnt_init
    best_wr   = 0.0
    best_state_dict = copy.deepcopy(online.state_dict())

    for epoch in range(epochs):
        game   = Gridworld(size=4, mode=mode)
        state1 = get_state(game)
        n_buf  = NStepBuffer(N_STEP, GAMMA)
        beta   = beta_start + (beta_end - beta_start) * epoch / epochs

        for mov in range(MAX_MOVES):
            step_cnt += 1
            qval = online(state1.to(device))
            action_ = np.random.randint(0, 4) if random.random() < epsilon \
                      else int(torch.argmax(qval).item())

            game.makeMove(ACTION_SET[action_])
            state2       = get_state(game)
            reward, done = get_reward_done(game)

            n_buf.add((state1, action_, reward, state2, done))
            nstep_trans = n_buf.get()
            if nstep_trans is not None:
                per.add(nstep_trans)

            state1 = state2

            if len(per) >= BATCH_SIZE:
                batch, indices, weights = per.sample(BATCH_SIZE, beta)
                w_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

                s1b = torch.cat([e[0] for e in batch]).to(device)
                ab  = torch.tensor([e[1] for e in batch], dtype=torch.long).to(device)
                rb  = torch.tensor([e[2] for e in batch], dtype=torch.float32).to(device)
                s2b = torch.cat([e[3] for e in batch]).to(device)
                db  = torch.tensor([e[4] for e in batch], dtype=torch.float32).to(device)

                Q1 = online(s1b)
                with torch.no_grad():
                    a_star = online(s2b).argmax(dim=1, keepdim=True)
                    Q2_val = target(s2b).gather(1, a_star).squeeze()

                gamma_n = GAMMA ** N_STEP
                Y = rb + gamma_n * (1 - db) * Q2_val
                X = Q1.gather(1, ab.unsqueeze(1)).squeeze()

                td_errors = (X - Y.detach()).abs().detach().cpu().numpy()
                per.update_priorities(indices, td_errors)

                loss = (w_tensor * nn.functional.smooth_l1_loss(
                    X, Y.detach(), reduction='none')).mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                optimizer.step()
                losses.append(loss.item())

            if step_cnt % SYNC_FREQ == 0:
                target.load_state_dict(online.state_dict())

            if done:
                break

        n_buf.reset()
        win_hist.append(1 if (done and reward > 0) else 0)
        if epsilon > epsilon_end:
            epsilon -= (epsilon_start - epsilon_end) / epochs

        # ── 改進 B：Best checkpoint ─────────────────────────────────────────
        if len(win_hist) >= 200:
            wr = float(np.mean(win_hist[-200:]))
            if wr > best_wr:
                best_wr = wr
                best_state_dict = copy.deepcopy(online.state_dict())

        if (epoch + 1) % 500 == 0:
            wr = np.mean(win_hist[-300:]) * 100 if len(win_hist) >= 300 \
                 else np.mean(win_hist) * 100
            print(f"  [{mode}] Epoch {epoch+1:>5d} | ε={epsilon:.3f} | "
                  f"β={beta:.3f} | WinRate(300): {wr:.1f}% | best={best_wr*100:.1f}%")

    return losses, win_hist, epsilon, step_cnt, best_wr, best_state_dict


# ── 初始化 ─────────────────────────────────────────────────────────────────────
online    = RainbowNet().to(device)
target    = copy.deepcopy(online).to(device)
optimizer = torch.optim.Adam(online.parameters(), lr=LR)
per       = PrioritizedReplay(MEM_SIZE)
all_losses, all_wins = [], []
total_steps = 0

# ── 改進 C：Curriculum Learning ──────────────────────────────────────────────
print("=" * 60)
print(f"Phase 1 — Curriculum: player mode ({EPOCHS_WARM} epochs)")
print("(學習追 goal、避 pit，建立基礎策略)")
print("=" * 60)

all_losses, all_wins, eps, total_steps, best_wr_warm, _ = train_phase(
    online, target, optimizer, per,
    mode='player', epochs=EPOCHS_WARM,
    epsilon_start=EPSILON, epsilon_end=0.2,  # player mode 結束時還保留較多探索
    beta_start=BETA_START, beta_end=0.6,
    step_cnt_init=total_steps,
    win_hist=all_wins, losses=all_losses,
)
print(f"\nPhase 1 done. Best win rate: {best_wr_warm*100:.1f}%\n")

print("=" * 60)
print(f"Phase 2 — Main: random mode ({EPOCHS_MAIN} epochs)")
print("(全隨機佈局，遷移 phase 1 知識)")
print("=" * 60)

# 沿用相同 network 權重（遷移學習），epsilon 從 phase 1 結束值繼續
all_losses, all_wins, eps, total_steps, best_wr_main, best_sd = train_phase(
    online, target, optimizer, per,
    mode='random', epochs=EPOCHS_MAIN,
    epsilon_start=eps,    # 銜接 phase 1 的 epsilon
    epsilon_end=EPS_MIN,
    beta_start=0.6, beta_end=BETA_END,
    step_cnt_init=total_steps,
    win_hist=all_wins, losses=all_losses,
)
print(f"\nPhase 2 done. Best win rate: {best_wr_main*100:.1f}%")


# ── 改進 B：載入最佳 checkpoint 做 greedy eval ─────────────────────────────
online.load_state_dict(best_sd)
online_cpu = online.cpu().eval()

print("\n" + "=" * 60)
print("Final Evaluation (greedy, 500 episodes, best checkpoint)")
print("=" * 60)
wins = 0
N_EVAL = 500
for _ in range(N_EVAL):
    game  = Gridworld(size=4, mode='random')
    state = get_state(game)
    for _ in range(MAX_MOVES):
        with torch.no_grad():
            action_ = int(torch.argmax(online_cpu(state)).item())
        game.makeMove(ACTION_SET[action_])
        state        = get_state(game)
        reward, done = get_reward_done(game)
        if done:
            wins += (reward > 0)
            break

print(f"Win rate: {wins/N_EVAL*100:.1f}%  (vs 原版 23%)")
print("\n改進清單：")
print("  A. 相對位置特徵 (8-dim)  — 64-dim one-hot → 直接編碼方向距離")
print("  B. Best checkpoint       — 自動儲存 rolling win rate 最高的模型")
print(f"  C. Curriculum learning   — player {EPOCHS_WARM} ep → random {EPOCHS_MAIN} ep")
print("  + 原版 Rainbow: Double DQN + Dueling + PER + N-step")


# ── Plot ──────────────────────────────────────────────────────────────────────
def smooth(x, w=200):
    return np.convolve(x, np.ones(w)/w, mode='valid')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

win_arr = np.array(all_wins)
axes[0].plot(smooth(win_arr))
axes[0].axvline(EPOCHS_WARM, color='gray', linestyle='--', alpha=0.7, label='player→random')
axes[0].set_title('Win Rate (200-ep avg) — Improved Rainbow')
axes[0].set_xlabel('Episode'); axes[0].set_ylabel('Win Rate'); axes[0].legend()

axes[1].plot(np.convolve(all_losses, np.ones(1000)/1000, mode='valid'))
axes[1].set_title('Training Loss (1000-step avg)')
axes[1].set_xlabel('Step')

plt.tight_layout()
plt.savefig('hw3_4_improved_results.png', dpi=120)
plt.show()
print("Saved hw3_4_improved_results.png")
