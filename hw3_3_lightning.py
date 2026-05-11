"""
HW3-3: DQN with PyTorch Lightning — Random Mode [30%]

Converts the vanilla DQN training loop into a LightningModule.
Training techniques integrated:
  • Gradient clipping   (gradient_clip_val=1.0)
  • LR scheduling       (StepLR: decay by 0.5 every 1000 epochs)
  • Target network sync (every SYNC_FREQ steps)
  • ε-greedy decay      (from 1.0 → 0.05 over training)

Install: pip install pytorch-lightning
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from collections import deque
import random
import copy

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from Gridworld import Gridworld

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── Hyper-parameters ─────────────────────────────────────────────────────────
MODE       = 'random'
LR         = 1e-3
GAMMA      = 0.9
EPSILON    = 1.0
EPS_MIN    = 0.05
EPOCHS     = 5000
MEM_SIZE   = 5000
BATCH_SIZE = 256
MAX_MOVES  = 50
SYNC_FREQ  = 300
ACTION_SET = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}


# ── Dueling DQN (same as HW3-2, works better for random mode) ─────────────────
class DuelingDQN(nn.Module):
    def __init__(self, state_dim=64, action_dim=4):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
        )
        self.value     = nn.Linear(128, 1)
        self.advantage = nn.Linear(128, action_dim)

    def forward(self, x):
        feat = self.feature(x)
        V    = self.value(feat)
        A    = self.advantage(feat)
        return V + A - A.mean(dim=1, keepdim=True)


# ── Replay Buffer as a Dataset ────────────────────────────────────────────────
class ReplayDataset(Dataset):
    """Wraps a deque so DataLoader can sample from it."""
    def __init__(self, buffer: deque):
        self._buf = buffer

    def __len__(self):
        return len(self._buf)

    def __getitem__(self, idx):
        s1, a, r, s2, done = self._buf[idx]
        return (s1.squeeze(0),
                torch.tensor(a,    dtype=torch.long),
                torch.tensor(r,    dtype=torch.float32),
                s2.squeeze(0),
                torch.tensor(done, dtype=torch.float32))


# ── Lightning Module ──────────────────────────────────────────────────────────
class DQNLightning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.online  = DuelingDQN()
        self.target  = copy.deepcopy(self.online)
        self.target.load_state_dict(self.online.state_dict())
        self.loss_fn  = nn.SmoothL1Loss()          # Huber loss — more stable
        self.replay   = deque(maxlen=MEM_SIZE)
        self.epsilon  = EPSILON
        self._step    = 0
        self._wins    = []

        # Pre-fill replay buffer before first training step
        self._populate_replay()

    # ── Replay population ──────────────────────────────────────────────────
    def _get_state(self, game):
        s = game.board.render_np().reshape(1, 64).astype(np.float32)
        s += np.random.rand(1, 64).astype(np.float32) / 100.0
        return torch.from_numpy(s)

    def _reward_done(self, game):
        r = game.reward()
        if r == 1:  return 10.0, True
        if r == -1: return -10.0, True
        return -1.0, False

    def _populate_replay(self, n=BATCH_SIZE * 2):
        while len(self.replay) < n:
            game   = Gridworld(size=4, mode=MODE)
            state1 = self._get_state(game)
            for _ in range(MAX_MOVES):
                action_ = np.random.randint(0, 4)
                game.makeMove(ACTION_SET[action_])
                state2        = self._get_state(game)
                reward, done  = self._reward_done(game)
                self.replay.append((state1, action_, reward, state2, done))
                state1 = state2
                if done:
                    break

    def collect_experience(self):
        """Run one full episode and add transitions to replay."""
        game   = Gridworld(size=4, mode=MODE)
        state1 = self._get_state(game)
        episode_win = False
        for _ in range(MAX_MOVES):
            with torch.no_grad():
                if random.random() < self.epsilon:
                    action_ = np.random.randint(0, 4)
                else:
                    action_ = int(torch.argmax(self.online(state1.to(self.device))).item())
            game.makeMove(ACTION_SET[action_])
            state2        = self._get_state(game)
            reward, done  = self._reward_done(game)
            self.replay.append((state1, action_, reward, state2, done))
            state1 = state2
            if done:
                episode_win = (reward > 0)
                break
        self._wins.append(int(episode_win))
        if self.epsilon > EPS_MIN:
            self.epsilon -= (EPSILON - EPS_MIN) / EPOCHS

    # ── DataLoader ─────────────────────────────────────────────────────────
    def train_dataloader(self):
        dataset = ReplayDataset(self.replay)
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, drop_last=True)

    # ── Training step ──────────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        self._step += 1
        # Collect new experience every step
        self.collect_experience()

        s1, a, r, s2, done = batch

        Q1     = self.online(s1)
        with torch.no_grad():
            # Double DQN target
            a_star = self.online(s2).argmax(dim=1, keepdim=True)
            Q2_val = self.target(s2).gather(1, a_star).squeeze()

        Y = r + GAMMA * (1 - done) * Q2_val
        X = Q1.gather(1, a.unsqueeze(1)).squeeze()

        loss = self.loss_fn(X, Y)

        # Sync target network
        if self._step % SYNC_FREQ == 0:
            self.target.load_state_dict(self.online.state_dict())

        # Logging
        self.log('train_loss', loss, prog_bar=True)
        if len(self._wins) >= 100:
            wr = float(np.mean(self._wins[-100:]))
            self.log('win_rate_100', wr, prog_bar=True)

        return loss

    # ── Optimizer + LR scheduler ───────────────────────────────────────────
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.online.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1000, gamma=0.5
        )
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    model = DQNLightning()

    checkpoint_cb = ModelCheckpoint(
        monitor='win_rate_100', mode='max',
        filename='best-{epoch:02d}-{win_rate_100:.2f}',
        save_top_k=1, verbose=True,
    )

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        gradient_clip_val=1.0,
        accelerator=accelerator,
        devices=1,
        callbacks=[checkpoint_cb],
        enable_progress_bar=True,
        log_every_n_steps=50,
    )
    print(f"Using device: {accelerator.upper()}")

    trainer.fit(model)

    # ── Final evaluation ───────────────────────────────────────────────────
    online = model.online.cpu()
    online.eval()
    ACTION_SET = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    wins = 0
    N_EVAL = 500
    for _ in range(N_EVAL):
        game  = Gridworld(size=4, mode=MODE)
        s     = game.board.render_np().reshape(1, 64).astype(np.float32)
        state = torch.from_numpy(s)
        for _ in range(MAX_MOVES):
            with torch.no_grad():
                action_ = int(torch.argmax(online(state)).item())
            game.makeMove(ACTION_SET[action_])
            s     = game.board.render_np().reshape(1, 64).astype(np.float32)
            state = torch.from_numpy(s)
            r, done = model._reward_done(game)
            if done:
                wins += (r > 0)
                break

    print(f"\nFinal win rate ({N_EVAL} eps): {wins/N_EVAL*100:.1f}%")
    print("Training techniques used:")
    print("  • Gradient clipping  : clip_val=1.0")
    print("  • LR scheduling      : StepLR (×0.5 every 1000 epochs)")
    print("  • Huber loss         : SmoothL1Loss (more robust than MSE)")
    print("  • Double DQN update  : reduce overestimation")
    print("  • Dueling network    : better value estimation")
    print("  • Target network     : sync every", SYNC_FREQ, "steps")
