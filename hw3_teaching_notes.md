# HW3 深度教學筆記：DQN 及其變體

> 本文件為 AI 對學生的完整教學紀錄，以中文撰寫，涵蓋原理解說、設計思路與深度討論。

---

## 目錄

1. [強化學習基礎回顧](#1-強化學習基礎回顧)
2. [為什麼需要 DQN？從 Q-Table 到神經網路](#2-為什麼需要-dqn從-q-table-到神經網路)
3. [HW3-1：Naive DQN 詳解](#3-hw3-1naive-dqn-詳解)
4. [HW3-2：Double DQN 與 Dueling DQN](#4-hw3-2double-dqn-與-dueling-dqn)
5. [HW3-3：為什麼要用 PyTorch Lightning？](#5-hw3-3為什麼要用-pytorch-lightning)
6. [HW3-4：Rainbow DQN 深度剖析](#6-hw3-4rainbow-dqn-深度剖析)
7. [三種 GridWorld 模式的策略差異](#7-三種-gridworld-模式的策略差異)
8. [常見問題與除錯建議](#8-常見問題與除錯建議)

---

## 1. 強化學習基礎回顧

### 1.1 核心概念

強化學習（Reinforcement Learning, RL）是讓智能體（Agent）透過與環境互動來學習最佳決策的方法。

```
Agent ──(action a)──→ Environment
  ↑                        |
  └──(state s, reward r)───┘
```

| 符號 | 名稱 | 在 GridWorld 中的意義 |
|------|------|----------------------|
| s    | State（狀態） | 棋盤上所有物件的位置 |
| a    | Action（動作） | 上下左右移動 |
| r    | Reward（獎勵） | +10（到達終點）/ −10（掉入陷阱）/ −1（每步） |
| π    | Policy（策略） | 在每個狀態下應選哪個動作 |
| γ    | Discount factor | 0.9，代表未來獎勵比當前獎勵稍微不重要 |

### 1.2 為什麼需要 Discount Factor（γ）？

想像你在 GridWorld 裡，最終獎勵是 +10，但需要走 5 步才到達。
- **沒有 γ（γ=1）**：不管走多少步，+10 就是 +10，模型無法判斷「快點到」比「慢慢到」更好
- **有 γ（γ=0.9）**：走 5 步後的 +10，對當前狀態的價值是 0.9⁵ × 10 ≈ 5.9；走 3 步的話是 0.9³ × 10 ≈ 7.3

**γ 讓 agent 學會「越快完成越好」。**

### 1.3 Q-Function 是什麼？

Q(s, a) 代表「在狀態 s 下執行動作 a，之後遵循最佳策略能獲得的總未來獎勵」。

最佳策略就是：
$$\pi^*(s) = \arg\max_a Q(s, a)$$

只要有了 Q-function，策略就確定了：每個狀態都選 Q 值最大的動作。

---

## 2. 為什麼需要 DQN？從 Q-Table 到神經網路

### 2.1 傳統 Q-Table 的問題

傳統 Q-learning 用一張大表格來記錄每個 (s, a) 的 Q 值：

```
State | Up  | Down | Left | Right
------|-----|------|------|------
(0,0) | ... | ...  | ...  | ...
(0,1) | ... | ...  | ...  | ...
...
```

**在 static mode**：只有 16 個 player 位置 × 4 個動作 = 64 個格子 → 可行

**在 random mode**：Player 位置 × Goal 位置 × Pit 位置 × Wall 位置 × 4 動作 = 16⁴ × 4 = **262,144 個格子** → 太大了，而且大多數從未被訪問過

### 2.2 DQN 的核心思想

用神經網路 **近似** Q-function：

$$Q(s, a) \approx f_\theta(s, a)$$

輸入狀態 s（64 維向量），輸出所有動作的 Q 值（4 個數字）：

```python
# 一次 forward pass 得到所有 4 個動作的 Q 值
q_values = model(state)   # shape: [1, 4]
# 選 Q 值最大的動作
action = q_values.argmax()
```

**為什麼這樣更好？**
- 神經網路可以泛化：沒見過的狀態也能預測合理的 Q 值
- 參數量固定，不隨 state space 爆炸

### 2.3 Bellman Equation：DQN 的訓練目標

DQN 的訓練目標來自 **Bellman Optimality Equation**：

$$Q^*(s, a) = \mathbb{E}\left[r + \gamma \max_{a'} Q^*(s', a')\right]$$

這個式子說的是：「現在這個 (s, a) 的最優 Q 值 = 立即獎勵 r，加上下一個狀態能得到的最大未來價值」

訓練時，我們用這個式子製造「目標值」（Y），然後讓網路輸出趨近 Y：

```
目標 Y = r + γ × max(Q_target(s'))    ← 用 target 網路算
預測 X = Q_online(s, a)               ← 用 online 網路算
損失 L = MSE(X, Y)                    ← 反向傳播更新 online 網路
```

---

## 3. HW3-1：Naive DQN 詳解

### 3.1 為什麼 Naive DQN 在 Static Mode 能成功？

Static mode 只有 16 種不同的 player 位置，goal/pit/wall 永遠在同一個地方。

這相當於一個非常簡單的導航問題：從任意位置出發，學會往左上角走。神經網路只需要記住 16 種情況的最佳動作，而不是泛化到未見過的情況。

**為什麼加 noise 到 state？**

```python
state = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
```

原始 state 是二進位向量（全是 0 和 1）。在 static mode 的早期訓練，如果兩個不同的 state 恰好讓網路輸出相同的 Q 值，就會陷入局部最優。加入微小 noise 可以打破這種對稱性，讓訓練更穩定。

### 3.2 Experience Replay 的深層原理

**問題：為什麼不直接用最新的經驗訓練？**

假設 agent 在走廊裡連續走了 20 步，這 20 個 transition 高度相關。如果直接用這批資料訓練，相當於對同一種情況反覆學習，會 overfitting 到當前軌跡，破壞之前學到的知識。

**Replay Buffer 的解法：**
1. 把 transition 存進 buffer（像日記一樣）
2. 每次從整個 buffer 裡**隨機抽取** mini-batch
3. 這樣每個 batch 包含來自不同時期、不同情境的經驗 → 打破相關性

```python
# 存入 buffer
replay.append((state1, action_, reward, state2, done))

# 隨機抽取 mini-batch（關鍵！）
minibatch = random.sample(replay, batch_size)
```

**直覺類比：** 你學習一門語言，如果每天只練昨天的對話，進步很慢。但如果每天從過去所有課程裡隨機複習，進步更快且更扎實。

### 3.3 Target Network 的深層原理

**問題：為什麼訓練不穩定？**

DQN 訓練有一個根本的矛盾：

```
計算目標 Y 時用的 Q_target
           ↓
    就是當前正在更新的網路 Q_online
           ↓
  每更新一步，目標 Y 就跟著變
           ↓
      像追一個移動的靶
```

這叫做 "Moving Target Problem"，會導致訓練震盪甚至發散。

**解法：Target Network**

```python
# 建立兩個網路
model  = build_model()          # online 網路：每步更新
target = copy.deepcopy(model)   # target 網路：每 200 步才更新一次

# 計算目標 Y 時，用固定的 target 網路
with torch.no_grad():
    Q2 = target(state2)   # target 不反向傳播
Y = reward + gamma * Q2.max()

# 每 200 步同步一次
if step % sync_freq == 0:
    target.load_state_dict(model.state_dict())
```

**直覺類比：** 考試時，答案卷（target）是固定的。你每次交作業（online 更新）都對照這份答案卷。每 200 次作業後，老師更新一次答案卷（sync）。這樣你的學習目標是穩定的。

---

## 4. HW3-2：Double DQN 與 Dueling DQN

### 4.1 Double DQN：解決過高估計問題

#### 為什麼 Vanilla DQN 會過估 Q 值？

Vanilla DQN 的目標：

$$Y = r + \gamma \cdot \underbrace{\max_{a'} Q_\theta(s', a')}_{\text{同一個網路選 + 評分}}$$

假設網路對 3 個動作的 Q 值估計有隨機誤差：
```
真實 Q 值：[3.0, 5.0, 2.0]
網路估計：  [3.5, 4.8, 2.3]  （有正負誤差）
```

`max()` 永遠會挑**估計偏高**的那個：max([3.5, 4.8, 2.3]) = 4.8（而不是真正最大的 5.0）

但如果誤差是 `[3.5, 5.5, 2.3]`，max 就變成 5.5，過估更嚴重。

**統計上，max 會系統性地挑被高估的動作 → 每步都累積過估 → Q 值越來越虛高**

#### Double DQN 的修正

把「選動作」和「評分」拆開：

```python
# Step 1：用 online 網路選「看起來最好的動作」
a_star = online(s2).argmax(dim=1, keepdim=True)

# Step 2：用 target 網路評估這個動作的真實價值
Q2_val = target(s2).gather(1, a_star).squeeze()

# 目標 Y
Y = r + gamma * Q2_val
```

**為什麼這樣能修正？**

- Online 網路和 target 網路的誤差是獨立的
- Online 偶爾高估動作 A，但 target 可能認為動作 A 沒那麼好 → 兩個網路的誤差互相抵銷
- 統計上消除了系統性偏差

**效果對比：**
| | Vanilla DQN | Double DQN |
|---|---|---|
| Q 值估計 | 系統性偏高 | 接近真實值 |
| 收斂速度 | 較慢（因為偏差） | 較快 |
| Player mode 表現 | 60-70% | 75-85% |

### 4.2 Dueling DQN：為什麼要拆分 V 和 A？

#### 核心洞察

在很多狀態下，**「這個狀態好不好」** 和 **「哪個動作最好」** 是兩個獨立的問題。

**例子：** 在 GridWorld 的中央格（遠離所有特殊格子）：
- V(s)：這個狀態本來就平淡，大概值 −3（距離目標還很遠）
- A(s, 上) = +0.5（往目標方向走更好）
- A(s, 右) = −0.1（稍微偏離）
- A(s, 下) = −0.3（遠離目標）

只要知道 V(s) ≈ −3，再加上各動作的相對優勢，就能得到精確的 Q 值，不需要對每個 (s, a) 組合都從頭學。

#### 數學定義

$$Q(s, a) = V(s) + \left[A(s, a) - \frac{1}{|A|}\sum_{a'} A(s, a')\right]$$

**為什麼要減去 mean(A)？**

如果不減，V 和 A 可以互相抵消：
- V = 0, A = [3, 5, 2] → Q = [3, 5, 2]
- V = 5, A = [−2, 0, −3] → Q = [3, 5, 2]

兩個不同的 (V, A) 給出相同的 Q，網路無法確定應該學哪一個（不可識別性問題）。

減去 mean 後，A 的均值強制為 0，V 就有唯一的含義了。

#### 網路結構

```python
class DuelingDQN(nn.Module):
    def __init__(self):
        # 共享特徵提取層
        self.feature = nn.Sequential(
            nn.Linear(64, 150), nn.ReLU(),
            nn.Linear(150, 100), nn.ReLU(),
        )
        # 兩個獨立的頭
        self.value     = nn.Linear(100, 1)      # 輸出 V(s)：單一數值
        self.advantage = nn.Linear(100, 4)      # 輸出 A(s,·)：4 個動作的優勢

    def forward(self, x):
        feat = self.feature(x)
        V = self.value(feat)                    # shape: [batch, 1]
        A = self.advantage(feat)                # shape: [batch, 4]
        Q = V + A - A.mean(dim=1, keepdim=True) # shape: [batch, 4]
        return Q
```

#### 為什麼 Dueling 特別適合 Player Mode？

在 player mode，goal/pit/wall 位置固定。當 player 靠近 goal 的格子時，不管往哪個方向移動，這個狀態本身就很有價值（V(s) 很高）。

Dueling 網路可以：
1. 快速學到「靠近 goal 的狀態 = 高 V(s)」
2. 再細化「在這個狀態，往 goal 方向那個動作的 A 值更高」

而 Vanilla DQN 必須對每個 (靠近 goal 的位置, 動作) 組合分別學習，效率低很多。

---

## 5. HW3-3：為什麼要用 PyTorch Lightning？

### 5.1 原始 PyTorch 的痛點

寫一個完整的 DQN 訓練迴圈，原始 PyTorch 需要處理：

```python
# 這些全都要手動寫
for epoch in range(EPOCHS):
    # 環境互動
    # replay buffer 管理
    # mini-batch 抽樣
    # forward pass
    # 計算 loss
    optimizer.zero_grad()
    loss.backward()
    # 手動梯度裁剪（如果需要）
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    # 手動 LR scheduler
    scheduler.step()
    # 手動存 checkpoint
    if best_reward:
        torch.save(model.state_dict(), 'best.pth')
    # 手動 logging
```

這些「工程程式碼」和「研究程式碼」混在一起，很難閱讀和維護。

### 5.2 Lightning 的設計哲學

Lightning 把訓練拆成幾個清晰的部分：

```
LightningModule
├── training_step()        ← 你只需要寫這個：怎麼算 loss
├── configure_optimizers() ← optimizer 和 scheduler 的設定
└── forward()              ← 模型的前向傳播

Trainer                    ← 所有工程細節都在這裡
├── gradient_clip_val      ← 梯度裁剪
├── max_epochs             ← 訓練輪數
├── callbacks              ← checkpoint、early stopping 等
└── log_every_n_steps      ← logging 頻率
```

### 5.3 訓練技巧的原理

#### 梯度裁剪（Gradient Clipping）

**問題：** 當 TD error 很大時（例如一開始 Q 值估計很差），loss 很大，梯度也很大，一次更新幅度太大，直接破壞已學到的知識。

**解法：** 把梯度的 L2 norm 裁剪到最大值 1.0

```python
# Trainer 會自動在 loss.backward() 後執行：
torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
```

#### 學習率排程（LR Scheduling）

**問題：** 訓練初期需要大步探索，後期需要小步精調。固定 lr=1e-3 在後期可能讓模型震盪。

**解法：** StepLR — 每 1000 個 epoch 把 lr 乘以 0.5

```
Epoch   0-1000:  lr = 1e-3
Epoch 1000-2000: lr = 5e-4
Epoch 2000-3000: lr = 2.5e-4
```

#### Huber Loss（SmoothL1Loss）

**問題：** MSE 對離群值非常敏感。如果某個 transition 的 TD error 特別大（比如第一次看到 reward），loss 會暴增，梯度也暴增。

**Huber Loss 的行為：**
$$L_\delta(y) = \begin{cases} \frac{1}{2}x^2 & \text{if } |x| \leq 1 \\ |x| - \frac{1}{2} & \text{otherwise} \end{cases}$$

- 小誤差（|e| ≤ 1）：和 MSE 一樣，平滑梯度
- 大誤差（|e| > 1）：和 L1 一樣，梯度恆定，不爆炸

---

## 6. HW3-4：Rainbow DQN 深度剖析

### 6.1 為什麼 Random Mode 特別難？

在 random mode，每個 episode 的格局都不一樣。模型不能只記住「從 (0,3) 走到 (0,0)」，而是必須學到抽象策略：

> 「不管我在哪裡，我應該朝 Goal 方向走，同時避開 Pit 和 Wall」

這需要 state representation 能同時表達所有物件的相對位置，而且訓練效率要夠高（不然 state space 太大，見過的情況太少）。

### 6.2 Prioritized Experience Replay（PER）深度解析

#### 核心思想

標準 Replay Buffer 對所有 transition 一視同仁。但是：

- 剛發現 Pit 時，那個 transition 的 TD error 很大（非常出乎意料） → 應該多學幾次
- 走到普通格子的 transition，TD error 接近 0（一點都不意外） → 學一次就夠了

**PER 的優先度公式：**
$$p_i = |\delta_i| + \varepsilon$$

其中 δ 是 TD error（預測值和目標值的差距），ε 是避免 priority=0 的小常數。

**抽樣機率：**
$$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$$

α 控制 prioritization 的程度：
- α = 0：均勻抽樣（等同標準 replay）
- α = 1：完全按 priority 抽樣
- 通常用 α = 0.6（折衷）

#### Importance Sampling 修正

PER 改變了資料分佈（高 TD error 的 transition 被抽更多次），會引入 bias。

需要用 importance sampling 修正：
$$w_i = \left(\frac{1}{N \cdot P(i)}\right)^\beta$$

β 從 0.4 開始線性增加到 1.0（訓練結束時完全修正偏差）。

Loss 計算時乘上 wi：
```python
loss = (w_tensor * smooth_l1_loss(X, Y, reduction='none')).mean()
```

#### SumTree 資料結構

如果每次都重新計算所有 transition 的抽樣機率，複雜度是 O(N)，太慢。

SumTree 是一個二元樹：
- 葉子節點：每個 transition 的 priority
- 內部節點：子樹的 priority 總和
- 根節點：所有 priority 的總和

```
         [總和=24]
        /          \
    [左=10]        [右=14]
    /    \          /    \
  [4]   [6]      [8]    [6]
```

抽樣時，先生成一個 0 到總和之間的隨機數，再沿樹往下找對應的 transition，複雜度 O(log N)。

### 6.3 N-step Returns 深度解析

#### 為什麼 1-step Bootstrap 有問題？

標準 Bellman equation 只看一步：
$$Y_1 = r_t + \gamma Q(s_{t+1})$$

問題：Q(s_{t+1}) 在訓練初期很不準確（隨機初始化）。用不準確的估計當目標，相當於「瞎子領路」。

#### N-step 的解法

等真實走了 n 步後再 bootstrap：
$$Y_n = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots + \gamma^{n-1} r_{t+n-1} + \gamma^n Q(s_{t+n})$$

以 n=3 為例：
```
Y_3 = r₀ + 0.99×r₁ + 0.99²×r₂ + 0.99³×Q(s₃)
```

**優點：**
- 前 n 步用真實 reward（準確）
- 只在 s_{t+n} 才 bootstrap（減少不準確估計的影響）
- 更快的 credit assignment（reward 更快傳回初始動作）

**缺點：**
- n 太大：用太多真實步驟，bias 小但 variance 大（每個 episode 的 n-step return 變異大）
- n=3 是經驗上的折衷點

### 6.4 Rainbow 的各組件貢獻

以下是各組件對性能提升的相對貢獻（基於 Hessel et al., 2018 的消融實驗）：

| 移除的組件 | 對最終性能的影響 |
|-----------|----------------|
| 移除 PER | 最大負面影響 |
| 移除 N-step | 第二大負面影響 |
| 移除 Distributional | 第三大 |
| 移除 Dueling | 中等負面影響 |
| 移除 Double DQN | 較小影響（因為 Distributional 已部分解決此問題） |
| 移除 Noisy Nets | 較小影響 |

在我們的 GridWorld 版本（沒有 Distributional 和 Noisy Nets），**PER 和 N-step 是最關鍵的兩個改進**。

---

## 7. 三種 GridWorld 模式的策略差異

### 7.1 Static Mode：記憶問題

Static mode 的最佳策略是確定的：

```
最佳路徑（從 (0,3) 出發）：
(0,3) → 左 → (0,2) → 左 → (0,1) ← 這裡是 Pit！要繞路
(0,3) → 下 → (1,3) → 左 → (1,2) → 左 → (1,0) → 上 → (0,0) 到達 Goal！
```

DQN 只需要「記住」這條路，不需要泛化。Naive DQN 就能達到 95%+ 的成功率。

### 7.2 Player Mode：泛化問題

Goal/Pit/Wall 固定，但 Player 每次從不同位置出發。DQN 需要學到「相對位置」的概念：

- 如果 Player 在 Goal 的右邊 → 往左
- 如果 Player 在 Goal 的下方 → 往上
- 同時要繞過 Pit 和 Wall

Dueling DQN 在這裡特別有效，因為它能快速學到：
- V(s) 隨著 Player 距 Goal 的距離而遞減
- A(s, 朝向 Goal) 永遠是正值

### 7.3 Random Mode：理解問題

所有物件都隨機，DQN 必須學到真正的「抽象理解」：

1. **從 state 中提取相對位置**：Player 在 Goal 的哪個方向？
2. **避免障礙**：Pit 在路徑上嗎？Wall 擋路嗎？
3. **最短路徑規劃**：如何最快到達 Goal？

State 的 64 維表示（4 個物件 × 16 個位置）已經包含了所有需要的信息，但 DQN 需要夠多的訓練和夠高的模型容量才能學會。

**為什麼 Rainbow 在 random mode 表現最好：**
- PER 確保每次遇到 Goal/Pit 的重要 transition 都被充分學習
- N-step return 讓 reward 訊號更快傳播回起始狀態
- Dueling 讓「接近 Goal」的狀態 V(s) 高，不管是哪個 Goal

---

## 8. 常見問題與除錯建議

### Q1：訓練了很久但 win rate 還是 0%，怎麼辦？

**可能原因與解法：**

1. **ε 太大，從未利用已學的知識**
   ```python
   # 確認 epsilon 有在下降
   print(f"Epoch {epoch}: epsilon = {epsilon:.3f}")
   ```

2. **Replay buffer 太小，同一批資料一直重複**
   - 增大 `MEM_SIZE`，例如從 1000 改成 5000

3. **Reward 設計問題**
   - 確認 goal 的 reward 夠大（+10），step penalty 夠小（-1）
   - 如果 +1 和 -1 太接近，agent 可能認為走到哪都一樣

4. **Max moves 太小**
   - 如果 `MAX_MOVES = 20`，agent 可能還沒到 goal 就被強制結束
   - 試試改成 `MAX_MOVES = 50` 或 `100`

### Q2：Loss 一直震盪不收斂？

1. **Learning rate 太大** → 試試 `lr = 5e-4` 或 `1e-4`
2. **Target network 同步太頻繁** → 增大 `SYNC_FREQ`，例如 500 或 1000
3. **沒有梯度裁剪** → 加入 `torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)`

### Q3：Random mode 就是學不好？

Random mode 真的很難，以下是改進建議：

1. **訓練更多 epoch**：從 5000 改成 10000-20000
2. **增大網路**：把 hidden size 從 150/100 改成 256/256
3. **加入 PER**：讓 goal/pit 的 transition 被更多學習
4. **調小 γ**：random mode 中，長期規劃更難，試試 γ=0.95

### Q4：Double DQN 和 Dueling DQN 哪個更重要？

根據論文和實驗：
- **Double DQN** 對於**穩定訓練**更重要（防止 Q 值爆炸）
- **Dueling DQN** 對於**收斂後的最終性能**更重要

兩個加在一起通常比各自單獨用好。在我們的程式碼裡，Dueling DQN 同時使用了 Double DQN 的更新方式。

### Q5：為什麼要用 `render_np()` 而不是直接用座標？

`render_np()` 返回的是一個 4×4×4 的二進位張量：
- 4 個 channel，每個 channel 代表一個物件
- 每個 channel 是 4×4 的 grid，物件所在位置為 1，其餘為 0

相比直接用座標 (row, col)，這種表示有幾個優點：
1. **包含更多資訊**：直接表示了哪個格子有什麼，不需要額外 embedding
2. **適合神經網路**：神經網路比較擅長處理 one-hot 型的輸入
3. **在 random mode 自動包含所有物件位置**：不需要特別處理

---

## 總結

| 作業 | 核心技術 | 解決的問題 | 關鍵程式碼改動 |
|------|----------|-----------|--------------|
| HW3-1 | Experience Replay + Target Network | 訓練不穩定、樣本相關性 | `deque` + `copy.deepcopy` |
| HW3-2 | Double DQN + Dueling DQN | 過高估計 + 泛化能力 | `argmax` 分離 + 雙 head 架構 |
| HW3-3 | PyTorch Lightning | 工程化 + 訓練技巧 | `LightningModule` + `Trainer` |
| HW3-4 | Rainbow DQN | 樣本效率 + 長期規劃 | `SumTree` + `NStepBuffer` |

每個改進都針對前一版本的**具體弱點**，理解這個脈絡，是理解 DQN 演化史的關鍵。

---

*本教學筆記由 AI 輔助撰寫，用於 Deep Reinforcement Learning 課程作業說明。*
