# HW3：DQN 及其變體

> **課程**：深度強化學習  
> **環境**：GridWorld（4×4），來自 [DRL in Action](https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction)  
> **Demo 網站**：[GitHub Pages](https://an7172799-ship-it.github.io/HW3-DQN-and-its-variants/)

---

## 📁 檔案結構

```
├── GridBoard.py              # 棋盤底層引擎（來自 DRL in Action repo）
├── Gridworld.py              # GridWorld 環境（三種模式）
├── hw3_1_naive_dqn.py        # HW3-1：Naive DQN + Experience Replay（static 模式）
├── hw3_2_double_dueling.py   # HW3-2：Double DQN 與 Dueling DQN（player 模式）
├── hw3_3_lightning.py        # HW3-3：PyTorch Lightning DQN（random 模式）
├── hw3_4_rainbow.py          # HW3-4：Rainbow DQN（random 模式）[加分題]
├── requirements.txt
├── conversation_log.md       # 與 AI 的完整對話紀錄
├── hw3_teaching_notes.md     # 中文深度教學筆記
└── docs/
    └── index.html            # 互動式 Demo 網站
```

---

## 🌍 環境：GridWorld

4×4 的方格世界，包含四個物件：玩家 `P`、目標 `+`、陷阱 `-`、牆壁 `W`。

```
static 模式的初始格局：
+  -  .  P
.  W  .  .
.  .  .  .
.  .  .  .
```

| 物件 | 符號 | 效果 |
|------|------|------|
| 目標 | `+`  | +10 獎勵，結束遊戲（獲勝） |
| 陷阱 | `-`  | −10 獎勵，結束遊戲（失敗） |
| 牆壁 | `W`  | 無法通過，原地不動 |
| 玩家 | `P`  | 由 agent 控制 |

### 三種模式

| 模式 | 玩家位置 | 其他物件 | 狀態維度 | 難度 |
|------|---------|---------|---------|------|
| `static` | 固定 (0,3) | 固定 | 64 | 低 ⭐ |
| `player` | 隨機 | 固定 | 64 | 中 ⭐⭐ |
| `random` | 隨機 | 全部隨機 | 64 | 高 ⭐⭐⭐ |

**狀態表示**：`game.board.render_np().reshape(64)`  
→ 4 個物件各有一個 4×4 的 binary channel，展平成 64 維向量

---

## HW3-1：Naive DQN — Static 模式 [30%]

### DQN 是什麼？

Deep Q-Network（DQN）用神經網路來近似 Q-function Q(s, a)，取代傳統的 Q-Table。並加入兩個關鍵的穩定訓練技巧：

1. **Experience Replay（經驗回放）** — 將 transition `(s, a, r, s', done)` 存入 buffer，每次從中隨機抽取 mini-batch 訓練，打破時序相關性
2. **Target Network（目標網路）** — 凍結的網路副本，用來計算 Bellman target，每 N 步才同步一次，防止「移動的靶」問題

### 網路架構

```
State（64 維）→ Linear(150) → ReLU → Linear(100) → ReLU → Linear(4)
                                                           ↑
                                              每個動作的 Q 值
```

### Bellman 更新公式

$$Q(s,a) \leftarrow r + \gamma \cdot \max_{a'} Q_{\text{target}}(s', a') \cdot (1 - \text{done})$$

### 訓練參數

| 參數 | 值 |
|------|-----|
| Optimizer | Adam，lr=1e-3 |
| γ（折扣因子） | 0.9 |
| ε 衰減 | 1.0 → 0.1（2000 個 epoch） |
| Replay buffer 大小 | 1000 |
| Batch size | 200 |
| Target 同步頻率 | 每 200 步 |
| 獎勵設計 | 目標=+10、陷阱=−10、每步=−1 |

### 執行方式

```bash
python hw3_1_naive_dqn.py
```

---

## HW3-2：Double DQN 與 Dueling DQN — Player 模式 [40%]

### Vanilla DQN 的問題：過高估計（Overestimation）

Vanilla DQN 的目標值：
$$Y = r + \gamma \max_{a'} Q_{\text{target}}(s', a')$$

`max` 運算在網路有雜訊時，會系統性地高估 Q 值，導致訓練不穩定。

---

### Double DQN

**核心思想**：將動作的**選擇**與**評估**拆開由兩個網路負責。

$$Y^{\text{Double}} = r + \gamma \cdot Q_{\text{target}}\!\left(s',\; \arg\max_{a'} Q_{\text{online}}(s', a')\right)$$

- Online 網路 → 負責**選擇**最好的動作
- Target 網路 → 負責**評估**該動作的價值
- 兩個網路的誤差互相抵銷，消除系統性偏差

```python
# Vanilla DQN
Q2_val = target(s2).max(dim=1).values

# Double DQN  ← 只改這兩行
a_star = online(s2).argmax(dim=1, keepdim=True)   # online 選動作
Q2_val = target(s2).gather(1, a_star).squeeze()    # target 評估
```

---

### Dueling DQN

**核心思想**：將 Q 值分解為**狀態價值** V(s) 與**動作優勢** A(s,a)。

$$Q(s, a) = V(s) + \left[ A(s,a) - \frac{1}{|A|} \sum_{a'} A(s, a') \right]$$

- **V(s)**：這個狀態本身有多好？（與動作無關）
- **A(s,a)**：動作 a 比平均動作好多少？
- 減去 mean(A) 確保 V 和 A 可以被唯一確定

```
State（64）→ Linear(150) → ReLU → Linear(100) → ReLU
                                                  ├─ Value Head     → Linear(1)   → V(s)
                                                  └─ Advantage Head → Linear(4)   → A(s,·)
                                                                            ↓
                                              Q = V + A − mean(A)
```

### 三種模型比較

| 模型 | 解決過估問題 | 更快的狀態評估 | 適用模式 |
|------|:-----------:|:------------:|---------|
| Vanilla DQN | ✗ | ✗ | static |
| Double DQN | ✅ | ✗ | player |
| Dueling DQN | ✅ | ✅ | player / random |

### 執行方式

```bash
python hw3_2_double_dueling.py
# 同時訓練三種模型，結果儲存至 hw3_2_results.png
```

---

## HW3-3：PyTorch Lightning — Random 模式 [30%]

### 為什麼要用 Lightning？

PyTorch Lightning 將**研究程式碼**（模型與 loss）與**工程程式碼**（訓練迴圈、logging、checkpoint）明確分離，並透過 `Trainer` 參數輕鬆啟用各種訓練技巧。

### 核心轉換對比

```python
# 原始 PyTorch                         # PyTorch Lightning
for epoch in range(EPOCHS):       →    class DQNLightning(pl.LightningModule):
    for step in ...:                        def training_step(self, batch, idx):
        loss = compute_loss()                   return compute_loss(batch)
        optimizer.zero_grad()
        loss.backward()                     def configure_optimizers(self):
        optimizer.step()                        return {"optimizer": ...,
                                                        "lr_scheduler": ...}
```

### 整合的訓練技巧

| 技巧 | 實作方式 | 目的 |
|------|---------|------|
| **梯度裁剪** | `Trainer(gradient_clip_val=1.0)` | 防止梯度爆炸 |
| **學習率排程** | `StepLR(step_size=1000, gamma=0.5)` | 訓練後期精細調整 |
| **Huber Loss** | `SmoothL1Loss` 取代 MSE | 對離群 TD error 更穩健 |
| **Double DQN** | Online 選，Target 評 | 降低過估 |
| **Dueling 網路** | V(s) + A(s,a) 架構 | 更好的價值估計 |
| **Target 網路** | 每 300 步同步 | 穩定訓練目標 |

### 執行方式

```bash
pip install pytorch-lightning
python hw3_3_lightning.py
```

---

## HW3-4（加分題）：Rainbow DQN — Random 模式

### Rainbow 是什麼？

Rainbow（Hessel et al., 2017）將多個 DQN 改進組合成一個 agent。本作業實作其中 4 個：

| # | 組件 | 功能 |
|---|------|------|
| 1 | **Double DQN** | 消除過估偏差 |
| 2 | **Dueling Network** | 分離 V(s) 與 A(s,a) |
| 3 | **Prioritized Experience Replay（PER）** | 對高 TD error 的 transition 更多訓練 |
| 4 | **N-step Returns** | 多步 bootstrap，加速 credit assignment |

### Prioritized Experience Replay（PER）

標準 replay 均勻抽樣。PER 根據 **TD error** 決定抽樣機率：

$$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}, \quad p_i = |\delta_i| + \varepsilon$$

使用 **SumTree** 資料結構，實現 O(log n) 的優先抽樣。

並以 **Importance Sampling** 修正非均勻抽樣引入的偏差：

$$w_i = \left(\frac{1}{N \cdot P(i)}\right)^\beta, \quad \beta: 0.4 \to 1.0$$

### N-step Returns

$$Y_n = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n \max_a Q(s_{t+n}, a)$$

n=3：走 3 步真實 reward 再 bootstrap，減少對不準確 Q 估計的依賴。

### 執行方式

```bash
python hw3_4_rainbow.py
```

---

## 🔧 安裝與執行

```bash
pip install -r requirements.txt
```

依序執行所有實驗：
```bash
python hw3_1_naive_dqn.py        # static 模式
python hw3_2_double_dueling.py   # player 模式（三種模型比較）
python hw3_3_lightning.py        # random 模式（Lightning 版）
python hw3_4_rainbow.py          # random 模式（Rainbow，加分）
```

---

## 📊 預期結果

| 模型 | 模式 | 預期勝率 |
|------|------|:-------:|
| Naive DQN | static | ~95%+ |
| Vanilla DQN | player | ~60–70% |
| Double DQN | player | ~75–85% |
| Dueling DQN | player | ~80–90% |
| Lightning DQN | random | ~50–70% |
| Rainbow DQN | random | ~65–80% |

> 實際結果因隨機種子與訓練時間而有所不同。

---

## 📚 參考文獻

- Mnih et al. (2015). *Human-level control through deep reinforcement learning*. Nature.
- van Hasselt et al. (2016). *Deep Reinforcement Learning with Double Q-learning*. AAAI.
- Wang et al. (2016). *Dueling Network Architectures for Deep Reinforcement Learning*. ICML.
- Schaul et al. (2016). *Prioritized Experience Replay*. ICLR.
- Hessel et al. (2018). *Rainbow: Combining Improvements in Deep Reinforcement Learning*. AAAI.
- Zai & Brown (2020). *Deep Reinforcement Learning in Action*. Manning.
