# 與 AI 的完整對話紀錄
## HW3: DQN and its Variants

---

## 對話開始

---

**學生：**

（上傳作業說明截圖）

Homework 3: DQN and its variants  
Total: 100%

1. Setup & Reference  
   - Base your work on the DRL in Action (English) GitHub repo  
   - Use the updated starter code provided by the instructor as your baseline

2. HW3-1: Naive DQN for static mode [30%]  
3. HW3-2: Enhanced DQN Variants for player mode [40%]（Double DQN、Dueling DQN）  
4. HW3-3: Enhance DQN for random mode WITH Training Tips [30%]（轉成 Keras 或 PyTorch Lightning）  
5. HW3-4（加分題）Rainbow DQN 解 Random Mode GridWorld

先分析，再教你怎麼做  
static mode / player mode 解不出來要用高級  
random mode 用到可以解就好了

---

**AI：**

讓我先抓取那個 repo 的實際程式碼作為基礎。

（抓取 GitHub repo：`https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction`）

找到了原始程式碼位置在 Chapter 3，關鍵檔案：
- `Environments/GridBoard.py`
- `Environments/Gridworld.py`
- `Chapter 3/Ch3_book.ipynb`

---

**AI 分析（Analysis First）：**

**GridWorld 環境說明**

```
4x4 Grid:
(0,0)Goal  (0,1)Pit   (0,2).     (0,3)Player
(1,0).     (1,1)Wall  (1,2).     (1,3).
(2,0).     (2,1).     (2,2).     (2,3).
(3,0).     (3,1).     (3,2).     (3,3).
```

**為什麼 static mode 容易解、random mode 難？**

| Mode   | State Space             | 難度 | 需要什麼              |
|--------|-------------------------|------|----------------------|
| static | 固定格局，16 個位置     | 低   | 基本 DQN 即可        |
| player | 16 個起點，其餘固定     | 中   | Double/Dueling 幫助泛化 |
| random | 所有物件隨機 → 巨大空間 | 高   | 豐富的 state + Rainbow |

---

**學生：**

要以 `https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/tree/master` 這個為基礎

---

**AI：**

讓我先抓取那個 repo 的實際程式碼作為基礎。

（成功抓取原始碼後確認）：
- DQN 內容在 **Chapter 3**（不是 Chapter 4）
- State = `render_np().reshape(64)`（4 channels × 4×4 grid）
- Reward：+1 goal, −1 pit, 0 step
- 原版架構：64 → 150 → 100 → 4

---

**AI 建立的檔案：**

1. **`GridBoard.py`** — 原版棋盤底層（BoardPiece、GridBoard class）
2. **`Gridworld.py`** — 原版環境（static/player/random 三種模式）
3. **`hw3_1_naive_dqn.py`** — Naive DQN + Experience Replay + Target Network（static mode）
4. **`hw3_2_double_dueling.py`** — Double DQN + Dueling DQN 比較（player mode）
5. **`hw3_3_lightning.py`** — PyTorch Lightning 版本（random mode）
6. **`hw3_4_rainbow.py`** — Rainbow DQN（random mode，加分題）

---

**學生：**

這些東西你可以幫我做成 demo site 嗎？因為這課堂作業是要丟到 GitHub 的，而且需要附上詳細的解釋及說明。要丟到 `https://github.com/an7172799-ship-it/HW3-DQN-and-its-variants.git`

---

**AI：**

讓我幫你建立完整的 GitHub repo，包含 demo site（GitHub Pages）、詳細文件，然後 push 上去。

**建立的內容：**

1. **`README.md`** — 完整技術文件（英文），包含：
   - 三種模式說明表格
   - 每個 HW 的 Bellman 公式
   - 架構圖
   - 執行方式

2. **`docs/index.html`** — GitHub Pages 互動 demo site：
   - 可操作的 GridWorld 遊戲（支援鍵盤方向鍵）
   - 三種模式切換
   - 架構圖（SVG）
   - MathJax 數學公式
   - Chart.js 訓練曲線
   - 每個 DQN 變體的程式碼對比

3. **`requirements.txt`** + **`.gitignore`**

**Push 到 GitHub：**

遇到帳號問題（本機存的是 `sembeiiiii`，需要 `an7172799-ship-it` 帳號）。

解決方式：
- 方法 1：`gh auth login` 重新登入
- 方法 2：用 Personal Access Token

**GitHub Pages 啟用步驟：**
1. Repo → Settings → Pages
2. Branch: `main` → Folder: `/docs`
3. Save

---

**學生：**

已經 pull 好了，但是需要補充兩個檔案：
1. 我跟你的完整對話紀錄
2. 你指導我如何完成這項作業（解釋原理及深度討論，需要用中文）

---

**AI：**

兩個檔案同時建立：
- `conversation_log.md`（本檔案）— 完整對話紀錄
- `hw3_teaching_notes.md` — 中文深度教學文件

---

## 總結：本次作業完成的所有檔案

| 檔案 | 說明 |
|------|------|
| `GridBoard.py` | 原版 repo 棋盤底層 |
| `Gridworld.py` | 原版環境（三種模式） |
| `hw3_1_naive_dqn.py` | HW3-1: Naive DQN（static） |
| `hw3_2_double_dueling.py` | HW3-2: Double + Dueling DQN（player） |
| `hw3_3_lightning.py` | HW3-3: PyTorch Lightning（random） |
| `hw3_4_rainbow.py` | HW3-4: Rainbow DQN（random，加分） |
| `hw3_4_improved.py` | HW3-4 改進版：8-dim 特徵 + Checkpoint + Curriculum |
| `README.md` | 完整中文技術文件（含訓練結果） |
| `docs/index.html` | GitHub Pages 互動 demo site |
| `requirements.txt` | Python 依賴套件 |
| `conversation_log.md` | 本對話紀錄 |
| `hw3_teaching_notes.md` | 中文深度教學筆記 |

---

## 後續對話：HW3-4 改進與 Demo Site 更新

---

**學生：**

我想問 HW4 可以用更好的解法去嘗試嗎？

---

**AI：**

可以，原版 Rainbow 有三個核心問題：

**問題診斷：**
1. **無 Checkpoint 保存** — 訓練途中 peak 67.7%，但最終模型已漂移，greedy eval 只剩 23%
2. **State Representation 差** — 64-dim one-hot 迫使網路自己學空間關係
3. **無課程學習** — 直接面對最難的 random mode，前期探索效率極低

**三大改進方案（`hw3_4_improved.py`）：**

**A. 相對位置特徵（最關鍵）**
```python
# 原版：64-dim one-hot（網路要自己學距離概念）
# 改進：8-dim 相對向量（直接給距離和方向）
feat = [player_row, player_col,
        Δ到goal_row,  Δ到goal_col,
        Δ到pit_row,   Δ到pit_col,
        Δ到wall_row,  Δ到wall_col]
```

**B. Best Checkpoint 保存**
```python
if np.mean(win_hist[-200:]) > best_wr:
    best_wr = np.mean(win_hist[-200:])
    best_state_dict = copy.deepcopy(online.state_dict())
```

**C. Curriculum Learning**
```
player mode (1500 epoch) → 學會追 goal 避 pit
         ↓ 遷移權重
random mode (5000 epoch) → 把已學策略泛化
```

**預期結果對比：**
| 版本 | Greedy Eval |
|---|---|
| 原版 Rainbow | 23% |
| 加 Checkpoint | ~50-60% |
| 加相對特徵 | ~70-80% |
| 完整改進版 | ~90%+ |

---

**學生：**

好像可以解就好，所以現在這份就可以了，那你也要去更新 Demo Site

---

**AI：**

更新 `docs/index.html` 的 HW3-4 區塊，新增：
- 問題分析三張 card（無 checkpoint / state 差 / 無課程學習）
- 改進版對照表（原版 vs 改進版的三項升級）
- 各版本結果比較（Lightning 95.8% vs Rainbow 原版 23% vs 改進版預期 90%+）
- 將 Chart 換成反映實際結果的模擬曲線（Rainbow 峰值後微幅下滑 vs Lightning 穩定收斂）
- 修正 SumTree code snippet（1-indexed 版本）
- 修正超參數表（n=2, γ=0.95）

---

**學生：**

其實 HW3-1 ~ 3-4 都有可以詳細補充的內容吧？

---

**AI：**

對，全部補齊：

**HW3-1 新增：**
- 最佳路徑分析（為什麼要先往下繞，不能直接往左）
- Experience Replay 和 Target Network 的「問題→解法」說明
- ε-greedy 衰減公式
- 超參數設定表
- 完整訓練 log（Epoch 200→2000，最終 100%）

**HW3-2 新增：**
- Player mode vs static mode 難度差異說明
- Overestimation 的直覺解釋（像「永遠選看起來最重的」）
- Double DQN 的「解耦」概念
- 更詳細的 Dueling 架構圖（含每層 shape）
- 三種模型完整訓練 log 與比較表
- 說明為什麼三種最終都 100%，真正差距在 random mode

**HW3-3 新增：**
- Three mode 難度對比表（記憶 / 路徑規劃 / 策略泛化）
- ReplayDataset 設計說明（為什麼要把 deque 包成 Dataset）
- 四大技術的**原因**解釋（不只說是什麼，說為什麼需要）
- GPU 加速說明（`accelerator='gpu'`）
- 完整訓練 log（Epoch 500→5000，最終 95.8%）
- 解釋為什麼 Lightning 95.8% > Rainbow 23%（ModelCheckpoint 是關鍵）

---

**學生：**

要（push 到 GitHub）

---

**AI：**

```
git add docs/index.html hw3_4_improved.py
git commit -m "Enrich demo site for all sections and add improved Rainbow DQN"
git push → commit 4948101
```

---

**學生：**

阿 GitHub Page 要更新，還有我們的對話紀錄也要

---

**AI：**

GitHub Pages 在 push 後會自動從 `docs/` 資料夾部署，通常 1-2 分鐘後生效。

更新 `conversation_log.md`（本檔案）補入所有後續對話，再 commit push。
