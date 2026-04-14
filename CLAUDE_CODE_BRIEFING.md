# PARL PROJECT — COMPLETE CLAUDE CODE BRIEFING
# Phase-Aware Reinforcement Learning Based Page Replacement Policy
# CSD 204 Operating Systems Project — IIT, 2025
# ============================================================
# READ THIS ENTIRE FILE BEFORE WRITING ANY CODE.
# This document contains everything Claude Code needs to build the
# project from top to bottom with zero external context.
# ============================================================


## ══════════════════════════════════════════════════════════
## PART 1: PROJECT IDENTITY
## ══════════════════════════════════════════════════════════

Project Title:   Phase-Aware Reinforcement Learning Based Page Replacement
                 Policy for Adaptive Memory Management
Course:          CSD 204 — Operating Systems
Team:            Pranav Gupta, Ananya Singh, Rohan Mehta, Kiran Patel
Due (Final):     May 16, 2025
Language:        Python 3.11
Deep Learning:   PyTorch 2.0
Total Marks:     100


## ══════════════════════════════════════════════════════════
## PART 2: THE CORE PROBLEM (WHY THIS PROJECT EXISTS)
## ══════════════════════════════════════════════════════════

Modern applications (databases, ML training jobs, microservices) pass
through distinct memory access PHASES during execution:

  Phase 1 — Initialization:  cold start, many compulsory misses
  Phase 2 — Steady state:    high temporal locality (LRU works well)
  Phase 3 — Sequential scan: no locality, LRU thrashes badly
  Phase 4 — Random access:   frequency matters, LFU works better
  Phase 5 — Burst/hotspot:   small working set, CLOCK is fine

The problem: classical algorithms (LRU, CLOCK, LFU) use STATIC heuristics.
They cannot detect phase transitions and cannot switch strategy mid-run.
Result: when a workload shifts from phase 2 to phase 3, LRU's miss rate
spikes 30-60% above what an optimal dynamic policy would achieve.

Our solution: build a system that (1) detects the current phase in real time,
and (2) uses a trained RL agent to pick the best replacement policy for
the detected phase — updating continuously as the workload evolves.


## ══════════════════════════════════════════════════════════
## PART 3: SYSTEM ARCHITECTURE (3 MODULES)
## ══════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────┐
│                     PARL SYSTEM                             │
│                                                             │
│  ┌─────────────────┐    ┌──────────────────┐               │
│  │  PAGE REFERENCE │───>│  PHASE DETECTION │               │
│  │  TRACE / SIM    │    │  ENGINE (PDE)    │               │
│  └─────────────────┘    └────────┬─────────┘               │
│                                  │ phase_vector             │
│                         ┌────────▼──────────┐              │
│                         │  DQN RL AGENT     │              │
│                         │  (PyTorch)        │              │
│                         └────────┬──────────┘              │
│                                  │ policy_id / weights      │
│                         ┌────────▼──────────┐              │
│                         │  POLICY EXECUTION │              │
│                         │  LAYER (PEL)      │              │
│                         └───────────────────┘              │
│                                  │ evict_page()             │
│                         ┌────────▼──────────┐              │
│                         │  MEMORY SIMULATOR │              │
│                         │  (cache state)    │              │
│                         └───────────────────┘              │
└─────────────────────────────────────────────────────────────┘

MODULE 1 — Memory Simulator (simulator/):
  - Discrete-event page reference trace player
  - Configurable cache size (e.g., 64, 128, 256, 512 pages)
  - Tracks: hits, misses, miss rate, access sequence
  - Pluggable policy interface: sim.set_policy(policy_instance)

MODULE 2 — Phase Detection Engine (phase_detection/):
  - Runs over a sliding window of W=1000 page accesses
  - Computes feature vector F = [miss_rate, ird_mean, ird_std, entropy, 
                                   reuse_distance_p50, unique_pages_ratio]
  - Applies mini-batch k-means (k=5 phases) to cluster F into phase_id
  - Outputs: phase_id (0-4) + phase_embedding (continuous vector)

MODULE 3 — DQN RL Agent (rl_agent/):
  - State:  [phase_embedding (8-dim), miss_rate, cache_pressure, time_since_switch]
  - Action: discrete — choose from {LRU, CLOCK, LFU, ARC, HYBRID_LRU_LFU}
  - Reward: delta_hit_rate vs. baseline LRU over last 100 accesses
  - Network: FC(input_dim, 256) -> ReLU -> FC(256, 256) -> ReLU -> FC(256, n_actions)
  - Training: DQN with experience replay (buffer=50000), epsilon-greedy exploration
  - Target network update: every 500 steps


## ══════════════════════════════════════════════════════════
## PART 4: FILE/FOLDER STRUCTURE TO BUILD
## ══════════════════════════════════════════════════════════

parl/
├── README.md
├── requirements.txt
├── run_experiment.py          ← main entry point
├── train_agent.py             ← RL training loop
├── evaluate.py                ← benchmarking script
│
├── simulator/
│   ├── __init__.py
│   ├── cache.py               ← Cache class (fixed-size page table)
│   ├── trace_player.py        ← loads + replays .trace files
│   └── metrics.py             ← HitRateTracker, LatencyEstimator
│
├── policies/
│   ├── __init__.py
│   ├── base.py                ← abstract Policy class
│   ├── lru.py                 ← LRU (doubly-linked list + hash map, O(1))
│   ├── clock.py               ← CLOCK (circular buffer, reference bit)
│   ├── lfu.py                 ← LFU (min-heap + freq map, O(log n))
│   ├── arc.py                 ← ARC (T1/T2/B1/B2 lists)
│   ├── lirs.py                ← LIRS (LIR/HIR stack + queue)
│   ├── opt.py                 ← OPT (offline, needs full trace pre-scan)
│   └── hybrid.py              ← HybridPolicy (weighted combination)
│
├── phase_detection/
│   ├── __init__.py
│   ├── feature_extractor.py   ← computes sliding window features
│   ├── phase_classifier.py    ← mini-batch k-means phase clustering
│   └── phase_buffer.py        ← ring buffer for recent accesses
│
├── rl_agent/
│   ├── __init__.py
│   ├── dqn_model.py           ← PyTorch DQN network definition
│   ├── replay_buffer.py       ← experience replay buffer
│   ├── agent.py               ← DQNAgent (select_action, train_step, update_target)
│   └── reward.py              ← reward function definition
│
├── traces/
│   ├── synthetic/
│   │   └── generate_traces.py ← generates phased synthetic traces
│   └── spec/                  ← placeholder for SPEC CPU 2017 traces
│
├── experiments/
│   ├── baseline_comparison.py ← compare all policies head-to-head
│   ├── phase_transition_test.py
│   └── ablation_study.py
│
└── results/
    └── plots.py               ← matplotlib figure generation


## ══════════════════════════════════════════════════════════
## PART 5: DETAILED IMPLEMENTATION SPECS
## ══════════════════════════════════════════════════════════

### 5.1 Cache (simulator/cache.py)

class Cache:
    def __init__(self, capacity: int, policy: Policy)
    def access(self, page_id: int) -> bool   # returns True=hit, False=miss
    def get_stats(self) -> dict              # {'hits': int, 'misses': int, 'miss_rate': float}
    def reset(self)
    # internally calls policy.on_hit(page_id) or policy.on_miss(page_id, evict_page)

### 5.2 Policy Interface (policies/base.py)

class Policy(ABC):
    @abstractmethod
    def on_access(self, page_id: int, is_hit: bool) -> Optional[int]
    # Returns: page_id to evict (on miss), or None (on hit)
    
    @abstractmethod  
    def reset(self)

### 5.3 LRU (policies/lru.py)

Use OrderedDict for O(1) access, move_to_end on hit, popitem(last=False) on evict.
Do NOT use a simple list (O(n) deletion).

### 5.4 CLOCK (policies/clock.py)

Circular list of (page_id, reference_bit) pairs.
Clock hand advances until it finds reference_bit=0 → evict that page.
On access hit: set reference_bit=1.
On miss: advance hand, evict first page with ref_bit=0, insert new page.

### 5.5 LFU (policies/lfu.py)

Maintain:
  - freq_map: {page_id: frequency}
  - freq_to_pages: {freq: OrderedDict of page_ids}  ← for LRU tie-breaking
  - min_freq: int  (tracks current minimum frequency)
On hit: increment freq, move page from freq_to_pages[old_freq] to [new_freq]
On miss/evict: remove page at freq_to_pages[min_freq] (oldest in that freq bucket)

### 5.6 Feature Extractor (phase_detection/feature_extractor.py)

Window W = 1000 accesses (configurable).

Features computed on each window:
  f1 = miss_rate            = misses / total_accesses in window
  f2 = ird_mean             = mean inter-reference distance
  f3 = ird_std              = std dev of IRD
  f4 = entropy              = -sum(p*log2(p)) of page access distribution
  f5 = reuse_distance_p50   = median reuse distance (distance since last access)
  f6 = unique_pages_ratio   = unique pages in window / window size

IRD: for each page, track last_seen[page_id]; IRD = current_idx - last_seen[page_id]
Reuse distance: same as IRD but only computed for re-accessed pages.

### 5.7 Phase Classifier (phase_detection/phase_classifier.py)

Use sklearn.cluster.MiniBatchKMeans(n_clusters=5, random_state=42)
Fit on first 10 windows (warm-up period) using .partial_fit()
Then call .predict() on each subsequent feature vector
Output: phase_id (0-4) as integer
Also output phase_embedding = cluster_center[phase_id] as float numpy array

### 5.8 DQN Network (rl_agent/dqn_model.py)

import torch
import torch.nn as nn

class DQNNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    def forward(self, x): return self.net(x)

state_dim = 8 (phase_embedding) + 1 (miss_rate) + 1 (cache_fullness) + 1 (steps_since_switch) = 11
action_dim = 5 (LRU, CLOCK, LFU, ARC, HYBRID)

### 5.9 Replay Buffer (rl_agent/replay_buffer.py)

class ReplayBuffer:
    def __init__(self, capacity=50000)
    def push(self, state, action, reward, next_state, done)
    def sample(self, batch_size=64) -> tuple of tensors
    def __len__(self)
Use collections.deque(maxlen=capacity) internally.

### 5.10 DQN Agent (rl_agent/agent.py)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995)
    
    def select_action(self, state: np.ndarray) -> int
        # epsilon-greedy: with prob epsilon return random action
        # otherwise return argmax Q(state, ·)
    
    def train_step(self) -> Optional[float]
        # if buffer < batch_size: return None
        # sample batch, compute target Q, compute loss, backprop
        # returns loss value
    
    def update_target_network(self)
        # hard copy: target_net.load_state_dict(policy_net.state_dict())
    
    def decay_epsilon(self)
        # epsilon = max(epsilon_end, epsilon * epsilon_decay)

Loss = MSE(Q(s,a), r + gamma * max_a' Q_target(s', a') * (1 - done))
Optimizer: Adam(lr=1e-3)
Target network update: every 500 steps (not every step — important!)

### 5.11 Reward Function (rl_agent/reward.py)

def compute_reward(current_hit_rate: float, 
                   baseline_lru_hit_rate: float,
                   policy_switch_cost: float = 0.01) -> float:
    """
    Reward = improvement over LRU baseline, with small switch penalty.
    Encourages the agent to beat LRU but discourages unnecessary switching.
    """
    improvement = current_hit_rate - baseline_lru_hit_rate
    return improvement - policy_switch_cost

Hit rate is computed over last 100 accesses (rolling window).
Baseline LRU runs in shadow mode — parallel LRU cache updated on same trace.

### 5.12 Synthetic Trace Generator (traces/synthetic/generate_traces.py)

Generate multi-phase traces for development/testing:

Phase 0 (Init):       zipf distribution (alpha=0.8), 2000 accesses, page_space=500
Phase 1 (Steady):     zipf distribution (alpha=1.2), 5000 accesses, page_space=200
Phase 2 (Sequential): sequential scan pages [0..299] repeated, 3000 accesses
Phase 3 (Random):     uniform random over page_space=1000, 3000 accesses
Phase 4 (Hotspot):    80% accesses to 20 hot pages + 20% random, 4000 accesses

Concatenate phases and save as plain text:
  One page_id per line (integer), e.g.:
  42
  17
  42
  ...

Save to traces/synthetic/mixed_phase.trace


## ══════════════════════════════════════════════════════════
## PART 6: TRAINING PROCEDURE
## ══════════════════════════════════════════════════════════

File: train_agent.py

Algorithm:
1. Load trace (synthetic or SPEC)
2. Initialize Cache(capacity=256, policy=LRU) — this is the PARL cache
3. Initialize shadow Cache(capacity=256, policy=LRU) — for baseline reward
4. Initialize PhaseDetectionEngine(window_size=1000)
5. Initialize DQNAgent(state_dim=11, action_dim=5)
6. Initialize ReplayBuffer(capacity=50000)

For each access in trace:
  a. shadow_cache.access(page_id)  — track baseline hit rate
  b. cache.access(page_id)         — PARL cache access, current policy
  c. phase_engine.update(page_id, is_hit)
  
  Every 100 accesses:
    state = build_state(phase_embedding, miss_rate, cache_fullness, steps_since_switch)
    action = agent.select_action(state)
    cache.set_policy(POLICIES[action])
    reward = compute_reward(cache.hit_rate_100, shadow.hit_rate_100)
    replay_buffer.push(prev_state, prev_action, reward, state, done=False)
    loss = agent.train_step()
    agent.decay_epsilon()
    
    if step % 500 == 0:
      agent.update_target_network()

Training stops after full trace or max_steps=500000.
Save model: torch.save(agent.policy_net.state_dict(), 'models/dqn_parl.pt')
Log metrics every 1000 steps to results/training_log.csv


## ══════════════════════════════════════════════════════════
## PART 7: EVALUATION PROCEDURE
## ══════════════════════════════════════════════════════════

File: evaluate.py

Policies to compare:
  - OPT (offline optimal, requires full trace pre-scan)
  - LRU
  - CLOCK
  - LFU
  - ARC
  - LIRS
  - PARL (our system — trained DQN agent)

For each policy × trace × cache_size:
  Run full trace, collect:
    - Overall miss rate
    - Miss rate per phase (phases labeled by ground truth from generator)
    - Adaptation latency (how quickly miss rate recovers after phase change)
    - Total runtime (to measure overhead)

Cache sizes to test: [64, 128, 256, 512] pages
Traces to test:
  - synthetic/mixed_phase.trace (generated)
  - synthetic/phase_shift_abrupt.trace (sudden transition)
  - SPEC CPU 2017 traces (if available, else skip gracefully)

Output: results/evaluation_results.csv with columns:
  policy, trace, cache_size, miss_rate, phase, adaptation_latency_steps


## ══════════════════════════════════════════════════════════
## PART 8: PLOTS TO GENERATE
## ══════════════════════════════════════════════════════════

File: results/plots.py

Figure 1: Overall miss rate comparison — bar chart
  X-axis: Policy names
  Y-axis: Miss rate (%)
  Groups: cache_size (64, 128, 256, 512)

Figure 2: Miss rate over time with phase annotations — line chart
  X-axis: Access number (time)
  Y-axis: Rolling miss rate (window=500)
  Lines: LRU, ARC, PARL
  Shaded regions: ground-truth phases (different colors)

Figure 3: Per-phase miss rate heatmap — seaborn heatmap
  Rows: Policies
  Columns: Phases (0-4)
  Values: Miss rate

Figure 4: Training curve — line chart
  X-axis: Training step
  Y-axis: Smoothed reward (EMA alpha=0.95)

Figure 5: Adaptation latency box plot
  X-axis: Policy
  Y-axis: Steps until miss rate recovers to within 5% of phase-optimal

Use matplotlib + seaborn. Save all plots as results/fig1.png ... fig5.png


## ══════════════════════════════════════════════════════════
## PART 9: requirements.txt
## ══════════════════════════════════════════════════════════

numpy>=1.24
torch>=2.0
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
pandas>=2.0
tqdm>=4.65


## ══════════════════════════════════════════════════════════
## PART 10: KEY IMPLEMENTATION GOTCHAS
## ══════════════════════════════════════════════════════════

1. LRU with OrderedDict:
   - on HIT:  d.move_to_end(key)
   - on MISS: if len(d) >= capacity: d.popitem(last=False)  ← removes OLDEST
   - Never use a plain dict (no ordering in older Python) or list (O(n))

2. CLOCK reference bit reset:
   - Only set ref_bit=1 on ACCESS (hit or miss-then-insert)
   - Do NOT reset ref_bit on every clock tick; only when hand sweeps past

3. LFU min_freq maintenance:
   - On new page insert: min_freq = 1 always
   - On existing page freq increment: min_freq ONLY needs update if
     freq_to_pages[old_min_freq] becomes empty

4. ARC list sizes:
   - T1 + T2 <= cache_capacity (live pages)
   - B1 + B2 can exceed cache_capacity (ghost entries, no data stored)
   - Parameter p (target T1 size) adapts: 
     increase p if miss in B1, decrease p if miss in B2

5. Phase detection warm-up:
   - MiniBatchKMeans needs at least n_clusters samples before predicting
   - Return phase_id=0 during warm-up period (first 10 windows)
   - Use partial_fit() not fit() to support online updates

6. DQN training stability:
   - Do NOT train until replay buffer has >= 1000 transitions
   - Clip gradients: torch.nn.utils.clip_grad_norm_(params, max_norm=10)
   - Use separate policy_net and target_net (do NOT update target every step)

7. Reward normalization:
   - hit_rate is in [0, 1]; reward is thus in roughly [-1, 1]
   - No additional normalization needed for this scale
   - If reward explodes during training, add reward clipping: reward = clip(reward, -1, 1)

8. Trace file format (CRITICAL — be consistent):
   - One integer page_id per line
   - Page IDs start at 0
   - No header line
   - Example: echo "0\n1\n0\n2\n1" > test.trace

9. OPT implementation:
   - Pre-scan entire trace to build next_use[i] = index of next use of trace[i]
   - On eviction: remove page whose next_use is furthest (infinity if never used again)
   - This is O(n) pre-scan + O(capacity) per eviction

10. Avoiding circular imports:
    - simulator/cache.py imports from policies/base.py only (not specific policies)
    - run_experiment.py imports everything and wires them together
    - rl_agent/ imports from policies/ for the POLICIES dict only


## ══════════════════════════════════════════════════════════
## PART 11: run_experiment.py (main entry point)
## ══════════════════════════════════════════════════════════

Usage examples:
  python run_experiment.py --mode train --trace traces/synthetic/mixed_phase.trace
  python run_experiment.py --mode eval  --trace traces/synthetic/mixed_phase.trace
  python run_experiment.py --mode compare --trace traces/synthetic/mixed_phase.trace
  python run_experiment.py --mode generate_traces

Arguments:
  --mode:       train | eval | compare | generate_traces
  --trace:      path to .trace file
  --cache_size: integer (default 256)
  --model_path: path to saved .pt file (for eval mode)
  --output_dir: where to save results (default ./results)


## ══════════════════════════════════════════════════════════
## PART 12: BUILD ORDER (do this in sequence)
## ══════════════════════════════════════════════════════════

Step 1: traces/synthetic/generate_traces.py
        → Run it: python traces/synthetic/generate_traces.py
        → Verify: traces/synthetic/mixed_phase.trace exists, ~17000 lines

Step 2: policies/base.py → policies/lru.py → policies/clock.py → policies/lfu.py
        → Unit test each policy on tiny cache (capacity=3) with known trace
        → Known trace test: pages [1,2,3,4,1,2,5,1,2,3,4,5]
        → LRU should give 8 misses on that trace with cache=4

Step 3: simulator/cache.py → simulator/trace_player.py → simulator/metrics.py
        → Integration test: run LRU on mixed_phase.trace, print miss rate

Step 4: policies/arc.py, policies/lirs.py, policies/opt.py
        → These are harder; implement after Step 3 is solid

Step 5: phase_detection/feature_extractor.py → phase_detection/phase_classifier.py
        → Unit test: feed 5000 synthetic accesses, verify 5 distinct phase_ids emerge

Step 6: rl_agent/dqn_model.py → rl_agent/replay_buffer.py → rl_agent/agent.py
        → Unit test: forward pass with random state tensor, verify output shape=(batch, 5)

Step 7: train_agent.py
        → Train for 50k steps, verify reward trend is non-decreasing (approximately)
        → Save model checkpoint

Step 8: evaluate.py → results/plots.py
        → Run comparison, generate all 5 figures

Step 9: run_experiment.py
        → Wire everything behind the CLI interface


## ══════════════════════════════════════════════════════════
## PART 13: EXPECTED RESULTS (for report writing)
## ══════════════════════════════════════════════════════════

Based on related work, we expect:
  - OPT:    lowest miss rate (theoretical bound, ~15-25% on mixed traces)
  - PARL:   within 10-20% of OPT; 15-30% better than LRU on phase-heavy traces
  - ARC:    ~10-15% better than LRU overall
  - LIRS:   ~8-12% better than LRU on sequential-heavy traces
  - LRU:    baseline reference
  - CLOCK:  ~same as LRU (slightly worse in theory, similar in practice)
  - LFU:    worse than LRU on short traces (cold start), better on long stable ones

PARL's key advantage should show in Figure 2: lower miss rate in phase
transition zones (between phases) compared to all static policies.


## ══════════════════════════════════════════════════════════
## PART 14: GRADING RUBRIC (what the professor cares about)
## ══════════════════════════════════════════════════════════

Based on CSD 204 project requirements (100 marks total):
  - Preliminary Report (this doc):  20 marks — literature survey completeness
  - Mid-project Report:             20 marks — working baseline + initial results
  - Final Report:                   30 marks — complete system + analysis
  - Oral Presentation:              20 marks — understanding of baselines + RL
  - Code Quality:                   10 marks — clean, documented, runnable

Professor explicitly loves:
  - "memory management using ML" framing
  - Clear comparison against classical baselines
  - Evidence that you understand WHY each algorithm behaves as it does
  - Phase detection as a first-class contribution (novel angle)
  - Actual experimental data (even if on synthetic traces)


## ══════════════════════════════════════════════════════════
## PART 15: REFERENCE PAPERS (for code + report)
## ══════════════════════════════════════════════════════════

[1] Belady (1966) — OPT algorithm
[2] Denning (1968) — Working Set model
[3] Carr & Hennessy (1981) — WSClock
[4] Corbato (1968) — CLOCK (reference bit mechanism)
[5] Denning (1970) — Virtual memory / LRU analysis
[6] Megiddo & Modha (2003, USENIX FAST) — ARC
[7] Jiang & Zhang (2002, SIGMETRICS) — LIRS
[8] Johnson & Shasha (1994, VLDB) — 2Q
[9] Einziger et al. (2017, ACM ToS) — TinyLFU
[10] Shi et al. (2018, HotNets) — DeepCache
[11] Vietri et al. (2018, HotStorage) — LeCaR (ML-based)
[12] Lykouris & Vassilvitskii (2018, ICML) — ML-augmented caching theory
[13] Song et al. (2022, IEEE Trans. Computers) — RL for cache replacement
[14] Liu et al. (2021, SYSTOR) — RL for SSD caching
[15] Berger et al. (2018, SIGMETRICS) — Practical caching bounds


## ══════════════════════════════════════════════════════════
## END OF BRIEFING
## ══════════════════════════════════════════════════════════
# You now have everything needed. Start with Part 12 (Build Order).
# Ask no clarifying questions — all design decisions are specified above.
# If something is ambiguous, default to the simpler implementation.
