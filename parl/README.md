# PARL — Phase-Aware Reinforcement Learning Based Page Replacement Policy

**CSD 204 Operating Systems | SNU**  
Team: Pranav Gupta(2410110241), Shambhavi Sharma(2410110313), Mohak Joshi(2410110204)

---

## What is this?

Modern applications (databases, ML jobs, microservices) go through distinct memory access **phases** during execution:

| Phase | Pattern | Best Policy |
|-------|---------|-------------|
| 0 — Init | Cold start, many misses | Any |
| 1 — Steady | High temporal locality | LRU |
| 2 — Sequential scan | No locality, LRU thrashes | CLOCK / OPT |
| 3 — Random access | Frequency matters | LFU |
| 4 — Hotspot | Small hot working set | ARC |

Classical algorithms (LRU, CLOCK, LFU) use **static rules** and cannot adapt to phase changes.

**PARL** solves this by:
1. **Detecting the current phase** using a sliding-window feature extractor + online k-means clustering
2. **Using a DQN RL agent** to pick the optimal replacement policy for the current phase
3. **Updating continuously** as the workload evolves

---

## Architecture

```
Page Reference Trace
       ↓
Phase Detection Engine (PDE)
  - 6 features: miss_rate, IRD mean/std, entropy, reuse_dist_p50, unique_pages_ratio
  - Mini-batch k-means (k=5 phases)
  - Outputs: phase_id + phase_embedding (8D)
       ↓
DQN RL Agent (PyTorch)
  - State: [phase_embedding(8), miss_rate, cache_fullness, steps_since_switch]
  - Actions: {LRU, CLOCK, LFU, ARC, HYBRID}
  - Reward: PARL hit rate − LRU baseline hit rate
       ↓
Policy Execution Layer
  - Executes chosen policy on the cache
  - Hot-swaps policy when agent decides to switch
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic traces
python run_experiment.py --mode generate_traces

# 3. Train the RL agent (~5-10 min on CPU)
python run_experiment.py --mode train

# 4. Evaluate all policies
python run_experiment.py --mode eval

# 5. Generate all figures
python run_experiment.py --mode plots

# Or run everything at once:
python run_experiment.py --mode all
```

---

## File Structure

```
parl/
├── run_experiment.py          ← Main CLI entry point
├── train_agent.py             ← DQN training loop
├── evaluate.py                ← Benchmarking all policies
├── requirements.txt
│
├── simulator/
│   ├── cache.py               ← Fixed-size page cache
│   ├── trace_player.py        ← Loads .trace files
│   └── metrics.py             ← Hit rate tracking
│
├── policies/
│   ├── lru.py                 ← LRU (OrderedDict, O(1))
│   ├── clock.py               ← CLOCK (reference bit)
│   ├── lfu.py                 ← LFU (frequency buckets)
│   ├── arc.py                 ← ARC (adaptive T1/T2 balance)
│   ├── lirs.py                ← LIRS (inter-reference recency)
│   ├── opt.py                 ← OPT (offline optimal, Belady)
│   └── hybrid.py              ← Weighted LRU + LFU blend
│
├── phase_detection/
│   ├── feature_extractor.py   ← Sliding window → 6 features
│   ├── phase_classifier.py    ← Online k-means → phase_id
│   └── phase_buffer.py        ← Ring buffer for recent accesses
│
├── rl_agent/
│   ├── dqn_model.py           ← PyTorch DQN (3-layer FC, 256 hidden)
│   ├── replay_buffer.py       ← Experience replay (50k transitions)
│   ├── agent.py               ← DQNAgent: select, train, update_target
│   └── reward.py              ← Reward = improvement over LRU baseline
│
├── traces/
│   └── synthetic/
│       └── generate_traces.py ← Multi-phase synthetic trace generator
│
├── experiments/
│   ├── baseline_comparison.py
│   ├── phase_transition_test.py
│   └── ablation_study.py
│
└── results/
    └── plots.py               ← Generates fig1.png ... fig5.png
```

---

## Expected Results

| Policy | Miss Rate (cache=256) | Notes |
|--------|----------------------|-------|
| OPT    | ~20-30%              | Theoretical lower bound |
| PARL   | ~25-40%              | 15-30% better than LRU on phase-heavy traces |
| ARC    | ~35-45%              | ~10-15% better than LRU |
| LIRS   | ~38-48%              | Better on sequential phases |
| LRU    | ~45-55%              | Baseline |
| LFU    | ~48-60%              | Suffers on cold starts |
| CLOCK  | ~47-57%              | Close to LRU in practice |

PARL's key advantage: **lower miss rate at phase transition boundaries** (Figure 2).

---

## References

1. Belady (1966) — OPT algorithm
2. Megiddo & Modha (2003, USENIX FAST) — ARC  
3. Jiang & Zhang (2002, SIGMETRICS) — LIRS
4. Song et al. (2022, IEEE Trans. Computers) — RL for cache replacement
5. Lykouris & Vassilvitskii (2018, ICML) — ML-augmented caching theory

(See preliminary_report.pdf for full 15-reference bibliography)
