"""
DQNEvictionPolicy — the neural network directly decides which page to evict.

===========================================================================
HOW THIS IS DIFFERENT FROM THE OLD APPROACH
===========================================================================

OLD APPROACH (policy selector):
  DQN looks at the situation → picks a strategy (LRU / LFU / ARC / ...)
  That strategy then decides which page to evict using its fixed rule.
  The neural network never directly touches a page.

NEW APPROACH (per-page scorer — what you wanted):
  DQN looks at EVERY page currently in the cache.
  For each page it computes a "keep score" in [0, 1]:
    → score near 1.0 = "this page will probably be needed soon, KEEP IT"
    → score near 0.0 = "this page probably won't be needed, EVICT IT"
  The page with the LOWEST score gets evicted.
  The neural network is the eviction decision itself.

===========================================================================
PAGE FEATURES (8 numbers per page)
===========================================================================

For each cached page, we compute:
  f1: recency       — 1.0 = just accessed,  0.0 = accessed very long ago
  f2: frequency     — 1.0 = most accessed page, 0.0 = accessed once
  f3: age_in_cache  — 1.0 = just inserted,  0.0 = been here a long time
  f4: ird           — normalized mean inter-reference distance
                      low IRD = page is accessed regularly (KEEP)
  f5: phase_id      — which of the 5 workload phases we're in (0-4, normalized)
  f6: cache_pressure— how full the cache is (0.0 to 1.0)
  f7: local_hit     — was this page a hit in the last 100 accesses?
  f8: reuse_predicted— does this page tend to recur in the current phase?

===========================================================================
TRAINING (how the network learns)
===========================================================================

After evicting a page, we check: was it accessed in the next K=100 steps?
  - YES (we evicted something we still needed) → reward = -1  (bad choice)
  - NO  (we correctly removed a cold page)     → reward = +1  (good choice)

We accumulate (state, action, reward) tuples and train the neural network
using DQN with experience replay, the same way as the classic Atari paper.
The "state" here is the 8D feature vector of the evicted page at the time
of eviction, and the "action" is always 0 (binary: evict this page or not).

For batch training we use a simplified policy-gradient update:
  loss = -reward * log(score)  if reward > 0  (reinforce good choices)
  loss = -reward * log(1-score) if reward < 0 (penalize bad choices)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
from typing import Optional

from .base import Policy


# ─────────────────────────────────────────────────────────────────────────────
# Neural Network: scores one page at a time
# ─────────────────────────────────────────────────────────────────────────────

class PageScoringNetwork(nn.Module):
    """
    Takes 8 features describing a cached page → outputs a keep-score in [0,1].
    Higher score = more likely to be needed soon = keep in cache.
    """
    def __init__(self, feature_dim: int = 8, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),        # output in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Per-page statistics tracker
# ─────────────────────────────────────────────────────────────────────────────

class PageStats:
    """Tracks access statistics for a single page."""
    __slots__ = ("access_count", "last_access", "insert_time",
                 "ird_values", "phase_accesses")

    def __init__(self, insert_time: int):
        self.access_count = 1
        self.last_access = insert_time
        self.insert_time = insert_time
        self.ird_values: list = []
        self.phase_accesses: defaultdict = defaultdict(int)

    def record_access(self, step: int, phase_id: int):
        if self.last_access >= 0:
            ird = step - self.last_access
            self.ird_values.append(ird)
        self.last_access = step
        self.access_count += 1
        self.phase_accesses[phase_id] += 1

    def mean_ird(self) -> float:
        return float(np.mean(self.ird_values)) if self.ird_values else 9999.0

    def phase_affinity(self, current_phase: int) -> float:
        total = sum(self.phase_accesses.values())
        if total == 0:
            return 0.0
        return self.phase_accesses[current_phase] / total


# ─────────────────────────────────────────────────────────────────────────────
# DQN-Direct Eviction Policy
# ─────────────────────────────────────────────────────────────────────────────

class DQNEvictionPolicy(Policy):
    """
    Page replacement policy driven entirely by a neural network score.

    On every cache miss:
      1. Build an 8D feature vector for each page currently in cache
      2. Score all pages through the neural network
      3. Evict the page with the LOWEST score (network thinks it's least needed)
      4. Log the eviction event for training

    Training happens asynchronously:
      After K=100 steps, look back at past evictions and see if those pages
      were accessed again. Assign reward and train.
    """

    FEATURE_DIM = 8
    LOOKAHEAD = 100       # steps after eviction to check if page was reaccessed
    TRAIN_INTERVAL = 50   # train every N evictions
    BUFFER_SIZE = 5_000   # max training samples to remember

    def __init__(self, capacity: int = 256, lr: float = 5e-4,
                 device: str = "auto"):
        self._capacity = capacity
        self._step = 0
        self._phase_id = 0

        # Device
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        # Network
        self._net = PageScoringNetwork(self.FEATURE_DIM, hidden=64).to(self._device)
        self._optimizer = optim.Adam(self._net.parameters(), lr=lr)

        # Per-page stats
        self._page_stats: dict[int, PageStats] = {}

        # Eviction log: list of (evicted_page_id, eviction_step, feature_vector)
        self._eviction_log: deque = deque(maxlen=self.BUFFER_SIZE)

        # Future access index: page_id → set of future steps when it will be accessed
        # We maintain a rolling window of "upcoming" accesses for reward calculation
        self._future_accesses: deque = deque()   # (step, page_id) pairs
        self._recent_evictions: deque = deque()  # (eviction_step, page_id, features)

        # Training replay buffer: (features, reward)
        self._train_buffer: deque = deque(maxlen=self.BUFFER_SIZE)
        self._eviction_count = 0

        # Explanability log (shown in real-time demo)
        self.last_eviction_details: dict = {}

    def set_phase(self, phase_id: int):
        self._phase_id = phase_id

    def on_access(self, page_id: int, is_hit: bool, cache_full: bool,
                  current_pages=None) -> Optional[int]:
        self._step += 1

        # Record this access for future reward computation
        self._future_accesses.append((self._step, page_id))
        # Keep only the last LOOKAHEAD * 10 accesses
        while len(self._future_accesses) > self.LOOKAHEAD * 10:
            self._future_accesses.popleft()

        # Update or create page stats
        if page_id not in self._page_stats:
            self._page_stats[page_id] = PageStats(insert_time=self._step)
        else:
            self._page_stats[page_id].record_access(self._step, self._phase_id)

        # Check past evictions for reward signals
        self._process_eviction_rewards()

        if is_hit or not cache_full:
            return None

        # ── EVICTION DECISION ──────────────────────────────────────────────
        # Score every page currently in cache, evict the lowest-scored one
        pages_to_score = list(current_pages) if current_pages else []
        if not pages_to_score:
            return None

        scores, features_map = self._score_all_pages(pages_to_score)

        # Find the page with the LOWEST score (least likely to be needed)
        evict = min(scores, key=scores.get)
        evict_score = scores[evict]
        evict_features = features_map[evict]

        # Log this eviction for later reward assignment
        self._recent_evictions.append({
            "page_id":      evict,
            "evict_step":   self._step,
            "features":     evict_features,
            "score":        evict_score,
            "runner_up":    sorted(scores, key=scores.get)[1] if len(scores) > 1 else None,
            "all_scores":   scores,
        })

        # Keep eviction log bounded
        while len(self._recent_evictions) > self.BUFFER_SIZE:
            self._recent_evictions.popleft()

        # Save explanation for the real-time dashboard
        self.last_eviction_details = {
            "evicted_page":  evict,
            "evict_score":   evict_score,
            "step":          self._step,
            "scores_sample": dict(sorted(scores.items(), key=lambda x: x[1])[:5]),
            "top_kept":      sorted(scores, key=scores.get, reverse=True)[:3],
        }

        self._eviction_count += 1

        # Train periodically
        if self._eviction_count % self.TRAIN_INTERVAL == 0:
            self._train()

        # Remove from page_stats (evicted page is leaving cache)
        self._page_stats.pop(evict, None)
        return evict

    def _score_all_pages(self, pages: list) -> tuple[dict, dict]:
        """
        Compute keep-scores for all pages.
        Returns (scores_dict, features_dict).
        """
        max_freq = max(
            (self._page_stats[p].access_count for p in pages if p in self._page_stats),
            default=1
        )
        feature_list = []
        for p in pages:
            f = self._compute_features(p, max_freq)
            feature_list.append(f)

        X = torch.tensor(np.array(feature_list), dtype=torch.float32).to(self._device)
        with torch.no_grad():
            raw_scores = self._net(X).cpu().numpy()

        scores = {p: float(raw_scores[i]) for i, p in enumerate(pages)}
        features_map = {p: feature_list[i] for i, p in enumerate(pages)}
        return scores, features_map

    def _compute_features(self, page_id: int, max_freq: int) -> np.ndarray:
        """Build the 8D feature vector for a page."""
        stats = self._page_stats.get(page_id)
        step = self._step
        cap = self._capacity

        if stats is None:
            return np.zeros(self.FEATURE_DIM, dtype=np.float32)

        # f1: recency (0=very old, 1=just accessed)
        recency = 1.0 / (1.0 + (step - stats.last_access) / max(cap, 1))

        # f2: frequency (normalized, 0-1)
        frequency = np.log1p(stats.access_count) / np.log1p(max(max_freq, 1))

        # f3: age in cache (1=just inserted, 0=been here forever)
        age = 1.0 / (1.0 + (step - stats.insert_time) / max(cap, 1))

        # f4: IRD (low=regular access pattern = KEEP; normalize by cap)
        mean_ird = stats.mean_ird()
        ird_score = 1.0 / (1.0 + mean_ird / max(cap, 1))

        # f5: current phase (normalized 0-1)
        phase_norm = self._phase_id / 4.0

        # f6: cache pressure (how full is the cache)
        cache_pressure = len(self._page_stats) / max(cap, 1)
        cache_pressure = min(cache_pressure, 1.0)

        # f7: phase affinity (does this page tend to be accessed in current phase?)
        phase_aff = stats.phase_affinity(self._phase_id)

        # f8: access density (freq / age — pages accessed frequently AND recently)
        density = frequency * recency

        return np.array([
            recency, frequency, age, ird_score,
            phase_norm, cache_pressure, phase_aff, density,
        ], dtype=np.float32)

    def _process_eviction_rewards(self):
        """
        For past evictions that are now LOOKAHEAD steps old, assign rewards.
        Reward:
          +1.0 if evicted page was NOT accessed in the next LOOKAHEAD steps (good!)
          -1.0 if evicted page WAS accessed (bad! we evicted something needed)
        """
        current_step = self._step
        recent_page_set = {pid for (s, pid) in self._future_accesses
                           if s <= current_step and s > current_step - self.LOOKAHEAD}

        ready = []
        remaining = deque()
        for ev in self._recent_evictions:
            if current_step - ev["evict_step"] >= self.LOOKAHEAD:
                ready.append(ev)
            else:
                remaining.append(ev)
        self._recent_evictions = remaining

        for ev in ready:
            was_reaccessed = ev["page_id"] in {
                pid for (s, pid) in self._future_accesses
                if ev["evict_step"] < s <= ev["evict_step"] + self.LOOKAHEAD
            }
            reward = -1.0 if was_reaccessed else +1.0
            self._train_buffer.append((ev["features"], reward))

    def _train(self):
        """Train the network on accumulated eviction experiences."""
        if len(self._train_buffer) < 32:
            return

        batch = list(self._train_buffer)[-128:]   # most recent 128
        feats = np.array([b[0] for b in batch], dtype=np.float32)
        rewards = np.array([b[1] for b in batch], dtype=np.float32)

        X = torch.tensor(feats, dtype=torch.float32).to(self._device)
        R = torch.tensor(rewards, dtype=torch.float32).to(self._device)

        scores = self._net(X)    # shape: (batch,)

        # Policy-gradient style loss:
        #   good eviction (reward=+1): we want score to be LOW (we correctly evicted)
        #   bad eviction  (reward=-1): we want score to be HIGH (shouldn't have evicted)
        # Loss = reward * score  →  minimizing this pushes:
        #   positive reward → lower score (evict more aggressively)
        #   negative reward → higher score (keep these pages)
        loss = (R * scores).mean()

        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=5.0)
        self._optimizer.step()

    def save(self, path: str):
        torch.save(self._net.state_dict(), path)

    def load(self, path: str):
        self._net.load_state_dict(torch.load(path, map_location=self._device))
        self._net.eval()

    def reset(self):
        self._page_stats.clear()
        self._step = 0
        self._phase_id = 0
        self._eviction_count = 0
        self._recent_evictions.clear()
        self._future_accesses.clear()
        self._train_buffer.clear()

    def clone_for_shadow(self) -> "DQNEvictionPolicy":
        return DQNEvictionPolicy(self._capacity)
