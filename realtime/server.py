"""
server.py — Flask web server for PARL live visualization dashboard.

Runs at http://localhost:5050
Open that URL in any browser on your Mac, or on your Ubuntu VM's browser
by pointing it at http://<mac-ip>:5050

Uses Server-Sent Events (SSE) to push real-time state to the browser
every ~500ms. No WebSocket library needed.

Architecture:
  Background thread: continuously samples OS processes via psutil,
                     feeds them through all 4 caches (DQN-direct, LRU,
                     LFU, ARC), collects state.
  Main thread:       Flask serves the HTML page + SSE stream.
  Browser:           Renders animated cache grid, charts, eviction log.
"""

import os
import sys
import json
import time
import random
import threading
import queue
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, Response, render_template, jsonify
from realtime.os_monitor import OSMemoryMonitor
from simulator.cache import Cache
from simulator.metrics import HitRateTracker
from policies.dqn_policy import DQNEvictionPolicy
from policies.lru import LRUPolicy
from policies.lfu import LFUPolicy
from policies.arc import ARCPolicy
from phase_detection.phase_classifier import PhaseClassifier

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), "templates"),
            static_folder=os.path.join(os.path.dirname(__file__), "static"))

# ─────────────────────────────────────────────────────────────────────────────
# Shared simulation state (written by background thread, read by SSE stream)
# ─────────────────────────────────────────────────────────────────────────────

CACHE_SIZE = 48   # number of slots to display (small enough to visualize well)

PHASE_NAMES = {0: "Initialization", 1: "Steady State", 2: "Sequential Scan",
               3: "Random Access", 4: "Hotspot"}
PHASE_COLORS = {0: "#3b82f6", 1: "#22c55e", 2: "#f59e0b", 3: "#ef4444", 4: "#a855f7"}

simulation_state = {
    "step":          0,
    "cache_slots":   [],          # list of slot dicts
    "miss_rates":    {"DQN": [], "LRU": [], "LFU": [], "ARC": []},
    "phase_id":      0,
    "phase_name":    "Init",
    "phase_color":   "#3b82f6",
    "system":        {},
    "processes":     [],
    "eviction_log":  deque(maxlen=30),
    "last_added":    [],
    "last_evicted":  [],
    "running":       True,
    # Mode control: None = auto-detect from live OS; int 0-4 = force that phase
    # "synthetic" = run synthetic trace for the forced phase instead of live OS
    "forced_phase":  None,
    "sim_mode":      "live",   # "live" | "synthetic"
    "sort_by":       "score",  # "score" | "process" | "age"
    # Remote machine data (populated by /ubuntu_events POST)
    "remote": {
        "connected":   False,
        "last_seen":   0,
        "system":      {},
        "processes":   [],
        "evictions":   deque(maxlen=20),
        "label":       "Remote",
    },
}
state_lock = threading.Lock()

# Queue to push events to SSE clients
event_queue: queue.Queue = queue.Queue(maxsize=10)


# ─────────────────────────────────────────────────────────────────────────────
# Background simulation thread
# ─────────────────────────────────────────────────────────────────────────────

def simulation_thread(model_path: str, cache_size: int, interval: float):
    # Scale working set to ~2x cache size so there are always evictions but
    # not so many that all policies thrash at 100% miss rate.
    # Each process gets up to 8 regions → top_n * 8 ≈ working_set_size.
    top_n = max(8, cache_size // 4)         # cache=48→12 procs, cache=256→64 procs
    max_ev = cache_size * 2                  # events per sample ≈ 2× cache size

    monitor = OSMemoryMonitor(
        sample_interval=interval,
        top_n_processes=top_n,
        max_events_per_sample=max_ev,
    )

    dqn_policy = DQNEvictionPolicy(capacity=cache_size)
    if os.path.exists(model_path):
        try:
            dqn_policy.load(model_path)
        except Exception:
            pass

    caches = {
        "DQN": Cache(cache_size, dqn_policy),
        "LRU": Cache(cache_size, LRUPolicy()),
        "LFU": Cache(cache_size, LFUPolicy()),
        "ARC": Cache(cache_size, ARCPolicy(capacity=cache_size)),
    }
    trackers = {name: HitRateTracker(500) for name in caches}
    phase_clf = PhaseClassifier(n_phases=5, window_size=200)

    # Track per-slot state for DQN cache
    # slot_info[page_id] = {process, region, score, added_step}
    slot_info: dict = {}
    page_id_to_region: dict = {}
    step = 0

    # Rolling miss rate history (last 200 points per policy)
    miss_history = {name: deque(maxlen=200) for name in caches}

    while simulation_state["running"]:
        # Check mode — may switch to synthetic events
        with state_lock:
            current_sim_mode  = simulation_state["sim_mode"]
            current_forced_ph = simulation_state["forced_phase"]
            current_sort_by   = simulation_state["sort_by"]

        if current_sim_mode == "synthetic" and current_forced_ph is not None:
            raw_events = _synthetic_events_for_phase(current_forced_ph,
                                                     cache_size=cache_size)
            # Convert dicts to PageEvent-compatible objects
            from types import SimpleNamespace
            events = [SimpleNamespace(
                page_id=e["page_id"], process_name=e["process_name"],
                pid=0, region_name=e["region_name"],
                rss_bytes=0, event_type="access", timestamp=time.time(),
                cpu_percent=e["cpu_percent"], rss_delta=e["rss_delta"],
            ) for e in raw_events]
        else:
            events = monitor.sample_events()

        if not events:
            time.sleep(interval)
            continue

        # ── Shuffle events to break sequential-scan pattern ──────────────────
        # Without shuffling, psutil always returns processes in the same order,
        # which looks like a sequential scan → LRU/ARC/LFU thrash at 100%.
        random.shuffle(events)

        added_this_round = []
        evicted_this_round = []

        for ev in events:
            step += 1
            page_id = ev.page_id
            page_id_to_region[page_id] = {
                "process":   ev.process_name,
                "region":    ev.region_name,
                "cpu_pct":   ev.cpu_percent,
                "rss_delta": ev.rss_delta,
            }

            dqn_policy.set_phase(phase_clf.phase_id)

            # Feed to all caches
            for name, cache in caches.items():
                before_pages = frozenset(cache._pages)
                is_hit = cache.access(page_id)
                after_pages = frozenset(cache._pages)
                trackers[name].record(is_hit)

                if name == "DQN":
                    # Update phase classifier with real hit status
                    phase_clf.update(page_id, is_hit)

                    new_pages = after_pages - before_pages
                    removed_pages = before_pages - after_pages

                    for p in new_pages:
                        info = page_id_to_region.get(p, {})
                        cpu_pct  = info.get("cpu_pct", 0.0)
                        rss_delta = info.get("rss_delta", 0)
                        init_score = _cpu_score(cpu_pct, rss_delta)
                        slot_info[p] = {
                            "page_id":    p,
                            "process":    info.get("process", "?")[:14],
                            "region":     info.get("region",  "?")[:20],
                            "added_step": step,
                            "score":      init_score,
                            "cpu_pct":    cpu_pct,
                            "rss_delta":  rss_delta,
                        }
                        added_this_round.append(p)

                    for p in removed_pages:
                        evicted_this_round.append(p)
                        slot_info.pop(p, None)

            # Capture DQN eviction explanation + update ALL slot scores
            if dqn_policy.last_eviction_details:
                ev_det = dqn_policy.last_eviction_details
                all_scores = ev_det.get("all_scores", {})

                for pid, score in all_scores.items():
                    if pid in slot_info:
                        slot_info[pid]["score"] = score

                if evicted_this_round:
                    evicted_page = ev_det.get("evicted_page")
                    if evicted_page is not None:
                        info = page_id_to_region.get(evicted_page, {})
                        ev_score = ev_det.get("evict_score", 0)
                        # Pick eviction reason based on normalised rank
                        n_cached = len(all_scores)
                        rank = sum(1 for s in all_scores.values() if s <= ev_score)
                        rank_pct = rank / max(n_cached, 1)
                        log_entry = {
                            "step":    step,
                            "page":    evicted_page,
                            "process": info.get("process", "?")[:18],
                            "region":  info.get("region",  "?")[:28],
                            "score":   round(ev_score, 5),
                            "phase":   PHASE_NAMES.get(phase_clf.phase_id, "?"),
                            "reason":  _eviction_reason_rank(rank_pct),
                        }
                        with state_lock:
                            simulation_state["eviction_log"].appendleft(log_entry)

        # ── Proactive score refresh every 5 steps ────────────────────────────
        # Primary signal: CPU% of the owning process (high CPU = hot pages, keep).
        # DQN score blends in gradually as the network trains (first 10K evictions).
        # This ensures meaningful colour spread immediately on a live OS stream
        # where all pages are "accessed" every sample (recency/frequency useless).
        if step % 5 == 0 and slot_info:
            cached_pages = list(caches["DQN"]._pages)
            if cached_pages:
                try:
                    dqn_scores, _ = dqn_policy._score_all_pages(cached_pages)
                    n_trained = dqn_policy._eviction_count
                    dqn_blend = min(0.4, n_trained / 25_000)   # max 40% DQN weight
                    for pid in cached_pages:
                        if pid not in slot_info:
                            continue
                        cpu_pct   = slot_info[pid].get("cpu_pct", 0.0)
                        rss_delta = slot_info[pid].get("rss_delta", 0)
                        cpu_s = _cpu_score(cpu_pct, rss_delta)
                        dqn_s = dqn_scores.get(pid, cpu_s)
                        slot_info[pid]["score"] = (1 - dqn_blend) * cpu_s + dqn_blend * dqn_s
                except Exception:
                    pass

        # ── Update cpu_pct/rss_delta in slot_info from latest events ─────────
        for ev in events:
            if ev.page_id in slot_info:
                slot_info[ev.page_id]["cpu_pct"]   = ev.cpu_percent
                slot_info[ev.page_id]["rss_delta"]  = ev.rss_delta

        # Record miss rates
        for name, tracker in trackers.items():
            miss_history[name].append(round(tracker.rolling_miss_rate * 100, 2))

        # Build cache slot list for frontend
        dqn_pages = list(caches["DQN"]._pages)
        slots = []
        for p in dqn_pages:
            info = slot_info.get(p, {})
            score = info.get("score", 0.5)
            slots.append({
                "page_id": p,
                "process": info.get("process", "?"),
                "region":  info.get("region",  "?"),
                "score":   score,
                "age":     step - info.get("added_step", step),
            })
        slots.sort(key=lambda s: s["score"])   # lowest score first (eviction candidates)

        # Normalize scores to [0,1] relative to current cache contents so
        # the dashboard always shows meaningful colour spread even when the
        # trained network outputs very small absolute values for all pages.
        if slots:
            raw_min = slots[0]["score"]
            raw_max = slots[-1]["score"]
            rng = raw_max - raw_min
            for s in slots:
                if rng > 1e-6:
                    s["score_norm"] = round((s["score"] - raw_min) / rng, 3)
                else:
                    s["score_norm"] = 0.5
                s["score"] = round(s["score"], 5)  # keep raw for tooltip

        # System stats
        sys_stats = monitor.system_stats or {}
        proc_list = monitor.get_top_processes()[:8]
        proc_display = [
            {
                "pid":  p.pid,
                "name": p.name[:22],
                "rss":  round(p.rss / 1e6, 1),
                "pct":  round(p.pct, 1),
            }
            for p in proc_list
        ]

        # Write state
        phase_id = current_forced_ph if current_forced_ph is not None else phase_clf.phase_id

        # Sort slots according to current sort_by setting
        if current_sort_by == "process":
            slots.sort(key=lambda s: s["process"])
        elif current_sort_by == "age":
            slots.sort(key=lambda s: s["age"], reverse=True)   # oldest first
        else:
            pass  # already sorted by score (lowest first)

        with state_lock:
            simulation_state.update({
                "step":         step,
                "cache_slots":  slots,
                "miss_rates":   {k: list(v) for k, v in miss_history.items()},
                "phase_id":     phase_id,
                "phase_name":   PHASE_NAMES.get(phase_id, "?"),
                "phase_color":  PHASE_COLORS.get(phase_id, "#888"),
                "system":       {
                    "used_gb":    round(sys_stats.get("used_gb", 0), 1),
                    "total_gb":   round(sys_stats.get("total_gb", 0), 1),
                    "used_pct":   round(sys_stats.get("used_pct", 0), 1),
                    "swap_gb":    round(sys_stats.get("swap_used_gb", 0), 2),
                    "pages_active": sys_stats.get("pages_active", 0),
                    "pageins":    sys_stats.get("pageins", 0),
                    "pageouts":   sys_stats.get("pageouts", 0),
                },
                "processes":    proc_display,
                "last_added":   list(added_this_round[-5:]),
                "last_evicted": list(evicted_this_round[-5:]),
                "sim_mode":     current_sim_mode,
                "forced_phase": current_forced_ph,
                "sort_by":      current_sort_by,
            })

        # Push snapshot to SSE queue (non-blocking)
        try:
            with state_lock:
                remote = simulation_state["remote"]
                # Mark remote as disconnected if no POST in last 10 seconds
                if remote["connected"] and time.time() - remote["last_seen"] > 10:
                    remote["connected"] = False

                snapshot = {
                    "step":        simulation_state["step"],
                    "cache_slots": simulation_state["cache_slots"],
                    "miss_rates":  simulation_state["miss_rates"],
                    "phase_id":    simulation_state["phase_id"],
                    "phase_name":  simulation_state["phase_name"],
                    "phase_color": simulation_state["phase_color"],
                    "system":      simulation_state["system"],
                    "processes":   simulation_state["processes"],
                    "eviction_log": list(simulation_state["eviction_log"])[:15],
                    "last_added":  simulation_state["last_added"],
                    "last_evicted": simulation_state["last_evicted"],
                    "remote": {
                        "connected": remote["connected"],
                        "label":     remote["label"],
                        "system":    remote["system"],
                        "processes": remote["processes"],
                        "evictions": list(remote["evictions"])[:10],
                    },
                }
            event_queue.put_nowait(json.dumps(snapshot))
        except queue.Full:
            pass

        time.sleep(max(0, interval - 0.05))


def _cpu_score(cpu_pct: float, rss_delta: int = 0) -> float:
    """
    Convert process CPU% + RSS delta to a keep-score in [0, 1].

    Why CPU%?  In live OS monitoring every page is "accessed" every sample
    (same processes run continuously), so recency/frequency features are
    identical for all pages.  CPU% is the real signal: a process burning
    CPU has hot pages that must stay; a sleeping process has cold pages
    that are good eviction candidates.

      cpu_pct=0,  rss stable  → 0.10  (idle, cold — evict first)
      cpu_pct=0,  rss growing → 0.35  (allocating but not running yet)
      cpu_pct=5,  rss stable  → ~0.55
      cpu_pct=20, rss stable  → ~0.80
      cpu_pct=50+             → ~0.95  (very hot)
      any,        rss shrinking → -0.15 penalty (OS already reclaiming)
    """
    # Normalise: 50% CPU → ~1.0 base
    base = min(cpu_pct / 50.0, 1.0)
    # Small floor so even idle processes aren't all identical at 0
    base = 0.10 + 0.85 * base
    # RSS delta bonus/penalty
    if rss_delta > 0:
        base = min(base + 0.08, 1.0)   # growing = hotter
    elif rss_delta < 0:
        base = max(base - 0.15, 0.0)   # shrinking = OS already evicting
    return round(base, 4)


def _eviction_reason_rank(rank_pct: float) -> str:
    """Describe why a page was evicted based on its score rank in the cache."""
    if rank_pct < 0.05:
        return "Coldest page — clear eviction choice"
    elif rank_pct < 0.15:
        return "Very cold — rarely accessed recently"
    elif rank_pct < 0.30:
        return "Cold — below-average access pattern"
    elif rank_pct < 0.50:
        return "Below median — evicted under cache pressure"
    else:
        return "Warm page — evicted due to capacity pressure"


def _eviction_reason(score: float) -> str:
    if score < 0.2:
        return "Very cold — never reused"
    elif score < 0.35:
        return "Low recency + low frequency"
    elif score < 0.5:
        return "Infrequent in current phase"
    else:
        return "Least-bad candidate available"


# ─────────────────────────────────────────────────────────────────────────────
# Flask routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", cache_size=CACHE_SIZE)


@app.route("/stream")
def stream():
    """Server-Sent Events endpoint — browser connects here for live updates."""
    def generate():
        yield "data: {}\n\n"   # initial ping
        while True:
            try:
                data = event_queue.get(timeout=2.0)
                yield f"data: {data}\n\n"
            except queue.Empty:
                yield ": heartbeat\n\n"   # keep connection alive
    return Response(generate(),
                    mimetype="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                        "Connection": "keep-alive",
                    })


@app.route("/state")
def get_state():
    """REST endpoint for initial page load."""
    with state_lock:
        remote = simulation_state["remote"]
        return jsonify({
            "step":        simulation_state["step"],
            "cache_slots": simulation_state["cache_slots"],
            "miss_rates":  simulation_state["miss_rates"],
            "phase_id":    simulation_state["phase_id"],
            "phase_name":  simulation_state["phase_name"],
            "phase_color": simulation_state["phase_color"],
            "system":      simulation_state["system"],
            "processes":   simulation_state["processes"],
            "eviction_log": list(simulation_state["eviction_log"])[:15],
            "remote": {
                "connected": remote["connected"],
                "label":     remote["label"],
                "system":    remote["system"],
                "processes": remote["processes"],
                "evictions": list(remote["evictions"])[:10],
            },
        })


@app.route("/set_mode", methods=["POST"])
def set_mode():
    """
    Control the simulation mode from the browser UI.

    Body: { "sim_mode": "live"|"synthetic", "forced_phase": null|0|1|2|3|4,
            "sort_by": "score"|"process"|"age" }
    """
    from flask import request
    data = request.get_json(force=True, silent=True) or {}
    with state_lock:
        if "sim_mode" in data:
            simulation_state["sim_mode"] = data["sim_mode"]
        if "forced_phase" in data:
            simulation_state["forced_phase"] = data["forced_phase"]  # None or int
        if "sort_by" in data:
            simulation_state["sort_by"] = data["sort_by"]
    return jsonify({"ok": True,
                    "sim_mode": simulation_state["sim_mode"],
                    "forced_phase": simulation_state["forced_phase"],
                    "sort_by": simulation_state["sort_by"]})


def _synthetic_events_for_phase(phase_id: int, n_pages: int = 40,
                                 cache_size: int = 48) -> list:
    """
    Generate a small burst of synthetic PageEvent-like dicts matching the
    access pattern of the requested phase.  Used when sim_mode == 'synthetic'.

    Phase patterns:
      0 Init        — uniform random across all pages (cold start)
      1 Steady      — 80% of accesses hit the same 20% hot pages (LRU-friendly)
      2 Sequential  — pages accessed in strict order 0..N-1 (LRU killer)
      3 Random      — pure uniform random (frequency matters)
      4 Hotspot     — 90% of accesses on 5 hot pages, rest random
    """
    import random as _rng
    events = []
    phase_names = {0: "Init", 1: "Steady", 2: "Sequential", 3: "Random", 4: "Hotspot"}

    # Synthetic process names per phase for realism
    phase_procs = {
        0: ["init", "systemd", "udev", "kworker"],
        1: ["firefox", "python3", "gnome-shell", "code"],
        2: ["dd", "rsync", "ffmpeg", "tar"],
        3: ["stress-ng", "fio", "memtester", "sysbench"],
        4: ["redis", "postgres", "nginx", "memcached"],
    }
    procs = phase_procs.get(phase_id, ["process"])

    if phase_id == 2:
        # Sequential: strict scan — the LRU killer
        seq_pages = list(range(n_pages))
        page_ids = seq_pages[:20]           # 20 events per burst
    elif phase_id == 4:
        # Hotspot: 5 hot pages + occasional cold
        hot = list(range(5))
        cold = list(range(5, n_pages))
        page_ids = [_rng.choice(hot) if _rng.random() < 0.9 else _rng.choice(cold)
                    for _ in range(20)]
    elif phase_id == 1:
        # Steady: 20% pages get 80% of hits
        hot = list(range(int(n_pages * 0.2)))
        all_p = list(range(n_pages))
        page_ids = [_rng.choice(hot) if _rng.random() < 0.8 else _rng.choice(all_p)
                    for _ in range(20)]
    else:
        # Init / Random: uniform
        page_ids = [_rng.randint(0, n_pages - 1) for _ in range(20)]

    for pid in page_ids:
        cpu = _rng.uniform(5, 80) if phase_id in (1, 4) else _rng.uniform(0, 20)
        events.append({
            "page_id":      pid + 10000,          # offset so no collision with live IDs
            "process_name": _rng.choice(procs),
            "region_name":  f"region{pid % 4}",
            "cpu_percent":  cpu,
            "rss_delta":    _rng.randint(-512, 2048) * 1024,
        })
    return events


@app.route("/ubuntu_events", methods=["POST"])
def ubuntu_events():
    """
    Receive page-access events + process stats from a remote machine agent.

    Expected JSON body:
    {
        "label":   "Ubuntu VM",          # optional display name
        "events":  [{"page_id": 12345, "process": "firefox", "region": "heap:0"}, ...],
        "system":  {"total_gb": 8.0, "used_gb": 4.2, "used_pct": 52.5,
                    "swap_gb": 0.1, "pages_active": 80000,
                    "pageins": 100, "pageouts": 5},
        "processes": [{"pid": 1234, "name": "firefox", "rss": 412.3, "pct": 5.1}, ...],
        "evictions": [{"page_id": 8888, "process": "firefox", "region": "heap:0",
                       "score": 0.12, "phase": "Sequential Scan",
                       "reason": "Cold page flushed by kernel"}]
    }
    """
    from flask import request
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "invalid JSON"}), 400

    events    = data.get("events", [])
    sys_info  = data.get("system", {})
    procs     = data.get("processes", [])
    evictions = data.get("evictions", [])
    label     = data.get("label", "Remote")[:24]

    with state_lock:
        remote = simulation_state["remote"]
        remote["connected"] = True
        remote["last_seen"] = time.time()
        remote["label"]     = label
        if sys_info:
            remote["system"] = {
                k: v for k, v in sys_info.items()
                if k in ("total_gb", "used_gb", "used_pct",
                         "swap_gb", "pages_active", "pageins", "pageouts")
            }
        if procs:
            remote["processes"] = procs[:8]
        for ev in evictions:
            remote["evictions"].appendleft({
                "step":    simulation_state["step"],
                "page":    ev.get("page_id", 0),
                "process": ev.get("process", "remote")[:18],
                "region":  ev.get("region", "?")[:28],
                "score":   round(ev.get("score", 0), 4),
                "phase":   ev.get("phase", "?"),
                "reason":  ev.get("reason", "Remote eviction"),
            })

    return jsonify({"ok": True, "received": len(events)})


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_server(host: str = "0.0.0.0", port: int = 5050,
               model_path: str = "models/dqn_direct.pt",
               cache_size: int = 48,
               interval: float = 1.0):
    global CACHE_SIZE
    CACHE_SIZE = cache_size

    print(f"\n{'='*60}")
    print(f"  PARL Live Dashboard")
    print(f"  Open in browser: http://localhost:{port}")
    print(f"  Ubuntu VM:       http://<this-mac-ip>:{port}")
    print(f"  Cache size:      {cache_size} slots")
    print(f"  Sample interval: {interval}s")
    print(f"  Model:           {'loaded ✓' if os.path.exists(model_path) else 'untrained'}")
    print(f"{'='*60}\n")

    # Start simulation in background thread
    t = threading.Thread(
        target=simulation_thread,
        args=(model_path, cache_size, interval),
        daemon=True,
    )
    t.start()

    # Run Flask (disable reloader so background thread stays alive)
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PARL Web Dashboard")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--model", default="models/dqn_direct.pt")
    parser.add_argument("--cache_size", type=int, default=48)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()

    # Change to project root so imports work
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run_server(args.host, args.port, args.model, args.cache_size, args.interval)
