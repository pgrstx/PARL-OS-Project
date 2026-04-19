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
}
state_lock = threading.Lock()

# Queue to push events to SSE clients
event_queue: queue.Queue = queue.Queue(maxsize=10)


# ─────────────────────────────────────────────────────────────────────────────
# Background simulation thread
# ─────────────────────────────────────────────────────────────────────────────

def simulation_thread(model_path: str, cache_size: int, interval: float):
    monitor = OSMemoryMonitor(
        sample_interval=interval,
        top_n_processes=30,
        max_events_per_sample=cache_size * 3,
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
        events = monitor.sample_events()
        if not events:
            time.sleep(interval)
            continue

        added_this_round = []
        evicted_this_round = []

        for ev in events:
            step += 1
            page_id = ev.page_id
            page_id_to_region[page_id] = {
                "process": ev.process_name,
                "region":  ev.region_name,
            }

            # Update phase classifier
            phase_clf.update(page_id, False)
            dqn_policy.set_phase(phase_clf.phase_id)

            # Feed to all caches
            for name, cache in caches.items():
                before_pages = frozenset(cache._pages)
                is_hit = cache.access(page_id)
                after_pages = frozenset(cache._pages)
                trackers[name].record(is_hit)

                # Track what was added/evicted in DQN cache
                if name == "DQN":
                    new_pages = after_pages - before_pages
                    removed_pages = before_pages - after_pages
                    for p in new_pages:
                        info = page_id_to_region.get(p, {})
                        slot_info[p] = {
                            "page_id": p,
                            "process": info.get("process", "?")[:14],
                            "region":  info.get("region", "?")[:20],
                            "added_step": step,
                            "score": 0.5,
                        }
                        added_this_round.append(p)

                    for p in removed_pages:
                        evicted_this_round.append(p)
                        slot_info.pop(p, None)

            # Capture DQN eviction explanation
            if dqn_policy.last_eviction_details and evicted_this_round:
                ev_det = dqn_policy.last_eviction_details
                evicted_page = ev_det.get("evicted_page")
                all_scores = ev_det.get("all_scores", {})

                # Update scores in slot_info
                for pid, score in all_scores.items():
                    if pid in slot_info:
                        slot_info[pid]["score"] = score

                # Build eviction log entry
                if evicted_page is not None:
                    info = page_id_to_region.get(evicted_page, {})
                    log_entry = {
                        "step":    step,
                        "page":    evicted_page,
                        "process": info.get("process", "?")[:18],
                        "region":  info.get("region",  "?")[:28],
                        "score":   round(ev_det.get("evict_score", 0), 4),
                        "phase":   PHASE_NAMES.get(phase_clf.phase_id, "?"),
                        "reason":  _eviction_reason(ev_det.get("evict_score", 0.5)),
                    }
                    with state_lock:
                        simulation_state["eviction_log"].appendleft(log_entry)

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
                "score":   round(score, 3),
                "age":     step - info.get("added_step", step),
            })
        slots.sort(key=lambda s: s["score"])   # lowest score first (eviction candidates)

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
        phase_id = phase_clf.phase_id
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
            })

        # Push snapshot to SSE queue (non-blocking)
        try:
            with state_lock:
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
                }
            event_queue.put_nowait(json.dumps(snapshot))
        except queue.Full:
            pass

        time.sleep(max(0, interval - 0.05))


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
        })


@app.route("/ubuntu_events", methods=["POST"])
def ubuntu_events():
    """
    Receive page-access events from the Ubuntu VM agent (ubuntu_agent.py).

    Expected JSON body:
    {
        "events": [
            {"page_id": 12345, "process": "firefox", "region": "heap:0"},
            ...
        ],
        "system": {
            "total_gb": 8.0, "used_gb": 4.2, "used_pct": 52.5,
            "swap_gb": 0.1, "pages_active": 80000,
            "pageins": 100, "pageouts": 5
        }
    }
    """
    from flask import request
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "invalid JSON"}), 400

    events = data.get("events", [])
    sys_override = data.get("system", {})

    if events:
        # Inject Ubuntu events into the event queue as a special snapshot marker
        # so the background SSE stream picks them up.
        try:
            with state_lock:
                # Merge system stats if provided by Ubuntu agent
                if sys_override:
                    simulation_state["system"].update({
                        k: v for k, v in sys_override.items()
                        if k in ("total_gb", "used_gb", "used_pct",
                                 "swap_gb", "pages_active", "pageins", "pageouts")
                    })
                # Append to eviction log if the payload includes evictions
                for ev in data.get("evictions", []):
                    log_entry = {
                        "step":    simulation_state["step"],
                        "page":    ev.get("page_id", 0),
                        "process": ev.get("process", "ubuntu")[:18],
                        "region":  ev.get("region", "?")[:28],
                        "score":   round(ev.get("score", 0), 4),
                        "phase":   ev.get("phase", "?"),
                        "reason":  ev.get("reason", "Ubuntu kernel eviction"),
                    }
                    simulation_state["eviction_log"].appendleft(log_entry)
        except Exception:
            pass

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
