"""
dashboard.py — Live terminal dashboard showing PARL working on real OS pages.

What you'll see (plain English):
  ┌─ System Memory ────────────────────────────────────────────────┐
  │ RAM: 12.4 / 16.0 GB used (77.5%)   Swap: 0.2 GB              │
  │ Pages active: 1,234,567  |  Page faults: 892  |  Pageouts: 12 │
  └────────────────────────────────────────────────────────────────┘
  ┌─ Top Processes by RSS ─────────────────────────────────────────┐
  │ PID    Process          RSS       % RAM   Page Faults          │
  │ 1234   Google Chrome    2.1 GB    13.1%   45,231               │
  │ 5678   Xcode            1.8 GB    11.2%   12,098               │
  └────────────────────────────────────────────────────────────────┘
  ┌─ PARL Cache Simulation (256 page slots) ───────────────────────┐
  │ Applying DQN-direct eviction to real OS page accesses          │
  │                                                                │
  │ LAST EVICTION DECISION (step 1,234):                          │
  │   ✗ EVICTED: Chrome:heap        score=0.08 ← lowest score     │
  │   ✓ KEPT:    Xcode:__TEXT       score=0.94 (freq+recency high) │
  │   ✓ KEPT:    Python:stack       score=0.87                     │
  │                                                                │
  │ WHY was Chrome:heap evicted?                                   │
  │   recency=0.12  (not accessed recently)                        │
  │   frequency=0.03 (rarely used)                                 │
  │   phase_affinity=0.05 (not typical for current phase)         │
  │   age_in_cache=0.08 (been in cache a long time unused)        │
  └────────────────────────────────────────────────────────────────┘
  ┌─ Miss Rate Comparison (last 500 accesses) ─────────────────────┐
  │ DQN-direct: 28.3%  LRU: 34.7%  LFU: 31.2%  ARC: 32.1%       │
  └────────────────────────────────────────────────────────────────┘

Run with:  python realtime/dashboard.py
Stop with: Ctrl+C
"""

import os
import sys
import time
import threading
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich.columns import Columns
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("Install rich for the full dashboard: pip install rich")

from realtime.os_monitor import OSMemoryMonitor
from simulator.cache import Cache
from simulator.metrics import HitRateTracker
from policies.dqn_policy import DQNEvictionPolicy
from policies.lru import LRUPolicy
from policies.lfu import LFUPolicy
from policies.arc import ARCPolicy
from phase_detection.phase_classifier import PhaseClassifier


class LiveDashboard:
    """
    Runs multiple caches (DQN-direct, LRU, LFU, ARC) in parallel on real
    OS page access events, and displays a live comparison terminal dashboard.
    """

    CACHE_SIZE = 64    # simulate a small cache so evictions happen frequently

    def __init__(self, sample_interval: float = 1.0,
                 model_path: str = "models/dqn_direct.pt"):
        self._monitor = OSMemoryMonitor(
            sample_interval=sample_interval,
            top_n_processes=8,
            max_events_per_sample=30,
        )

        # Caches running in parallel on the same page events
        self._dqn_policy = DQNEvictionPolicy(capacity=self.CACHE_SIZE)
        self._caches = {
            "DQN-direct": Cache(self.CACHE_SIZE, self._dqn_policy),
            "LRU":        Cache(self.CACHE_SIZE, LRUPolicy()),
            "LFU":        Cache(self.CACHE_SIZE, LFUPolicy()),
            "ARC":        Cache(self.CACHE_SIZE, ARCPolicy(capacity=self.CACHE_SIZE)),
        }
        self._trackers = {name: HitRateTracker(500) for name in self._caches}

        # Phase detection
        self._phase_clf = PhaseClassifier(n_phases=5, window_size=200)

        # Load trained model if available
        if os.path.exists(model_path):
            try:
                self._dqn_policy.load(model_path)
                self._model_loaded = True
            except Exception:
                self._model_loaded = False
        else:
            self._model_loaded = False

        # State for display
        self._step = 0
        self._recent_events: deque = deque(maxlen=10)
        self._last_eviction = {}
        self._running = True

        # Phase name map
        self._phase_names = {
            0: "Init", 1: "Steady", 2: "Sequential", 3: "Random", 4: "Hotspot"
        }

    def _process_event(self, page_id: int, process_name: str,
                       region_name: str, event_type: str):
        """Feed one page event to all caches."""
        self._step += 1
        self._phase_clf.update(page_id, False)
        self._dqn_policy.set_phase(self._phase_clf.phase_id)

        for name, cache in self._caches.items():
            is_hit = cache.access(page_id)
            self._trackers[name].record(is_hit)

        # Capture last DQN eviction for display
        if self._dqn_policy.last_eviction_details:
            self._last_eviction = dict(self._dqn_policy.last_eviction_details)
            self._last_eviction["region_name"] = region_name
            self._last_eviction["process_name"] = process_name

        self._recent_events.appendleft({
            "step": self._step,
            "page_id": page_id,
            "process": process_name,
            "region": region_name[:35],
            "type": event_type,
        })

    def _make_display(self) -> str:
        """Build the display string (plain text if no rich)."""
        lines = []
        lines.append("=" * 70)
        lines.append(" PARL — Live Page Eviction Monitor")
        lines.append(f" Model: {'loaded ✓' if self._model_loaded else 'untrained (run train first)'}")
        lines.append("=" * 70)

        # System stats
        stats = self._monitor.system_stats
        if stats:
            lines.append(f"\n[SYSTEM MEMORY]")
            lines.append(f"  RAM: {stats.get('used_gb',0):.1f} / {stats.get('total_gb',0):.1f} GB "
                         f"({stats.get('used_pct',0):.1f}% used)")
            swap = stats.get('swap_used_gb', 0)
            if swap > 0:
                lines.append(f"  Swap: {swap:.2f} GB in use ⚠")
            if 'pages_active' in stats:
                lines.append(f"  Active pages: {stats['pages_active']:,}  |  "
                             f"Inactive: {stats.get('pages_inactive',0):,}  |  "
                             f"Free: {stats.get('pages_free',0):,}")
            if 'pageins' in stats:
                lines.append(f"  Pageins: {stats['pageins']:,}  |  "
                             f"Pageouts: {stats.get('pageouts',0):,}")

        # Miss rate comparison
        lines.append(f"\n[MISS RATE COMPARISON — last 500 page events, cache={self.CACHE_SIZE} slots]")
        for name, tracker in self._trackers.items():
            mr = tracker.rolling_miss_rate * 100
            bar = "█" * int(mr / 5) + "░" * (20 - int(mr / 5))
            marker = " ← OUR AI" if name == "DQN-direct" else ""
            lines.append(f"  {name:<12} {mr:5.1f}%  [{bar}]{marker}")

        # Phase
        phase_id = self._phase_clf.phase_id
        phase_name = self._phase_names.get(phase_id, "?")
        lines.append(f"\n[CURRENT PHASE] {phase_id}: {phase_name}  |  Step: {self._step:,}")

        # Last DQN eviction explanation
        if self._last_eviction:
            ev = self._last_eviction
            lines.append(f"\n[LAST DQN EVICTION DECISION — step {ev.get('step',0):,}]")
            lines.append(f"  Process: {ev.get('process_name','?')} | Region: {ev.get('region_name','?')}")

            evicted_page = ev.get("evicted_page")
            evict_score = ev.get("evict_score", 0)
            lines.append(f"\n  ✗ EVICTED  page_id={evicted_page}   score={evict_score:.4f}")
            lines.append( "            ^ LOWEST score = least likely to be needed soon")

            top_kept = ev.get("top_kept", [])
            all_scores = ev.get("all_scores", {})
            if top_kept:
                lines.append(f"\n  ✓ KEPT (highest scored pages):")
                for p in top_kept[:3]:
                    s = all_scores.get(p, 0)
                    lines.append(f"    page_id={p}  score={s:.4f}")

            lines.append(f"\n  HOW THE SCORE IS COMPUTED (8 features, 0=bad 1=good):")
            lines.append( "  ┌────────────────────────────────────────────────────┐")
            lines.append( "  │ recency     — accessed recently?  (0=old, 1=fresh) │")
            lines.append( "  │ frequency   — accessed often?     (0=rare, 1=hot)  │")
            lines.append( "  │ age_in_cache— new in cache?        (0=old, 1=new)   │")
            lines.append( "  │ ird_score   — regular access pattern?               │")
            lines.append( "  │ phase_norm  — which phase are we in? (0-4)          │")
            lines.append( "  │ cache_press — how full is cache?   (0=empty 1=full) │")
            lines.append( "  │ phase_aff   — this page typical for current phase?  │")
            lines.append( "  │ density     — freq × recency combined               │")
            lines.append( "  └────────────────────────────────────────────────────┘")
            lines.append( "  Neural net combines these 8 numbers → single keep-score")
            lines.append( "  The page with LOWEST keep-score gets evicted.")

        # Recent events
        lines.append(f"\n[RECENT PAGE EVENTS]")
        for ev_info in list(self._recent_events)[:5]:
            t = "FAULT" if ev_info["type"] == "fault" else \
                "EVICT" if ev_info["type"] == "evict" else "access"
            lines.append(f"  [{t:6}] {ev_info['process']:<18} {ev_info['region']}")

        lines.append("\n  (Ctrl+C to stop)")
        return "\n".join(lines)

    def _make_rich_display(self):
        """Build rich renderables for a pretty dashboard."""
        # System memory panel
        stats = self._monitor.system_stats
        mem_text = Text()
        if stats:
            used = stats.get("used_gb", 0)
            total = stats.get("total_gb", 0)
            pct = stats.get("used_pct", 0)
            color = "red" if pct > 85 else "yellow" if pct > 70 else "green"
            mem_text.append(f"RAM: {used:.1f} / {total:.1f} GB  ", style="bold")
            mem_text.append(f"({pct:.1f}% used)", style=f"bold {color}")
            swap = stats.get("swap_used_gb", 0)
            if swap > 0.1:
                mem_text.append(f"  ⚠ Swap: {swap:.2f} GB", style="bold red")
            if "pages_active" in stats:
                mem_text.append(f"\nActive: {stats['pages_active']:,}  "
                               f"Inactive: {stats.get('pages_inactive',0):,}  "
                               f"Free: {stats.get('pages_free',0):,}", style="dim")
            if "pageins" in stats:
                mem_text.append(f"\nPageins: {stats['pageins']:,}  "
                               f"Pageouts: {stats.get('pageouts',0):,}", style="dim")
        sys_panel = Panel(mem_text, title="[bold cyan]System Memory[/]",
                          border_style="cyan", box=box.ROUNDED)

        # Miss rate table
        mr_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        mr_table.add_column("Policy", style="bold", min_width=12)
        mr_table.add_column("Miss Rate", justify="right", min_width=10)
        mr_table.add_column("Bar", min_width=25)
        mr_table.add_column("", min_width=12)

        for name, tracker in self._trackers.items():
            mr = tracker.rolling_miss_rate * 100
            filled = int(mr / 4)
            bar = "█" * filled + "░" * (25 - filled)
            color = "green" if name == "DQN-direct" else "white"
            note = "[bold green]← OUR AI[/]" if name == "DQN-direct" else ""
            style = f"bold {color}" if name == "DQN-direct" else color
            mr_table.add_row(name, f"{mr:.1f}%", bar, note, style=style)

        phase_id = self._phase_clf.phase_id
        phase_name = self._phase_names.get(phase_id, "?")
        mr_panel = Panel(mr_table,
                         title=f"[bold yellow]Miss Rates — cache={self.CACHE_SIZE} slots | "
                               f"Phase {phase_id}: {phase_name} | Step {self._step:,}[/]",
                         border_style="yellow", box=box.ROUNDED)

        # Eviction explanation panel
        ev = self._last_eviction
        ev_text = Text()
        if ev:
            ev_text.append("LAST EVICTION DECISION\n", style="bold white")
            proc = ev.get("process_name", "?")
            region = ev.get("region_name", "?")
            ev_text.append(f"Process: {proc}  Region: {region}\n", style="dim")

            evicted_page = ev.get("evicted_page", "?")
            evict_score = ev.get("evict_score", 0)
            ev_text.append(f"\n  ✗ EVICTED  ", style="bold red")
            ev_text.append(f"page_id={evicted_page}  score={evict_score:.4f}\n",
                          style="red")
            ev_text.append("    ↑ lowest score = DQN thinks this page won't be "
                          "needed soon\n", style="dim red")

            top_kept = ev.get("top_kept", [])
            all_scores = ev.get("all_scores", {})
            if top_kept:
                ev_text.append(f"\n  ✓ KEPT (highest scored):\n", style="bold green")
                for p in top_kept[:3]:
                    s = all_scores.get(p, 0)
                    ev_text.append(f"    page_id={p}  score={s:.4f}\n",
                                  style="green")

            ev_text.append("\n  HOW EVICTION SCORE WORKS:\n", style="bold")
            ev_text.append("  8 features → neural network → keep-score [0,1]\n", style="dim")
            ev_text.append("  recency + frequency + age + regularity + phase_affinity\n",
                          style="dim")
            ev_text.append("  LOWEST score gets evicted (unlike LRU which only uses recency)\n",
                          style="dim")
        else:
            ev_text.append("Waiting for first eviction...", style="dim")
            ev_text.append("\n\nThe DQN will start making eviction decisions once\n"
                          "the cache fills up (after first ~64 unique pages).", style="dim")

        ev_panel = Panel(ev_text, title="[bold magenta]DQN Eviction Explanation[/]",
                        border_style="magenta", box=box.ROUNDED)

        # Recent events table
        ev_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        ev_table.add_column("Step", justify="right", min_width=6)
        ev_table.add_column("Type", min_width=7)
        ev_table.add_column("Process", min_width=16)
        ev_table.add_column("Memory Region", min_width=35)

        for ev_info in list(self._recent_events)[:6]:
            t = ev_info["type"]
            style = "red" if t == "fault" else "yellow" if t == "evict" else "green"
            label = {"fault": "FAULT", "evict": "EVICT", "access": "access"}.get(t, t)
            ev_table.add_row(
                str(ev_info["step"]),
                f"[{style}]{label}[/]",
                ev_info["process"],
                ev_info["region"],
            )

        events_panel = Panel(ev_table,
                            title="[bold blue]Recent OS Page Events[/]",
                            border_style="blue", box=box.ROUNDED)

        return sys_panel, mr_panel, ev_panel, events_panel

    def run(self, duration_sec: float = float("inf")):
        """Run the dashboard."""
        print("\nStarting PARL Live Monitor...")
        print(f"Cache size: {self.CACHE_SIZE} pages | Comparing: DQN-direct vs LRU vs LFU vs ARC")
        print("Sampling OS memory every second. Ctrl+C to stop.\n")

        if HAS_RICH:
            self._run_rich(duration_sec)
        else:
            self._run_plain(duration_sec)

    def _run_plain(self, duration_sec):
        import time
        start = time.time()
        for event in self._monitor.stream_events(duration_sec):
            self._process_event(
                event.page_id, event.process_name,
                event.region_name, event.event_type,
            )
            # Refresh display every 20 events
            if self._step % 20 == 0:
                os.system("clear" if os.name == "posix" else "cls")
                print(self._make_display())

    def _run_rich(self, duration_sec):
        monitor = self._monitor
        process_fn = self._process_event
        get_display = self._make_rich_display
        step_ref = self

        with Live(refresh_per_second=2, screen=True) as live:
            from rich.layout import Layout
            from rich.console import Group

            for event in monitor.stream_events(duration_sec):
                process_fn(
                    event.page_id, event.process_name,
                    event.region_name, event.event_type,
                )
                if step_ref._step % 5 == 0:
                    sys_panel, mr_panel, ev_panel, events_panel = get_display()
                    live.update(Group(sys_panel, mr_panel, ev_panel, events_panel))


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="PARL Live Monitor — watch DQN evict real OS pages")
    parser.add_argument("--cache_size", type=int, default=64,
                        help="Simulated cache size in page slots (default: 64)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Sampling interval in seconds (default: 1.0)")
    parser.add_argument("--model", default="models/dqn_direct.pt",
                        help="Path to trained DQN model")
    parser.add_argument("--duration", type=float, default=float("inf"),
                        help="Run for this many seconds (default: forever)")
    args = parser.parse_args()

    dashboard = LiveDashboard(
        sample_interval=args.interval,
        model_path=args.model,
    )
    dashboard.CACHE_SIZE = args.cache_size
    try:
        dashboard.run(args.duration)
    except KeyboardInterrupt:
        print("\nStopped.")
        print(f"\nFinal miss rates after {dashboard._step} events:")
        for name, tracker in dashboard._trackers.items():
            print(f"  {name:<14} {tracker.overall_miss_rate*100:.1f}%")


if __name__ == "__main__":
    main()
