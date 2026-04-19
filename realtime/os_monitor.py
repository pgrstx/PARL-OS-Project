"""
os_monitor.py — Samples real OS memory data using psutil.

What this does (plain English):
  Your Mac has thousands of processes running. Each process uses RAM pages.
  When RAM is full, the OS must decide which pages to evict to disk (swap).
  This module:
    1. Reads the current list of running processes + their memory usage
    2. Samples which memory regions (pages) are currently resident in RAM
    3. Converts the raw process data into page-access events our simulator understands
    4. Also records system-wide page fault rates (when pages HAD to be loaded from disk)

  On macOS, we use:
    - psutil.virtual_memory()  → total/used/free RAM + page fault rate
    - psutil.Process.memory_info() → per-process RSS, VMS, page faults
    - psutil.Process.memory_maps() → which files/libraries each process has mapped

  We turn each (process, memory_region) pair into a synthetic "page_id" that
  our cache simulator can track. When a region's RSS increases, we count it
  as a "page access". When RSS drops, those pages were evicted.
"""

import os
import sys
import time
import hashlib
import platform
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

try:
    import psutil
except ImportError:
    print("psutil not found. Install with: pip install psutil")
    sys.exit(1)

IS_MACOS = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"


@dataclass
class PageEvent:
    """A single page access event from the OS."""
    page_id: int           # synthetic page ID (hash of process+region)
    process_name: str
    pid: int
    region_name: str       # e.g. "Chrome: heap", "Python: stack", "[anon]"
    rss_bytes: int         # resident set size of this region
    event_type: str        # "access" | "evict" | "fault"
    timestamp: float
    cpu_percent: float = 0.0   # CPU % of owning process at sample time
    rss_delta: int = 0         # RSS change since last sample (+ = growing, - = shrinking)


@dataclass
class ProcessMemorySnapshot:
    """Memory snapshot of one process at one point in time."""
    pid: int
    name: str
    rss: int               # resident set size in bytes
    vms: int               # virtual memory size
    pct: float             # % of total RAM
    page_faults: int       # cumulative page faults
    regions: list = field(default_factory=list)  # list of (region_name, rss)
    cpu_percent: float = 0.0


class OSMemoryMonitor:
    """
    Continuously samples OS memory state and converts it to page access events.

    Usage:
        monitor = OSMemoryMonitor(sample_interval=0.5)
        for event in monitor.stream_events():
            print(event)
    """

    # How many pages we "simulate" — treat each 4KB block as a page
    PAGE_SIZE = 4096

    def __init__(self, sample_interval: float = 1.0,
                 top_n_processes: int = 10,
                 max_events_per_sample: int = 50):
        self._interval = sample_interval
        self._top_n = top_n_processes
        self._max_events = max_events_per_sample

        # Previous snapshot for computing deltas (access = RSS increase)
        self._prev_snapshot: dict[int, ProcessMemorySnapshot] = {}
        self._prev_region_rss: dict[int, dict[str, int]] = defaultdict(dict)

        # Page ID registry: maps (pid, region) → stable integer page_id
        self._page_registry: dict[tuple, int] = {}
        self._next_page_id = 0

        # Access history for miss rate computation
        self._access_history: deque = deque(maxlen=500)

        # System-wide stats
        self.system_stats = {}

    def _get_page_id(self, pid: int, region: str) -> int:
        key = (pid, region)
        if key not in self._page_registry:
            self._page_registry[key] = self._next_page_id
            self._next_page_id += 1
        return self._page_registry[key]

    def get_top_processes(self) -> list[ProcessMemorySnapshot]:
        """
        Get top N processes by RSS memory usage.
        Works without root on macOS/Linux — uses memory_info() which is
        always accessible, rather than memory_maps() which needs privileges.
        """
        snapshots = []
        attrs = ["pid", "name", "memory_info", "memory_percent", "status",
                 "cpu_percent", "num_threads"]
        for proc in psutil.process_iter(attrs):
            try:
                info = proc.info
                if info.get("status") == psutil.STATUS_ZOMBIE:
                    continue
                mi = info.get("memory_info")
                if mi is None or mi.rss == 0:
                    continue
                snap = ProcessMemorySnapshot(
                    pid=info["pid"],
                    name=(info["name"] or "?")[:20],
                    rss=mi.rss,
                    vms=mi.vms,
                    pct=info.get("memory_percent") or 0.0,
                    page_faults=getattr(mi, "pfaults", getattr(mi, "num_page_faults", 0)),
                )
                snap.cpu_percent = info.get("cpu_percent") or 0.0
                # Break RSS into logical "regions" by size buckets so we get more
                # granular page events per process (simulates sub-process page tracking)
                n_buckets = max(1, min(8, mi.rss // (1024 * 1024)))  # 1 bucket per MB, max 8
                for bucket in range(n_buckets):
                    bucket_rss = mi.rss // n_buckets
                    region_name = f"{snap.name}:region{bucket}"
                    snap.regions.append((region_name, bucket_rss))

                snapshots.append(snap)
            except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
                continue

        snapshots.sort(key=lambda s: s.rss, reverse=True)
        return snapshots[:self._top_n]

    def get_system_vm_stats(self) -> dict:
        """Get system-wide virtual memory statistics."""
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        stats = {
            "total_gb":     vm.total / 1e9,
            "used_gb":      vm.used  / 1e9,
            "available_gb": vm.available / 1e9,
            "used_pct":     vm.percent,
            "swap_used_gb": swap.used / 1e9,
            "swap_pct":     swap.percent,
        }
        # macOS: get page fault stats via vm_stat if available
        if IS_MACOS:
            try:
                import subprocess
                out = subprocess.check_output(["vm_stat"], text=True)
                for line in out.splitlines():
                    if "Pages free" in line:
                        stats["pages_free"] = int(line.split(":")[1].strip().rstrip("."))
                    elif "Pages active" in line:
                        stats["pages_active"] = int(line.split(":")[1].strip().rstrip("."))
                    elif "Pages inactive" in line:
                        stats["pages_inactive"] = int(line.split(":")[1].strip().rstrip("."))
                    elif "Pageins" in line:
                        stats["pageins"] = int(line.split(":")[1].strip().rstrip("."))
                    elif "Pageouts" in line:
                        stats["pageouts"] = int(line.split(":")[1].strip().rstrip("."))
            except Exception:
                pass
        return stats

    def sample_events(self) -> list[PageEvent]:
        """
        Take one sample, compare to previous, emit PageEvents.

        Logic:
          - For each top process, get its current memory regions
          - If a region's RSS grew since last sample → that's a page ACCESS
          - If a region's RSS shrank → pages were EVICTED
          - New regions (RSS > 0 but not seen before) → page FAULT (loaded from disk)
        """
        events = []
        now = time.time()
        processes = self.get_top_processes()
        self.system_stats = self.get_system_vm_stats()

        for proc_snap in processes:
            pid = proc_snap.pid
            name = proc_snap.name[:20]
            cpu_pct = proc_snap.cpu_percent

            # Use pre-computed regions from ProcessMemorySnapshot (no memory_maps needed)
            regions = proc_snap.regions if proc_snap.regions else [(f"{name}:total", proc_snap.rss)]

            prev_regions = self._prev_region_rss.get(pid, {})

            for region_name, curr_rss in regions:
                if curr_rss == 0:
                    continue

                prev_rss = prev_regions.get(region_name, 0)
                page_id = self._get_page_id(pid, region_name)
                rss_delta = curr_rss - prev_rss

                if prev_rss == 0 and curr_rss > 0:
                    ev_type = "fault"
                elif curr_rss > prev_rss:
                    ev_type = "access"
                elif curr_rss < prev_rss:
                    ev_type = "evict"
                else:
                    ev_type = "access"

                events.append(PageEvent(
                    page_id=page_id,
                    process_name=name,
                    pid=pid,
                    region_name=region_name,
                    rss_bytes=curr_rss,
                    event_type=ev_type,
                    timestamp=now,
                    cpu_percent=cpu_pct,
                    rss_delta=rss_delta,
                ))

                if len(events) >= self._max_events:
                    break

            # Update tracking
            self._prev_region_rss[pid] = dict(regions)

            if len(events) >= self._max_events:
                break

        return events

    def stream_events(self, duration_sec: float = float("inf")):
        """Generator that yields PageEvent batches at regular intervals."""
        start = time.time()
        while time.time() - start < duration_sec:
            events = self.sample_events()
            for ev in events:
                yield ev
            time.sleep(self._interval)
