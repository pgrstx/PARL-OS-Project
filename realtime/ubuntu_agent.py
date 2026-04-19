"""
ubuntu_agent.py — Run this on your Ubuntu VM (inside UTM).

What it does:
  Reads real Linux kernel memory events from /proc and /sys,
  sends them to the PARL dashboard running on your Mac host.

  On Linux we can access much richer data than on macOS:
    /proc/vmstat        — kernel page fault counters (pgfault, pgmajfault, etc.)
    /proc/meminfo       — RAM breakdown (Active, Inactive, Dirty pages, etc.)
    /proc/PID/statm     — per-process page counts (total, resident, shared, text, data)
    /proc/PID/status    — process name, PID, VmRSS, VmSwap, etc.
    /proc/PID/pagemap   — which virtual pages are in physical RAM (needs root)

  It also optionally uses cgroups v2 to create a memory-limited container
  and issues madvise() hints to the kernel, which is as close to "replacing
  the page replacement algorithm" as you can get in userspace.

Setup on Ubuntu VM:
  1. pip3 install psutil requests
  2. python3 ubuntu_agent.py --host <mac-ip> --port 5050

The agent will stream page events to the dashboard over HTTP.
The dashboard will show Ubuntu VM processes alongside macOS ones.
"""

import os
import sys
import time
import json
import socket
import struct
import argparse
import threading
from collections import defaultdict, deque
from pathlib import Path

try:
    import psutil
    import requests
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("Install deps: pip3 install psutil requests")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Linux-specific memory readers
# ─────────────────────────────────────────────────────────────────────────────

def read_vmstat() -> dict:
    """Read /proc/vmstat — kernel page fault counters."""
    stats = {}
    try:
        with open("/proc/vmstat") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    stats[parts[0]] = int(parts[1])
    except Exception:
        pass
    return stats


def read_meminfo() -> dict:
    """Read /proc/meminfo — system memory breakdown."""
    info = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip().split()[0]  # strip "kB"
                    info[key] = int(val) * 1024         # convert to bytes
    except Exception:
        pass
    return info


def read_process_pages(pid: int) -> dict | None:
    """
    Read /proc/PID/statm — page-level stats for one process.
    Returns: total, resident, shared, text, data pages.
    """
    try:
        with open(f"/proc/{pid}/statm") as f:
            vals = list(map(int, f.read().split()))
        page_size = os.sysconf("SC_PAGE_SIZE")
        return {
            "total_pages":    vals[0],
            "resident_pages": vals[1],   # pages in physical RAM
            "shared_pages":   vals[2],
            "text_pages":     vals[3],   # code
            "data_pages":     vals[5],   # heap + stack
            "rss_bytes":      vals[1] * page_size,
        }
    except Exception:
        return None


def read_process_status(pid: int) -> dict | None:
    """Read /proc/PID/status — human-readable process info."""
    info = {}
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                k, _, v = line.partition(":")
                info[k.strip()] = v.strip()
    except Exception:
        return None
    return info


def get_all_pids() -> list[int]:
    """List all PIDs from /proc."""
    pids = []
    for entry in Path("/proc").iterdir():
        try:
            if entry.name.isdigit():
                pids.append(int(entry.name))
        except Exception:
            pass
    return pids


# ─────────────────────────────────────────────────────────────────────────────
# Optional: madvise-based page control
# ─────────────────────────────────────────────────────────────────────────────

MADV_DONTNEED  = 4    # tell kernel: this region can be freed
MADV_WILLNEED  = 3    # tell kernel: this region will be needed soon
MADV_COLD      = 20   # (Linux 5.4+) move to cold list, evict sooner
MADV_PAGEOUT   = 21   # (Linux 5.4+) force eviction to swap

def advise_page(addr: int, length: int, advice: int) -> bool:
    """
    Call madvise() to give the kernel hints about a memory range.
    Requires: ctypes (built-in), and the memory range must be accessible.

    This is how our PARL agent can ACTUALLY influence which pages
    the Linux kernel keeps or evicts — without modifying the kernel!
    """
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        result = libc.madvise(ctypes.c_void_p(addr), ctypes.c_size_t(length), ctypes.c_int(advice))
        return result == 0
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# cgroup v2 memory controller
# ─────────────────────────────────────────────────────────────────────────────

CGROUP_BASE = "/sys/fs/cgroup/parl_sim"

def setup_cgroup(memory_limit_mb: int = 256) -> bool:
    """
    Create a cgroup v2 with a memory limit.
    Any process placed in this cgroup will be limited to memory_limit_mb.
    This creates real memory pressure so our PARL agent can observe evictions.

    Requires root (sudo).
    """
    try:
        os.makedirs(CGROUP_BASE, exist_ok=True)
        limit_bytes = memory_limit_mb * 1024 * 1024
        with open(f"{CGROUP_BASE}/memory.max", "w") as f:
            f.write(str(limit_bytes))
        with open(f"{CGROUP_BASE}/memory.swap.max", "w") as f:
            f.write("0")   # disable swap for clean page eviction testing
        print(f"Created cgroup at {CGROUP_BASE} with {memory_limit_mb}MB limit")
        return True
    except PermissionError:
        print("Note: cgroup setup requires root. Running in observation-only mode.")
        return False
    except Exception as e:
        print(f"cgroup setup failed: {e}")
        return False


def add_pid_to_cgroup(pid: int) -> bool:
    """Move a process into the PARL cgroup."""
    try:
        with open(f"{CGROUP_BASE}/cgroup.procs", "w") as f:
            f.write(str(pid))
        return True
    except Exception:
        return False


def read_cgroup_memory_events() -> dict:
    """Read memory events from cgroup (evictions, OOM, etc.)"""
    events = {}
    try:
        with open(f"{CGROUP_BASE}/memory.events") as f:
            for line in f:
                k, v = line.split()
                events[k] = int(v)
    except Exception:
        pass
    return events


# ─────────────────────────────────────────────────────────────────────────────
# Main agent loop
# ─────────────────────────────────────────────────────────────────────────────

class UbuntuAgent:
    """
    Collects Linux memory events and POSTs them to the PARL dashboard server.
    The dashboard server then feeds them through DQN-direct and all classic
    policies, and renders the visualization.
    """

    def __init__(self, host: str, port: int, interval: float = 1.0):
        self._url = f"http://{host}:{port}/ubuntu_events"
        self._interval = interval
        self._prev_vmstat = {}
        self._prev_proc_pages: dict[int, int] = {}  # pid → prev resident pages
        self._page_registry: dict[tuple, int] = {}
        self._next_id = 10_000_000   # start high to avoid collision with macOS IDs

    def _get_page_id(self, pid: int, region: str) -> int:
        key = (pid, region)
        if key not in self._page_registry:
            self._page_registry[key] = self._next_id
            self._next_id += 1
        return self._page_registry[key]

    def sample(self) -> dict:
        """Take one memory snapshot, compute deltas, return event list."""
        events = []
        vmstat = read_vmstat()
        meminfo = read_meminfo()

        # Compute page fault delta since last sample
        pgfault      = vmstat.get("pgfault", 0)      - self._prev_vmstat.get("pgfault", 0)
        pgmajfault   = vmstat.get("pgmajfault", 0)   - self._prev_vmstat.get("pgmajfault", 0)
        pgpgin       = vmstat.get("pgpgin", 0)        - self._prev_vmstat.get("pgpgin", 0)
        pgpgout      = vmstat.get("pgpgout", 0)       - self._prev_vmstat.get("pgpgout", 0)
        self._prev_vmstat = vmstat

        system_info = {
            "source":        "ubuntu_vm",
            "hostname":      socket.gethostname(),
            "total_bytes":   meminfo.get("MemTotal", 0),
            "free_bytes":    meminfo.get("MemFree", 0),
            "available_bytes": meminfo.get("MemAvailable", 0),
            "active_bytes":  meminfo.get("Active", 0),
            "inactive_bytes": meminfo.get("Inactive", 0),
            "dirty_bytes":   meminfo.get("Dirty", 0),
            "swap_used":     meminfo.get("SwapTotal", 0) - meminfo.get("SwapFree", 1),
            "pgfault_delta": pgfault,
            "pgmajfault_delta": pgmajfault,
            "pgpgin_delta":  pgpgin,
            "pgpgout_delta": pgpgout,
        }

        # Per-process page events
        processes = []
        for proc in psutil.process_iter(["pid", "name", "memory_info", "status"]):
            try:
                if proc.info["status"] == psutil.STATUS_ZOMBIE:
                    continue
                mi = proc.info["memory_info"]
                if mi is None or mi.rss == 0:
                    continue

                pid  = proc.info["pid"]
                name = (proc.info["name"] or "?")[:20]
                curr_resident = mi.rss // 4096   # resident pages

                prev_resident = self._prev_proc_pages.get(pid, curr_resident)
                delta_pages = curr_resident - prev_resident
                self._prev_proc_pages[pid] = curr_resident

                # More detailed region info from /proc
                proc_pages = read_process_pages(pid)

                processes.append({
                    "pid":            pid,
                    "name":           name,
                    "rss_bytes":      mi.rss,
                    "resident_pages": curr_resident,
                    "delta_pages":    delta_pages,
                    "text_pages":     proc_pages.get("text_pages", 0) if proc_pages else 0,
                    "data_pages":     proc_pages.get("data_pages", 0) if proc_pages else 0,
                })

                # Generate page events based on deltas
                # Each "region" gets its own page_id for the simulator
                n_buckets = max(1, min(6, curr_resident // 100))
                for b in range(n_buckets):
                    region = f"{name}:r{b}"
                    page_id = self._get_page_id(pid, region)
                    bucket_delta = delta_pages // n_buckets

                    event_type = "fault" if bucket_delta > 5 else \
                                 "evict" if bucket_delta < -5 else "access"

                    events.append({
                        "page_id":      page_id,
                        "process_name": name,
                        "pid":          pid,
                        "region_name":  region,
                        "rss_bytes":    mi.rss // n_buckets,
                        "event_type":   event_type,
                    })

            except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
                continue

        processes.sort(key=lambda p: p["rss_bytes"], reverse=True)

        return {
            "system":    system_info,
            "processes": processes[:20],
            "events":    events[:200],
        }

    def run(self):
        """Main loop: sample → POST to dashboard."""
        print(f"Ubuntu agent started. Sending to {self._url}")
        print(f"Hostname: {socket.gethostname()}")
        print(f"Interval: {self._interval}s")
        print("Ctrl+C to stop\n")

        while True:
            try:
                data = self.sample()
                try:
                    resp = requests.post(self._url, json=data, timeout=3)
                    if resp.status_code == 200:
                        n_ev = len(data["events"])
                        sys_d = data["system"]
                        used = (sys_d["total_bytes"] - sys_d["available_bytes"]) / 1e9
                        total = sys_d["total_bytes"] / 1e9
                        print(f"  [{socket.gethostname()}] {n_ev} events | "
                              f"RAM {used:.1f}/{total:.1f}GB | "
                              f"pgfaults={sys_d['pgfault_delta']} | "
                              f"pgmajfaults={sys_d['pgmajfault_delta']}")
                except requests.exceptions.ConnectionError:
                    print(f"  Can't reach {self._url} — is the server running on the Mac?")
            except KeyboardInterrupt:
                print("\nAgent stopped.")
                break
            except Exception as e:
                print(f"  Error: {e}")

            time.sleep(self._interval)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone mode — run without connecting to dashboard (local only)
# ─────────────────────────────────────────────────────────────────────────────

def run_local():
    """Print memory events locally — useful for testing on Ubuntu without the Mac."""
    print("Running in local mode (no dashboard connection).\n")

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from simulator.cache import Cache
        from policies.dqn_policy import DQNEvictionPolicy
        from policies.lru import LRUPolicy

        dqn = DQNEvictionPolicy(capacity=64)
        cache_dqn = Cache(64, dqn)
        cache_lru = Cache(64, LRUPolicy())
        agent = UbuntuAgent("localhost", 5050, 1.0)

        print("Running PARL simulation on Ubuntu memory events...")
        print(f"{'Step':>6} {'Events':>7} {'DQN miss':>10} {'LRU miss':>10} {'Pageins':>8}")
        print("-" * 50)

        for step in range(30):
            data = agent.sample()
            for ev in data["events"]:
                cache_dqn.access(ev["page_id"])
                cache_lru.access(ev["page_id"])

            sys_d = data["system"]
            print(f"{step:>6} {len(data['events']):>7} "
                  f"{cache_dqn.miss_rate*100:>9.1f}% "
                  f"{cache_lru.miss_rate*100:>9.1f}% "
                  f"{sys_d.get('pgpgin_delta', 0):>8}")
            time.sleep(1)

    except ImportError:
        # PARL not installed on VM, just show system info
        vmstat = read_vmstat()
        meminfo = read_meminfo()
        print(f"Total RAM:   {meminfo.get('MemTotal',0)//1024//1024} GB")
        print(f"Used RAM:    {(meminfo['MemTotal']-meminfo['MemAvailable'])//1024//1024} GB")
        print(f"Active:      {meminfo.get('Active',0)//1024//1024} GB")
        print(f"Inactive:    {meminfo.get('Inactive',0)//1024//1024} GB")
        print(f"Page faults: {vmstat.get('pgfault',0):,}")
        print(f"Major faults:{vmstat.get('pgmajfault',0):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PARL Ubuntu VM Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Send events to Mac dashboard at 192.168.64.1:5050
  python3 ubuntu_agent.py --host 192.168.64.1 --port 5050

  # Run locally (no Mac dashboard needed)
  python3 ubuntu_agent.py --local

  # Setup cgroup memory limit (run as root)
  sudo python3 ubuntu_agent.py --setup-cgroup --cgroup-limit 512

  # Find your Mac's IP for UTM (run on Mac):
  #   ifconfig | grep "inet " | grep -v 127.0
        """
    )
    parser.add_argument("--host",  default="192.168.64.1",
                        help="Mac host IP (UTM default gateway: 192.168.64.1)")
    parser.add_argument("--port",  type=int, default=5050)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--local", action="store_true",
                        help="Run in local mode (no dashboard)")
    parser.add_argument("--setup-cgroup", action="store_true",
                        help="Create memory-limited cgroup (needs root)")
    parser.add_argument("--cgroup-limit", type=int, default=256,
                        help="Cgroup memory limit in MB (default: 256)")
    args = parser.parse_args()

    if args.setup_cgroup:
        setup_cgroup(args.cgroup_limit)
        sys.exit(0)

    if args.local:
        run_local()
    else:
        agent = UbuntuAgent(args.host, args.port, args.interval)
        try:
            agent.run()
        except KeyboardInterrupt:
            print("\nStopped.")
