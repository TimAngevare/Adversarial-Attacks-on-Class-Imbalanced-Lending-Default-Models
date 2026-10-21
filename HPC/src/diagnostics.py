# Record task resources, stage timings and memory use.
import json
import os
import platform
import resource
import socket
import subprocess
import threading
import time
from contextlib import contextmanager

THREAD_ENV_VARS = [
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "NUM_OMP_THREADS", "HQ_CPUS", "HQ_PIN", "HQ_JOB_ID", "HQ_TASK_ID",
    "SLURM_JOB_ID", "SLURM_CPUS_ON_NODE", "SLURM_JOB_NODELIST", "LOCAL_DATA_DIR",
]


# Format core numbers as ranges, for example 0-2,5-6
def _ranges(cpus):
    cpus = sorted(cpus)
    out, start = [], None
    for i, c in enumerate(cpus):
        if start is None:
            start = c
        if i == len(cpus) - 1 or cpus[i + 1] != c + 1:
            out.append(f"{start}-{c}" if c != start else f"{c}")
            start = None
    return ",".join(out)


# Read the number of CPUs available to the process
def _nproc():
    try:
        return int(subprocess.run(["nproc"], capture_output=True, text=True, timeout=5).stdout.strip())
    except Exception:
        return None


# Read available node memory in GiB
def _mem_available_gib():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    return round(int(line.split()[1]) / 1024**2, 1)
    except Exception:
        pass
    return None


# Report assigned cores, thread settings and CPU affinity
def runtime_report(n_cores, label=""):
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    info = {
        "label": label,
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "n_cores_requested": n_cores,
        "nproc": _nproc(),
        "sched_getaffinity_count": len(affinity),
        "sched_getaffinity": _ranges(affinity),
        "os_cpu_count": os.cpu_count(),
        "mem_available_gib": _mem_available_gib(),
        "env": {k: os.environ.get(k) for k in THREAD_ENV_VARS},
    }
    try:
        import torch
        info["torch_num_threads"] = torch.get_num_threads()
        info["torch_num_interop_threads"] = torch.get_num_interop_threads()
    except Exception as e:
        info["torch_num_threads"] = f"unavailable ({e})"
    try:
        from threadpoolctl import threadpool_info
        info["threadpools"] = [
            {k: p.get(k) for k in ("user_api", "internal_api", "num_threads", "prefix")}
            for p in threadpool_info()
        ]
    except Exception as e:
        info["threadpools"] = f"unavailable ({e})"

    print("=" * 72)
    print(f"RUNTIME DIAGNOSTICS {label}")
    print(f"  host={info['hostname']}  python={info['python']}  mem_available={info['mem_available_gib']} GiB")
    print(f"  n_cores (requested)       = {n_cores}")
    print(f"  nproc                     = {info['nproc']}")
    print(f"  os.sched_getaffinity(0)   = {info['sched_getaffinity_count']} cpus [{info['sched_getaffinity']}]")
    print(f"  os.cpu_count()            = {info['os_cpu_count']}")
    print(f"  torch.get_num_threads()   = {info['torch_num_threads']}")
    for k, v in info["env"].items():
        print(f"  {k:<25} = {v}")
    if isinstance(info["threadpools"], list):
        for p in info["threadpools"]:
            print(f"  threadpool {p['internal_api']:<10} ({p['user_api']}) threads={p['num_threads']}")
    if info["nproc"] and n_cores and info["nproc"] != n_cores:
        print(f"  WARNING: nproc ({info['nproc']}) differs from n_cores ({n_cores}); check --cpus and pinning.")
    print("=" * 72, flush=True)
    return info


# Measure stage wall time, CPU time and CPU efficiency
class StageTimer:

    def __init__(self, n_cores, memory_monitor=None):
        self.n_cores = n_cores
        self.stages = {}
        self.memory = memory_monitor

    @staticmethod
    # Sum CPU time used by the process and its completed children
    def _cpu():
        s = resource.getrusage(resource.RUSAGE_SELF)
        c = resource.getrusage(resource.RUSAGE_CHILDREN)
        return s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime

    @contextmanager
    # Measure runtime, CPU efficiency and peak memory for one stage
    def stage(self, name):
        wall0, cpu0 = time.perf_counter(), self._cpu()
        if self.memory:
            self.memory.take_stage_peak_gib()
        print(f"[{time.strftime('%H:%M:%S')}] stage '{name}' started", flush=True)
        try:
            yield
        finally:
            wall = time.perf_counter() - wall0
            cpu = self._cpu() - cpu0
            eff = cpu / (wall * self.n_cores) if wall > 0 and self.n_cores else None
            mem = None
            if self.memory:
                self.memory.sample()
                mem = self.memory.take_stage_peak_gib()
            self.stages[name] = {"wall_seconds": round(wall, 2), "cpu_seconds": round(cpu, 2),
                                 "cpu_efficiency": round(eff, 3) if eff is not None else None,
                                 "peak_family_memory_gib": mem}
            print(f"[{time.strftime('%H:%M:%S')}] stage '{name}' done: wall={wall:.1f}s "
                  f"cpu={cpu:.1f}s efficiency={eff:.0%} of {self.n_cores} cores"
                  + (f" peak memory (task + workers)={mem} GiB" if mem is not None else ""), flush=True)

    # Collect stage timings and peak memory measurements
    def summary(self):
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2   # KiB -> GiB on Linux
        peak_children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024**2
        out = {"stages": self.stages,
               "peak_rss_gib": round(peak, 2),
               "peak_rss_largest_child_gib": round(peak_children, 2)}
        if self.memory:
            out["peak_family_memory_gib"] = round(self.memory.peak_kib / 1024**2, 2)
        return out


# Group process IDs by parent process
def _children_map():
    children = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
            children.setdefault(ppid, []).append(int(entry))
        except (OSError, ValueError, IndexError):
            pass
    return children


# Read proportional memory use, using RSS when PSS is unavailable
def _proportional_memory_kib(pid):
    for path, key in ((f"/proc/{pid}/smaps_rollup", "Pss:"), (f"/proc/{pid}/status", "VmRSS:")):
        try:
            with open(path) as f:
                for line in f:
                    if line.startswith(key):
                        return int(line.split()[1])
        except OSError:
            continue
    return 0


# Sample peak memory for the task and its child processes
class FamilyMemoryMonitor:

    def __init__(self, interval=5.0):
        self.interval = interval
        self.peak_kib = 0
        self.stage_peak_kib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    # Measure memory across the task and its child processes
    def sample(self):
        root = os.getpid()
        children = _children_map()
        total, stack = 0, [root]
        while stack:
            pid = stack.pop()
            total += _proportional_memory_kib(pid)
            stack.extend(children.get(pid, []))
        self.peak_kib = max(self.peak_kib, total)
        self.stage_peak_kib = max(self.stage_peak_kib, total)
        return total

    # Sample memory at the configured interval
    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                self.sample()
            except Exception:
                pass

    # Start memory sampling in the background
    def start(self):
        self.sample()
        self._thread.start()
        return self

    # Signal the memory sampling thread to stop
    def stop(self):
        self._stop.set()

    # Return the stage memory peak in GiB and reset it
    def take_stage_peak_gib(self):
        peak = self.stage_peak_kib
        self.stage_peak_kib = 0
        return round(peak / 1024**2, 2)


# Write the diagnostics to a JSON file
def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
