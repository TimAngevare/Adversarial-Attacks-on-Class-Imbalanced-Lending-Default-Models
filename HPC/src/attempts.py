# Track task attempts and record which stages produced valid outputs.
import json
import os
import shutil
import socket
import time

PIPELINE_VERSION = "v2"

STAGE_FILES = {
    "cv": ["metrics.csv"],
    "shap": ["shap.csv"],
    "fairness": ["fairness.csv", "final_metrics.csv"],
    "attacks": ["attack_summary.csv", "attack_records.parquet"],
    "pricing": ["pricing.csv"],
}


# Create a run ID from the timestamp, job IDs and process ID
def new_run_id():
    parts = [time.strftime("%Y%m%d-%H%M%S"),
             os.environ.get("SLURM_JOB_ID", "noslurm"),
             f"hq{os.environ.get('HQ_JOB_ID', '-')}.{os.environ.get('HQ_TASK_ID', '-')}",
             f"pid{os.getpid()}"]
    return "_".join(parts)


# Write metadata through a temporary file and replace it atomically
def _write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)   # never leaves a half-written meta.json


class Attempt:
    def __init__(self, out_dir, identity, stages):
        self.out_dir = out_dir
        self.meta_path = os.path.join(out_dir, "meta.json")
        self.run_id = new_run_id()
        os.makedirs(out_dir, exist_ok=True)

        identity = {**identity, "pipeline_version": PIPELINE_VERSION}
        meta = {}
        if os.path.exists(self.meta_path):
            with open(self.meta_path) as f:
                meta = json.load(f)
        if meta.get("identity") != identity:
            # Archive the task directory when its configuration or version differs
            self._move_all_to(f"stale_{time.strftime('%Y%m%d-%H%M%S')}")
            meta = {}

        # Archive the preceding attempt's error file
        error_path = os.path.join(out_dir, "error.txt")
        if os.path.exists(error_path):
            os.makedirs(os.path.join(out_dir, "errors"), exist_ok=True)
            previous = (meta.get("attempts") or [{}])[-1].get("run_id", "unknown")
            shutil.move(error_path, os.path.join(out_dir, "errors", f"error_{previous}.txt"))

        records = meta.get("stage_records", {})
        for stage in stages:
            records.pop(stage, None)
            for name in STAGE_FILES.get(stage, []):
                path = os.path.join(out_dir, name)
                if os.path.exists(path):
                    os.remove(path)

        attempts = meta.get("attempts", [])
        attempts.append({"run_id": self.run_id, "host": socket.gethostname(),
                         "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                         "stages": list(stages), "status": "running"})
        self.meta = {**meta, "identity": identity, "stage_records": records,
                     "attempts": attempts, "status": "running", "current_run_id": self.run_id}
        self.save()

    # Archive task files when the recorded identity no longer matches
    def _move_all_to(self, folder):
        entries = [e for e in os.listdir(self.out_dir) if not e.startswith("stale_")]
        if not entries:
            return
        target = os.path.join(self.out_dir, folder)
        os.makedirs(target, exist_ok=True)
        for e in entries:
            shutil.move(os.path.join(self.out_dir, e), os.path.join(target, e))

    def save(self):
        _write(self.meta_path, self.meta)

    def update(self, **fields):
        self.meta.update(fields)
        self.save()

    # Update completed-stage timings and peak memory in meta.json
    def record_timings(self, summary):
        stages = dict(self.meta.get("stages", {}))
        for name, values in summary.get("stages", {}).items():
            stages[name] = {**values, "run_id": self.run_id}
        rest = {k: v for k, v in summary.items() if k != "stages"}
        self.meta.update(rest, stages=stages)
        peaks = [v.get("peak_family_memory_gib") for v in stages.values()
                 if v.get("peak_family_memory_gib") is not None]
        if peaks:
            self.meta["peak_family_memory_gib"] = max(peaks)
        self.save()

    # Check stage outputs and record successful completion
    def stage_done(self, stage, **details):
        missing = [n for n in STAGE_FILES.get(stage, []) if not os.path.exists(os.path.join(self.out_dir, n))]
        if missing:
            raise RuntimeError(f"Stage {stage} finished without writing {missing}")
        self.meta["stage_records"][stage] = {"run_id": self.run_id,
                                             "finished": time.strftime("%Y-%m-%d %H:%M:%S"), **details}
        self.save()

    # Save the final attempt status and any error details
    def finish(self, status, error=None):
        self.meta["status"] = status
        self.meta["attempts"][-1].update(status=status, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        if error:
            self.meta["attempts"][-1]["error"] = error.strip().splitlines()[-1] if error.strip() else ""
            with open(os.path.join(self.out_dir, "error.txt"), "w") as f:
                f.write(error)
        self.save()
