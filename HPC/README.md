# HPC pipeline (Snellius, SLURM + HyperQueue)

This folder runs the experiments used in the paper on Snellius. `Notebook_Final_submission.ipynb` loads the collected results for analysis and figure generation. The pipeline prepares the data, tunes the models, runs the experiments and collects their outputs. Run the commands below from this `HPC/` folder.

## Phases

| Phase | Script | Runs as | What it does |
|---|---|---|---|
| A | `setup.py` (`setup_job.sh`) | 1 exclusive genoa job | Data preparation as in notebook Chapters 2-3; fixed borrower-level train/test pools; five prevalence datasets; grid search on the training pool of the 10% dataset |
| B | `main.py` (`hq_task.sh`) | 100 HQ tasks, 16 cores each | One experiment (prevalence × model × sampler): CV + SHAP, final model + fairness, attacks |
| B' | `sensitivity.py` (`hq_task.sh sensitivity.py`) | 30 HQ tasks, 4 cores each | Pricing sensitivity of notebook Section 6.1.6 |
| Test | `smoke_job.sh`, `scaling_job.sh` | shared genoa jobs | Parallel vs serial attacks, memory per model; scaling with 1-8 cores |
| C | `aggregate.py` | login node | Collects results, checks them, writes `outputs/`, task report and quick-look plots |

### Phase A outputs (`data/processed/`, `config/`)

`X_full_{frac}.parquet`, `y_full_{frac}.parquet`, `pricing_{frac}.parquet` (index = borrower id, the row number in `accepted.csv`), `train_ids.parquet`, `test_ids.parquet`, `bias_{frac}.json`, `feature_names.json`, `preprocessing_report.json` (dropped columns per pruning step, implied interest-rate check, pool sizes), `config/best_params.json`.

### Phase B: one task (`results_v2/exp_XXXX/`)

Each experiment writes its outputs to `results_v2/exp_XXXX/`. Collection checks the task identity and completed-stage records before reading those files. Smoke and scaling tests use separate folders under `results/`.

```
main.py --experiment_id=N
  ├─ runtime diagnostics: nproc, os.sched_getaffinity(0), torch.get_num_threads(),
  │  OMP_NUM_THREADS and the other thread variables, HQ_CPUS, threadpools   (stdout + meta.json)
  ├─ cv:        5-fold CV on the training pool          -> metrics.csv
  ├─ shap:      last CV fold                            -> shap.csv
  ├─ final_fit: training pool minus 20% validation split, F1-optimal threshold
  ├─ fairness:  DIR, TPR/FPR gaps, bootstrap intervals on the test pool
  │                                                     -> fairness.csv, final_metrics.csv
  └─ attacks:   grid on every eligible default (+ exact reference for Logistic Regression),
                GA and ZOO on 100 sampled defaults, in parallel
                                                        -> attack_summary.csv, attack_records.parquet
meta.json: identity (experiment, configuration, pipeline version), one entry per
           attempt (run id, host, stages, status, last error line), stage records
           (which run wrote each stage's files), stage wall time, CPU time and
           CPU efficiency, peak memory, attack workers, KMeansSMOTE fallbacks.
error.txt: traceback of the latest attempt, only if that attempt failed.
```

### Attempts, retries and stale files (`src/attempts.py`)

Every run of `main.py` or `sensitivity.py` is an attempt with its own run id (time, SLURM job, HQ job and task, pid). At the start of an attempt:

- if `meta.json` describes another configuration or an older pipeline version, the whole folder moves to `stale_<time>/`;
- an `error.txt` from an earlier attempt moves to `errors/error_<run id>.txt`, so a retry that succeeds is reported as `ok`;
- the output files and stage records of the stages it is about to run are deleted.

A stage gets its record only after all its files exist. `aggregate.py` reads a file only if its stage has a record and the folder's identity matches the task id; anything else is listed under "Ignored or inconsistent files". `outputs/provenance.csv` lists, for every collected file, the task folder, stage and run id that produced it (kept separate so that `outputs/*.csv` have exactly the notebook's columns).

Experiment id → configuration: `id = 20 × prevalence index + 5 × model index + sampler index`, with prevalences `0.01, 0.02, 0.05, 0.1, 0.2`, models `XGBClassifier, LogisticRegression, RandomForestClassifier, NeuralNetClassifier` and samplers `Baseline, ClassWeight, RandomUnderSampler, SMOTEENN, KMeansSMOTE`. This is the notebook's loop order, so each task uses the notebook's attack seeds (configuration number = id + 1).

### What runs in parallel

- **Across experiments:** 100 HQ tasks share a 192-core genoa node. With 16 cores per task, CPU capacity allows up to 12 tasks; memory requests can reduce concurrency.
- **Inside a task:** model training and prediction use the task's cores (XGBoost `n_jobs`, Random Forest `n_jobs`, torch threads, resamplers `n_jobs`). CV folds run sequentially.
- **Attacks:** GA and ZOO distribute sampled borrowers across worker processes, using one model thread per worker. Seeds are assigned per borrower before distributing jobs. XGBoost, Logistic Regression and neural-network workers use `spawn`; Random Forest workers use `fork` to share model memory. `ATTACK_START_METHOD` sets these choices. If the pool fails, the task records `attack_pool_error` in `meta.json` and completes the searches serially. The grid and LR references evaluate batches in the main process using the task's model threads. `smoke_job.sh` compares parallel and serial results before the full run.

Running attacks as separate HQ tasks would mean fitting every final model twice or writing models of several GB to disk, so they stay inside the experiment task.

## Running it

```bash
# Run from HPC/ on the login node
module load 2023 && module load HyperQueue/0.19.0

# Create the log directory before submitting jobs
mkdir -p logs

# 1. Phase A (once): preprocessing + grid search
sbatch setup_job.sh

# 2. Before the full run: smoke test and scaling test (see below)
sbatch smoke_job.sh
sbatch scaling_job.sh

# 3. HyperQueue server (keep running)
nohup hq server start &

# 4. Submit experiment and pricing sensitivity arrays
bash submit_experiments.sh

# 5. Workers on a genoa node
sbatch hyperq_job_hpc.sh

# 6. After all tasks finished
python3.11 aggregate.py
```

Re-running one experiment (for example after a timeout) is `hq submit --array <id> --cpus=16 --resource mem=<MB> ... hq_task.sh` with the model's values from `submit_experiments.sh`. The attempt records determine which stage outputs remain valid (see Attempts above). To rerun only some stages and keep the files of the others, add `--env STAGES="final attacks"` to `hq submit` (`hq_task.sh` passes it to `main.py --stages`).

Every job script stops with a non-zero exit code when a module, the virtual environment or the Python step fails: `setup_job.sh` prints `Setup complete.` only after `setup.py` succeeded, `hyperq_job_hpc.sh` returns the exit code of the HQ worker, `hq_task.sh` returns the exit code of `main.py`/`sensitivity.py`, so HQ marks the task as failed.

### Memory

`hyperq_job_hpc.sh` gives the worker a `mem` scheduling resource of 330,000 MB (`WORKER_MEM_MB`). `submit_experiments.sh` requests memory per task: XGBoost 20,000 MB, Logistic Regression 16,000 MB, Random Forest 48,000 MB and neural network 24,000 MB. HyperQueue starts a task when both its core and memory requests fit. These requests guide scheduling and are not enforced limits. After the smoke test, compare them with the measured peak for each model, convert units where needed and allow about 20% headroom.

Peak memory is measured for the whole task: `FamilyMemoryMonitor` sums the proportional set size (PSS) of the task and all its child processes (attack workers, joblib workers) every 5 s. PSS divides shared pages among the processes that share them, so forked Random Forest workers are not counted 16 times. `meta.json` and `task_report.csv` give the peak per stage and for the whole task; `peak_rss_gib` (main process only) is kept for comparison.

## Checking a run

| When | Command | Shows |
|---|---|---|
| Start of every task | `logs/<id>.out` (top) | `nproc`, `os.sched_getaffinity(0)`, `torch.get_num_threads()`, `OMP_NUM_THREADS`, `HQ_CPUS`, BLAS/OpenMP thread pools. A warning is printed if `nproc` differs from the task's cores |
| During the run | `squeue -u $USER`, `hq job progress <hq_job>` | queue and task progress |
| During the run | `bash watch_node.sh <slurm_jobid>` | live `top -H` on the worker node (`srun --overlap` into the running job) |
| During the run | `logs/monitor_<slurm_jobid>.log` | every 10 min: load average, memory, `top -H` (threads), per-process threads/CPU/RSS, HQ jobs |
| During the run | `logs/<id>.out` | stage start/end with wall time and CPU efficiency, attack timings |
| After the run | `bash post_run_report.sh <slurm_jobid> [hq_job ...]` | `seff` (CPU efficiency of the worker job), `sacct`, failed HQ tasks of every HQ job (or the ones given), then `aggregate.py` |
| After the run | `outputs/task_report.csv` | per task: status, number of attempts, run id, completed stages, host, cores, `nproc`, torch threads, wall time, CPU efficiency and peak memory (task + workers) per stage, attack workers and start method, KMeansSMOTE fallbacks, last error line |
| After the run | `outputs/provenance.csv` | which run wrote every collected file |

For a task assigned 16 cores, check for `nproc = 16`, 16 cpus in the affinity list, `torch.get_num_threads() = 16` and `OMP_NUM_THREADS = 16`. In `top -H`, each task should have about 16 busy threads; many more than 16 threads per task at 100% means oversubscription. A low CPU efficiency in the `attacks` stage with `attack workers = 1` means the pool fell back to serial (see `attack_pool_error` in `meta.json`).

## Smoke test (parallel attacks, memory)

```bash
sbatch smoke_job.sh                    # experiments 80, 85, 90, 95: each model, Baseline, 20% prevalence
sbatch smoke_job.sh "90" 16 32         # only Random Forest, 16 workers, 32 GA/ZOO borrowers
python3.11 compare_attack_runs.py      # rerun the comparison on results/smoke/
```

For one experiment per model, the job fits the final model and runs the attacks twice on 32 GA/ZOO borrowers: with 16 workers and serially. `compare_attack_runs.py` fails the job unless both runs completed, the pool used its configured start method (fork for Random Forest) without falling back, and the attack records agree per method and borrower (success, changed features, queries; scores within 1e-8). It also prints the GA/ZOO speed-up and the peak memory of task plus workers, which sets the memory requests in `submit_experiments.sh`. The 20% prevalence condition provides a high-row-count case for checking memory use.

## Scaling test (XGBoost with 1, 2, 4 and 8 cores)

```bash
sbatch scaling_job.sh                  # experiment 60 = XGBoost + Baseline at 10% prevalence
sbatch scaling_job.sh 60 "1 2 4 8 16"  # other id or core counts
python3.11 scaling_report.py --exp_id 60
```

The job asks for 24 cores (a 1/8 genoa node, the smallest shared unit, so it costs the same as 8 cores) and runs the experiment once per core count, each pinned with `taskset` to its own cores and writing to `results/scaling/`. `scaling_report.py` prints wall seconds per stage, the speed-up relative to 1 core and the parallel efficiency (speed-up / cores). Use the stage timings and efficiency to assess whether additional cores improve runtime enough to justify their allocation.

Each run starts in an empty folder and its exit code goes to `results/scaling/status_exp_<id>.tsv`. A failed run does not stop the others, but `scaling_report.py` uses only completed runs (exit code 0, status `ok`, every stage recorded), lists the others under "runs left out", and computes speed-up only if the 1-core run completed. The job exits non-zero if any run failed.

## Configuration (`src/config.py`)

| Constant | Default | Meaning |
|---|---|---|
| `MINORITY_FRACS` | `[0.01, 0.02, 0.05, 0.1, 0.2]` | prevalences |
| `TEST_SIZE` / `VALIDATION_SIZE` | `0.20` / `0.20` | fixed test pool / validation split of the training pool |
| `TUNING_FRACTION` | `0.1` | prevalence used for the grid search |
| `N_ATTACK_CASES` | `100` | GA/ZOO borrowers per experiment (`None` = all eligible) |
| `GRID_LEVELS` | `21` | grid attack levels per feature (1,261 candidates) |
| `SEPARABILITY_TOL` | `1e-6` | tolerance (log-odds) for the exact reference check and for ties in the attack checks |
| `ZOO_SKIP_MODELS` | empty | models for which ZOO is skipped (the notebook runs ZOO everywhere) |
| `ATTACK_START_METHOD` | spawn; fork for Random Forest | how GA/ZOO workers start |
| `KMEANS_SMOTE_DENSITY_EXPONENT` | `2` | same as the notebook (Section 5.1) |
| `KMEANS_SMOTE_ALLOW_FALLBACK` | `False` | a failed KMeans-SMOTE fit fails the task; see below |
| `RESULTS_DIR` | `results_v2` | task folders |
| `PRICING_*` | 3 variants, Logistic Regression, Baseline + ClassWeight | pricing sensitivity grid |

## KMeans-SMOTE and the exact reference

**KMeans-SMOTE** uses `MiniBatchKMeans` with 30 clusters, `cluster_balance_threshold=0.01` and density exponent 2. The fixed exponent avoids unstable cluster weights on this dataset. A failed fit fails the task by default. If `KMEANS_SMOTE_ALLOW_FALLBACK` is enabled, a failed fit uses plain SMOTE and receives the separate label `KMeansSMOTE (SMOTE fallback)`. The task report records these replacements.

**LR reference.** The coefficient-based calculation separates the log-odds into terms for each mutable feature and its dependent feature. It evaluates allowed endpoints and stationary points, then combines up to two beneficial changes. Model predictions verify the calculated log-odds within `SEPARABILITY_TOL`. This gives the optimum within the implemented action space, subject to numerical tolerances. Notebook Section 7.4 explains the reference, and its synthetic tests compare it with a fine grid.

## Differences from the notebook

- **KMeansSMOTE:** none (see above).

- **Neural network CV:** HPC tasks set the torch seed using `RANDOM_STATE`. The notebook's CV path does not explicitly set this seed, so repeated neural-network fits can differ. The paper figures use the supplied HPC results.
