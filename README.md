# Adversarial Attacks on Class-Imbalanced Lending-Default Models

The final notebook analyses predictive performance, fairness and adversarial robustness using the supplied HPC results. Model fitting and attack execution are disabled by default.

## Run the notebook

1. Open `Notebook_Final_submission.ipynb` from the project root.
2. If packages are missing, uncomment the installation command in Section 1.2.1 and run it once. Comment it out again and restart the kernel.
3. Keep the execution settings below for local preprocessing and analysis.
4. Run the notebook. Section 1.2.2 downloads the accepted-loans data directly from Kaggle, imports it into pandas and reuses the supplied HPC results for analysis. No manual CSV download is required.

```python
RUN_DATA_PREPARATION = True
RUN_EXPERIMENTS = False
RUN_SYNTHETIC_TESTS = False
```

## Dataset

Source: [All Lending Club loan data on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club), by wordsforthewise.

The standard workflow retrieves the raw data directly through `kagglehub`. With `RUN_DATA_PREPARATION = True`, the notebook downloads only `accepted_2007_to_2018Q4.csv.gz` from dataset version 3, extracts it to `Data/accepted.csv` and loads it into pandas. The `Data/` folder and CSV are created automatically. Later runs reuse the local CSV; deleting it causes the notebook to recreate it from the Kaggle download or its cache.

The compressed file is about 393 MB and expands to about 1.68 GB. These raw files are excluded from Git. The first download needs an internet connection and enough space for the compressed file and extracted CSV. KaggleHub stores the compressed download in its local cache.

If Kaggle requests authentication or consent, follow the [official KaggleHub instructions](https://github.com/Kaggle/kagglehub#authenticate).

## Supplied HPC results

Keep these files in `outputs/`:

- `model_results.csv`
- `shap_importances.csv`
- `fairness_results.csv`
- `final_model_test_metrics.csv`
- `attack_summary.csv`
- `attack_records.parquet`
- `pricing_sensitivity.csv`

The HPC implementation is the source of the reported results. Instructions for reproducing the experiments are in [the HPC README](HPC_updated_submission/HPC/README.md).

Chapter 9 saves the paper figures to `figures/`. The evaluation chapters save additional plots and tables to `notebook_analysis/`.

## Files excluded from Git

The `.gitignore` excludes the raw dataset, temporary experiment folders, caches, local credentials, `task_report.csv` and `provenance.csv`. These files can remain on your computer. The scientific result files listed above remain included.
