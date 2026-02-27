<p align="center">
  <img width="100rem" alt="SurvHub Logo" src="https://github.com/taltstidl/survhub/blob/main/assets/logo.svg" />
  <h1 align="center">SurvHub: A Survival Analysis Benchmark</h1>
</p>

SurvHub is a living benchmarking ecosystem for survival analysis from tabular data. It is currently limited to time-invariant (static) covariates and time-independent estimates — although we aim to extend it to time-varying (dynamic) covariates and time-dependent estimates in future updates.

![SurvHub Leaderboard](https://github.com/taltstidl/survhub/blob/main/assets/leaderboard.svg)

## Running benchmarks

Execute the following command to evaluate a single model on a single dataset using 5-fold cross-validation, repeated with five different seeds:

```bash
python bench.py -m model_name -d dataset_name [--tuned]
```

* `model_name` is the name of the machine learning model (currently supports `coxnet`, `rsf`, `gbse`, `ssvm`, `deepsurv`, `rankdeepsurv`, `deepweisurv1`, `deepweisurv2`, `deephit`, `tabpfn`, and `popsicl`)
* `dataset_name` is either the name or the zero-based index of the dataset (for valid names and indices refer to `summary.csv` supplied in this repository)
* `--tuned` is an optional flag that determines whether hyperparameters should be tuned, only valid for classic machine learning models

If the runtime exceeds server-defined limits, you may alternatively use the following command to evaluate each fold separately:

```bash
for i in {1..25}
do
    python bench.py -m model_name -d dataset_name [--tuned] -i "$i"
done
```

Of course, you will need to adapt this depending on how your cluster is set up, e.g., by submitting a separate Slurm job for each fold.

## Plotting results

The benchmark results are published in this repository in the `benchmark` folder. Execute the following command to generate the plots used in the paper (currently under revision):

```bash
python plot.py
```

This will save all plots as PDF files and export leaderboards as CSV files. The leaderboard relies on the benchmark evaluation scripts provided by [TabArena](https://github.com/autogluon/tabarena/tree/main/bencheval).