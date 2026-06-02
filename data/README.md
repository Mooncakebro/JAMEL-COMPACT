# Data Directory

This directory is reserved for local datasets and generated parquet files.

Expected release layout:

```text
data/
  react/
    react-text/<app>/trajectory.parquet
    react-vision/<app>/trajectory.parquet
  jamel_sft_data/
    jamel_memory_sft_train.parquet
    jamel_memory_sft_val.parquet
  jamel_react_full_sft/
    jamel_memory_sft_train.parquet
    jamel_memory_sft_val.parquet
```

Large parquet datasets are intentionally not committed. Download or generate
them locally, then pass paths with `REACT_ROOT`, `TRAIN_FILE`, or `VAL_FILE`.
