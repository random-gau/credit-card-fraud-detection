# Data and modelling decisions

| # | Decision | Why | Leakage risk addressed |
|---|---|---|---|
| 1 | Split 60/20/20 train/val/test, stratified on Class, seed 42 | Equal fraud rate per split; val reserved for threshold tuning; test touched once | Test never used for selection |
| 2 | Drop exact duplicate rows before splitting | Same transaction must not appear in train and test | Memorised rows inflating test scores |
| 3 | All scaling inside model Pipelines (fit on train / train fold only) | Scaler statistics must not see val/test | Preprocessing leakage |
| 4 | Amount -> log1p then StandardScaler; Time -> sin/cos of 24 h cycle; V1-V28 standardised (linear models only) | Skewed Amount; raw Time is near an ID; penalised LR needs comparable scales | None (row-wise transforms) |
| 5 | Random stratified split, not time-based | Standard benchmark for this dataset | Model trains on some later transactions: listed as a limitation |
