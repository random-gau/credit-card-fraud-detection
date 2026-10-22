# Data and modelling decisions

| # | Decision | Why | Leakage risk addressed |
|---|---|---|---|
| 1 | Split 60/20/20 train/val/test, stratified on Class, seed 42 | Equal fraud rate per split; val reserved for threshold tuning; test touched once | Test never used for selection |
| 2 | Drop exact duplicate rows before splitting | Same transaction must not appear in train and test | Memorised rows inflating test scores |
| 3 | All scaling inside model Pipelines (fit on train / train fold only) | Scaler statistics must not see val/test | Preprocessing leakage |
| 4 | Amount -> log1p then StandardScaler; Time -> sin/cos of 24 h cycle; V1-V28 standardised (linear models only) | Skewed Amount; raw Time is near an ID; penalised LR needs comparable scales | None (row-wise transforms) |
| 5 | Random stratified split, not time-based | Standard benchmark for this dataset | Model trains on some later transactions: listed as a limitation |
| 6 | Use 5-fold CV on train for strategy and model selection; validation only for thresholds; test once | Val and test have only ~95 frauds each, so single-split scores are noisy | Keeps val and test independent of selection |
| 7 | Drop SMOTE and undersampling; keep class weights for the baseline | CV PR-AUC: none 0.736, SMOTE 0.736, class weights 0.734, undersampling 0.504 (worst on every fold) | Samplers were inside the CV pipeline, so this comparison is leak-free |
| 8 | XGBoost as best model; scale_pos_weight searched | Beat logistic regression on 5/5 CV folds (0.850 vs 0.734). Train PR-AUC 1.000: memorises training frauds, noted as a limitation | Search on train folds only |
| 9 | Threshold = highest recall with precision >= 0.80 on validation, per model | No cost model; 0.80 is a stated assumption. XGBoost 0.094, logistic regression 0.99999 | Tuned on val, frozen before test |
| 10 | Test set used once (guard file); not re-tuned after seeing that t=0.5 gave higher test F1 | Re-tuning after seeing test results would turn test into a second validation set | Test leakage |
| 11 | SHAP via XGBoost's own TreeSHAP (pred_contribs), additivity verified | Exact, and avoids shap/xgboost version mismatches | None (no fitting) |
