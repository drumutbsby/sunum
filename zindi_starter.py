# =====================================================================
# Zindi — Financial Stress Prediction Challenge (July Starter Track)
# Hedef: liquidity_stress_next_30d  (ikili sınıflandırma: 0 / 1)
# Model: LightGBM + 5-fold Cross Validation
# Google Colab'da calisir. Once Train.csv, Test.csv, SampleSubmission.csv yukle.
# =====================================================================

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# ---- 1) Veriyi oku --------------------------------------------------
train = pd.read_csv("Train.csv")
test  = pd.read_csv("Test.csv")
sample = pd.read_csv("SampleSubmission.csv")

TARGET = "liquidity_stress_next_30d"
ID_COL = "ID"

print("Train:", train.shape, "| Test:", test.shape)
print("\nHedef dagilimi (0 vs 1):")
print(train[TARGET].value_counts(normalize=True).round(4))

# ---- 2) Kategorik sutunlari sayiya cevir ----------------------------
cat_cols = ["gender", "region", "smartphone", "segment", "earning_pattern"]

for c in cat_cols:
    # train + test'i birlestirip ortak kategori kodlamasi yap
    train[c] = train[c].astype("category")
    test[c]  = pd.Categorical(test[c], categories=train[c].cat.categories)
    train[c] = train[c].cat.codes
    test[c]  = test[c].cat.codes

# ---- 3) Ozellik listesi (ID ve hedef haric) -------------------------
features = [col for col in train.columns if col not in [TARGET, ID_COL]]
X = train[features]
y = train[TARGET]
X_test = test[features]

# ---- 4) 5-fold CV ile LightGBM egit --------------------------------
params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 50,
    "verbose": -1,
    "seed": 42,
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))
oof = np.zeros(len(X))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    dtr  = lgb.Dataset(X.iloc[tr_idx], y.iloc[tr_idx])
    dval = lgb.Dataset(X.iloc[val_idx], y.iloc[val_idx])

    model = lgb.train(
        params, dtr,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )

    oof[val_idx] = model.predict(X.iloc[val_idx])
    test_preds  += model.predict(X_test) / skf.n_splits
    print(f"Fold {fold+1} AUC: {roc_auc_score(y.iloc[val_idx], oof[val_idx]):.5f}")

print(f"\n>>> Genel CV AUC: {roc_auc_score(y, oof):.5f}")

# ---- 5) Submission dosyasi olustur ----------------------------------
# SampleSubmission'in sutun adlarini otomatik kullan (format garantisi)
sub = sample.copy()
pred_col = [c for c in sample.columns if c != ID_COL][0]  # hedef sutun adi

# ID sirasini SampleSubmission ile eslestir
test_map = dict(zip(test[ID_COL], test_preds))
sub[pred_col] = sub[ID_COL].map(test_map)

# NOT: Cogu Zindi yarismasi OLASILIK ister (AUC/LogLoss icin) -> boyle birak.
# Eger 0/1 ETIKET istiyorsa asagidaki satirin basindaki # isaretini kaldir:
# sub[pred_col] = (sub[pred_col] > 0.5).astype(int)

sub.to_csv("submission.csv", index=False)
print("\nsubmission.csv hazir! Ilk satirlar:")
print(sub.head())
