# =====================================================================
# Zindi — Financial Stress Prediction Challenge (July Starter Track)
# GELISMIS VERSIYON — Feature Engineering + LightGBM 5-fold CV
#
# Hedef : liquidity_stress_next_30d (binary)
# Cikti : Target sutununa OLASILIK (metrik = AUC / LogLoss)
# Not   : M1 = en yeni ay, M6 = en eski ay
#
# Colab: once Train.csv, Test.csv, SampleSubmission.csv yukle, sonra calistir.
# =====================================================================

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

TARGET = "liquidity_stress_next_30d"
ID_COL = "ID"

train = pd.read_csv("Train.csv")
test  = pd.read_csv("Test.csv")
sample = pd.read_csv("SampleSubmission.csv")
print("Train:", train.shape, "| Test:", test.shape)
print("Hedef dagilimi:\n", train[TARGET].value_counts(normalize=True).round(4))

# ---- Feature engineering -------------------------------------------
# Islem tipleri ve her ay olculen 4 metrik ailesi
TX_TYPES = ["paybill", "merchantpay", "transfer_from_bank",
            "mm_send", "received", "deposit", "withdraw"]
# her tip icin sayac sutununun adi farkli (companies/merchants/banks/...)
COUNT_SUFFIX = {
    "paybill": "companies", "merchantpay": "merchants",
    "transfer_from_bank": "banks", "mm_send": "recipients",
    "received": "senders", "deposit": "agents", "withdraw": "agents",
}
MONTHS = [1, 2, 3, 4, 5, 6]

def add_features(df):
    df = df.copy()

    # Her islem tipi + metrik icin 6 ay boyunca ozet istatistikler
    for tx in TX_TYPES:
        for metric in ["volume", "total_value", "highest_amount", COUNT_SUFFIX[tx]]:
            cols = [f"m{m}_{tx}_{metric}" for m in MONTHS]
            cols = [c for c in cols if c in df.columns]
            if not cols:
                continue
            vals = df[cols]
            df[f"{tx}_{metric}_sum"]   = vals.sum(axis=1)
            df[f"{tx}_{metric}_mean"]  = vals.mean(axis=1)
            df[f"{tx}_{metric}_std"]   = vals.std(axis=1)
            df[f"{tx}_{metric}_max"]   = vals.max(axis=1)
            # trend: en yeni ay (m1) - en eski ay (m6)
            df[f"{tx}_{metric}_trend"] = df[cols[0]] - df[cols[-1]]
            # aktif ay sayisi (sifir olmayan)
            df[f"{tx}_{metric}_active_months"] = (vals > 0).sum(axis=1)

    # ---- Bakiye ozellikleri (daily_avg_bal) ----
    bal_cols = [f"m{m}_daily_avg_bal" for m in MONTHS if f"m{m}_daily_avg_bal" in df.columns]
    if bal_cols:
        bal = df[bal_cols]
        df["bal_mean"]  = bal.mean(axis=1)
        df["bal_std"]   = bal.std(axis=1)          # volatilite = stres sinyali
        df["bal_min"]   = bal.min(axis=1)
        df["bal_max"]   = bal.max(axis=1)
        df["bal_trend"] = df[bal_cols[0]] - df[bal_cols[-1]]  # dususte mi?
        df["bal_low_months"] = (bal < bal.mean(axis=1).values[:, None]).sum(axis=1)

    # ---- Net nakit akisi (para giris - cikis) ----
    inflow  = df.get("received_total_value_sum", 0) + df.get("deposit_total_value_sum", 0) \
              + df.get("transfer_from_bank_total_value_sum", 0)
    outflow = df.get("mm_send_total_value_sum", 0) + df.get("withdraw_total_value_sum", 0) \
              + df.get("paybill_total_value_sum", 0) + df.get("merchantpay_total_value_sum", 0)
    df["total_inflow"]  = inflow
    df["total_outflow"] = outflow
    df["net_flow"]      = inflow - outflow
    df["outflow_ratio"] = outflow / (inflow + 1)   # 1'e yakin/ustu = stres

    return df

train = add_features(train)
test  = add_features(test)

# ---- Kategorikleri sayiya cevir ------------------------------------
cat_cols = ["gender", "region", "smartphone", "segment", "earning_pattern"]
for c in cat_cols:
    train[c] = train[c].astype("category")
    test[c]  = pd.Categorical(test[c], categories=train[c].cat.categories)
    train[c] = train[c].cat.codes
    test[c]  = test[c].cat.codes

features = [c for c in train.columns if c not in [TARGET, ID_COL]]
X, y, X_test = train[features], train[TARGET], test[features]
print("Toplam ozellik sayisi:", len(features))

# ---- LightGBM 5-fold CV --------------------------------------------
params = {
    "objective": "binary", "metric": "auc",
    "learning_rate": 0.02, "num_leaves": 64,
    "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1,
    "min_child_samples": 60, "lambda_l1": 1.0, "lambda_l2": 1.0,
    "verbose": -1, "seed": 42,
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))
oof = np.zeros(len(X))

for fold, (tr, va) in enumerate(skf.split(X, y)):
    model = lgb.train(
        params,
        lgb.Dataset(X.iloc[tr], y.iloc[tr]),
        num_boost_round=3000,
        valid_sets=[lgb.Dataset(X.iloc[va], y.iloc[va])],
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)],
    )
    oof[va]     = model.predict(X.iloc[va])
    test_preds += model.predict(X_test) / skf.n_splits
    print(f"Fold {fold+1} AUC: {roc_auc_score(y.iloc[va], oof[va]):.5f}")

print(f"\n>>> Genel CV AUC: {roc_auc_score(y, oof):.5f}")

# ---- Onemli ozellikler (bilgi amacli) ------------------------------
imp = pd.Series(model.feature_importance(), index=features).sort_values(ascending=False)
print("\nEn onemli 15 ozellik:\n", imp.head(15))

# ---- Submission (OLASILIK) -----------------------------------------
pred_col = [c for c in sample.columns if c != ID_COL][0]   # = "Target"
sub = sample.copy()
sub[pred_col] = sub[ID_COL].map(dict(zip(test[ID_COL], test_preds)))
sub.to_csv("submission.csv", index=False)
print("\nsubmission.csv hazir:\n", sub.head())
