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
from sklearn.metrics import roc_auc_score, log_loss

TARGET = "liquidity_stress_next_30d"
ID_COL = "ID"

train = pd.read_csv("Train.csv")
test  = pd.read_csv("Test.csv")
sample = pd.read_csv("SampleSubmission.csv")
print("Train (ham):", train.shape, "| Test:", test.shape)

# --- Hedefi bos (NaN) olan satirlari egitimden cikar ---
n_before = len(train)
train = train.dropna(subset=[TARGET]).reset_index(drop=True)
train[TARGET] = train[TARGET].astype(int)
print(f"Hedefi bos {n_before - len(train)} satir atildi -> egitim: {len(train)} satir")
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
    # Yeni ozellikleri bir dict'te topla, sonunda TEK seferde ekle
    # (boylece "DataFrame is highly fragmented" uyarisi ve yavaslama olmaz)
    feats = {}

    # Her islem tipi + metrik icin 6 ay boyunca ozet istatistikler
    for tx in TX_TYPES:
        for metric in ["volume", "total_value", "highest_amount", COUNT_SUFFIX[tx]]:
            cols = [f"m{m}_{tx}_{metric}" for m in MONTHS if f"m{m}_{tx}_{metric}" in df.columns]
            if not cols:
                continue
            vals = df[cols]
            feats[f"{tx}_{metric}_sum"]           = vals.sum(axis=1)
            feats[f"{tx}_{metric}_mean"]          = vals.mean(axis=1)
            feats[f"{tx}_{metric}_std"]           = vals.std(axis=1)
            feats[f"{tx}_{metric}_max"]           = vals.max(axis=1)
            feats[f"{tx}_{metric}_trend"]         = df[cols[0]] - df[cols[-1]]  # m1(yeni)-m6(eski)
            feats[f"{tx}_{metric}_active_months"] = (vals > 0).sum(axis=1)

    # ---- Bakiye ozellikleri (daily_avg_bal) ----
    bal_cols = [f"m{m}_daily_avg_bal" for m in MONTHS if f"m{m}_daily_avg_bal" in df.columns]
    if bal_cols:
        bal = df[bal_cols]
        feats["bal_mean"]       = bal.mean(axis=1)
        feats["bal_std"]        = bal.std(axis=1)          # volatilite = stres sinyali
        feats["bal_min"]        = bal.min(axis=1)
        feats["bal_max"]        = bal.max(axis=1)
        feats["bal_trend"]      = df[bal_cols[0]] - df[bal_cols[-1]]  # dususte mi?
        feats["bal_low_months"] = (bal.lt(bal.mean(axis=1), axis=0)).sum(axis=1)

    # ---- Net nakit akisi (para giris - cikis) ----
    inflow  = feats.get("received_total_value_sum", 0) + feats.get("deposit_total_value_sum", 0) \
              + feats.get("transfer_from_bank_total_value_sum", 0)
    outflow = feats.get("mm_send_total_value_sum", 0) + feats.get("withdraw_total_value_sum", 0) \
              + feats.get("paybill_total_value_sum", 0) + feats.get("merchantpay_total_value_sum", 0)
    feats["total_inflow"]  = inflow
    feats["total_outflow"] = outflow
    feats["net_flow"]      = inflow - outflow
    feats["outflow_ratio"] = outflow / (inflow + 1)   # 1'e yakin/ustu = stres

    # Tek seferde birlestir
    return pd.concat([df, pd.DataFrame(feats, index=df.index)], axis=1)

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
# Metrik = Log Loss (%60) + ROC-AUC (%40). Log Loss baskin oldugu icin
# ONCE binary_logloss'a gore erken durdurma yapiyoruz (first_metric_only).
params = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],   # ilk metrik = logloss -> erken durdurma buna gore
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
        num_boost_round=5000,
        valid_sets=[lgb.Dataset(X.iloc[va], y.iloc[va])],
        callbacks=[lgb.early_stopping(200, first_metric_only=True),  # logloss'a gore dur
                   lgb.log_evaluation(0)],
    )
    oof[va]     = model.predict(X.iloc[va])
    test_preds += model.predict(X_test) / skf.n_splits
    print(f"Fold {fold+1} | LogLoss: {log_loss(y.iloc[va], oof[va]):.5f} "
          f"| AUC: {roc_auc_score(y.iloc[va], oof[va]):.5f}")

cv_ll  = log_loss(y, oof)
cv_auc = roc_auc_score(y, oof)
print(f"\n>>> Genel CV  |  LogLoss: {cv_ll:.5f}  (dusuk=iyi)  |  AUC: {cv_auc:.5f}  (yuksek=iyi)")

# ---- Onemli ozellikler (bilgi amacli) ------------------------------
imp = pd.Series(model.feature_importance(), index=features).sort_values(ascending=False)
print("\nEn onemli 15 ozellik:\n", imp.head(15))

# ---- Submission (OLASILIK) -----------------------------------------
pred_col = [c for c in sample.columns if c != ID_COL][0]   # = "Target"
sub = sample.copy()
sub[pred_col] = sub[ID_COL].map(dict(zip(test[ID_COL], test_preds)))
sub.to_csv("submission.csv", index=False)
print("\nsubmission.csv hazir:\n", sub.head())
