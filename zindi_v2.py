# =====================================================================
# Zindi — Financial Stress Prediction  |  v2 (ILK IYILESTIRME)
# Yeni: zengin ozellikler (momentum, oranlar, bakiye kalibrasyonu)
#       + 3-seed averaging (varyans dusurur -> LogLoss & AUC iyilesir)
# Metrik: LogLoss (%60) + ROC-AUC (%40) -> cikti = OLASILIK
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

n0 = len(train)
train = train.dropna(subset=[TARGET]).reset_index(drop=True)
train[TARGET] = train[TARGET].astype(int)
print(f"Egitim: {len(train)} satir (atilan: {n0-len(train)}) | Test: {len(test)}")
print("Hedef:", dict(train[TARGET].value_counts(normalize=True).round(3)))

TX_TYPES = ["paybill", "merchantpay", "transfer_from_bank",
            "mm_send", "received", "deposit", "withdraw"]
COUNT_SUFFIX = {"paybill":"companies","merchantpay":"merchants","transfer_from_bank":"banks",
                "mm_send":"recipients","received":"senders","deposit":"agents","withdraw":"agents"}
MONTHS = [1,2,3,4,5,6]

def add_features(df):
    feats = {}
    # ---- Her islem tipi + metrik icin ozetler + MOMENTUM ----
    for tx in TX_TYPES:
        for metric in ["volume","total_value","highest_amount",COUNT_SUFFIX[tx]]:
            cols = [f"m{m}_{tx}_{metric}" for m in MONTHS if f"m{m}_{tx}_{metric}" in df.columns]
            if len(cols) < 6:
                continue
            vals = df[cols]
            feats[f"{tx}_{metric}_sum"]           = vals.sum(axis=1)
            feats[f"{tx}_{metric}_mean"]          = vals.mean(axis=1)
            feats[f"{tx}_{metric}_std"]           = vals.std(axis=1)
            feats[f"{tx}_{metric}_max"]           = vals.max(axis=1)
            feats[f"{tx}_{metric}_trend"]         = df[cols[0]] - df[cols[-1]]       # m1(yeni)-m6(eski)
            feats[f"{tx}_{metric}_active_months"] = (vals > 0).sum(axis=1)
            # MOMENTUM: son 3 ay (m1,m2,m3) ort. - eski 3 ay (m4,m5,m6) ort.
            recent = df[[cols[0],cols[1],cols[2]]].mean(axis=1)
            old    = df[[cols[3],cols[4],cols[5]]].mean(axis=1)
            feats[f"{tx}_{metric}_momentum"] = recent - old

    # ---- Bakiye ozellikleri (en guclu sinyal grubu) ----
    bal_cols = [f"m{m}_daily_avg_bal" for m in MONTHS if f"m{m}_daily_avg_bal" in df.columns]
    if len(bal_cols) == 6:
        bal = df[bal_cols]
        bmean = bal.mean(axis=1)
        feats["bal_mean"]      = bmean
        feats["bal_std"]       = bal.std(axis=1)
        feats["bal_min"]       = bal.min(axis=1)
        feats["bal_max"]       = bal.max(axis=1)
        feats["bal_last"]      = df[bal_cols[0]]                       # en yeni ay bakiyesi
        feats["bal_trend"]     = df[bal_cols[0]] - df[bal_cols[-1]]
        feats["bal_cv"]        = bal.std(axis=1) / (bmean + 1)         # oynaklik katsayisi
        feats["bal_min_to_mean"] = bal.min(axis=1) / (bmean + 1)      # dip / ortalama
        feats["bal_low_months"]  = bal.lt(bmean, axis=0).sum(axis=1)
        feats["bal_recent_vs_old"] = df[bal_cols[:3]].mean(axis=1) - df[bal_cols[3:]].mean(axis=1)

    # ---- Net nakit akisi + kritik oranlar ----
    g = feats.get
    inflow  = g("received_total_value_sum",0) + g("deposit_total_value_sum",0) + g("transfer_from_bank_total_value_sum",0)
    outflow = g("mm_send_total_value_sum",0) + g("withdraw_total_value_sum",0) + g("paybill_total_value_sum",0) + g("merchantpay_total_value_sum",0)
    spend   = g("paybill_total_value_sum",0) + g("merchantpay_total_value_sum",0)
    feats["total_inflow"]   = inflow
    feats["total_outflow"]  = outflow
    feats["net_flow"]       = inflow - outflow
    feats["outflow_ratio"]  = outflow / (inflow + 1)
    feats["spend_to_inflow"]= spend / (inflow + 1)
    feats["withdraw_to_deposit"] = g("withdraw_total_value_sum",0) / (g("deposit_total_value_sum",0) + 1)
    feats["send_to_received"]    = g("mm_send_total_value_sum",0) / (g("received_total_value_sum",0) + 1)
    # toplam etkinlik (kac islem-ay aktif)
    act_cols = [k for k in feats if k.endswith("_active_months")]
    feats["total_active_months"] = sum(feats[k] for k in act_cols)

    return pd.concat([df, pd.DataFrame(feats, index=df.index)], axis=1)

train = add_features(train)
test  = add_features(test)

cat_cols = ["gender","region","smartphone","segment","earning_pattern"]
for c in cat_cols:
    train[c] = train[c].astype("category")
    test[c]  = pd.Categorical(test[c], categories=train[c].cat.categories)
    train[c] = train[c].cat.codes
    test[c]  = test[c].cat.codes

features = [c for c in train.columns if c not in [TARGET, ID_COL]]
X, y, X_test = train[features], train[TARGET], test[features]
print("Ozellik sayisi:", len(features))

# ---- 3-seed x 5-fold LightGBM (seed averaging) ----------------------
base = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "learning_rate": 0.015, "num_leaves": 48,
    "feature_fraction": 0.65, "bagging_fraction": 0.8, "bagging_freq": 1,
    "min_child_samples": 80, "lambda_l1": 1.0, "lambda_l2": 2.0,
    "verbose": -1,
}
SEEDS = [42, 2024, 777]
oof = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for seed in SEEDS:
    params = dict(base, seed=seed)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y):
        model = lgb.train(
            params,
            lgb.Dataset(X.iloc[tr], y.iloc[tr]),
            num_boost_round=6000,
            valid_sets=[lgb.Dataset(X.iloc[va], y.iloc[va])],
            callbacks=[lgb.early_stopping(200, first_metric_only=True), lgb.log_evaluation(0)],
        )
        oof[va]     += model.predict(X.iloc[va]) / len(SEEDS)
        test_preds  += model.predict(X_test) / (len(SEEDS) * skf.n_splits)
    print(f"  seed {seed} bitti")

cv_ll  = log_loss(y, oof)
cv_auc = roc_auc_score(y, oof)
print(f"\n>>> v2 CV  |  LogLoss: {cv_ll:.5f}  |  AUC: {cv_auc:.5f}")
print("    (v1 idi: LogLoss 0.31050 | AUC 0.85265)")

# ---- Submission -----------------------------------------------------
pred_col = [c for c in sample.columns if c != ID_COL][0]
sub = sample.copy()
sub[pred_col] = sub[ID_COL].map(dict(zip(test[ID_COL], test_preds)))
sub.to_csv("submission.csv", index=False)
print("\nsubmission.csv hazir:\n", sub.head())

# Colab'da otomatik indir (istemezsen bu 2 satiri sil):
try:
    from google.colab import files
    files.download("submission.csv")
except Exception:
    pass
