# =====================================================================
# Zindi — Financial Stress Prediction  |  v3 (ENSEMBLE)
# LightGBM + CatBoost + XGBoost  ->  blend agirliklari LB metrigini
# (0.4*AUC - LogLoss) dogrudan maksimize edecek sekilde optimize edilir.
# Colab: en basta -> !pip install catboost xgboost lightgbm -q
# =====================================================================

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
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
print(f"Egitim: {len(train)} (atilan: {n0-len(train)}) | Test: {len(test)}")

TX_TYPES = ["paybill","merchantpay","transfer_from_bank","mm_send","received","deposit","withdraw"]
COUNT_SUFFIX = {"paybill":"companies","merchantpay":"merchants","transfer_from_bank":"banks",
                "mm_send":"recipients","received":"senders","deposit":"agents","withdraw":"agents"}
MONTHS = [1,2,3,4,5,6]

def add_features(df):
    feats = {}
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
            feats[f"{tx}_{metric}_trend"]         = df[cols[0]] - df[cols[-1]]
            feats[f"{tx}_{metric}_active_months"] = (vals > 0).sum(axis=1)
            feats[f"{tx}_{metric}_momentum"]      = df[[cols[0],cols[1],cols[2]]].mean(axis=1) - df[[cols[3],cols[4],cols[5]]].mean(axis=1)
    bal_cols = [f"m{m}_daily_avg_bal" for m in MONTHS if f"m{m}_daily_avg_bal" in df.columns]
    if len(bal_cols) == 6:
        bal = df[bal_cols]; bmean = bal.mean(axis=1)
        feats["bal_mean"]=bmean; feats["bal_std"]=bal.std(axis=1)
        feats["bal_min"]=bal.min(axis=1); feats["bal_max"]=bal.max(axis=1)
        feats["bal_last"]=df[bal_cols[0]]; feats["bal_trend"]=df[bal_cols[0]]-df[bal_cols[-1]]
        feats["bal_cv"]=bal.std(axis=1)/(bmean+1); feats["bal_min_to_mean"]=bal.min(axis=1)/(bmean+1)
        feats["bal_low_months"]=bal.lt(bmean,axis=0).sum(axis=1)
        feats["bal_recent_vs_old"]=df[bal_cols[:3]].mean(axis=1)-df[bal_cols[3:]].mean(axis=1)
    g = feats.get
    inflow  = g("received_total_value_sum",0)+g("deposit_total_value_sum",0)+g("transfer_from_bank_total_value_sum",0)
    outflow = g("mm_send_total_value_sum",0)+g("withdraw_total_value_sum",0)+g("paybill_total_value_sum",0)+g("merchantpay_total_value_sum",0)
    spend   = g("paybill_total_value_sum",0)+g("merchantpay_total_value_sum",0)
    feats["total_inflow"]=inflow; feats["total_outflow"]=outflow; feats["net_flow"]=inflow-outflow
    feats["outflow_ratio"]=outflow/(inflow+1); feats["spend_to_inflow"]=spend/(inflow+1)
    feats["withdraw_to_deposit"]=g("withdraw_total_value_sum",0)/(g("deposit_total_value_sum",0)+1)
    feats["send_to_received"]=g("mm_send_total_value_sum",0)/(g("received_total_value_sum",0)+1)
    feats["total_active_months"]=sum(feats[k] for k in list(feats) if k.endswith("_active_months"))
    return pd.concat([df, pd.DataFrame(feats, index=df.index)], axis=1)

train = add_features(train); test = add_features(test)
for c in ["gender","region","smartphone","segment","earning_pattern"]:
    train[c] = train[c].astype("category")
    test[c]  = pd.Categorical(test[c], categories=train[c].cat.categories)
    train[c] = train[c].cat.codes; test[c] = test[c].cat.codes

features = [c for c in train.columns if c not in [TARGET, ID_COL]]
X, y, X_test = train[features], train[TARGET], test[features]
print("Ozellik sayisi:", len(features))

# AYNI fold bolunmesi 3 modelde de kullanilir (blend OOF gecerli olsun)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(X, y))

def cv_lightgbm():
    oof=np.zeros(len(X)); tp=np.zeros(len(X_test))
    p={"objective":"binary","metric":["binary_logloss","auc"],"learning_rate":0.015,
       "num_leaves":48,"feature_fraction":0.65,"bagging_fraction":0.8,"bagging_freq":1,
       "min_child_samples":80,"lambda_l1":1.0,"lambda_l2":2.0,"verbose":-1,"seed":42}
    for tr,va in folds:
        m=lgb.train(p,lgb.Dataset(X.iloc[tr],y.iloc[tr]),num_boost_round=6000,
            valid_sets=[lgb.Dataset(X.iloc[va],y.iloc[va])],
            callbacks=[lgb.early_stopping(200,first_metric_only=True),lgb.log_evaluation(0)])
        oof[va]=m.predict(X.iloc[va]); tp+=m.predict(X_test)/len(folds)
    return oof,tp

def cv_xgboost():
    oof=np.zeros(len(X)); tp=np.zeros(len(X_test))
    p={"objective":"binary:logistic","eval_metric":"logloss","eta":0.02,"max_depth":6,
       "subsample":0.8,"colsample_bytree":0.7,"min_child_weight":5,
       "reg_lambda":2.0,"reg_alpha":1.0,"seed":42}
    dtest=xgb.DMatrix(X_test)
    for tr,va in folds:
        dtr=xgb.DMatrix(X.iloc[tr],label=y.iloc[tr]); dva=xgb.DMatrix(X.iloc[va],label=y.iloc[va])
        m=xgb.train(p,dtr,num_boost_round=6000,evals=[(dva,"v")],
                    early_stopping_rounds=200,verbose_eval=False)
        it=(0,m.best_iteration+1)
        oof[va]=m.predict(dva,iteration_range=it); tp+=m.predict(dtest,iteration_range=it)/len(folds)
    return oof,tp

def cv_catboost():
    oof=np.zeros(len(X)); tp=np.zeros(len(X_test))
    for tr,va in folds:
        m=CatBoostClassifier(iterations=6000,learning_rate=0.02,depth=6,l2_leaf_reg=3,
            loss_function="Logloss",eval_metric="Logloss",random_seed=42,
            early_stopping_rounds=200,verbose=False)
        m.fit(X.iloc[tr],y.iloc[tr],eval_set=(X.iloc[va],y.iloc[va]),use_best_model=True)
        oof[va]=m.predict_proba(X.iloc[va])[:,1]; tp+=m.predict_proba(X_test)[:,1]/len(folds)
    return oof,tp

def score(yy,pp):  # LB metrigi yaklasimi (yuksek=iyi)
    return 0.4*roc_auc_score(yy,pp) - log_loss(yy,pp)

print("\nLightGBM egitiliyor..."); oof_l,tp_l = cv_lightgbm()
print("XGBoost egitiliyor...");   oof_x,tp_x = cv_xgboost()
print("CatBoost egitiliyor...");  oof_c,tp_c = cv_catboost()

for name,o in [("LightGBM",oof_l),("XGBoost",oof_x),("CatBoost",oof_c)]:
    print(f"  {name:9s} | LogLoss {log_loss(y,o):.5f} | AUC {roc_auc_score(y,o):.5f}")

# ---- Blend agirliklarini LB metrigine gore optimize et --------------
best=(-9,None)
for wl in np.arange(0,1.001,0.1):
    for wc in np.arange(0,1.001-wl,0.1):
        wx=1-wl-wc
        s=score(y, wl*oof_l+wc*oof_c+wx*oof_x)
        if s>best[0]: best=(s,(round(wl,2),round(wc,2),round(wx,2)))
wl,wc,wx = best[1]
oof_b = wl*oof_l+wc*oof_c+wx*oof_x
tp_b  = wl*tp_l +wc*tp_c +wx*tp_x
print(f"\nEn iyi blend agirliklari -> LGB:{wl} CAT:{wc} XGB:{wx}")
print(f">>> v3 ENSEMBLE CV | LogLoss: {log_loss(y,oof_b):.5f} | AUC: {roc_auc_score(y,oof_b):.5f}")
print("    (v2 idi: LogLoss 0.28201 | AUC 0.88091)")

# ---- Submission -----------------------------------------------------
pred_col = [c for c in sample.columns if c != ID_COL][0]
sub = sample.copy()
sub[pred_col] = sub[ID_COL].map(dict(zip(test[ID_COL], np.clip(tp_b,1e-6,1-1e-6))))
sub.to_csv("submission.csv", index=False)
print("\nsubmission.csv hazir:\n", sub.head())
try:
    from google.colab import files; files.download("submission.csv")
except Exception:
    pass
