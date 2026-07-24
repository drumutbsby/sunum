# =====================================================================
# Zindi — Financial Stress Prediction  |  v15 (YAVAS-UZUN CEKIRDEK)
#
# Lab-3 kesfi: Optuna lr aramasi 0.02'nin altina hic inmemisti.
# Yavas-uzun rejim (LGB lr=0.005, CAT lr=0.01, 30k tur, sabir 600)
# 5-fold/1-seed'de bile stack'e +0.00297 kazandirdi (M5=0.11901).
# v15 = ayni cekirdegi TAM pipeline'a tasir: 10-fold x 5 seed x 2 model
# + 4 adayli secim (HAM/KALIBRE/STACK/STACK+KALIBRE) + guvenlik.
#
# Gecilecek hedef (v13): LogLoss 0.24278 | AUC 0.91154 | LB 0.12184
# Beklenti: LB ~0.1245-0.1250 (public ~0.722-0.723)
#
# !!! SURE: 100 yavas fit -> 8 vCPU'da ~3-5 SAAT. Kesinlikle ARKA
# PLANDA yurut (ucretli Colab ozelligi), geceye birak.
# Hizli mod gerekirse: N_FOLDS=5, SEEDS=[42,2024,777] (~1.5 saat)
#
# Colab: !pip install catboost lightgbm -q
# =====================================================================

import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

TARGET = "liquidity_stress_next_30d"
ID_COL = "ID"

train = pd.read_csv("Train.csv")
test  = pd.read_csv("Test.csv")
sample = pd.read_csv("SampleSubmission.csv")

n0 = len(train)
train = train.dropna(subset=[TARGET]).reset_index(drop=True)
train[TARGET] = train[TARGET].astype(int)
print(f"Egitim: {len(train)} (atilan: {n0-len(train)}) | Test: {len(test)}")

# ---------------- 1) MUSTERI PARMAK IZI ------------------------------
PROF_COLS = ["arpu","age","gender","region","smartphone","segment",
             "earning_pattern","x_90_d_activity_rate"]

def make_cust_id(df):
    return df[PROF_COLS].astype(str).agg("|".join, axis=1)

train["cust_id"] = make_cust_id(train)
test["cust_id"]  = make_cust_id(test)

n_shared = len(set(train["cust_id"]) & set(test["cust_id"]))
print(f"Benzersiz musteri: train={train['cust_id'].nunique()}, test={test['cust_id'].nunique()}")
print(f"Train ve test'te ORTAK musteri: {n_shared}  <-- sinyal kaynagi")

# ---------------- 2) Davranis ozellikleri (v4/v5 ile ayni) -----------
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
            vals = df[cols]; mean6 = vals.mean(axis=1)
            feats[f"{tx}_{metric}_sum"]=vals.sum(axis=1); feats[f"{tx}_{metric}_mean"]=mean6
            feats[f"{tx}_{metric}_std"]=vals.std(axis=1); feats[f"{tx}_{metric}_max"]=vals.max(axis=1)
            feats[f"{tx}_{metric}_trend"]=df[cols[0]]-df[cols[-1]]
            feats[f"{tx}_{metric}_active_months"]=(vals>0).sum(axis=1)
            recent3 = df[[cols[0],cols[1],cols[2]]].mean(axis=1)
            old3    = df[[cols[3],cols[4],cols[5]]].mean(axis=1)
            feats[f"{tx}_{metric}_momentum"]=recent3-old3
            feats[f"{tx}_{metric}_last_vs_mean"]=df[cols[0]]/(mean6+1)
            # EDA-B1/B2: dusus ORANI (fark degil) — olcek bagimsiz cokus sinyali
            feats[f"{tx}_{metric}_recent_over_old"]=(recent3+1)/(old3+1)
    bal_cols=[f"m{m}_daily_avg_bal" for m in MONTHS if f"m{m}_daily_avg_bal" in df.columns]
    if len(bal_cols)==6:
        bal=df[bal_cols]; bmean=bal.mean(axis=1)
        feats["bal_mean"]=bmean; feats["bal_std"]=bal.std(axis=1)
        feats["bal_min"]=bal.min(axis=1); feats["bal_max"]=bal.max(axis=1)
        feats["bal_last"]=df[bal_cols[0]]; feats["bal_trend"]=df[bal_cols[0]]-df[bal_cols[-1]]
        feats["bal_cv"]=bal.std(axis=1)/(bmean+1); feats["bal_min_to_mean"]=bal.min(axis=1)/(bmean+1)
        feats["bal_low_months"]=bal.lt(bmean,axis=0).sum(axis=1)
        feats["bal_recent_vs_old"]=df[bal_cols[:3]].mean(axis=1)-df[bal_cols[3:]].mean(axis=1)
        feats["bal_last_vs_mean"]=df[bal_cols[0]]/(bmean+1)
        feats["bal_last_vs_max"]=df[bal_cols[0]]/(bal.max(axis=1)+1)
        feats["bal_m1_minus_m2"]=df[bal_cols[0]]-df[bal_cols[1]]
        # ---- EDA-B1: COKUS ozellikleri (uretecin ana sinyali) ----
        b_recent = df[bal_cols[:3]].mean(axis=1)   # m1-m3
        b_old    = df[bal_cols[3:]].mean(axis=1)   # m4-m6
        feats["bal_recent_over_old"] = (b_recent+1)/(b_old+1)     # <1 = cokus
        feats["bal_min_recent_over_old"] = (df[bal_cols[:3]].min(axis=1)+1)/(b_old+1)
        # 6 aylik dogrusal EGIM (m1=t6 en yeni ... m6=t1 en eski)
        w = np.array([2.5,1.5,0.5,-0.5,-1.5,-2.5])/17.5
        feats["bal_slope6"] = sum(w[i]*df[bal_cols[i]] for i in range(6))
        feats["bal_slope6_norm"] = feats["bal_slope6"]/(bmean+1)   # olcekten bagimsiz egim
        # tepe noktasindan dusus (drawdown)
        feats["bal_drawdown"] = (bal.max(axis=1)-df[bal_cols[0]])/(bal.max(axis=1)+1)
        # ---- EDA-B3: arpu-normalize bakiye (zenginlige gore konum) ----
        feats["bal_mean_per_arpu"] = bmean/(df["arpu"]+1)
        feats["bal_last_per_arpu"] = df[bal_cols[0]]/(df["arpu"]+1)
    g=feats.get
    inflow=g("received_total_value_sum",0)+g("deposit_total_value_sum",0)+g("transfer_from_bank_total_value_sum",0)
    outflow=g("mm_send_total_value_sum",0)+g("withdraw_total_value_sum",0)+g("paybill_total_value_sum",0)+g("merchantpay_total_value_sum",0)
    spend=g("paybill_total_value_sum",0)+g("merchantpay_total_value_sum",0)
    feats["total_inflow"]=inflow; feats["total_outflow"]=outflow; feats["net_flow"]=inflow-outflow
    feats["outflow_ratio"]=outflow/(inflow+1); feats["spend_to_inflow"]=spend/(inflow+1)
    feats["withdraw_to_deposit"]=g("withdraw_total_value_sum",0)/(g("deposit_total_value_sum",0)+1)
    feats["send_to_received"]=g("mm_send_total_value_sum",0)/(g("received_total_value_sum",0)+1)
    feats["total_active_months"]=sum(feats[k] for k in list(feats) if k.endswith("_active_months"))
    def m1v(tx): return df.get(f"m1_{tx}_total_value",0)
    m1_in=m1v("received")+m1v("deposit")+m1v("transfer_from_bank")
    m1_out=m1v("mm_send")+m1v("withdraw")+m1v("paybill")+m1v("merchantpay")
    feats["m1_inflow"]=m1_in; feats["m1_outflow"]=m1_out; feats["m1_net_flow"]=m1_in-m1_out
    feats["m1_outflow_ratio"]=m1_out/(m1_in+1)
    feats["m1_netflow_vs_bal"]=(m1_in-m1_out)/(feats.get("bal_last",0)+1)

    # ============ v11 SEMANTIK OZELLIKLER (GELIR RITMI) ============
    n = len(df)
    # aylik GELIR toplamlari ve gelir islem sayilari (m1=en yeni)
    inc_val = [df[f"m{m}_received_total_value"]+df[f"m{m}_deposit_total_value"]
               +df[f"m{m}_transfer_from_bank_total_value"] for m in MONTHS]
    inc_vol = [df[f"m{m}_received_volume"]+df[f"m{m}_deposit_volume"]
               +df[f"m{m}_transfer_from_bank_volume"] for m in MONTHS]
    inc_mat = np.column_stack([v.values for v in inc_val])

    # S2: gelir duzenliligi (dusuk CV = duzenli maasli)
    feats["income_mean"] = inc_mat.mean(axis=1)
    feats["income_cv"]   = inc_mat.std(axis=1)/(inc_mat.mean(axis=1)+1)
    # S4: gelir soku — son ay geliri / 6 aylik medyan
    feats["income_shock"] = inc_mat[:,0]/(np.median(inc_mat,axis=1)+1)

    # S3: sifir-gelir serileri (en uzun + en guncel)
    zv = [(v.values==0).astype(float) for v in inc_vol]
    best=np.zeros(n); cur=np.zeros(n); rec=np.ones(n); recent=np.zeros(n)
    for i in range(6):
        cur = (cur+zv[i])*zv[i]           # sifir olmayan ayda sifirlanir
        best = np.maximum(best,cur)
        rec = rec*zv[i]                    # m1'den itibaren kesintisiz sifir
        recent += rec
    feats["income_zero_streak_max"]    = best
    feats["income_zero_streak_recent"] = recent

    # S1: "maas gelmedi" bayragi — aylik kazancli + son ay geliri sifir
    feats["missed_income_m1"] = (inc_vol[0].values==0).astype(int)
    feats["missed_salary_flag"] = ((df["earning_pattern"]=="Monthly Earner").values
                                   & (inc_vol[0].values==0)).astype(int)

    # S5: bakiye dusus serisi + dip ayin konumu
    if len(bal_cols)==6:
        b=[df[c].values for c in bal_cols]      # b[0]=m1 en yeni
        run=np.ones(n); streak=np.zeros(n)
        for i in range(5):
            run = run*(b[i] < b[i+1])           # yeni < eski = dusus
            streak += run
        feats["bal_decline_streak"] = streak     # kac aydir ustuste dusuyor
        feats["bal_argmin_month"]   = np.argmin(np.column_stack(b),axis=1)+1  # 1=dip su an

    # S6: EDA'nin en korele ham ailelerine dogrusal EGIM
    w = np.array([2.5,1.5,0.5,-0.5,-1.5,-2.5])/17.5
    for tx,metric in [("deposit","agents"),("received","senders"),("withdraw","agents"),
                      ("deposit","volume"),("received","volume")]:
        cols=[f"m{m}_{tx}_{metric}" for m in MONTHS]
        if all(c in df.columns for c in cols):
            feats[f"{tx}_{metric}_slope6"] = sum(w[i]*df[cols[i]] for i in range(6))

    return pd.concat([df, pd.DataFrame(feats, index=df.index)], axis=1)

train = add_features(train); test = add_features(test)

# ---------------- 3) MUSTERI-GRUBU ozellikleri (etiketsiz) -----------
# Train+test birlikte (labelsiz davranis ortalamasi) — izinli: hepsi
# yarismanin verdigi veri.
GRP_KEYS = ["bal_last","bal_mean","bal_trend","net_flow","m1_net_flow",
            "outflow_ratio","m1_outflow_ratio","total_active_months"]
allc = pd.concat([train[["cust_id"]+GRP_KEYS], test[["cust_id"]+GRP_KEYS]], ignore_index=True)
gstat = allc.groupby("cust_id")[GRP_KEYS].mean()
gsize = allc.groupby("cust_id").size().rename("grp_size")

for df in (train, test):
    df["grp_size"] = df["cust_id"].map(gsize).astype(float)
    for k in GRP_KEYS:
        df[f"grp_{k}_mean"]  = df["cust_id"].map(gstat[k])
        df[f"grp_{k}_delta"] = df[k] - df[f"grp_{k}_mean"]   # bu snapshot gruptan ne kadar sapiyor?

# Musterinin train'de kac etiketli snapshot'i var (te guvenilirligi)
n_lab = train.groupby("cust_id").size().rename("grp_n_labeled")
for df in (train, test):
    df["grp_n_labeled"] = df["cust_id"].map(n_lab).fillna(0).astype(float)

# ---- EDA-B4: musteri-ICI yuzdelik sira (bu snapshot, musterinin en
# kotu donemi mi?) — etiket kullanilmaz, tamamen davranissal ----------
RANK_KEYS = ["bal_last","bal_recent_over_old","bal_slope6_norm","net_flow","m1_net_flow","m1_outflow_ratio"]
allr = pd.concat([train[["cust_id"]+RANK_KEYS], test[["cust_id"]+RANK_KEYS]], ignore_index=True)
for k in RANK_KEYS:
    allr[f"grp_{k}_rank"] = allr.groupby("cust_id")[k].rank(pct=True)
n_tr_rows = len(train)
for k in RANK_KEYS:
    train[f"grp_{k}_rank"] = allr[f"grp_{k}_rank"].iloc[:n_tr_rows].values
    test[f"grp_{k}_rank"]  = allr[f"grp_{k}_rank"].iloc[n_tr_rows:].values

# ---------------- 4) OOF MUSTERI TARGET-ENCODING ---------------------
# Satirin KENDI etiketi kendi te'sine asla girmez (fold-disi hesap).
def oof_target_encode(cust_tr, y, cust_te, n_splits=5, seed=42, prior=20.0):
    gm = float(y.mean())
    te_tr = np.full(len(y), gm)
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(np.zeros(len(y)), y):
        agg = y.iloc[tr].groupby(cust_tr.iloc[tr].values).agg(["sum","count"])
        mp  = (agg["sum"] + prior*gm) / (agg["count"] + prior)
        te_tr[va] = cust_tr.iloc[va].map(mp).fillna(gm).values
    agg = y.groupby(cust_tr.values).agg(["sum","count"])
    mp  = (agg["sum"] + prior*gm) / (agg["count"] + prior)
    te_te = cust_te.map(mp).fillna(gm).values
    return te_tr, te_te

y_tmp = train[TARGET]
# Iki farkli smoothing: keskin (prior=3) + yumusak (prior=20)
te_tr3,  te_te3  = oof_target_encode(train["cust_id"], y_tmp, test["cust_id"], prior=3.0)
te_tr20, te_te20 = oof_target_encode(train["cust_id"], y_tmp, test["cust_id"], prior=20.0)
train["cust_te_p3"]  = te_tr3;  test["cust_te_p3"]  = te_te3
train["cust_te_p20"] = te_tr20; test["cust_te_p20"] = te_te20
print(f"cust_te_p3  aralik: [{te_tr3.min():.3f},{te_tr3.max():.3f}]")
print(f"cust_te_p20 aralik: [{te_tr20.min():.3f},{te_tr20.max():.3f}]")
# TANI: musteri-etiket sinyali tek basina ne kadar guclu?
from sklearn.metrics import roc_auc_score as _auc
print(f">>> TANI | te-alone OOF AUC: {_auc(y_tmp, te_tr3):.4f}  "
      f"(0.50=sinyal yok, >0.60=guclu musteri sinyali)")

# ---- S7: grup-ici Z-SKORLAR (kendi segmentine / kazanc tipine gore
# anormal cokus mu?) — istatistikler train+test uzerinden (etiketsiz) --
ZKEYS = ["bal_slope6_norm","bal_recent_over_old","bal_drawdown","income_cv","income_shock"]
for gcol in ["segment","earning_pattern"]:
    alls = pd.concat([train[[gcol]+ZKEYS], test[[gcol]+ZKEYS]], ignore_index=True)
    gstat_z = alls.groupby(gcol)[ZKEYS].agg(["mean","std"])
    for k in ZKEYS:
        mu, sd = gstat_z[(k,"mean")], gstat_z[(k,"std")]
        for df in (train, test):
            df[f"z_{gcol}_{k}"] = (df[k]-df[gcol].map(mu))/(df[gcol].map(sd)+1e-9)

# ---------------- 5) Kategorikler + ozellik listesi ------------------
for c in ["gender","region","smartphone","segment","earning_pattern"]:
    train[c] = train[c].astype("category")
    test[c]  = pd.Categorical(test[c], categories=train[c].cat.categories)
    train[c] = train[c].cat.codes; test[c] = test[c].cat.codes

features = [c for c in train.columns if c not in [TARGET, ID_COL, "cust_id"]]
X, y, X_test = train[features], train[TARGET], test[features]
print("Ozellik sayisi:", len(features))

# ---------------- 6a) v10 Optuna sonuclari (GOMULU — arama yok) ------
BEST_LGB = {
    "objective": "binary", "verbose": -1, "bagging_freq": 1,
    "learning_rate":     0.005,   # YAVAS-UZUN rejim (Lab-3 M2)
    "num_leaves":        28,
    "feature_fraction":  0.608453329394061,
    "bagging_fraction":  0.7117796418745417,
    "min_child_samples": 24,
    "lambda_l1":         0.6178288937936505,
    "lambda_l2":         0.018243960300491052,
}
BEST_CAT = {
    "learning_rate":       0.01,   # YAVAS-UZUN rejim (Lab-3 M4)
    "depth":               6,
    "l2_leaf_reg":         4.73195934284595,
    "random_strength":     0.5207174480416124,
    "bagging_temperature": 1.57688689814328,
}
print("\nLGB+CAT: v10 Optuna parametreleri gomulu (arama atlandi).")

# ---------------- 6b) Ensemble: 5-seed LGB(tuned) + 5-seed CAT(tuned) -
N_FOLDS = 10   # zaman darsa 5
SEEDS = [42, 2024, 777, 1453, 5]   # zaman darsa ilk 3'u birak

def cv_lgb_seeds():
    oof=np.zeros(len(X)); tp=np.zeros(len(X_test))
    for seed in SEEDS:
        p=dict(BEST_LGB, metric=["binary_logloss","auc"], seed=seed)  # Optuna parametreleri
        skf=StratifiedKFold(N_FOLDS,shuffle=True,random_state=seed)
        for tr,va in skf.split(X,y):
            m=lgb.train(p,lgb.Dataset(X.iloc[tr],y.iloc[tr]),num_boost_round=30000,
                valid_sets=[lgb.Dataset(X.iloc[va],y.iloc[va])],
                callbacks=[lgb.early_stopping(600,first_metric_only=True),lgb.log_evaluation(0)])
            oof[va]+=m.predict(X.iloc[va])/len(SEEDS)
            tp+=m.predict(X_test)/(len(SEEDS)*N_FOLDS)
        print(f"    LGB seed {seed} ok")
    return oof,tp

def cv_cat_seeds():
    oof=np.zeros(len(X)); tp=np.zeros(len(X_test))
    for seed in SEEDS:
        skf=StratifiedKFold(N_FOLDS,shuffle=True,random_state=seed)
        for tr,va in skf.split(X,y):
            m=CatBoostClassifier(iterations=30000, loss_function="Logloss",
                eval_metric="Logloss", random_seed=seed,
                early_stopping_rounds=600, verbose=False, **BEST_CAT)
            m.fit(X.iloc[tr],y.iloc[tr],eval_set=(X.iloc[va],y.iloc[va]),use_best_model=True)
            oof[va]+=m.predict_proba(X.iloc[va])[:,1]/len(SEEDS)
            tp+=m.predict_proba(X_test)[:,1]/(len(SEEDS)*N_FOLDS)
        print(f"    CAT seed {seed} ok")
    return oof,tp

def lb_score(yy,pp): return 0.4*roc_auc_score(yy,pp) - log_loss(yy,pp)

print("\nLightGBM..."); oof_l,tp_l = cv_lgb_seeds()
print("CatBoost...");  oof_c,tp_c = cv_cat_seeds()
for name,o in [("LGB",oof_l),("CAT",oof_c)]:
    print(f"  {name:4s} | LogLoss {log_loss(y,o):.5f} | AUC {roc_auc_score(y,o):.5f} | LB {lb_score(y,o):.5f}")

best=(-9,None)
for wl in np.arange(0,1.0001,0.05):
    sc_=lb_score(y, wl*oof_l+(1-wl)*oof_c)
    if sc_>best[0]: best=(sc_,round(wl,2))
wl = best[1]; wc = round(1-wl,2)
oof_b = wl*oof_l+wc*oof_c
tp_b  = wl*tp_l +wc*tp_c
print(f"\nEn iyi blend -> LGB:{wl} CAT:{wc}")
print(f"Ham blend    | LogLoss: {log_loss(y,oof_b):.5f} | AUC: {roc_auc_score(y,oof_b):.5f} | LB: {lb_score(y,oof_b):.5f}")

# ---------------- 7) CV-guvenli isotonic kalibrasyon -----------------
def cross_fit_calibrate(oof_raw, y, test_raw, n_splits=5, seed=42):
    skf_c = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_cal = np.zeros_like(oof_raw)
    for tr, va in skf_c.split(oof_raw.reshape(-1,1), y):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1-1e-6)
        iso.fit(oof_raw[tr], y.iloc[tr])
        oof_cal[va] = iso.predict(oof_raw[va])
    iso_full = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1-1e-6)
    iso_full.fit(oof_raw, y)
    return np.clip(oof_cal,1e-6,1-1e-6), np.clip(iso_full.predict(test_raw),1e-6,1-1e-6)

oof_cal, tp_cal = cross_fit_calibrate(oof_b, y, tp_b)
print(f"Kalibre blend| LogLoss: {log_loss(y,oof_cal):.5f} | AUC: {roc_auc_score(y,oof_cal):.5f} | LB: {lb_score(y,oof_cal):.5f}")

# ---- STACKING: OOF logitleri uzerinde cross-fit lojistik regresyon ---
from sklearn.linear_model import LogisticRegression

def logit(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p/(1-p))

def cross_fit_stack(oofs, y, tests, n_splits=5, seed=42):
    Ztr = np.column_stack([logit(o) for o in oofs])
    Zte = np.column_stack([logit(t) for t in tests])
    skf_s = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    oof_s = np.zeros(len(y))
    for tr, va in skf_s.split(Ztr, y):
        lr = LogisticRegression(C=1.0, max_iter=1000)
        lr.fit(Ztr[tr], y.iloc[tr])
        oof_s[va] = lr.predict_proba(Ztr[va])[:,1]
    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(Ztr, y)
    return np.clip(oof_s,1e-6,1-1e-6), np.clip(lr.predict_proba(Zte)[:,1],1e-6,1-1e-6)

oof_s, tp_s = cross_fit_stack([oof_l, oof_c], y, [tp_l, tp_c])
print(f"Stack (LR)   | LogLoss: {log_loss(y,oof_s):.5f} | AUC: {roc_auc_score(y,oof_s):.5f} | LB: {lb_score(y,oof_s):.5f}")

# 4. aday: stack ciktisini ayrica kalibre et (v9'da stack kazanmisti)
oof_sc, tp_sc = cross_fit_calibrate(oof_s, y, tp_s)
print(f"Stack+Kalibre| LogLoss: {log_loss(y,oof_sc):.5f} | AUC: {roc_auc_score(y,oof_sc):.5f} | LB: {lb_score(y,oof_sc):.5f}")

# ---- 4 aday arasindan LB metrigine gore otomatik secim ---------------
candidates = {
    "HAM":           (oof_b,   np.clip(tp_b,1e-6,1-1e-6)),
    "KALIBRE":       (oof_cal, tp_cal),
    "STACK":         (oof_s,   tp_s),
    "STACK+KALIBRE": (oof_sc,  tp_sc),
}
chosen = max(candidates, key=lambda k: lb_score(y, candidates[k][0]))
oof_f, tp_f = candidates[chosen]

print(f"\nSecilen: {chosen}")
print(f">>> v15 SLOW CV | LogLoss: {log_loss(y,oof_f):.5f} | AUC: {roc_auc_score(y,oof_f):.5f} | LB: {lb_score(y,oof_f):.5f}")
print("    (v13 idi: LogLoss 0.24278 | AUC 0.91154 | LB 0.12184  <- gecilecek hedef)")

# ---------------- 8) Submission --------------------------------------
pred_col = [c for c in sample.columns if c != ID_COL][0]
sub = sample.copy()
sub[pred_col] = sub[ID_COL].map(dict(zip(test[ID_COL], tp_f)))
# GUVENLIK: SampleSubmission'daki her ID icin tahmin uretilmis olmali.
# Bos kalan varsa dosyalar uyumsuz demektir (eski sample + yeni test gibi).
n_missing = int(sub[pred_col].isna().sum())
assert n_missing == 0, (f"{n_missing} ID icin tahmin yok! Zindi'den Train/Test/"
                        f"SampleSubmission'in HEPSINI guncel haliyle yeniden indir.")
sub.to_csv("submission.csv", index=False)
print("\nsubmission.csv hazir:\n", sub.head())
try:
    from google.colab import files; files.download("submission.csv")
except Exception:
    pass
