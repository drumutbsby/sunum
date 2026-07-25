# =====================================================================
# Zindi — v19 DATAIKU (5 SESLI KORO: gbdt + CAT + DART + GOSS + XTREES)
#
# Lab-7: boosting-modu cesitliligi stack'te +0.00146 verdi (esik ustu).
# v19 = v18 ozellik seti + 3 yeni stack sesi (hepsi 5 seed x 10 fold):
#   - LGB DART  (sabit 1200 tur, lr 0.03 — dart'ta erken durdurma yok)
#   - LGB GOSS  (bagging'siz, yavas rejim)
#   - LGB extra_trees (asiri rastgele bolmeler, yavas rejim)
# Stack 5 girisli; adaylar: HAM(LGB+CAT) / KALIBRE / STACK5 / STACK5+KAL
#
# Gecilecek hedef (v17/v18): LogLoss 0.24014 | AUC 0.91385 | LB 0.12540
# Beklenti: LB ~0.1261-0.1264 -> public ~0.724-0.726
# Sure (40 vCPU): ~90 dk — arka planda yurut
# =====================================================================

try:
    import dataiku
except Exception:
    dataiku = None
import os
from joblib import Parallel, delayed

THREADS_PER_MODEL = 4
N_WORKERS = max(1, (os.cpu_count() or 8) // THREADS_PER_MODEL)
print(f"vCPU: {os.cpu_count()} | paralel isci: {N_WORKERS} x {THREADS_PER_MODEL} thread")

import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.isotonic import IsotonicRegression
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

TARGET = "liquidity_stress_next_30d"
ID_COL = "ID"

DS_TRAIN = "Train_1"          # <- DSS dataset adlarin farkliysa burayi degistir
DS_TEST  = "Test"
DS_SAMPLE = "SampleSubmission"   # yoksa sorun degil, otomatik olusturulur

try:                                   # Dataiku ortami
    train = dataiku.Dataset(DS_TRAIN).get_dataframe()
    test  = dataiku.Dataset(DS_TEST).get_dataframe()
    print("Veri Dataiku dataset'lerinden okundu")
except Exception:                       # Colab / yerel CSV ortami
    train = pd.read_csv("Train.csv")
    test  = pd.read_csv("Test.csv")
    print("Veri CSV dosyalarindan okundu")

try:
    sample = dataiku.Dataset(DS_SAMPLE).get_dataframe()
    print(f"SampleSubmission dataset'inden okundu: {len(sample)} satir")
except Exception:
    try:
        sample = pd.read_csv("SampleSubmission.csv")
        print(f"SampleSubmission.csv okundu: {len(sample)} satir")
    except Exception:
        sample = pd.DataFrame({"ID": test["ID"].values, "Target": 0.5})
        print(f"SampleSubmission yok -> test ID'lerinden olusturuldu ({len(sample)} satir)")

# --- DSS tip guvenligi: CSV'den yuklenen datasetlerde kolonlar string
# kalabilir; kategorik/ID disindaki her seyi sayiya zorla ---------------
NON_NUM = {"gender","region","smartphone","segment","earning_pattern","ID"}
for _df in (train, test):
    for c in _df.columns:
        if c not in NON_NUM:
            _df[c] = pd.to_numeric(_df[c], errors="coerce")

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

# ---- v16: NET AKIS SERISI + FIRST-PASSAGE (Lab-4 kazananlari) --------
def add_net_series(d):
    d = d.copy(); nf = {}
    nets = []
    for m in range(1,7):
        inn  = sum_cols = None
        inn  = d[f"m{m}_received_total_value"]+d[f"m{m}_deposit_total_value"]+d[f"m{m}_transfer_from_bank_total_value"]
        outt = d[f"m{m}_mm_send_total_value"]+d[f"m{m}_withdraw_total_value"]+d[f"m{m}_paybill_total_value"]+d[f"m{m}_merchantpay_total_value"]
        nf[f"net_m{m}"] = inn - outt
        nets.append(nf[f"net_m{m}"])
    N = np.column_stack([v.values for v in nets])          # kolon 0 = m1 (en yeni)
    w = np.array([2.5,1.5,0.5,-0.5,-1.5,-2.5])/17.5
    nf["net_slope6"]        = N @ w
    nf["net_mean6"]         = N.mean(axis=1)
    nf["net_std6"]          = N.std(axis=1)
    nf["net_recent3"]       = N[:,:3].mean(axis=1)
    nf["net_old3"]          = N[:,3:].mean(axis=1)
    nf["net_recent_minus_old"] = nf["net_recent3"] - nf["net_old3"]
    nf["net_neg_months"]    = (N < 0).sum(axis=1)
    nf["net_neg_recent"]    = (N[:,:3] < 0).sum(axis=1)
    nf["net_cum3"]          = N[:,:3].sum(axis=1)
    nf["net_cum6"]          = N.sum(axis=1)
    nf["net_min"]           = N.min(axis=1)
    bl = d["bal_last"] if "bal_last" in d else d["m1_daily_avg_bal"]
    nf["net_cum3_over_bal"] = nf["net_cum3"] / (bl + 1)
    nf["net_m1_over_bal"]   = nf["net_m1"] / (bl + 1)
    return pd.concat([d, pd.DataFrame(nf, index=d.index)], axis=1)

def add_first_passage(d):
    d = d.copy(); fp = {}
    bl    = d["bal_last"] if "bal_last" in d else d["m1_daily_avg_bal"]
    drift = d["net_recent3"]                      # aylik beklenen net akis
    sig   = d["net_std6"].clip(lower=1.0)         # aylik akis oynakligi
    bmean = d["bal_mean"] if "bal_mean" in d else bl
    for name, thr in [("zero", 0.0), ("q25", 0.25), ("q50", 0.50)]:
        z = (thr*bmean - (bl + drift)) / sig      # 1 aylik ufuk
        fp[f"fp_z_{name}"] = z.clip(-8, 8)
        fp[f"fp_p_{name}"] = norm.cdf(fp[f"fp_z_{name}"])
    return pd.concat([d, pd.DataFrame(fp, index=d.index)], axis=1)

train = add_first_passage(add_net_series(train))
test  = add_first_passage(add_net_series(test))
print("Net-akis + first-passage ozellikleri eklendi.")

# ---- v17: P1 oran serisi + P2 cok-ufuk fp/projeksiyon (Lab-5) --------
def add_ratio_series(d):
    d=d.copy(); rf={}; ratios=[]
    for m in range(1,7):
        inn  = d[f"m{m}_received_total_value"]+d[f"m{m}_deposit_total_value"]+d[f"m{m}_transfer_from_bank_total_value"]
        outt = d[f"m{m}_mm_send_total_value"]+d[f"m{m}_withdraw_total_value"]+d[f"m{m}_paybill_total_value"]+d[f"m{m}_merchantpay_total_value"]
        r = outt/(inn+1)
        rf[f"outin_m{m}"]=r; ratios.append(r)
    R=np.column_stack([v.values for v in ratios])
    w=np.array([2.5,1.5,0.5,-0.5,-1.5,-2.5])/17.5
    rf["outin_slope6"]   = R @ w
    rf["outin_recent3"]  = R[:,:3].mean(axis=1)
    rf["outin_gt1_cnt"]  = (R>1).sum(axis=1)
    rf["outin_gt1_rec"]  = (R[:,:3]>1).sum(axis=1)
    rf["outin_max"]      = R.max(axis=1)
    rf["net_accel"]      = d["net_m1"]-d["net_m2"]
    rf["net_accel2"]     = (d["net_m1"]+d["net_m2"])/2 - (d["net_m3"]+d["net_m4"])/2
    return pd.concat([d,pd.DataFrame(rf,index=d.index)],axis=1)

def add_fp_multi(d):
    d=d.copy(); fp={}
    bl    = d["bal_last"]; bmean = d["bal_mean"]
    drift = d["net_recent3"]; sig = d["net_std6"].clip(lower=1.0)
    drift_proj = d["net_m1"] + d["net_slope6"]          # trend-projeksiyonlu drift
    # ek esikler h=1
    for name,thr in [("q10",0.10),("q75",0.75)]:
        z=((thr*bmean-(bl+drift))/sig).clip(-8,8)
        fp[f"fp_z_{name}"]=z; fp[f"fp_p_{name}"]=norm.cdf(z)
    # cok ufuk (h=2,3) esik=0
    for h in [2,3]:
        z=((0-(bl+drift*h))/(sig*np.sqrt(h))).clip(-8,8)
        fp[f"fp_z_h{h}"]=z; fp[f"fp_p_h{h}"]=norm.cdf(z)
    # projeksiyon: 3 aylik tahmini minimum bakiye
    pb1=bl+drift_proj; pb2=bl+2*drift_proj; pb3=bl+3*drift_proj
    pmin=np.minimum(np.minimum(pb1,pb2),pb3)
    fp["proj_min3"]=pmin
    fp["proj_min3_norm"]=pmin/(bmean+1)
    fp["proj_neg3"]=(pmin<0).astype(int)
    return pd.concat([d,pd.DataFrame(fp,index=d.index)],axis=1)

train = add_fp_multi(add_ratio_series(train))
test  = add_fp_multi(add_ratio_series(test))
print("Oran serisi + cok-ufuk fp/projeksiyon eklendi (P3 etkilesimler ELENDI).")

# ---- v18: RUNWAY + SOK (Lab-6 kazananlari; Q3 alt-sigma ELENDI) ------
OUT_FAMS = ["mm_send","withdraw","paybill","merchantpay"]
def add_runway(d):
    d=d.copy(); rf={}
    outs=[]
    for m in range(1,7):
        o = sum(d[f"m{m}_{tx}_total_value"] for tx in OUT_FAMS)
        outs.append(o)
    O = np.column_stack([v.values for v in outs])
    out_rec3 = O[:,:3].mean(axis=1)
    rf["out_recent3"]    = out_rec3
    rf["out_slope6"]     = O @ (np.array([2.5,1.5,0.5,-0.5,-1.5,-2.5])/17.5)
    rf["runway_m1"]      = d["bal_last"]/(O[:,0]+1)          # son ay giderine gore
    rf["runway_recent3"] = d["bal_last"]/(out_rec3+1)        # son 3 ay ortalamasina gore
    rf["runway_min"]     = d["bal_min"]/(out_rec3+1)         # dip bakiye ile
    rf["runway_lt1"]     = (rf["runway_recent3"]<1).astype(int)   # 1 aydan az nefes
    rf["runway_lt2"]     = (rf["runway_recent3"]<2).astype(int)
    return pd.concat([d,pd.DataFrame(rf,index=d.index)],axis=1)

def add_shock(d):
    d=d.copy(); sf={}
    hi1 = np.maximum.reduce([d[f"m1_{tx}_highest_amount"].values for tx in OUT_FAMS])
    hi3 = np.maximum.reduce([d[f"m{m}_{tx}_highest_amount"].values for tx in OUT_FAMS for m in [1,2,3]])
    sf["shock_m1"]     = hi1/(d["bal_last"]+1)               # tek islem bakiyeyi yutar mi
    sf["shock_rec3"]   = hi3/(d["bal_last"]+1)
    sf["shock_gt_bal"] = (hi1 > d["bal_last"]).astype(int)
    for tx in ["withdraw","mm_send","paybill"]:              # islem konsantrasyonu
        sf[f"conc_{tx}_m1"] = d[f"m1_{tx}_highest_amount"]/(d[f"m1_{tx}_total_value"]+1)
    return pd.concat([d,pd.DataFrame(sf,index=d.index)],axis=1)

train = add_shock(add_runway(train))
test  = add_shock(add_runway(test))
print("Runway + tek-islem soku eklendi.")




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

# ===================== LAB-8: MUTABAKAT + TOPLAM SERI + BUDAMA ======
# Onceki uc turun dersi: 5-fold/1-seed lab, VARYANS AZALTAN her seyi
# abartiyor; sadece GERCEK BILGI transfer oluyor. Bu yuzden lab-8:
#   (a) sadece BILGI ekleyen/cikaran adaylari test eder,
#   (b) kazanani URETIM kosullarinda (10-fold x 3 seed) DOGRULAR.
#
# R1) BAKIYE-AKIS MUTABAKATI: residual_m = (bal_m - bal_m+1) - net_m
#     -> mobil para disindaki gizli nakit hareketi (yeni bilgi)
# R2) AILELER-ARASI AYLIK TOPLAM SERILERI: aylik toplam islem sayisi,
#     tekil muhatap sayisi, aktif aile cesitliligi (+ egim/trend)
# R3) OLU OZELLIK BUDAMA: cust_te_* (AUC 0.5062) ve grp_* (snapshotlar
#     bagimsiz) ailelerini at
# D)  TANI: GroupKFold(musteri) vs rastgele KFold -> CV'miz iyimser mi?
#
# Sure (40 vCPU): eleme ~15 dk + dogrulama ~20 dk
# =====================================================================
from sklearn.model_selection import GroupKFold

BEST_LGB = {
    "objective": "binary", "verbose": -1, "bagging_freq": 1,
    "learning_rate":     0.005, "num_leaves": 19,
    "feature_fraction":  0.40233986364468105,
    "bagging_fraction":  0.7431947609091499,
    "min_child_samples": 103,
    "lambda_l1":         0.0914017685854035,
    "lambda_l2":         1.7137107277413957,
}
def lb_score(yy,pp): return 0.4*roc_auc_score(yy,pp) - log_loss(yy,pp)

# ===================== LAB-9: TAVAN TESHISI =========================
# Lab-8'de uc bilgi adayi da negatif cikti. Artik "daha ne bulabiliriz"
# yerine ASIL soruyu cevapliyoruz:
#
#   Modelimiz, KENDI olasilik dagiliminin izin verdigi maksimuma
#   ulasmis mi? Yani MODEL mi yetersiz, VERI mi?
#
# Yontem (sentetikte dogrulandi): y_sim ~ Bernoulli(p_hat) uretip
# AUC(y_sim, p_hat) hesaplanir. Bu, "p_hat dogru olsaydi mukemmel bir
# modelin alacagi AUC"dir = kendi-tutarli tavan.
#   gercek AUC ~ tavan  -> MODEL doymus, kazanc ancak YENI OZELLIKTEN
#   gercek AUC << tavan -> siralamada bosluk var, MODEL tarafinda is var
#
# Ayni sekilde LogLoss icin: entropi(p_hat) = indirgenemez alt sinir.
# Sure: ~4 dk (tek OOF kosusu + aninda analiz)
# =====================================================================

# ---- ortak degerlendirme --------------------------------------------
# NOT: fit fonksiyonu MODUL SEVIYESINDE ve veriyi global _LAB_X uzerinden
# aliyor (v19'un kanitlanmis deseni). Ic ice closure kullanilsaydi loky
# her gorevde ~180 MB'lik dataframe'i yeniden serilestirirdi.
_LAB_X = None

def _lab_fit(tr, va, seed):
    Xd = _LAB_X
    m = lgb.train(dict(BEST_LGB, metric=["binary_logloss","auc"], seed=seed,
                       num_threads=THREADS_PER_MODEL),
        lgb.Dataset(Xd.iloc[tr], y.iloc[tr]), num_boost_round=30000,
        valid_sets=[lgb.Dataset(Xd.iloc[va], y.iloc[va])],
        callbacks=[lgb.early_stopping(600, first_metric_only=True), lgb.log_evaluation(0)])
    return va, m.predict(Xd.iloc[va])

def eval_cv(Xd, tag, n_folds=5, seeds=(42,), groups=None, quiet=False):
    """Tum (seed, fold) isleri TEK Parallel cagrisinda toplanir; boylece
    5-fold'da bile 10 iscinin hepsi dolar (onceden yarisi bostaydi)."""
    global _LAB_X
    _LAB_X = Xd
    jobs = []
    for sd in seeds:
        if groups is None:
            splits = StratifiedKFold(n_folds, shuffle=True, random_state=sd).split(Xd, y)
        else:
            splits = GroupKFold(n_splits=n_folds).split(Xd, y, groups)
        jobs += [(tr, va, sd) for tr, va in splits]
    res = Parallel(n_jobs=N_WORKERS, verbose=0)(delayed(_lab_fit)(a,b,c) for a,b,c in jobs)
    oof = np.zeros(len(Xd))
    for va, pv in res:
        oof[va] += pv/len(seeds)
    lb = lb_score(y, oof)
    if not quiet:
        print(f"{tag:26s} | LogLoss {log_loss(y,oof):.5f} | AUC {roc_auc_score(y,oof):.5f} | LB {lb:.5f}")
    return lb, oof

# Eleme 2 seed ile: 5 fold x 2 seed = 10 is -> 10 iscinin hepsi dolar,
# duvar saati ayni kalir ama eleme gurultusu ~yariya iner.
SCREEN = dict(n_folds=5, seeds=(42, 2024))


print("\n" + "="*70)
print("OOF hesaplaniyor (5-fold x 2 seed)...")
print("="*70)
lb, oof = eval_cv(X, "mevcut set", n_folds=5, seeds=(42, 2024))
p = np.clip(oof, 1e-6, 1-1e-6)
auc_real = roc_auc_score(y, p)
ll_real  = log_loss(y, p)

rng9 = np.random.default_rng(123)
def self_ceiling(ph, reps=12):
    return float(np.mean([roc_auc_score((rng9.uniform(size=len(ph)) < ph).astype(int), ph)
                          for _ in range(reps)]))

ceil_auc = self_ceiling(p)
ent = float(np.mean(-(p*np.log(p) + (1-p)*np.log(1-p))))

print("\n" + "="*70)
print("TAVAN TESHISI")
print("="*70)
print(f"  gercek AUC              : {auc_real:.5f}")
print(f"  kendi-tutarli AUC tavani: {ceil_auc:.5f}")
print(f"  BOSLUK                  : {ceil_auc-auc_real:+.5f}")
print()
print(f"  gercek LogLoss          : {ll_real:.5f}")
print(f"  entropi (indirgenemez)  : {ent:.5f}")
print(f"  BOSLUK                  : {ll_real-ent:+.5f}")

gap = ceil_auc - auc_real
print("\n  YORUM:")
if gap < -0.004:
    print("  -> p_hat FAZLA SIKISIK: siralama kalitemiz, olasiliklarimizin")
    print("     gosterdiginden IYI. Sadece olasiliklari KESKINLESTIREREK")
    print("     LogLoss kazanilabilir — yeni ozellik gerekmeden.")
elif gap > 0.004:
    print("  -> p_hat FAZLA YAYVAN (asiri guvenli): yumusatma LogLoss'u")
    print("     iyilestirebilir.")
else:
    print("  -> p_hat kendi-tutarli. Siralamada bosluk yok; kazanc ancak")
    print("     YENI BILGIDEN (yeni ozellikten) gelebilir.")

# ---- KESKINLESTIRME TARAMASI: bedava LogLoss kazanci var mi? --------
print("\n" + "="*70)
print("KESKINLESTIRME TARAMASI (logit x s) — AUC degismez, sadece LogLoss")
print("="*70)
lg0 = np.log(p/(1-p))
lb_base = lb_score(y, p)
best = (-9.0, 1.0)
for sc in np.arange(0.80, 1.461, 0.02):
    ps = np.clip(1/(1+np.exp(-lg0*sc)), 1e-6, 1-1e-6)
    v = lb_score(y, ps)
    if v > best[0]: best = (v, sc)
    if abs(sc - round(sc,1)) < 1e-9:
        print(f"   s={sc:.2f} -> LogLoss {log_loss(y,ps):.5f} | LB {v:.5f}")
bs = best[1]
p_sharp = np.clip(1/(1+np.exp(-lg0*bs)), 1e-6, 1-1e-6)
print(f"\n  en iyi s = {bs:.2f}")
print(f"  once : LogLoss {ll_real:.5f} | AUC {auc_real:.5f} | LB {lb_base:.5f}")
print(f"  sonra: LogLoss {log_loss(y,p_sharp):.5f} | AUC {roc_auc_score(y,p_sharp):.5f} | LB {best[0]:.5f}")
print(f"  BEDAVA KAZANC: {best[0]-lb_base:+.5f}")
if best[0]-lb_base > 0.0005:
    print("  -> ANLAMLI: v20'de stack ciktisina bu olcekleme uygulanmali")
else:
    print("  -> ihmal edilebilir: olasiliklarimiz zaten dogru olcekte")

# gercek yayilim tahmini: kendi-tutarli tavani gercek AUC'ye esitleyen s
lo, hi = 0.5, 4.0
mid = 1.0
for _ in range(26):
    mid = (lo+hi)/2
    pm = np.clip(1/(1+np.exp(-lg0*mid)), 1e-6, 1-1e-6)
    if self_ceiling(pm, reps=4) < auc_real: lo = mid
    else: hi = mid
sig_hat  = float(lg0.std())
sig_true = float((lg0*mid).std())
print(f"\n  p_hat logit yayilimi   : sigma = {sig_hat:.3f}")
print(f"  tahmini GERCEK yayilim : sigma = {sig_true:.3f}")
print(f"  1. sira icin gereken   : sigma ~ 2.74")
print(f"  kalan bilgi acigi      : ~%{100*(2.74/max(sig_true,1e-9)-1):.1f}")

# hangi satirlarda kaybediyoruz? kalibrasyon dilimleri
print("\n" + "="*70)
print("NEREDE KAYBEDIYORUZ? (tahmin dilimlerine gore LogLoss katkisi)")
print("="*70)
q = pd.qcut(p, 10, duplicates="drop")
d = pd.DataFrame({"p": p, "y": y.values,
                  "loss": -(y.values*np.log(p) + (1-y.values)*np.log(1-p))})
g = d.groupby(q, observed=True).agg(n=("y","size"), ort_p=("p","mean"),
                                    gercek=("y","mean"), toplam_kayip=("loss","sum"))
g["kayip_pay"] = (g["toplam_kayip"]/g["toplam_kayip"].sum()*100).round(1)
g["kalibrasyon"] = (g["gercek"]-g["ort_p"]).round(4)
print(g[["n","ort_p","gercek","kalibrasyon","kayip_pay"]].round(4).to_string())
print("\n  kalibrasyon sutunu ~0 olmali; kayip_pay toplam kaybin yuzdesi")
print("="*70)
