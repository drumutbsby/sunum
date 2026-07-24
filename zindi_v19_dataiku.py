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

# ---------------- 6a) Parametreler: RETUNE veya gomulu ---------------
RETUNE = False   # v17 aramasinin sonuclari asagida gomulu

BEST_LGB = {
    "objective": "binary", "verbose": -1, "bagging_freq": 1,
    "learning_rate":     0.005,                 # yavas rejim korunur
    "num_leaves":        19,
    "feature_fraction":  0.40233986364468105,
    "bagging_fraction":  0.7431947609091499,
    "min_child_samples": 103,
    "lambda_l1":         0.0914017685854035,
    "lambda_l2":         1.7137107277413957,
}
BEST_CAT = {
    "learning_rate":       0.01,                # yavas rejim korunur
    "depth":               8,
    "l2_leaf_reg":         10.779361932748845,
    "random_strength":     0.22948683681130552,
    "bagging_temperature": 0.36364993441420124,
}

if RETUNE:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    LGB_TRIALS, CAT_TRIALS = 20, 12

    def lgb_objective(trial):
        p={"objective":"binary","metric":"binary_logloss","verbose":-1,"seed":42,"bagging_freq":1,
           "learning_rate":trial.suggest_float("learning_rate",0.02,0.08,log=True),
           "num_leaves":trial.suggest_int("num_leaves",16,128),
           "feature_fraction":trial.suggest_float("feature_fraction",0.4,0.95),
           "bagging_fraction":trial.suggest_float("bagging_fraction",0.6,0.95),
           "min_child_samples":trial.suggest_int("min_child_samples",15,200),
           "lambda_l1":trial.suggest_float("lambda_l1",1e-2,10.0,log=True),
           "lambda_l2":trial.suggest_float("lambda_l2",1e-2,10.0,log=True)}
        lls=[]
        for i,(tr,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=42).split(X,y)):
            m=lgb.train(p,lgb.Dataset(X.iloc[tr],y.iloc[tr]),num_boost_round=3000,
                valid_sets=[lgb.Dataset(X.iloc[va],y.iloc[va])],
                callbacks=[lgb.early_stopping(100,verbose=False),lgb.log_evaluation(0)])
            lls.append(m.best_score["valid_0"]["binary_logloss"])
            trial.report(float(np.mean(lls)),i)
            if trial.should_prune(): raise optuna.TrialPruned()
        return float(np.mean(lls))

    def cat_objective(trial):
        p=dict(iterations=4000,loss_function="Logloss",eval_metric="Logloss",
               random_seed=42,verbose=False,early_stopping_rounds=100,
               learning_rate=trial.suggest_float("learning_rate",0.02,0.08,log=True),
               depth=trial.suggest_int("depth",4,8),
               l2_leaf_reg=trial.suggest_float("l2_leaf_reg",0.5,20.0,log=True),
               random_strength=trial.suggest_float("random_strength",0.1,5.0,log=True),
               bagging_temperature=trial.suggest_float("bagging_temperature",0.0,2.0))
        lls=[]
        for i,(tr,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=42).split(X,y)):
            m=CatBoostClassifier(**p)
            m.fit(X.iloc[tr],y.iloc[tr],eval_set=(X.iloc[va],y.iloc[va]),use_best_model=True)
            lls.append(log_loss(y.iloc[va],m.predict_proba(X.iloc[va])[:,1]))
            trial.report(float(np.mean(lls)),i)
            if trial.should_prune(): raise optuna.TrialPruned()
        return float(np.mean(lls))

    print(f"\nRETUNE: LGB araniyor ({LGB_TRIALS} deneme)...")
    st1=optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
    st1.optimize(lgb_objective,n_trials=LGB_TRIALS)
    print("LGB yeni parametreler:",st1.best_params)
    BEST_LGB=dict({"objective":"binary","verbose":-1,"bagging_freq":1},**st1.best_params)
    BEST_LGB["learning_rate"]=0.005          # yapi yeni, rejim YAVAS kalir

    print(f"RETUNE: CAT araniyor ({CAT_TRIALS} deneme)...")
    st2=optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
    st2.optimize(cat_objective,n_trials=CAT_TRIALS)
    print("CAT yeni parametreler:",st2.best_params)
    BEST_CAT=dict(st2.best_params); BEST_CAT["learning_rate"]=0.01
else:
    print("\nRETUNE=False: v16 parametreleri kullaniliyor.")

# ---------------- 6b) Ensemble: 5-seed LGB(tuned) + 5-seed CAT(tuned) -
N_FOLDS = 10   # zaman darsa 5
SEEDS = [42, 2024, 777, 1453, 5]   # zaman darsa ilk 3'u birak

def _fit_lgb(tr, va, seed):
    p=dict(BEST_LGB, metric=["binary_logloss","auc"], seed=seed,
           num_threads=THREADS_PER_MODEL)
    m=lgb.train(p,lgb.Dataset(X.iloc[tr],y.iloc[tr]),num_boost_round=30000,
        valid_sets=[lgb.Dataset(X.iloc[va],y.iloc[va])],
        callbacks=[lgb.early_stopping(600,first_metric_only=True),lgb.log_evaluation(0)])
    return va, m.predict(X.iloc[va]), m.predict(X_test)

def cv_lgb_seeds():
    jobs=[]
    for seed in SEEDS:
        skf=StratifiedKFold(N_FOLDS,shuffle=True,random_state=seed)
        for tr,va in skf.split(X,y):
            jobs.append((tr,va,seed))
    res=Parallel(n_jobs=N_WORKERS, verbose=5)(delayed(_fit_lgb)(tr,va,sd) for tr,va,sd in jobs)
    oof=np.zeros(len(X)); tp=np.zeros(len(X_test))
    for va,pv,pt in res:
        oof[va]+=pv/len(SEEDS); tp+=pt/(len(SEEDS)*N_FOLDS)
    print("    LGB paralel tamam")
    return oof,tp

def _fit_cat(tr, va, seed):
    m=CatBoostClassifier(iterations=30000, loss_function="Logloss",
        eval_metric="Logloss", random_seed=seed,
        early_stopping_rounds=600, verbose=False,
        thread_count=THREADS_PER_MODEL, **BEST_CAT)
    m.fit(X.iloc[tr],y.iloc[tr],eval_set=(X.iloc[va],y.iloc[va]),use_best_model=True)
    return va, m.predict_proba(X.iloc[va])[:,1], m.predict_proba(X_test)[:,1]

def cv_cat_seeds():
    jobs=[]
    for seed in SEEDS:
        skf=StratifiedKFold(N_FOLDS,shuffle=True,random_state=seed)
        for tr,va in skf.split(X,y):
            jobs.append((tr,va,seed))
    res=Parallel(n_jobs=N_WORKERS, verbose=5)(delayed(_fit_cat)(tr,va,sd) for tr,va,sd in jobs)
    oof=np.zeros(len(X)); tp=np.zeros(len(X_test))
    for va,pv,pt in res:
        oof[va]+=pv/len(SEEDS); tp+=pt/(len(SEEDS)*N_FOLDS)
    print("    CAT paralel tamam")
    return oof,tp

# ---- v19: uc yeni stack sesi (paralel) -------------------------------
def _fit_lgb_mode(tr, va, seed, mode):
    if mode=="dart":
        p=dict(BEST_LGB, boosting="dart", drop_rate=0.1, skip_drop=0.5,
               learning_rate=0.03, metric=["binary_logloss","auc"],
               seed=seed, num_threads=THREADS_PER_MODEL)
        m=lgb.train(p,lgb.Dataset(X.iloc[tr],y.iloc[tr]),num_boost_round=1200,
            valid_sets=[lgb.Dataset(X.iloc[va],y.iloc[va])],
            callbacks=[lgb.log_evaluation(0)])
    else:
        p=dict(BEST_LGB, metric=["binary_logloss","auc"], seed=seed,
               num_threads=THREADS_PER_MODEL)
        if mode=="goss":
            p["boosting"]="goss"; p.pop("bagging_fraction",None); p.pop("bagging_freq",None)
        elif mode=="xtrees":
            p["extra_trees"]=True
        m=lgb.train(p,lgb.Dataset(X.iloc[tr],y.iloc[tr]),num_boost_round=30000,
            valid_sets=[lgb.Dataset(X.iloc[va],y.iloc[va])],
            callbacks=[lgb.early_stopping(600,first_metric_only=True),lgb.log_evaluation(0)])
    return va, m.predict(X.iloc[va]), m.predict(X_test)

def cv_mode_seeds(mode):
    jobs=[]
    for seed in SEEDS:
        skf=StratifiedKFold(N_FOLDS,shuffle=True,random_state=seed)
        for tr,va in skf.split(X,y):
            jobs.append((tr,va,seed))
    res=Parallel(n_jobs=N_WORKERS, verbose=5)(delayed(_fit_lgb_mode)(tr,va,sd,mode) for tr,va,sd in jobs)
    oof=np.zeros(len(X)); tp=np.zeros(len(X_test))
    for va,pv,pt in res:
        oof[va]+=pv/len(SEEDS); tp+=pt/(len(SEEDS)*N_FOLDS)
    print(f"    {mode} paralel tamam")
    return oof,tp

def lb_score(yy,pp): return 0.4*roc_auc_score(yy,pp) - log_loss(yy,pp)

print("\nLightGBM..."); oof_l,tp_l = cv_lgb_seeds()
print("CatBoost...");  oof_c,tp_c = cv_cat_seeds()
print("DART...");      oof_d,tp_d = cv_mode_seeds("dart")
print("GOSS...");      oof_g,tp_g = cv_mode_seeds("goss")
print("XTREES...");    oof_x,tp_x = cv_mode_seeds("xtrees")
for name,o in [("LGB",oof_l),("CAT",oof_c),("DART",oof_d),("GOSS",oof_g),("XTR",oof_x)]:
    print(f"  {name:4s} | LogLoss {log_loss(y,o):.5f} | AUC {roc_auc_score(y,o):.5f} | LB {lb_score(y,o):.5f}")

# 5 SESLI agirlik aramasi: 0.1 adimli tam simpleks taramasi (1001 nokta),
# ardindan en iyinin cevresinde 0.025 adimli yerel rafinasyon. Tamamen
# deterministik; 40k satirda ~4100 nokta -> ~2 dakika (kosunun %2'si).
from itertools import product
VOICES = [("LGB",oof_l,tp_l), ("CAT",oof_c,tp_c), ("DART",oof_d,tp_d),
          ("GOSS",oof_g,tp_g), ("XTR",oof_x,tp_x)]
OOFS = np.column_stack([v[1] for v in VOICES])
TPS  = np.column_stack([v[2] for v in VOICES])

def _best_weights(step, center=None, span=None):
    n = OOFS.shape[1]
    if center is None:                      # tam simpleks taramasi
        k = int(round(1/step))
        grids = [np.arange(0, k+1) for _ in range(n-1)]
        best = (-9, None)
        for combo in product(*grids):
            if sum(combo) > k: continue
            w = np.array(list(combo) + [k-sum(combo)], dtype=float)/k
            sc_ = lb_score(y, OOFS @ w)
            if sc_ > best[0]: best = (sc_, w)
        return best
    lo = np.clip(center-span, 0, 1); hi = np.clip(center+span, 0, 1)
    grids = [np.arange(lo[i], hi[i]+1e-9, step) for i in range(n)]
    best = (-9, None)
    for combo in product(*grids):
        w = np.array(combo, dtype=float)
        tot = w.sum()
        if tot <= 0: continue
        w = w/tot                            # simplekse normalize et
        sc_ = lb_score(y, OOFS @ w)
        if sc_ > best[0]: best = (sc_, w)
    return best

_, w_coarse = _best_weights(0.10)
_, w_fine   = _best_weights(0.025, center=w_coarse, span=0.05)
W = w_fine if lb_score(y, OOFS @ w_fine) >= lb_score(y, OOFS @ w_coarse) else w_coarse
oof_b = OOFS @ W
tp_b  = TPS  @ W
print("\nEn iyi 5-sesli blend -> " + "  ".join(
      f"{VOICES[i][0]}:{W[i]:.3f}" for i in range(len(VOICES))))
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

oof_s, tp_s = cross_fit_stack([oof_l, oof_c, oof_d, oof_g, oof_x], y, [tp_l, tp_c, tp_d, tp_g, tp_x])
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
print(f">>> v19 CV | LogLoss: {log_loss(y,oof_f):.5f} | AUC: {roc_auc_score(y,oof_f):.5f} | LB: {lb_score(y,oof_f):.5f}")
print("    (v17/v18 idi: LogLoss 0.24014 | AUC 0.91385 | LB 0.12540  <- gecilecek hedef)")

# ---------------- 8) Submission --------------------------------------
pred_col = [c for c in sample.columns if c != ID_COL][0]
sub = sample.copy()
sub[pred_col] = sub[ID_COL].map(dict(zip(test[ID_COL], tp_f)))
# GUVENLIK: SampleSubmission'daki her ID icin tahmin uretilmis olmali.
# Bos kalan varsa dosyalar uyumsuz demektir (eski sample + yeni test gibi).
n_missing = int(sub[pred_col].isna().sum())
assert n_missing == 0, (f"{n_missing} ID icin tahmin yok! Zindi'den Train/Test/"
                        f"SampleSubmission'in HEPSINI guncel haliyle yeniden indir.")
print("\nsubmission hazir:\n", sub.head())
try:
    folder = dataiku.Folder("submissions")      # Flow'da bu adla Managed Folder olustur
    with folder.get_writer("submission.csv") as w:
        w.write(sub.to_csv(index=False).encode("utf-8"))
    print(">> 'submissions' klasorune yazildi — Flow'dan indirip Zindi'ye yukle.")
except Exception as e:
    sub.to_csv("submission.csv", index=False)
    print(f">> Managed Folder yok ({type(e).__name__}) — yerel diske yazildi; asagida INDIRME LINKI:")
    try:
        import base64
        from IPython.display import display, HTML
        b64 = base64.b64encode(sub.to_csv(index=False).encode("utf-8")).decode()
        display(HTML(f'<a download="submission.csv" href="data:text/csv;base64,{b64}">'
                     f'&#128229; submission.csv INDIR (tikla)</a>'))
    except Exception:
        pass
