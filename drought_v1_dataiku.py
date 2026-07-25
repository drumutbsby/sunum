# =====================================================================
# ITU DROUGHT v1 — DATAIKU (Global Water Storage / TWS t+1 tahmini)
#
# Onceki analizin DOGRULANMIS bulgulari uzerine kurulu:
#  1) Test maskeleme kurali: onceki ay da test'te ise TWS_t maskeli
#     -> test satirlari zincir olusturur, gercek ufuk 1-7 ay
#        (H1 ~%33, H2 ~%22, H3 ~%17, H4 ~%11, H5-7 ~%5.5'er)
#  2) Hedef her zaman takvim t+1; eksik (GRACE bosluk) aylarin TWS'si
#     hedeflerden geri kazanilir -> kesintisiz aylik eksen
#  3) Degisim alani uzamsal-duzgun (sinyalin ~%79'u buyuk olcekli)
#     -> cok olcekli komsu ortalama ozellikleri (r=1,2,4,8)
#  4) Gelecek kovaryati KULLANILMAZ (sizinti politikasi: Zindi
#     "use of data leaks" diskalifiye maddesi)
#  5) Ufuk basina ayri LightGBM modeli; egitim ornekleri test
#     zincirlerini taklit eder (anchor = T-h)
#
# Dataiku dataset adlari: Train_drought / Test_drought
# (SampleSubmission_drought varsa kolon adi/sirasi ondan alinir)
# =====================================================================
import os, io, base64, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.ndimage import uniform_filter

warnings.filterwarnings("ignore")

THREADS    = os.cpu_count() or 8
RADII      = (1, 2, 4, 8)     # komsu ortalama yaricaplari (hucre sayisi)
HMAX       = 7                # ufuk ustu siniri (zincir kuralindan)
VAL_MONTHS = 24               # dogrulama penceresi (train sonundan)
MIN_HIST   = 14               # hedef ay icin gereken asgari gecmis (ay)
MIN_DATA   = 200              # lgb min_data_in_leaf
SEED       = 42

# ---------------- 1) VERI ----------------
try:
    import dataiku
    tr = dataiku.Dataset("Train_drought").get_dataframe()
    te = dataiku.Dataset("Test_drought").get_dataframe()
    print("Veri Dataiku'dan okundu")
except Exception:
    tr = pd.read_csv("Train_drought.csv")
    te = pd.read_csv("Test_drought.csv")
    print("Veri CSV'den okundu")

tcols = [c for c in tr.columns if c not in te.columns
         and not c.lower().endswith("id") and tr[c].dtype.kind in "fc"]
assert len(tcols) == 1, f"hedef sutun belirsiz: {tcols}"
TARGET = tcols[0]
print(f"train {tr.shape} | test {te.shape} | hedef: {TARGET}")

COVS = ["SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t", "SOIL_MOISTURE_t"]
for d in (tr, te):
    d["time"] = pd.to_datetime(d["time"])

t0 = min(tr["time"].min(), te["time"].min())
def midx(s):
    return ((s.dt.year - t0.year) * 12 + (s.dt.month - t0.month)).astype(int)
tr["m"] = midx(tr["time"]); te["m"] = midx(te["time"])
n_m = int(max(tr["m"].max(), te["m"].max())) + 2      # +1: son hedef ayi

# hucre kimligi + duzenli izgara indeksi
cells = pd.concat([tr[["lat","lon"]], te[["lat","lon"]]]).drop_duplicates().reset_index(drop=True)
cells["cid"] = np.arange(len(cells))
tr = tr.merge(cells, on=["lat","lon"], how="left")
te = te.merge(cells, on=["lat","lon"], how="left")
n_c = len(cells)
LAT = cells["lat"].to_numpy(np.float32); LON = cells["lon"].to_numpy(np.float32)
step_la = np.diff(np.sort(cells["lat"].unique())).min()
step_lo = np.diff(np.sort(cells["lon"].unique())).min()
gi = np.rint((cells["lat"] - cells["lat"].min()) / step_la).astype(int).to_numpy()
gj = np.rint((cells["lon"] - cells["lon"].min()) / step_lo).astype(int).to_numpy()
NI, NJ = gi.max()+1, gj.max()+1
print(f"hucre: {n_c} | izgara: {NI}x{NJ} (adim {step_la}x{step_lo}) | ay ekseni: {n_m}")

# ---------------- 2) PANELLER ----------------
# P_full : hedefler dahil TWS (egitim y'si)   P_avail: tahmin aninda gorulebilir TWS
P_full  = np.full((n_c, n_m), np.nan, np.float32)
P_avail = np.full((n_c, n_m), np.nan, np.float32)

v = tr[np.isfinite(tr["TWS_t"].to_numpy(float))]
P_full[v["cid"].to_numpy(), v["m"].to_numpy()] = v["TWS_t"].to_numpy(np.float32)
P_full[tr["cid"].to_numpy(), tr["m"].to_numpy()+1] = tr[TARGET].to_numpy(np.float32)  # bosluk aylari geri kazanilir

te_start = int(te["m"].min())
tr_last  = int(tr["m"].max()) + 1            # train hedeflerinin ulastigi son ay
P_avail[:, :tr_last+1] = P_full[:, :tr_last+1]         # train doneminde her sey gorulebilir
u = te[np.isfinite(te["TWS_t"].to_numpy(float))]
P_avail[u["cid"].to_numpy(), u["m"].to_numpy()] = u["TWS_t"].to_numpy(np.float32)  # maskesiz test TWS

C = {}
for c in COVS:
    A = np.full((n_c, n_m), np.nan, np.float32)
    A[tr["cid"].to_numpy(), tr["m"].to_numpy()] = tr[c].to_numpy(np.float32)
    A[te["cid"].to_numpy(), te["m"].to_numpy()] = te[c].to_numpy(np.float32)
    C[c] = A

# ---------------- 3) MASKELEME KURALI DOGRULAMASI ----------------
te_key = set(zip(te["cid"].to_numpy(), te["m"].to_numpy()))
prev_in = np.fromiter(((c, m-1) in te_key for c, m in zip(te["cid"], te["m"])), bool, len(te))
masked  = ~np.isfinite(te["TWS_t"].to_numpy(float))
print("\nMASKELEME KURALI (beklenen: maskeli <=> onceki ay test'te):")
print(pd.crosstab(pd.Series(masked, name="maskeli"), pd.Series(prev_in, name="onceki_ay_testte")).to_string())

# ---------------- 4) TUREV PANELLER ----------------
calm = (np.arange(n_m) + t0.month - 1) % 12
clim = np.full((n_c, 12), np.nan, np.float32)          # iklimoloji: yalnizca train donemi
for k in range(12):
    ck = np.where((calm == k) & (np.arange(n_m) < te_start))[0]
    if len(ck): clim[:, k] = np.nanmean(P_full[:, ck], axis=1)
ANOM  = P_avail - clim[:, calm]
DELTA = np.full_like(P_avail, np.nan); DELTA[:, 1:] = P_avail[:, 1:] - P_avail[:, :-1]

def nb_panel(A, r):
    """her ay icin r-yaricapli NaN-farkindali komsu ortalamasi"""
    out  = np.full_like(A, np.nan)
    size = 2 * r + 1
    for m in range(A.shape[1]):
        G = np.full((NI, NJ), np.nan, np.float32); G[gi, gj] = A[:, m]
        fin = np.isfinite(G)
        if not fin.any(): continue
        s = uniform_filter(np.where(fin, G, 0.0), size=size, mode="constant")
        n = uniform_filter(fin.astype(np.float32), size=size, mode="constant")
        with np.errstate(invalid="ignore", divide="ignore"):
            R = np.where(n > 0, s / n, np.nan)
        out[:, m] = R[gi, gj]
    return out

print("\nKomsu panelleri hesaplaniyor (8 adet)...")
NB = {}
for r in RADII:
    NB[("anom", r)]  = nb_panel(ANOM, r)
    NB[("delta", r)] = nb_panel(DELTA, r)

# ---------------- 5) OZELLIK MOTORU ----------------
def feats(cid, a, T):
    """cid/a/T: ornek dizileri (hucre, anchor ayi, hedef ayi). Sadece <=a TWS + T-1 kovaryat."""
    F = {}
    ga = P_avail[cid, a]
    F["tws_a"]  = ga
    F["anom_a"] = ANOM[cid, a]
    for k in (1, 2, 3, 6, 12):
        ak = a - k
        F[f"d{k}"] = np.where(ak >= 0, ga - P_avail[cid, np.clip(ak, 0, None)], np.nan)
    idx = a[:, None] - np.arange(12)[None, :]
    W = np.where(idx >= 0, P_avail[cid[:, None], np.clip(idx, 0, None)], np.nan)
    F["roll6_m"]  = np.nanmean(W[:, :6], 1); F["roll6_s"]  = np.nanstd(W[:, :6], 1)
    F["roll12_m"] = np.nanmean(W, 1);        F["roll12_s"] = np.nanstd(W, 1)
    for key, r in NB:
        F[f"nb_{key}{r}"] = NB[(key, r)][cid, a]
    for c in COVS:                                     # satirin kendi kovaryatlari (T-1 ayi)
        F[c] = C[c][cid, T - 1]
    F["clim_T"]     = clim[cid, calm[T]]
    F["clim_shift"] = clim[cid, calm[T]] - clim[cid, calm[a]]
    F["lat"] = LAT[cid]; F["lon"] = LON[cid]
    F["msin"] = np.sin(2*np.pi*calm[T]/12).astype(np.float32)
    F["mcos"] = np.cos(2*np.pi*calm[T]/12).astype(np.float32)
    return pd.DataFrame(F)

# ---------------- 6) TEST UFUKLARI ----------------
avail_m = np.where(np.isfinite(P_avail), np.arange(n_m)[None, :], -1)
last_obs = np.maximum.accumulate(avail_m, axis=1)      # hucre basina son gozlemli ay <= m
cid_te = te["cid"].to_numpy(); m_te = te["m"].to_numpy()
a_te = last_obs[cid_te, m_te]
assert (a_te >= 0).all(), "anchor'siz test satiri var"
T_te = m_te + 1
h_te = T_te - a_te
h_clip = np.clip(h_te, 1, HMAX)
dist = pd.Series(h_te).value_counts().sort_index()
print("\nTEST UFUK DAGILIMI (beklenen: H1 ~%33 ... H7 ~%5.5):")
print((dist / len(te) * 100).round(1).to_string())
W_H = pd.Series(h_clip).value_counts().sort_index()    # agirlikli ozet icin

# ---------------- 7) UFUK BASINA EGITIM + TAHMIN ----------------
PARAMS = dict(objective="regression", metric="rmse", learning_rate=0.05,
              num_leaves=127, min_data_in_leaf=MIN_DATA, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=THREADS, seed=SEED, verbosity=-1)

Ts_all = np.arange(MIN_HIST, te_start)                 # hedef aylar: train donemi
pred = np.full(len(te), np.nan, np.float32)
rows = []
for h in range(1, HMAX + 1):
    gT = np.repeat(Ts_all, n_c); gc = np.tile(np.arange(n_c), len(Ts_all))
    ga = gT - h
    ok = np.isfinite(P_full[gc, gT]) & np.isfinite(P_avail[gc, ga])
    gc, ga, gT = gc[ok], ga[ok], gT[ok]
    X = feats(gc, ga, gT); y = P_full[gc, gT].astype(np.float64)
    isv = gT >= te_start - VAL_MONTHS
    dtr = lgb.Dataset(X[~isv], y[~isv])
    mdl = lgb.train(PARAMS, dtr, num_boost_round=3000,
                    valid_sets=[lgb.Dataset(X[isv], y[isv], reference=dtr)],
                    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    pv = mdl.predict(X[isv], num_iteration=mdl.best_iteration)
    rm  = float(np.sqrt(np.mean((pv - y[isv])**2)))
    rp  = float(np.sqrt(np.mean((X["tws_a"][isv] - y[isv])**2)))   # kalicilik tabani
    # nihai: tum train orneklerinde yeniden egit
    rounds = max(50, int(mdl.best_iteration * 1.15))
    mdl_f = lgb.train(PARAMS, lgb.Dataset(X, y), num_boost_round=rounds)
    sel = h_clip == h
    if sel.any():
        Xt = feats(cid_te[sel], a_te[sel], T_te[sel])
        pred[sel] = mdl_f.predict(Xt).astype(np.float32)
    rows.append((h, int(len(y)), int(mdl.best_iteration), rm, rp, int(sel.sum())))
    print(f"H={h} | n={len(y):>9,} | tur={mdl.best_iteration:>4} | val RMSE {rm:.4f} | kalicilik {rp:.4f} | test satiri {sel.sum():,}")

assert np.isfinite(pred).all(), "NaN tahmin var"

# ---------------- 8) OZET + GONDERIM ----------------
tab = pd.DataFrame(rows, columns=["H","n_egitim","tur","model_RMSE","kalicilik_RMSE","n_test"])
w = tab["n_test"] / tab["n_test"].sum()
wm = float((tab["model_RMSE"]     * w).sum())
wp = float((tab["kalicilik_RMSE"] * w).sum())

sub = pd.DataFrame({"ID": te["ID"].to_numpy(), TARGET: pred})
try:
    ss = dataiku.Dataset("SampleSubmission_drought").get_dataframe()
    assert set(ss["ID"]) == set(sub["ID"]), "SampleSubmission ID kumesi uyusmuyor"
    sub = ss[["ID"]].merge(sub, on="ID", how="left")
    sub.columns = list(ss.columns[:2])
    print("SampleSubmission_drought ile hizalandi")
except Exception as e:
    print(f"(SampleSubmission hizalamasi atlandi: {type(e).__name__})")
assert sub.iloc[:, 1].notna().all() and len(sub) == len(te)

out_name = "submission_drought_v1.csv"
written = False
try:
    fld = dataiku.Folder("submissions")
    with fld.get_writer(out_name) as wr_:
        wr_.write(sub.to_csv(index=False).encode())
    print(f"Yazildi: submissions/{out_name}"); written = True
except Exception:
    try:
        dataiku.Dataset("submission_drought").write_with_schema(sub)
        print("Yazildi: submission_drought dataset'i (Flow'dan Export edebilirsin)"); written = True
    except Exception:
        sub.to_csv(out_name, index=False)
        print(f"Yerel CSV yazildi: {os.path.abspath(out_name)}")
        try:
            from IPython.display import HTML, display
            b64 = base64.b64encode(sub.to_csv(index=False).encode()).decode()
            display(HTML(f'<a download="{out_name}" href="data:text/csv;base64,{b64}">{out_name} INDIR</a>'))
            written = True
        except Exception:
            pass

print("\n" + "=" * 70)
print("OZET  <<< SADECE BU BLOGU YAPISTIRMAN YETER >>>")
print("=" * 70)
print(tab.round(4).to_string(index=False))
print(f"\nTest-agirlikli model RMSE : {wm:.4f}")
print(f"Test-agirlikli kalicilik  : {wp:.4f}")
print(f"Kazanc                    : %{100*(1 - wm/wp):.2f}  (onceki yerel calisma: %13.3)")
print(f"Gonderim: {len(sub):,} satir | kolonlar {list(sub.columns)} | yazildi: {written}")
print("=" * 70)
