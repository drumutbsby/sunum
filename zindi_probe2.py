# =====================================================================
# SONDA v2 (~2-3 dk, egitim yok)
#  A) Pencere kaymasi k=1..4 icin ortusme testi (3 aile birden esles-
#     tirilir: bakiye + received + deposit -> sahte eslesme ihtimali ~0)
#  B) Satir-sirasi sizintisi: hedef, satir indeksiyle iliskili mi?
#  C) ID sizintisi: ID'nin alfabetik sirasi hedefle iliskili mi?
# =====================================================================
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

TARGET = "liquidity_stress_next_30d"
tr = pd.read_csv("Train.csv"); te = pd.read_csv("Test.csv")
PROF = ["arpu","age","gender","region","smartphone","segment","earning_pattern","x_90_d_activity_rate"]
for d in (tr, te):
    d["cust_id"] = d[PROF].astype(str).agg("|".join, axis=1)

tr["src"]="train"; te["src"]="test"; te[TARGET]=np.nan
allc = pd.concat([tr, te], ignore_index=True)

FAMS = ["daily_avg_bal", "received_total_value", "deposit_total_value"]
COLS = {f: [f"m{m}_{f}" for m in range(1,7)] for f in FAMS}

print("="*68)
print("A) PENCERE KAYMASI TESTLERI (k = kac ay kayma)")
for k in [1, 2, 3, 4]:
    matched = 0
    nxt = np.full(len(allc), np.nan)
    for cid, g in allc.groupby("cust_id", sort=False):
        idx = g.index.to_numpy()
        if len(idx) < 2: continue
        # B (yeni) [k:] == A (eski) [:6-k]  — 3 aile ust uste
        Amat = np.column_stack([g[COLS[f]].to_numpy()[:, :6-k] for f in FAMS])
        Bmat = np.column_stack([g[COLS[f]].to_numpy()[:, k:]   for f in FAMS])
        for a in range(len(idx)):
            for b in range(len(idx)):
                if a == b: continue
                if np.allclose(Bmat[b], Amat[a], rtol=1e-6, atol=1e-6):
                    nxt[idx[a]] = allc.at[idx[b], "m1_daily_avg_bal"]
                    matched += 1
    cov = np.isfinite(nxt[(allc.src=="train").to_numpy()]).mean()
    line = f"  k={k}: eslesme={matched:6d} | train kapsami: %{100*cov:.1f}"
    if matched > 500:
        m = (allc.src=="train").to_numpy() & np.isfinite(nxt)
        auc = roc_auc_score(allc.loc[m, TARGET].astype(int), -nxt[m])
        line += f" | tek-ozellik AUC: {auc:.4f}  <-- SIZINTI!"
    print(line)

print("\nB) SATIR SIRASI SIZINTISI (train)")
y = tr[TARGET].astype(int).to_numpy()
idx_auc = roc_auc_score(y, np.arange(len(tr)))
print(f"  AUC(hedef ~ satir indeksi): {idx_auc:.4f}  (0.50=iliski yok)")
# blok yapisi: ardisik 100'luk bloklarin stres orani varyansi
blocks = pd.Series(y).groupby(np.arange(len(y))//1000).mean()
print(f"  1000'lik blok stres oranlari min/max: {blocks.min():.3f} / {blocks.max():.3f} (taban {y.mean():.3f})")

print("\nC) ID SIZINTISI")
order = tr["ID"].rank().to_numpy()
print(f"  AUC(hedef ~ ID alfabetik sira): {roc_auc_score(y, order):.4f}  (0.50=iliski yok)")
# ID'nin hex kismindan sayi turet
hexv = tr["ID"].str.replace("ID_","",regex=False).apply(lambda s: int(s[:8],16) if all(c in '0123456789ABCDEF' for c in s[:8]) else np.nan)
if hexv.notna().all():
    print(f"  AUC(hedef ~ ID hex degeri):     {roc_auc_score(y, hexv):.4f}")

print("\nD) MUSTERI SNAPSHOT BENZERLIGI (pencereler ortusmuyorsa ne kadar farkli?)")
# ayni musterinin iki snapshot'inin bal serileri arasi korelasyon dagilimi
cors = []
rng = np.random.default_rng(0)
sample_cids = rng.choice(allc["cust_id"].unique(), 300, replace=False)
bal = COLS["daily_avg_bal"]
for cid in sample_cids:
    g = allc[allc.cust_id==cid]
    M = g[bal].to_numpy()
    for a in range(len(M)):
        for b in range(a+1, len(M)):
            ca = np.corrcoef(M[a], M[b])[0,1]
            if np.isfinite(ca): cors.append(ca)
cors = np.array(cors)
print(f"  ortalama korelasyon: {cors.mean():.3f} | medyan: {np.median(cors):.3f}")
print("  (~0 = snapshotlar bagimsiz uretilmis; yuksek = ortak seri var, baska hizada)")
print("="*68)
print("Ciktinin tamamini yapistir. Sonra LAB'i kostur (E1 tablolari kritik).")
