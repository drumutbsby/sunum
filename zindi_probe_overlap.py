# =====================================================================
# SONDA: Ortusen pencere sizintisi var mi?  (~1-2 dk, egitim yok)
# Hipotez: ayni musterinin snapshot'lari ardisik aylar -> B.m2==A.m1
# Dogrulanirsa: A'nin hedefi (sonraki 30 gun) = B'nin m1 ayi -> ALTIN
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

bal = [f"m{m}_daily_avg_bal" for m in range(1,7)]
# B, A'dan 1 ay sonra ise: B[m2..m6] == A[m1..m5]
B_cols, A_cols = bal[1:], bal[:5]

next_m1  = np.full(len(allc), np.nan)   # satirin "sonraki ay" bakiyesi (varsa)
has_next = np.zeros(len(allc))
n_checked = n_matched = 0

for cid, g in allc.groupby("cust_id", sort=False):
    idx = g.index.to_numpy()
    if len(idx) < 2:
        continue
    Bm = g[B_cols].to_numpy(); Am = g[A_cols].to_numpy()
    for a in range(len(idx)):
        for b in range(len(idx)):
            if a == b: continue
            n_checked += 1
            if np.allclose(Bm[b], Am[a], rtol=1e-6, atol=1e-6):
                # idx[b], idx[a]'dan 1 ay sonraki snapshot
                next_m1[idx[a]]  = allc.at[idx[b], "m1_daily_avg_bal"]
                has_next[idx[a]] = 1
                n_matched += 1

allc["next_m1_bal"] = next_m1
allc["has_next"]    = has_next

print("="*68)
print(f"Kontrol edilen sirali cift: {n_checked} | Eslesen (1-ay-ardisik): {n_matched}")
cov_tr = allc.loc[allc.src=='train','has_next'].mean()
cov_te = allc.loc[allc.src=='test' ,'has_next'].mean()
print(f"'Sonraki ay' bulunan satir orani -> TRAIN: %{100*cov_tr:.1f} | TEST: %{100*cov_te:.1f}")

m = (allc.src=="train") & (allc.has_next==1)
if m.sum() > 100:
    yv = allc.loc[m, TARGET].astype(int)
    nb = allc.loc[m, "next_m1_bal"]
    print(f"\nKapsanan train satiri: {m.sum()} | stres orani: {yv.mean():.3f}")
    print("\nSONRAKI AY BAKIYESI -> STRES ORANI (desiller):  <-- sizinti kaniti burada")
    print(yv.groupby(pd.qcut(nb, 10, duplicates='drop')).agg(['mean','count']).round(3).to_string())
    auc = roc_auc_score(yv, -nb)          # dusuk gelecek-bakiye = stres beklenir
    print(f"\n>>> TEK OZELLIK AUC (-next_m1_bal, kapsanan satirlar): {auc:.4f}")
    print("    0.75+ ise HIPOTEZ DOGRU -> v15 zaman-cizgisi ozellikleriyle geliyor")
    # oran versiyonu da olc (mevcut bakiyeye gore dusus)
    ratio = nb / (allc.loc[m,'m1_daily_avg_bal'] + 1)
    auc2 = roc_auc_score(yv, -ratio)
    print(f">>> TEK OZELLIK AUC (-next/current orani):            {auc2:.4f}")
else:
    print("\nEslesme yok/az -> hipotez YANLIS (pencereler ortusmuyor).")
    print("Bu da bilgi: top-6'nin sinyali baska yerde, lab sonuclarina donecegiz.")
print("="*68)
print("Bu ciktinin tamamini yapistir.")
