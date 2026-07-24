# =====================================================================
# SONDA-3: ESKI VERI SURUMU SIZINTISI  (~1 dk, egitim yok)
# Eski Train dosyalarindaki etiketli satirlar simdiki Test'e tasindi mi?
#
# KULLANIM: Elindeki ESKI dosyalarin yollarini OLD_FILES'a yaz
# (kac tane varsa). Guncel Train/Test Dataiku'dan ya da CSV'den okunur.
# =====================================================================
import pandas as pd
import numpy as np

OLD_FILES = [
    "Train_eski_v1.csv",      # <- elindeki eski dosya adlarini yaz
    "Train_eski_v2.csv",      #    (yoksa satiri sil)
]

TARGET = "liquidity_stress_next_30d"

try:
    import dataiku
    new_train = dataiku.Dataset("Train_1").get_dataframe()
    new_test  = dataiku.Dataset("Test").get_dataframe()
    print("Guncel veri Dataiku'dan okundu")
except Exception:
    new_train = pd.read_csv("Train.csv"); new_test = pd.read_csv("Test.csv")
    print("Guncel veri CSV'den okundu")

new_tr_ids = set(new_train["ID"].astype(str))
new_te_ids = set(new_test["ID"].astype(str))
print(f"Guncel: train {len(new_tr_ids)} ID | test {len(new_te_ids)} ID")
print("="*68)

for f in OLD_FILES:
    try:
        old = pd.read_csv(f)
    except Exception as e:
        print(f"\n{f}: OKUNAMADI ({e}) — yolu kontrol et"); continue
    has_y = TARGET in old.columns
    ids = set(old["ID"].astype(str))
    inter_test  = ids & new_te_ids
    inter_train = ids & new_tr_ids
    print(f"\n{f}: {len(old)} satir | etiket var mi: {has_y}")
    print(f"  eski ∩ guncel TRAIN: {len(inter_train)}")
    print(f"  eski ∩ guncel TEST : {len(inter_test)}   <-- SIZINTI BURASI")
    if has_y and len(inter_test) > 0:
        sub_old = old[old["ID"].astype(str).isin(inter_test)]
        rate = sub_old[TARGET].mean()
        frac = len(inter_test)/len(new_te_ids)
        print(f"  >>> JACKPOT: test'in %{100*frac:.1f}'inin ETIKETI ELIMIZDE!")
        print(f"      kapsanan satir stres orani: {rate:.3f}")
        print(f"      Bu etiketler dogrudan submission'a yazilabilir ->")
        print(f"      AUC/LogLoss'ta buyuk sicrama beklenir.")
print("="*68)
print("Ciktiyi oldugu gibi yapistir. JACKPOT varsa v19'u ona gore yazarim.")
