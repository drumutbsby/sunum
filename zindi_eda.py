# =====================================================================
# Zindi — DERIN VERI INCELEME (EDA)  |  Colab'da calistir, ciktiyi yapistir
# Cikti kompakt tutuldu: her blok modele dair bir soruya cevap verir.
# =====================================================================
import pandas as pd, numpy as np

pd.set_option("display.width", 140)
TARGET = "liquidity_stress_next_30d"

tr = pd.read_csv("Train.csv")
te = pd.read_csv("Test.csv")
print("="*70)
print("1) BOYUT & BUTUNLUK")
print(f"Train: {tr.shape} | Test: {te.shape}")
print(f"Train eksik hucre: {tr.isna().sum().sum()} | Test eksik: {te.isna().sum().sum()}")
print(f"Hedef dagilimi: {dict(tr[TARGET].value_counts(normalize=True).round(4))}")

PROF = ["arpu","age","gender","region","smartphone","segment","earning_pattern","x_90_d_activity_rate"]
tr["cust_id"] = tr[PROF].astype(str).agg("|".join, axis=1)
te["cust_id"] = te[PROF].astype(str).agg("|".join, axis=1)
print(f"\n2) MUSTERI YAPISI")
print(f"Benzersiz musteri: train={tr.cust_id.nunique()} test={te.cust_id.nunique()} "
      f"ortak={len(set(tr.cust_id)&set(te.cust_id))}")
print(f"Musteri basina snapshot: train={dict(tr.cust_id.value_counts().value_counts())} "
      f"test={dict(te.cust_id.value_counts().value_counts())}")
# ayni musterinin etiketleri ne kadar tutarli?
g = tr.groupby("cust_id")[TARGET]
agree = g.mean()
print(f"Musteri-ici etiket ortalamasi dagilimi (0=hep 0, 1=hep 1):")
print(agree.value_counts(bins=5, normalize=True).round(3).to_string())

print(f"\n3) KATEGORIK -> STRES ORANI (taban: {tr[TARGET].mean():.3f})")
for c in ["segment","earning_pattern","region","smartphone","gender"]:
    t = tr.groupby(c)[TARGET].agg(["mean","count"]).round(3).sort_values("mean")
    print(f"--- {c}:"); print(t.to_string())

print("\n4) HEDEFLE EN KORELE 25 SAYISAL OZELLIK (ham sutunlar)")
num = tr.select_dtypes("number").drop(columns=[TARGET])
cor = num.corrwith(tr[TARGET]).abs().sort_values(ascending=False).head(25)
print(cor.round(4).to_string())

print("\n5) BAKIYE PROFILI: stresli vs stressiz (medyanlar)")
bal_cols = [f"m{m}_daily_avg_bal" for m in range(1,7)]
print(tr.groupby(TARGET)[bal_cols].median().round(1).to_string())

print("\n6) SIFIR-ENFLASYONU (islem tipi hic kullanilmayan ay orani, train)")
for tx,cnt in [("paybill","m1_paybill_volume"),("merchantpay","m1_merchantpay_volume"),
               ("transfer_from_bank","m1_transfer_from_bank_volume"),("mm_send","m1_mm_send_volume"),
               ("received","m1_received_volume"),("deposit","m1_deposit_volume"),
               ("withdraw","m1_withdraw_volume")]:
    cols=[f"m{m}_{tx}_volume" for m in range(1,7)]
    zero_ay=(tr[cols]==0).values.mean()
    print(f"  {tx:20s}: aylarin %{100*zero_ay:.1f}'i sifir")

print("\n7) DEMOGRAFI: yas/arpu/aktivite — stresli vs stressiz (medyan)")
print(tr.groupby(TARGET)[["age","arpu","x_90_d_activity_rate"]].median().round(3).to_string())

print("\n8) TRAIN vs TEST DAGILIM KAYMASI (temel kolon medyanlari)")
key=["arpu","age","x_90_d_activity_rate","m1_daily_avg_bal","m6_daily_avg_bal"]
cmp=pd.DataFrame({"train":tr[key].median(),"test":te[key].median()}).round(2)
print(cmp.to_string())
print("="*70)
print("EDA BITTI - bu ciktinin tamamini kopyala/yapistir")
