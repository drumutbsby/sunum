# Laya değerlendirme paneli

`NandhaKishorM/laya` deposunun genel değerlendirmesi ve depodaki benchmark çıktılarının
tek sayfalık paneli. GitHub Pages ile yayımlanır (`.github/workflows/pages.yml`).

- `index.html` — panel (bağımlılık yok, `file://` ile de açılır)
- `data.js` — `tools/build_panel_data.py` tarafından laya klonundan üretilen veri
- `tools/build_panel_data.py` — veri üretici; `python docs/tools/build_panel_data.py <laya-klonu>`

Yayımlamak için depo ayarlarında **Settings → Pages → Source: GitHub Actions** seçilmelidir.

## Simülasyon

- `sim/scenarios.json` — 54 senaryo (tr, en, de, es, fr, hi, ar), iki soru seti (`destek`, `guvenlik`), beklenen etiketler
- `tools/simulate.py` — senaryoları gerçek checkpoint'lerle koşturur; `python docs/tools/simulate.py --threads 4`
- `sim/results.json` ve `sim_data.js` — ham cevaplar, olasılıklar, gecikme ve özet metrikler (panelin Simülasyon sekmesi)
