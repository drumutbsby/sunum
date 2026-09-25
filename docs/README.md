# Laya değerlendirme paneli

`NandhaKishorM/laya` deposunun genel değerlendirmesi ve depodaki benchmark çıktılarının
tek sayfalık paneli. GitHub Pages ile yayımlanır (`.github/workflows/pages.yml`).

- `index.html` — panel (bağımlılık yok, `file://` ile de açılır)
- `data.js` — `tools/build_panel_data.py` tarafından laya klonundan üretilen veri
- `tools/build_panel_data.py` — veri üretici; `python docs/tools/build_panel_data.py <laya-klonu>`

Yayımlamak için depo ayarlarında **Settings → Pages → Source: GitHub Actions** seçilmelidir.
