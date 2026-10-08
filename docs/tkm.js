/* TKM sekmesi — TKM-27B v1.0/v1.1 · Jev · Clef · TKM-E, 112.328 soru (100k havuz + GEL-2 + arkadaş).
   Veri: tkm_data.js (özet/kırılım/eşli), tkm_index.js (tüm satırlar, kompakt), tkm_ornek.js (1.500 tam senaryo, anında),
         tkm_full/<parça>.json (256 parça; TÜM 112.328 satırın tam metni, satıra tıklanınca ilgili parça indirilir ve önbelleğe alınır).
   Bağımsız IIFE: sayfanın sekme mekanizması #tkm bölümünü gösterince render eder. */
(function () {
  'use strict';
  const D = window.TKM_DATA, IX = window.TKM_INDEX || [], OR = window.TKM_ORNEK || [];
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const pct = (d, n) => n ? (100 * d / n).toLocaleString('tr-TR', { minimumFractionDigits: 1, maximumFractionDigits: 2 }) : '—';
  const fi = (v) => Number(v).toLocaleString('tr-TR');
  if (!D || !$('tkm')) return;
  const SK = ['v11', 'v10', 'jev', 'clef', 'tkme'];
  const RENK = { v11: 'var(--s3)', v10: '#6ac08f', jev: 'var(--s1)', clef: 'var(--s2)', tkme: 'var(--s4)' };
  const AD = D.meta.sistemler;
  const KUMELER = ['gel2', 'gel', 'train', 'dev', 'arkadas'];
  const KISA = { gel2: 'GEL-2 (10k, 13 yeni kalıp)', gel: 'GEL (5.435, görülmemiş arketip)', train: 'Havuz train (84.300)', dev: 'Havuz dev (10.265)', arkadas: 'Arkadaş (2.328)' };
  const col = {}; D.meta.kolonlar.forEach((k, i) => { col[k] = i; });
  const inv = {}; Object.keys(D.meta.sozluk).forEach((alan) => { inv[alan] = {}; Object.keys(D.meta.sozluk[alan]).forEach((k) => { inv[alan][D.meta.sozluk[alan][k]] = k; }); });
  const ORid = {}; OR.forEach((o) => { ORid[o.id] = o; });
  let rendered = false;

  // ---------- özet
  function ozetTablo() {
    let h = '<div class="scroll"><table><thead><tr><th>Küme</th><th>n</th>' + SK.map((s) => `<th style="color:${RENK[s]}">${esc(AD[s])}</th>`).join('') + '</tr></thead><tbody>';
    KUMELER.forEach((k) => {
      const o = D.ozet[k] || {}; const n = Math.max(...SK.map((s) => (o[s] || [0, 0])[1]));
      const vals = SK.map((s) => o[s] && o[s][1] ? 100 * o[s][0] / o[s][1] : null); const best = Math.max(...vals.filter((v) => v != null));
      h += `<tr><td><b>${esc(KISA[k])}</b></td><td>${fi(n)}</td>` + SK.map((s, i) => vals[i] == null ? '<td style="color:var(--muted)">—</td>' : `<td${vals[i] === best ? ' style="font-weight:700"' : ''}>${pct(o[s][0], o[s][1])}${o[s][1] !== n ? `<span style="color:var(--muted);font-size:11px"> (n ${fi(o[s][1])})</span>` : ''}</td>`).join('') + '</tr>';
    });
    return h + '</tbody></table></div>';
  }
  function esliTablo() {
    let h = '<div class="scroll"><table><thead><tr><th>Küme</th><th>Kıyas</th><th>Fark (puan)</th><th>GA95 (küme-bootstrap)</th><th>McNemar p</th></tr></thead><tbody>';
    KUMELER.forEach((k) => (D.esli[k] || []).forEach((e) => { h += `<tr><td>${esc(KISA[k])}</td><td>${esc(e[0])}</td><td><b>${esc(e[1])}</b></td><td>${esc(e[2])}</td><td>${esc(e[3])}</td></tr>`; }));
    return h + '</tbody></table></div>';
  }
  function tiles() {
    const g2 = D.ozet.gel2, g = D.ozet.gel, tr = D.ozet.train, ak = D.ozet.arkadas;
    const t = (l, v, d) => `<div class="tile"><div class="l">${l}</div><div class="v">${v}</div><div class="d">${d}</div></div>`;
    return t('GEL-2 — 13 hiç görülmemiş karar kalıbı (10.000)', pct(...g2.v11) + ' %', `TKM-27B v1.1 · Jev ${pct(...g2.jev)} · Clef ${pct(...g2.clef)} · v1.0 ${pct(...g2.v10)} · eşli v1.1−Jev +1,35 (p 0,002)`)
      + t('GEL — görülmemiş arketipler (5.435)', pct(...g.v11) + ' %', `TKM-27B v1.1 · Jev ${pct(...g.jev)} · Clef ${pct(...g.clef)} · v1.0 ${pct(...g.v10)} · TKM-E ${pct(...g.tkme)}`)
      + t('100k havuz — eğitim dilimi, sıfır-atış (84.300)', pct(...tr.jev) + ' %', `Jev · TKM-27B v1.0 ${pct(...tr.v10)} · Clef ${pct(...tr.clef)} · (v1.1 görmediği 20k: ${pct(...tr.v11)})`)
      + t('Arkadaş — bağımsız bankacılık dosyaları (2.328)', pct(...ak.v11) + ' %', `TKM-27B v1.1 · Jev ${pct(...ak.jev)} (p 0,52) · v1.0 ${pct(...ak.v10)} · tarih+sayı gerçekleriyle v1.1 95,19`);
  }

  // ---------- kırılım çubukları
  function kirilim() {
    const kume = $('tkmKume').value, alan = $('tkmAlan').value, kr = (D.kirilim[kume] || {})[alan] || {};
    const sira = Object.keys(kr).map((deger) => { const e = kr[deger]; const n = Math.max(...SK.map((s) => (e[s] || [0, 0])[1])); return { deger, n, e }; }).filter((x) => x.n >= 20).sort((a, b) => {
      const ref = (x) => x.e.v11 && x.e.v11[1] ? x.e.v11[0] / x.e.v11[1] : (x.e.v10 ? x.e.v10[0] / x.e.v10[1] : 0); return ref(a) - ref(b);
    });
    const sis = SK.filter((s) => sira.some((x) => x.e[s] && x.e[s][1]));
    let h = `<div class="tkm-legend">${sis.map((s) => `<span><i style="background:${RENK[s]}"></i>${esc(AD[s])}</span>`).join('')}</div>`;
    sira.forEach((x) => {
      h += `<div class="tkm-row"><div class="tkm-lbl" title="${esc(x.deger)}">${esc(x.deger)} <span>(n ${fi(x.n)})</span></div><div class="tkm-bars">` +
        sis.map((s) => { const v = x.e[s] && x.e[s][1] ? 100 * x.e[s][0] / x.e[s][1] : null; return v == null ? '' : `<div class="tkm-bar" title="${esc(AD[s])}: ${pct(x.e[s][0], x.e[s][1])} %"><i style="width:${v.toFixed(1)}%;background:${RENK[s]}"></i><b>${pct(x.e[s][0], x.e[s][1])}</b></div>`; }).join('') + '</div></div>';
    });
    $('tkmKirilim').innerHTML = h || '<div class="note">Bu küme için kırılım yok.</div>';
  }

  // ---------- senaryo gezgini (tüm indeks)
  function doldurSecenekler() {
    const alanlar = { tkmFKume: 'kume', tkmFArketip: 'arketip', tkmFTuzak: 'tuzak', tkmFDil: 'dil', tkmFTip: 'tip' };
    Object.keys(alanlar).forEach((id) => {
      const sel = $(id); const alan = alanlar[id]; const vals = Object.keys(D.meta.sozluk[alan]).sort();
      sel.innerHTML = `<option value="">${alan}: hepsi</option>` + vals.map((v) => `<option value="${esc(v)}">${esc(alan === 'kume' ? (KISA[v] || v) : v)}</option>`).join('');
    });
  }
  let sayfa = 0; const SAYFA = 40; let sonFiltre = [];
  function filtrele() {
    const al = (id, alan) => { const v = $(id).value; return v === '' ? null : D.meta.sozluk[alan][v]; };
    const fk = al('tkmFKume', 'kume'), fa = al('tkmFArketip', 'arketip'), ft = al('tkmFTuzak', 'tuzak'), fd = al('tkmFDil', 'dil'), fp = al('tkmFTip', 'tip'), son = $('tkmFSonuc').value;
    const j = col.jev, v = col.v11, v0 = col.v10;
    const biz = (r) => r[v] >= 0 ? r[v] : r[v0];           // bizim = v1.1 varsa v1.1, yoksa v1.0 (train dilimi)
    sonFiltre = IX.filter((r) => (fk == null || r[col.kume] === fk) && (fa == null || r[col.arketip] === fa) && (ft == null || r[col.tuzak] === ft) && (fd == null || r[col.dil] === fd) && (fp == null || r[col.tip] === fp) &&
      (son === '' || (son === 'jev_biz_yanlis' && r[j] === 1 && biz(r) === 0) || (son === 'biz_jev_yanlis' && r[j] === 0 && biz(r) === 1) || (son === 'ikisi_yanlis' && r[j] === 0 && biz(r) === 0) || (son === 'ikisi_dogru' && r[j] === 1 && biz(r) === 1)));
    sayfa = 0; cizTablo();
  }
  function cizTablo() {
    const n = sonFiltre.length; const say = {};
    SK.forEach((s) => { let d = 0, m = 0; sonFiltre.forEach((r) => { const x = r[col[s]]; if (x >= 0) { m++; d += x; } }); say[s] = [d, m]; });
    $('tkmSayac').innerHTML = `<b>${fi(n)}</b> soru · ` + SK.filter((s) => say[s][1]).map((s) => `<span style="color:${RENK[s]}">${esc(AD[s])}</span> ${pct(say[s][0], say[s][1])} % (n ${fi(say[s][1])})`).join(' · ');
    const dilim = sonFiltre.slice(sayfa * SAYFA, (sayfa + 1) * SAYFA);
    let h = '<table><thead><tr><th>küme</th><th>arketip</th><th>tuzak</th><th>dil</th><th>tip</th>' + SK.map((s) => `<th style="color:${RENK[s]}">${esc(AD[s].split(' ')[0])}${s === 'v11' ? ' v1.1' : s === 'v10' ? ' v1.0' : ''}</th>`).join('') + '<th>tam metin</th></tr></thead><tbody>';
    dilim.forEach((r) => {
      const id = r[col.id]; const var_ = !!ORid[id];
      h += `<tr class="tkm-r tkm-click" data-id="${esc(id)}" data-parca="${esc(r[col.parca])}"><td>${esc(inv.kume[r[col.kume]])}</td><td>${esc(inv.arketip[r[col.arketip]])}</td><td>${esc(inv.tuzak[r[col.tuzak]])}</td><td>${esc(inv.dil[r[col.dil]])}</td><td>${esc(inv.tip[r[col.tip]])}</td>` +
        SK.map((s) => { const x = r[col[s]], g = r[col['g_' + s]]; return x < 0 ? '<td style="color:var(--muted)">—</td>' : `<td title="güven ${g == null ? '—' : (100 * g).toFixed(0) + ' %'}">${x ? '<span style="color:var(--good-text)">✓</span>' : '<span style="color:var(--critical)">✗</span>'}<span style="color:var(--muted);font-size:11px"> ${g == null ? '' : Math.round(100 * g)}</span></td>`; }).join('') +
        `<td>${var_ ? '<span style="color:var(--s1)">aç ▸</span>' : '<span style="color:var(--s1)">aç ▸</span><span style="color:var(--muted);font-size:11px" title="Tam metin bu satır için ayrı bir parça dosyasından indirilir (~1 MB, bir kez)"> ⇣</span>'}</td></tr>`;
    });
    h += '</tbody></table>';
    const top = Math.max(1, Math.ceil(n / SAYFA));
    $('tkmTablo').innerHTML = h;
    $('tkmSayfa').innerHTML = `sayfa ${sayfa + 1}/${fi(top)} · <button type="button" id="tkmOnce"${sayfa === 0 ? ' disabled' : ''}>◀</button> <button type="button" id="tkmSonra"${sayfa + 1 >= top ? ' disabled' : ''}>▶</button>`;
    $('tkmOnce').onclick = () => { sayfa--; cizTablo(); }; $('tkmSonra').onclick = () => { sayfa++; cizTablo(); };
    document.querySelectorAll('#tkmTablo tr.tkm-click').forEach((tr) => tr.addEventListener('click', () => acSatir(tr.getAttribute('data-id'), tr.getAttribute('data-parca'), tr)));
  }
  // ---------- tam metin: örnek kümesinde yoksa parça dosyasından (tkm_full/<parca>.json) getir
  const PARCA = {};                                             // parca -> Promise<json>
  function parcaGetir(pc) {
    if (!PARCA[pc]) PARCA[pc] = fetch('tkm_full/' + pc + '.json', { cache: 'force-cache' }).then((r) => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); }).catch((e) => { delete PARCA[pc]; throw e; });
    return PARCA[pc];
  }
  function parcadanDetay(id, P) {
    const r = P.rows[id]; if (!r) return null;
    const sistem = {}; Object.keys(r.s).forEach((s) => { sistem[s] = { dogru: r.s[s].d, cevap: r.s[s].c, olasilik: r.s[s].p }; });
    return { id, kume: r.m[0], arketip: r.m[1], tuzak: r.m[2], dil: r.m[3], bant: r.m[4], tip: r.m[5], state: P.states[r.st] || '', soru: r.q, altin: r.a, sistem };
  }
  function acSatir(id, pc, tr) {
    if (ORid[id]) { detay(ORid[id]); return; }
    const hucre = tr && tr.lastElementChild; const eski = hucre ? hucre.innerHTML : '';
    if (hucre) hucre.innerHTML = '<span style="color:var(--muted)">indiriliyor…</span>';
    parcaGetir(pc).then((P) => { const o = parcadanDetay(id, P); if (hucre) hucre.innerHTML = eski; if (o) detay(o); else alert('Bu satırın tam metni parça dosyasında bulunamadı: ' + id); })
      .catch((e) => { if (hucre) hucre.innerHTML = eski; alert('Tam metin indirilemedi (' + e.message + '). Sayfa GitHub Pages üzerinden açıksa biraz sonra yeniden deneyin.'); });
  }
  function olasilikBar(ol, cevap, altin) {
    return Object.keys(ol).sort((a, b) => ol[b] - ol[a]).map((k) => `<div class="tkm-p${k === altin ? ' altin' : ''}${k === cevap ? ' secim' : ''}"><span class="k">${esc(k)}${k === altin ? ' ★' : ''}</span><span class="bar"><i style="width:${(100 * ol[k]).toFixed(1)}%"></i></span><span class="v">${(100 * ol[k]).toFixed(1).replace('.', ',')} %</span></div>`).join('');
  }
  function detay(o) {
    if (!o) return;
    const q = o.soru; let krit = '';
    if (q.type === 'choice' && q.criteria) krit = '<ul class="tkm-krit">' + Object.keys(q.criteria).map((k) => `<li><code>${esc(k)}</code>${q.criteria[k] ? ' — ' + esc(q.criteria[k]) : ''}</li>`).join('') + '</ul>';
    if (q.type === 'score' && q.criteria) krit = '<ol start="0" class="tkm-krit">' + q.criteria.map((c) => `<li>${esc(c)}</li>`).join('') + '</ol>';
    const sis = SK.filter((s) => o.sistem[s]).map((s) => { const r = o.sistem[s]; return `<div class="tkm-sis"><div class="tkm-sis-h" style="border-left-color:${RENK[s]}"><b>${esc(AD[s])}</b> → <code>${esc(r.cevap)}</code> ${r.dogru ? '<span style="color:var(--good-text)">✓ doğru</span>' : '<span style="color:var(--critical)">✗ yanlış</span>'}</div>${olasilikBar(r.olasilik || {}, String(r.cevap), String(o.altin))}</div>`; }).join('');
    $('tkmDetay').innerHTML = `<div class="card-head"><div class="t"><h3>${esc(o.id)}</h3><div class="sub">${esc(o.kume)} · ${esc(o.arketip)} · tuzak: ${esc(o.tuzak)} · ${esc(o.dil)} · ${esc(o.bant)} · ${esc(o.tip)}</div></div><button type="button" id="tkmKapat">kapat</button></div>
      <div class="tkm-grid"><div><div class="sub" style="margin-bottom:4px">STATE (dosya)</div><pre class="tkm-state">${esc(o.state)}</pre></div>
      <div><div class="sub" style="margin-bottom:4px">SORU (${esc(q.type)})</div><div class="tkm-soru">${esc(q.instructions)}</div>${krit}<div class="sub" style="margin:8px 0 4px">ALTIN: <code>${esc(o.altin)}</code></div>${sis}</div></div>`;
    $('tkmDetay').style.display = 'block'; $('tkmKapat').onclick = () => { $('tkmDetay').style.display = 'none'; };
    $('tkmDetay').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // ---------- otomatik oynatma
  let oynat = null; const sayac = {}; let gosterilen = 0;
  function oynatAdim() {
    const havuz = $('tkmOynaKume').value ? OR.filter((o) => o.kume === $('tkmOynaKume').value) : OR;
    if (!havuz.length) return;
    const o = havuz[Math.floor(Math.random() * havuz.length)]; gosterilen++;
    SK.forEach((s) => { if (o.sistem[s]) { sayac[s] = sayac[s] || [0, 0]; sayac[s][0] += o.sistem[s].dogru; sayac[s][1]++; } });
    $('tkmOynaSayac').innerHTML = `<b>${gosterilen}</b> senaryo · ` + SK.filter((s) => sayac[s]).map((s) => `<span style="color:${RENK[s]}">${esc(AD[s])}</span> ${pct(sayac[s][0], sayac[s][1])} %`).join(' · ');
    const q = o.soru;
    $('tkmOynaKart').innerHTML = `<div class="sub">${esc(o.kume)} · ${esc(o.arketip)} · tuzak: ${esc(o.tuzak)} · ${esc(o.dil)}</div><pre class="tkm-state" style="max-height:180px">${esc(o.state.slice(0, 900))}${o.state.length > 900 ? ' …' : ''}</pre><div class="tkm-soru"><b>${esc(q.type)}:</b> ${esc(q.instructions)}</div><div class="tkm-cevaplar">` +
      SK.filter((s) => o.sistem[s]).map((s) => `<span class="tkm-cv" style="border-color:${RENK[s]}">${esc(AD[s].split(' ')[0])}${s === 'v11' ? ' v1.1' : s === 'v10' ? ' v1.0' : ''}: <code>${esc(o.sistem[s].cevap)}</code> ${o.sistem[s].dogru ? '✓' : '✗'}</span>`).join('') + `<span class="tkm-cv" style="border-color:var(--ink)">altın: <code>${esc(o.altin)}</code></span></div>`;
  }
  function oynatBaslat() {
    if (oynat) { clearInterval(oynat); oynat = null; $('tkmOynaBtn').textContent = '▶ Başlat'; return; }
    oynatAdim(); oynat = setInterval(oynatAdim, Number($('tkmOynaHiz').value) * 1000); $('tkmOynaBtn').textContent = '⏸ Durdur';
  }

  function render() {
    if (rendered) return; rendered = true;
    $('tkmTiles').innerHTML = tiles(); $('tkmOzet').innerHTML = ozetTablo(); $('tkmEsli').innerHTML = esliTablo();
    $('tkmKume').innerHTML = KUMELER.map((k) => `<option value="${k}">${esc(KISA[k])}</option>`).join('');
    $('tkmKume').onchange = kirilim; $('tkmAlan').onchange = kirilim; kirilim();
    doldurSecenekler(); ['tkmFKume', 'tkmFArketip', 'tkmFTuzak', 'tkmFDil', 'tkmFTip', 'tkmFSonuc'].forEach((id) => { $(id).onchange = filtrele; });
    $('tkmFKume').value = 'gel2'; $('tkmFSonuc').value = 'jev_biz_yanlis'; filtrele();
    $('tkmOynaKume').innerHTML = '<option value="">hepsi</option>' + KUMELER.map((k) => `<option value="${k}">${esc(KISA[k])}</option>`).join('');
    $('tkmOynaBtn').onclick = oynatBaslat;
    $('tkmMeta').textContent = `Veri üretimi ${D.meta.uretim} · ${fi(IX.length)} satır indeks · tam metin: TÜM satırlar (${fi(OR.length)} örnek sayfayla gelir, kalanı tıklanınca 256 parçadan indirilir) · ${D.meta.not}`;
  }
  function gorunur() { return $('tkm').classList.contains('on'); }
  if (gorunur()) render();
  window.addEventListener('hashchange', () => { if (gorunur()) render(); });
  document.querySelectorAll('#tabs a[href="#tkm"]').forEach((a) => a.addEventListener('click', () => setTimeout(render, 0)));
})();
