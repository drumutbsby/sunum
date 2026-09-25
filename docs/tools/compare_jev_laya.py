#!/usr/bin/env python3
"""Jev (TypeSafe API) ile Laya'yı aynı senaryolarda, aynı sorularla karşılaştırır.

Kullanım:
    TYPESAFE_API_KEY=... python docs/tools/compare_jev_laya.py                # her ikisi
    python docs/tools/compare_jev_laya.py --backends laya                    # yalnızca Laya
    python docs/tools/compare_jev_laya.py --backends jev --merge             # yalnızca Jev, eski Laya sonuçlarını koru
    python docs/tools/compare_jev_laya.py --jev-model jev-1.13.0 --limit 5   # deneme

Senaryolar:
    docs/compare/scenarios_kredi.json   kredi tahsisi onay süreci (4 senaryo, 8 soru)
    docs/sim/scenarios.json             destek ve güvenlik senaryoları (54 senaryo)

Çıktı:
    docs/compare/results.json           her senaryo için iki backend'in ham cevapları, gecikme, kullanım
    docs/compare_data.js                panelin okuduğu özet (window.LAYA_COMPARE)

API anahtarı yalnızca TYPESAFE_API_KEY ortam değişkeninden okunur; hiçbir dosyaya yazılmaz.
"""
import argparse
import json
import os
import platform
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

here = os.path.dirname(os.path.abspath(__file__))
docs = os.path.normpath(os.path.join(here, ".."))
OUT_JSON = os.path.join(docs, "compare", "results.json")
OUT_JS = os.path.join(docs, "compare_data.js")

ap = argparse.ArgumentParser()
ap.add_argument("--backends", default="laya,jev", help="virgülle: laya, jev")
ap.add_argument("--jev-model", default=os.environ.get("TYPESAFE_DEFAULT_MODEL", "jev-latest"))
ap.add_argument("--device", default="cpu")
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--limit", type=int, default=0, help="yalnızca ilk N senaryo (deneme)")
ap.add_argument("--merge", action="store_true", help="mevcut results.json'daki diğer backend sonuçlarını koru")
ap.add_argument("--repeats", type=int, default=1, help="gecikme için tekrar sayısı; cevaplar ilk tekrardan alınır")
args = ap.parse_args()
backends = [b.strip() for b in args.backends.split(",") if b.strip()]


# ---------- senaryolar ----------
def load_scenarios():
    kredi = json.load(open(os.path.join(docs, "compare", "scenarios_kredi.json"), encoding="utf-8"))
    sim = json.load(open(os.path.join(docs, "sim", "scenarios.json"), encoding="utf-8"))
    qsets = {"kredi": kredi["question_set"], **sim["question_sets"]}
    items = []
    for sc in kredi["scenarios"]:
        items.append({**sc, "question_set": "kredi"})
    for sc in sim["scenarios"]:
        items.append(sc)
    return qsets, items


qsets, scenarios = load_scenarios()
if args.limit:
    scenarios = scenarios[: args.limit]


# ---------- ortak değerlendirme ----------
def argmax_key(probs):
    return max(probs, key=lambda k: probs[k])


def decide(answer):
    """Bir cevabı karşılaştırılabilir değere indir: choice -> etiket, noul -> bool, score -> en olası seviye (int)."""
    t = answer["type"]
    if t == "choice":
        return answer["choice"]
    if t == "noul":
        return answer["noul"] >= 0.5
    probs = answer["probabilities"]
    k = argmax_key(probs)
    if str(k).isdigit():
        return int(k)
    return list(probs.keys()).index(k)  # anahtarlar metin ise sırasını seviye kabul et


def p_max(answer):
    """Backend'den bağımsız tek güven ölçüsü: verilen cevabın olasılığı."""
    t = answer["type"]
    if t == "noul":
        return max(answer["noul"], 1.0 - answer["noul"])
    return max(answer["probabilities"].values())


def is_correct(decided, expected):
    if isinstance(expected, list):
        return decided in expected
    return decided == expected


# ---------- kararlas: kullanıcının kural seti (kredi senaryoları) ----------
def kararlas(answers):
    n = {k: v["noul"] for k, v in answers.items() if v["type"] == "noul"}
    s = {k: v for k, v in answers.items() if v["type"] == "score"}
    c = {k: v for k, v in answers.items() if v["type"] == "choice"}
    out = {
        "onay_seviyesi": c["onay_seviyesi"]["choice"],
        "risk_seviyesi": argmax_key(s["risk_seviyesi"]["probabilities"]),
        "aciliyet": argmax_key(s["aciliyet"]["probabilities"]),
    }
    risk = s["risk_seviyesi"]["score"]
    if not n["belgeler_tam"] > 0.5:
        out["islem"] = "RED ÖNERİSİ: belgeler eksik"
    elif not n["borçlanma_kapasitesi"] > 0.4:
        out["islem"] = "RED ÖNERİSİ: borçlanma kapasitesi yetersiz"
    elif not n["kredi_gecmisi_temiz"] > 0.5 and risk >= 1.5:
        out["islem"] = "GENEL MÜDÜRLÜĞE: risk yüksek"
    elif not n["kredi_amaci_uygun"] > 0.6:
        out["islem"] = "PAZARLAMA DANIŞMANLIĞI: kredi amacı sorgulanabilir"
    elif n["borçlanma_kapasitesi"] > 0.8 and n["belgeler_tam"] > 0.9 and risk < 0.8:
        out["islem"] = "ŞUBE ONAYI: düşük risk"
    else:
        out["islem"] = "GENEL MÜDÜRLÜĞE: orta risk"
    return out


# ---------- backend'ler ----------
def make_laya():
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    import laya
    from laya import Router
    t0 = time.time()
    router = Router(device=args.device, max_loaded=3)
    router.preload(["english", "multilingual"])
    load_s = round(time.time() - t0, 1)
    router.predict({"body": "warm up"}, qsets["destek"], model="english")
    router.predict({"body": "warm up"}, qsets["destek"], model="multilingual")
    info = {"laya": laya.__version__, "torch": torch.__version__, "device": args.device, "threads": torch.get_num_threads(), "load_s": load_s}

    def call(state, questions):
        t = time.perf_counter()
        res = router.predict(state, questions)
        ms = (time.perf_counter() - t) * 1000
        return {"model": "laya/" + res["routing"]["model"], "answers": res["answers"], "usage": res.get("usage"), "routing": dict(res["routing"])}, ms

    return call, info


def make_jev():
    key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not key:
        raise SystemExit("TYPESAFE_API_KEY ortam değişkeni boş; Jev backend'i koşulamaz.")
    from typesafe_sdk import Choice, Noul, Score, TypeSafeClient
    import typesafe_sdk
    client = TypeSafeClient(api_key=key, model=args.jev_model, timeout=60)
    key = None

    def to_sdk(questions):
        out = {}
        for qid, q in questions.items():
            if q["type"] == "noul":
                out[qid] = Noul(instructions=q["instructions"], criteria=q.get("criteria"))
            elif q["type"] == "choice":
                out[qid] = Choice(instructions=q["instructions"], criteria=q["criteria"])
            else:
                out[qid] = Score(instructions=q["instructions"], criteria=q["criteria"])
        return out

    def to_plain(resp):
        answers = {}
        for qid, a in resp.answers.items():
            d = a.model_dump() if hasattr(a, "model_dump") else dict(a.__dict__)
            answers[qid] = d
        usage = resp.usage.model_dump() if hasattr(resp.usage, "model_dump") else dict(resp.usage.__dict__)
        return {"model": resp.model, "answers": answers, "usage": usage}

    def call(state, questions):
        t = time.perf_counter()
        resp = client.system_one(state=state, questions=to_sdk(questions))
        ms = (time.perf_counter() - t) * 1000
        return to_plain(resp), ms

    return call, {"typesafe_sdk": typesafe_sdk.__version__ if hasattr(typesafe_sdk, "__version__") else "?", "jev_model": args.jev_model}


# ---------- koşu ----------
existing = None
if args.merge and os.path.exists(OUT_JSON):
    existing = json.load(open(OUT_JSON, encoding="utf-8"))
    existing_by_id = {e["id"]: e for e in existing["results"]}
else:
    existing_by_id = {}

runners, infos = {}, {}
for b in backends:
    runners[b], infos[b] = (make_laya() if b == "laya" else make_jev())
    print(f"{b}: hazır {infos[b]}", flush=True)

results = []
for i, sc in enumerate(scenarios):
    q = qsets[sc["question_set"]]
    entry = {"id": sc["id"], "lang": sc["lang"], "family": sc["family"], "question_set": sc["question_set"],
             "title": sc.get("title"), "state": sc["state"], "expected": sc["expected"], "runs": {}}
    if sc["id"] in existing_by_id:
        for b, r in existing_by_id[sc["id"]].get("runs", {}).items():
            if b not in backends:
                entry["runs"][b] = r
    for b in backends:
        try:
            resp, ms = runners[b](sc["state"], q)
            times = [ms]
            for _ in range(max(0, args.repeats - 1)):
                _, ms2 = runners[b](sc["state"], q)
                times.append(ms2)
            dec = {qid: decide(a) for qid, a in resp["answers"].items()}
            corr = {qid: is_correct(dec[qid], exp) for qid, exp in sc["expected"].items() if qid in dec}
            run = {"model": resp["model"], "ms": round(statistics.median(times), 1), "ms_samples": [round(x, 1) for x in times],
                   "answers": resp["answers"], "usage": resp.get("usage"), "decided": dec, "correct": corr,
                   "p_max": {qid: round(p_max(a), 4) for qid, a in resp["answers"].items()}}
            if "routing" in resp:
                run["routing"] = resp["routing"]
            if sc["question_set"] == "kredi":
                run["kararlas"] = kararlas(resp["answers"])
            entry["runs"][b] = run
            ok = sum(corr.values())
            print(f"[{i+1:02d}/{len(scenarios)}] {sc['id']:10s} {b:5s} {resp['model']:18s} {ok}/{len(corr)} doğru {run['ms']:8.1f} ms", flush=True)
        except Exception as e:  # bir senaryo düşerse koşu devam etsin; hata kaydedilsin
            entry["runs"][b] = {"error": f"{type(e).__name__}: {e}"[:300], "ms": None, "answers": {}, "decided": {}, "correct": {}, "p_max": {}}
            print(f"[{i+1:02d}/{len(scenarios)}] {sc['id']:10s} {b:5s} HATA {type(e).__name__}: {str(e)[:120]}", flush=True)
    results.append(entry)

# senaryolar limit ile kısıtlandıysa, eskileri de taşı
if existing and args.limit:
    seen = {e["id"] for e in results}
    results += [e for e in existing["results"] if e["id"] not in seen]

# ---------- metrikler ----------
all_backends = sorted({b for e in results for b in e["runs"]})


def summarize(b):
    by_q = defaultdict(lambda: [0, 0]); by_set = defaultdict(lambda: [0, 0]); by_lang = defaultdict(lambda: [0, 0]); by_fam = defaultdict(lambda: [0, 0])
    ms = []; ok = n = 0; full = 0; scen = 0; errors = 0; conf_ok = []; conf_bad = []; tokens_in = 0
    for e in results:
        r = e["runs"].get(b)
        if not r:
            continue
        if r.get("error"):
            errors += 1; continue
        scen += 1; ms.append(r["ms"])
        if r.get("usage") and isinstance(r["usage"], dict):
            tokens_in += r["usage"].get("input_tokens", 0) or 0
        allc = True
        for qid, c in r["correct"].items():
            by_q[qid][0] += c; by_q[qid][1] += 1
            by_set[e["question_set"]][0] += c; by_set[e["question_set"]][1] += 1
            by_lang[e["lang"]][0] += c; by_lang[e["lang"]][1] += 1
            by_fam[e["family"]][0] += c; by_fam[e["family"]][1] += 1
            ok += c; n += 1; allc &= c
            (conf_ok if c else conf_bad).append(r["p_max"][qid])
        full += allc
    pct = lambda d: {k: {"correct": v[0], "n": v[1], "accuracy": round(v[0] / v[1], 4) if v[1] else None} for k, v in sorted(d.items())}
    return {
        "accuracy": round(ok / n, 4) if n else None, "correct": ok, "n_decisions": n, "scenarios": scen, "errors": errors,
        "scenarios_fully_correct": full,
        "by_question": pct(by_q), "by_set": pct(by_set), "by_lang": pct(by_lang), "by_family": pct(by_fam),
        "latency_ms": {"p50": round(statistics.median(ms), 1), "p95": round(sorted(ms)[int(0.95 * (len(ms) - 1))], 1), "mean": round(statistics.mean(ms), 1)} if ms else None,
        "mean_p_max_when_correct": round(statistics.mean(conf_ok), 4) if conf_ok else None,
        "mean_p_max_when_wrong": round(statistics.mean(conf_bad), 4) if conf_bad else None,
        "input_tokens_total": tokens_in,
    }


def agreement():
    if "laya" not in all_backends or "jev" not in all_backends:
        return None
    by_q = defaultdict(lambda: [0, 0]); tot = [0, 0]; both_ok = both_bad = only_laya = only_jev = 0
    for e in results:
        a, b = e["runs"].get("laya"), e["runs"].get("jev")
        if not a or not b or a.get("error") or b.get("error"):
            continue
        for qid in a["decided"]:
            if qid not in b["decided"]:
                continue
            same = a["decided"][qid] == b["decided"][qid]
            by_q[qid][0] += same; by_q[qid][1] += 1; tot[0] += same; tot[1] += 1
            if qid in e["expected"]:
                ca, cb = a["correct"][qid], b["correct"][qid]
                both_ok += ca and cb; both_bad += (not ca) and (not cb); only_laya += ca and not cb; only_jev += cb and not ca
    return {"rate": round(tot[0] / tot[1], 4) if tot[1] else None, "n": tot[1],
            "by_question": {k: {"same": v[0], "n": v[1], "rate": round(v[0] / v[1], 4)} for k, v in sorted(by_q.items())},
            "both_correct": both_ok, "both_wrong": both_bad, "only_laya_correct": only_laya, "only_jev_correct": only_jev}


summary = {b: summarize(b) for b in all_backends}
meta = {
    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    "backends_run_now": backends, "backends": all_backends, "info": {**(existing["meta"]["info"] if existing else {}), **infos},
    "python": platform.python_version(), "machine": platform.machine(), "ci": bool(os.environ.get("GITHUB_ACTIONS")),
    "n_scenarios": len(results),
    "note": "Aynı durumlar ve aynı sorular iki backend'e verildi. Laya zero-shot, yerel CPU; Jev hosted API (gecikmeye ağ dahil). Beklenen etiketler elle yazıldı.",
}
payload = {"meta": meta, "question_sets": qsets, "summary": summary, "agreement": agreement(), "results": results}
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
json.dump(payload, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
with open(OUT_JS, "w", encoding="utf-8") as f:
    f.write("// Otomatik üretildi: docs/tools/compare_jev_laya.py — elle düzenlemeyin.\nwindow.LAYA_COMPARE = ")
    json.dump(payload, f, ensure_ascii=False)
    f.write(";\n")
print(json.dumps({b: {k: v for k, v in s.items() if k in ("accuracy", "scenarios_fully_correct", "latency_ms", "errors")} for b, s in summary.items()}, indent=1, ensure_ascii=False))
if payload["agreement"]:
    print("uyum:", payload["agreement"]["rate"], "her ikisi doğru:", payload["agreement"]["both_correct"], "yalnız laya:", payload["agreement"]["only_laya_correct"], "yalnız jev:", payload["agreement"]["only_jev_correct"])
