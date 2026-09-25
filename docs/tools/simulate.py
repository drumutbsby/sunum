#!/usr/bin/env python3
"""Laya'yı Jev tarzı tipli kararlarla senaryolar üzerinde koşturan simülasyon.

Kullanım:
    python docs/tools/simulate.py [--device cpu] [--threads N]

Girdi:  docs/sim/scenarios.json   (durumlar, soru setleri, beklenen cevaplar)
Çıktı:  docs/sim/results.json     (her senaryo için ham cevaplar, olasılıklar, gecikme)
        docs/sim_data.js          (panelin okuduğu özet: window.LAYA_SIM = {...})

Her senaryo üç kez cevaplanır: Router'ın seçtiği checkpoint ile (Jev'deki gibi tek çağrı),
ayrıca zorla `english` ve zorla `multilingual` ile; böylece yönlendirmenin etkisi görülür.
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
sim_dir = os.path.join(here, "..", "sim")
ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--threads", type=int, default=0)
args = ap.parse_args()

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch  # noqa: E402
if args.threads:
    torch.set_num_threads(args.threads)
torch.set_num_interop_threads(1)
import laya  # noqa: E402
from laya import Router  # noqa: E402

spec = json.load(open(os.path.join(sim_dir, "scenarios.json"), encoding="utf-8"))
qsets, scenarios = spec["question_sets"], spec["scenarios"]

t0 = time.time()
router = Router(device=args.device, max_loaded=3)
router.preload(["english", "multilingual"])
load_s = round(time.time() - t0, 1)
print(f"checkpoints yüklendi: {load_s} s", flush=True)

# ısınma
router.predict({"body": "warm up"}, qsets["destek"], model="english")
router.predict({"body": "warm up"}, qsets["destek"], model="multilingual")


def decide(answers, qdef):
    """Bir cevabı karşılaştırılabilir tek değere indir: choice -> etiket, noul -> bool, score -> argmax seviye."""
    out = {}
    for qid, a in answers.items():
        if a["type"] == "choice":
            out[qid] = a["choice"]
        elif a["type"] == "noul":
            out[qid] = a["noul"] >= 0.5
        else:
            probs = a["probabilities"]
            out[qid] = int(max(probs, key=lambda k: probs[k]))
    return out


results = []
for i, sc in enumerate(scenarios):
    q = qsets[sc["question_set"]]
    entry = {"id": sc["id"], "lang": sc["lang"], "family": sc["family"], "question_set": sc["question_set"],
             "state": sc["state"], "expected": sc["expected"], "runs": {}}
    route = router.route(sc["state"], q)
    entry["routing"] = dict(route)
    for mode in ("routed", "english", "multilingual"):
        kw = {} if mode == "routed" else {"model": mode}
        t = time.perf_counter()
        res = router.predict(sc["state"], q, **kw)
        ms = round((time.perf_counter() - t) * 1000, 1)
        dec = decide(res["answers"], q)
        correct = {qid: dec[qid] == sc["expected"][qid] for qid in sc["expected"]}
        entry["runs"][mode] = {"model": res["routing"]["model"] if mode == "routed" else mode, "ms": ms,
                               "answers": res["answers"], "usage": res.get("usage"), "decided": dec, "correct": correct}
    results.append(entry)
    r = entry["runs"]["routed"]
    ok = sum(r["correct"].values()); n = len(r["correct"])
    print(f"[{i+1:02d}/{len(scenarios)}] {sc['id']:10s} -> {r['model']:12s} {ok}/{n} doğru  {r['ms']:7.1f} ms", flush=True)

# ---------- metrikler ----------
def summarize(mode):
    by_q = defaultdict(lambda: [0, 0]); by_fam = defaultdict(lambda: [0, 0]); by_lang = defaultdict(lambda: [0, 0])
    by_set = defaultdict(lambda: [0, 0]); ms = []; conf_ok = []; conf_bad = []; gate = {"n": 0, "auto": 0, "auto_ok": 0}
    all_ok = 0; all_n = 0; full = 0
    for e in results:
        r = e["runs"][mode]; ms.append(r["ms"])
        allc = True
        for qid, c in r["correct"].items():
            by_q[qid][0] += c; by_q[qid][1] += 1
            by_fam[e["family"]][0] += c; by_fam[e["family"]][1] += 1
            by_lang[e["lang"]][0] += c; by_lang[e["lang"]][1] += 1
            by_set[e["question_set"]][0] += c; by_set[e["question_set"]][1] += 1
            all_ok += c; all_n += 1; allc &= c
            ac = r["answers"][qid]["answer_confidence"]
            (conf_ok if c else conf_bad).append(ac)
            gate["n"] += 1
            if ac >= 0.8:
                gate["auto"] += 1; gate["auto_ok"] += c
        full += allc
    pct = lambda d: {k: {"correct": v[0], "n": v[1], "accuracy": round(v[0] / v[1], 4)} for k, v in sorted(d.items())}
    return {
        "accuracy": round(all_ok / all_n, 4), "correct": all_ok, "n_decisions": all_n,
        "scenarios_fully_correct": full, "n_scenarios": len(results),
        "by_question": pct(by_q), "by_family": pct(by_fam), "by_lang": pct(by_lang), "by_set": pct(by_set),
        "latency_ms": {"p50": round(statistics.median(ms), 1), "p95": round(sorted(ms)[int(0.95 * (len(ms) - 1))], 1),
                       "mean": round(statistics.mean(ms), 1), "min": min(ms), "max": max(ms)},
        "mean_answer_confidence_when_correct": round(statistics.mean(conf_ok), 4) if conf_ok else None,
        "mean_answer_confidence_when_wrong": round(statistics.mean(conf_bad), 4) if conf_bad else None,
        "gate_0_8": {"automated_share": round(gate["auto"] / gate["n"], 4),
                     "accuracy_of_automated": round(gate["auto_ok"] / gate["auto"], 4) if gate["auto"] else None},
    }

summary = {m: summarize(m) for m in ("routed", "english", "multilingual")}
routed_models = defaultdict(int)
for e in results:
    routed_models[e["runs"]["routed"]["model"]] += 1

meta = {
    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    "laya": laya.__version__, "torch": torch.__version__, "device": args.device,
    "threads": torch.get_num_threads(), "cpu": platform.processor() or platform.machine(), "python": platform.python_version(),
    "load_s": load_s, "n_scenarios": len(scenarios), "routed_models": dict(routed_models),
    "note": "Zero-shot; ince ayar ve sıcaklık fit'i yok. Beklenen etiketler panel için elle yazıldı. Sıcaklıklar checkpoint'in gönderdiği gibi (kısıtlı).",
}
json.dump({"meta": meta, "question_sets": qsets, "summary": summary, "results": results},
          open(os.path.join(sim_dir, "results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
with open(os.path.join(here, "..", "sim_data.js"), "w", encoding="utf-8") as f:
    f.write("// Otomatik üretildi: docs/tools/simulate.py — elle düzenlemeyin.\nwindow.LAYA_SIM = ")
    json.dump({"meta": meta, "question_sets": qsets, "summary": summary, "results": results}, f, ensure_ascii=False)
    f.write(";\n")
print(json.dumps({m: {k: v for k, v in s.items() if k in ("accuracy", "scenarios_fully_correct", "latency_ms")} for m, s in summary.items()}, indent=1, ensure_ascii=False))
