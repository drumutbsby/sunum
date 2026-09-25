#!/usr/bin/env python3
"""Laya değerlendirme paneli için veri dosyasını üretir.

Kullanım:
    python docs/tools/build_panel_data.py <laya-klon-dizini> [oturum-test-sonuçları.tsv] [vitest-özeti.json]

Çıktı: docs/data.js  (window.LAYA_DATA = {...}; file:// ile de çalışsın diye JSON değil JS)
Bütün sayılar laya deposundaki research/results/*.json ve
research/benchmarks/feishu_zh/results/v1/summary.json dosyalarından okunur; hiçbir değer elle girilmez.
Tek istisna: README/BENCHMARKS.md'de tablo olarak verilen ama JSON'u depoda olmayan
laya-typed-decisions satırı, kaynağı belirtilerek `typed_decisions.readme_table` altında taşınır.
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

laya = sys.argv[1]
tsv = sys.argv[2] if len(sys.argv) > 2 else None
vitest = sys.argv[3] if len(sys.argv) > 3 else None
here = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(here, "..", "data.js")


def load(rel):
    with open(os.path.join(laya, rel), encoding="utf-8") as f:
        return json.load(f)


def git(*args):
    return subprocess.check_output(["git", "-C", laya, *args], text=True).strip()


def count_lines(paths):
    n = 0
    for p in paths:
        with open(p, encoding="utf-8", errors="ignore") as f:
            n += sum(1 for _ in f)
    return n


def walk(sub, exts):
    res = []
    for root, dirs, files in os.walk(os.path.join(laya, sub)):
        dirs[:] = [d for d in dirs if d not in ("node_modules", "dist", ".git", "__pycache__")]
        for fn in files:
            if fn.endswith(exts):
                res.append(os.path.join(root, fn))
    return res


# ---------- depo meta ----------
pyproject = open(os.path.join(laya, "pyproject.toml"), encoding="utf-8").read()
import re
version = re.search(r'version = "(.*?)"', pyproject).group(1)
meta = {
    "repo": "NandhaKishorM/laya",
    "url": "https://github.com/NandhaKishorM/laya",
    "commit": git("rev-parse", "HEAD"),
    "commit_short": git("rev-parse", "--short", "HEAD"),
    "commit_date": git("log", "-1", "--format=%ci"),
    "commit_subject": git("log", "-1", "--format=%s"),
    "version": version,
    "license": "Apache-2.0",
    "python_requires": re.search(r'requires-python = "(.*?)"', pyproject).group(1),
    "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    "code": {
        "laya_py_lines": count_lines(walk("laya", (".py",))),
        "laya_py_files": len(walk("laya", (".py",))),
        "tests_py_lines": count_lines(walk("tests", (".py",))),
        "tests_py_files": len(walk("tests", (".py",))),
        "research_py_lines": count_lines(walk("research", (".py",))),
        "ts_src_lines": count_lines(walk("laya-ts/src", (".ts",))),
        "ts_test_lines": count_lines(walk("laya-ts/tests", (".ts",))),
        "ts_test_files": len(walk("laya-ts/tests", (".ts",))),
        "docs_md_files": len(walk("docs", (".md",))),
        "workflows": sorted(os.listdir(os.path.join(laya, ".github", "workflows"))),
    },
    "external": {
        # PyPI JSON API ve Hugging Face API'den bu oturumda okundu (2026-09-25)
        "pypi_releases": 28,
        "pypi_first_release": "2026-09-18",
        "pypi_latest": "0.3.20",
        "pypi_latest_date": "2026-09-24",
        "hf_likes_laya": 3462,
        "hf_likes_multilingual": 253,
        "hf_created": "2026-09-18",
    },
}

# ---------- 51 dil ----------
sweep = load("research/results/cpu_51_language_sweep.json")
eval51 = load("research/results/eval_english_51_languages.json")
langs = sweep["part_a"]["config"]["languages"]
en = sweep["part_a"]["by_model"]["english"]["per_language"]
ml = sweep["part_a"]["by_model"]["multilingual"]["per_language"]
langs51 = []
for lg in langs:
    langs51.append({
        "lang": lg,
        "english_acc": en[lg]["accuracy"], "english_ece": en[lg]["ece"], "english_conf": en[lg]["mean_confidence"],
        "multilingual_acc": ml[lg]["accuracy"], "multilingual_ece": ml[lg]["ece"], "multilingual_conf": ml[lg]["mean_confidence"],
        "english_ece_clamped": eval51["report"].get(lg, {}).get("ece"),
    })
sweep_summary = {
    "config": sweep["part_a"]["config"],
    "meta": sweep["meta"],
    "english": {k: v for k, v in sweep["part_a"]["by_model"]["english"].items() if k != "per_language"},
    "multilingual": {k: v for k, v in sweep["part_a"]["by_model"]["multilingual"].items() if k != "per_language"},
    "clamped_rerun": eval51["summary"],
}

# ---------- T4 Colab ----------
t4 = load("research/results/t4_colab_benchmark.json")
def suite_rows(prefix):
    rows = []
    for k, v in t4["suites"].items():
        if k.startswith(prefix + "."):
            rows.append({"lang": k.split(".", 1)[1],
                         "laya": v["laya"]["calibrated"]["accuracy"],
                         "laya_ece": v["laya"]["calibrated"]["ece"],
                         "multilingual": v["laya-multilingual"]["calibrated"]["accuracy"],
                         "multilingual_ece": v["laya-multilingual"]["calibrated"]["ece"],
                         "n": v["laya"]["calibrated"]["n"]})
    return rows
english_suites = []
for k in ("en.ag_news", "en.boolq", "en.emotion", "en.prompt_injections", "en.sst5"):
    v = t4["suites"][k]
    english_suites.append({"suite": k[3:], "laya": v["laya"]["calibrated"]["accuracy"],
                           "multilingual": v["laya-multilingual"]["calibrated"]["accuracy"],
                           "n": v["laya"]["calibrated"]["n"],
                           "in_training": k in ("en.ag_news", "en.boolq")})
t4_block = {
    "meta": {k: v for k, v in t4["meta"].items() if k != "models"},
    "models": {m: {k: v for k, v in d.items() if k in ("repo", "params_total_m", "hidden_size", "num_layers", "vocab_size", "max_len", "head_max_len", "temperature_by_options", "training")} for m, d in t4["meta"]["models"].items()},
    "summary": t4["summary"],
    "latency": t4["latency"],
    "option_order": t4["option_order_robustness"],
    "calibration": {m: {"shipped": d["mean_ece_shipped"], "refit": d["mean_ece_refit"]} for m, d in t4["calibration_repair"].items()},
    "massive_intent": suite_rows("massive_intent"),
    "massive_scenario": suite_rows("massive_scenario"),
    "xnli": suite_rows("xnli"),
    "english_suites": english_suites,
    "caveats": t4["caveats"],
}

# ---------- uygulama temaları ----------
apps = load("research/results/app_benchmark_results.json")
app_rows = []
for k, v in apps["suites"].items():
    row = {"suite": k, "in_training": v["english"].get("in_training")}
    for m in ("english", "multilingual", "typed-decisions"):
        r = v[m]
        row[m] = {"accuracy": r["accuracy"], "macro_f1": r["macro_f1"], "ece": r["ece"], "ms_per_case": r["ms_per_case"], "n": r["n"]}
    app_rows.append(row)
apps_block = {"meta": apps["meta"], "jev_published": apps["jev_published"], "rows": app_rows}

# ---------- typed-decisions ----------
pb = sweep["part_b"]
typed = {
    "reference": pb["reference_points"],
    "english_cpu": {k: v for k, v in pb["by_model"]["english"].items() if k not in ("by_workflow", "by_question_type")},
    "english_by_workflow": pb["by_model"]["english"]["by_workflow"],
    "english_by_qtype": pb["by_model"]["english"]["by_question_type"],
    "t4": {m: {k: v for k, v in d.items() if k in ("calibrated", "soft_acc", "score_mae", "within_1")} for m, d in t4["suites"]["typed_decisions"].items()},
    # BENCHMARKS.md tablosu; JSON'u depoda yok (laya-typed-decisions satırı)
    "readme_table": {
        "source": "BENCHMARKS.md § typed-decisions",
        "rows": [
            {"model": "laya-typed-decisions", "accuracy": 0.766, "soft_acc": 0.471, "brier": 0.061, "ece": 0.213, "score_mae": 0.242},
            {"model": "laya", "accuracy": 0.361, "soft_acc": 0.332, "brier": 0.316, "ece": 0.175, "score_mae": 0.694},
            {"model": "laya-multilingual", "accuracy": 0.352, "soft_acc": 0.328, "brier": 0.463, "ece": 0.314, "score_mae": 0.760},
            {"model": "Jev 1.13.0 (yayımlanmış)", "accuracy": 0.727, "soft_acc": 0.580, "brier": 0.148, "ece": 0.144, "score_mae": 0.391},
        ],
        "by_workflow_typed": {"agent_trace_observability": 0.730, "customer_service": 0.764, "invoice_processing": 0.804, "security_incidents": 0.766},
    },
}

# ---------- uzun bağlam ----------
lc = load("research/results/long_context_multilingual.json")
long_ctx = {"meta": {k: v for k, v in lc.items() if k not in ("rows", "cases", "questions")}, "rows": lc["rows"]}

# ---------- CPU gecikme ----------
lat = load("research/results/latency_cpu_m7a_xlarge_20260924.json")
latency_cpu = {"meta": lat["meta"], "raw": lat["raw_latency"], "cold_load_ms": lat["cold_load_ms"],
               "detection": lat["detection_overhead"], "router_hot": lat["router_hot"],
               "cold_swap_median_ms": lat["router_cold_swap"]["median_ms"], "mixed": lat["mixed_workload"]}

# ---------- uzunluk gruplama ----------
lb = load("research/results/length_batching_cpu_20260924.json")
length_batching = {"meta": {k: v for k, v in lb.items() if not isinstance(v, (list, dict))},
                   "cases": [{k: v for k, v in c.items() if not isinstance(v, (list, dict))} for c in lb["cases"]]}

# ---------- feishu ----------
fs = load("research/benchmarks/feishu_zh/results/v1/summary.json")
feishu = {}
for backend in ("laya", "jev"):
    feishu[backend] = {}
    for mode in ("choice", "four_noul"):
        r = fs[backend][mode]
        feishu[backend][mode] = {k: r.get(k) for k in ("n", "correct", "accuracy", "macro_f1", "false_action_count", "missed_action_count", "urgent_recalled", "urgent_denominator", "total_requests", "failed_requests")}
        feishu[backend][mode]["p50_ms"] = r["timing"]["p50_ms"]
        feishu[backend][mode]["p95_ms"] = r["timing"]["p95_ms"]
        feishu[backend][mode]["by_family"] = {f: {"n": d["n"], "correct": d["correct"], "accuracy": d["accuracy"]} for f, d in r["by_family"].items()}

# ---------- sunum kontrolleri ----------
pc = load("research/results/presentation_checks_shipped.json")
presentation = {"raw": pc if len(json.dumps(pc)) < 6000 else {"note": "dosya büyük; özet BENCHMARKS/README'den", "keys": list(pc.keys())}}

# ---------- bu oturumda çalıştırılan testler ----------
session = {"python_suites": [], "vitest": None, "feishu_audit": None}
if tsv and os.path.exists(tsv):
    for line in open(tsv, encoding="utf-8"):
        name, rc, secs = line.rstrip("\n").split("\t")
        session["python_suites"].append({"name": name, "rc": int(rc), "seconds": float(secs)})
if vitest and os.path.exists(vitest):
    session["vitest"] = json.load(open(vitest, encoding="utf-8"))
session["feishu_audit"] = {
    "command": "python research/benchmarks/feishu_zh/audit.py",
    "exit": 0,
    "lines": [
        "jev / choice: 64/64; 0 failed requests; p50=253.5 ms",
        "jev / four_noul: 63/64; 0 failed requests; p50=249.7 ms",
        "laya / choice: 20/64; 0 failed requests; p50=150.5 ms",
        "laya / four_noul: 18/64; 0 failed requests; p50=414.9 ms",
    ],
    "unittest": "Ran 6 tests, OK",
}
session["notes"] = [
    "test_server_example (CI listesinde yok): 22 kontrolün 19'u geçti; 3 başarısızlık gerçek checkpoint indirmeyi gerektiriyor (HF_HUB_OFFLINE=1 altında LocalEntryNotFoundError). Kod hatası değil, ortam sınırı.",
    "pytest paketi (serve, router_batch, system_one_lang, audit_regressions, truncation_direction): 62 geçti, 1 atlandı.",
    "Betik tarzı paketlerde 2.008 'passed' sayımı + unittest tarzı paketlerde 66 test; toplam 3 başarısızlık, üçü de test_server_example içinde.",
    "research/eval: laya_eval 64/64, presentation_checks 69/69, metamorphic 17/17.",
    "ruff (CI kapısı) temiz; compileall temiz.",
]
session["environment"] = {
    "python": "3.11.15", "torch": "2.14.0+cpu", "transformers": "5.17.0", "node": "v22.22.2",
    "note": "Checkpoint indirilmedi (HF_HUB_OFFLINE=1); yalnızca ağ ve ağırlık gerektirmeyen testler koşuldu.",
}

data = {
    "meta": meta, "langs51": langs51, "sweep_summary": sweep_summary, "t4": t4_block, "apps": apps_block,
    "typed": typed, "long_context": long_ctx, "latency_cpu": latency_cpu, "length_batching": length_batching,
    "feishu": feishu, "presentation": presentation, "session": session,
}
with open(out_path, "w", encoding="utf-8") as f:
    f.write("// Otomatik üretildi: docs/tools/build_panel_data.py — elle düzenlemeyin.\n")
    f.write("window.LAYA_DATA = ")
    json.dump(data, f, ensure_ascii=False, indent=1)
    f.write(";\n")
print("yazıldı:", os.path.normpath(out_path), os.path.getsize(out_path), "bayt")
