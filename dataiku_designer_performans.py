# -*- coding: utf-8 -*-
# =============================================================================
# Dataiku Designer Node — Performans / Kaynak Kullanımı Raporu
# =============================================================================
# Dataiku Python notebook'unda çalışır ve notebook kernel'inin çalıştığı
# makinenin (varsayılan kurulumda DSS Designer node'u) anlık kaynak durumunu
# gösterir:
#   - Sistem bilgisi (makine adı, OS, uptime, DSS sürümü)
#   - CPU: çekirdek sayısı, anlık kullanım %, çekirdek bazlı kullanım, yük ort.
#   - RAM / Swap: toplam, kullanılan, kullanılabilir + bu notebook'un tüketimi
#   - Disk: kök, temp, çalışma dizini ve DSS data dizini (DIP_HOME) doluluğu
#   - GPU / VRAM: nvidia-smi üzerinden kullanım, sıcaklık, güç ve GPU süreçleri
#   - En çok RAM tüketen süreçler
#   - Opsiyonel: canlı izleme (belirli aralıklarla yenilenen tablo)
#
# Notlar:
#   - Notebook "containerized execution" ile çalışıyorsa değerler o konteynere
#     aittir; cgroup CPU/RAM limitleri tespit edilirse ayrıca gösterilir.
#   - psutil varsa kullanılır; yoksa /proc ve ps/nvidia-smi komutlarıyla çalışır.
#   - GPU bilgisi için makinede nvidia-smi kurulu olmalıdır (NVIDIA sürücüsü).
# =============================================================================

# %% 1) Kurulum ve yardımcı fonksiyonlar

import os
import platform
import shutil
import subprocess
import tempfile
import time
from datetime import datetime

import pandas as pd
from IPython.display import display

try:
    import psutil
    PSUTIL = True
except ImportError:
    psutil = None
    PSUTIL = False
    print("Not: psutil bulunamadı; /proc ve komut tabanlı yedek yöntemler kullanılacak.")


def okunur(bayt):
    """Bayt değerini okunur metne çevirir (ör. 15.6 GB)."""
    if bayt is None:
        return "-"
    bayt = float(bayt)
    for birim in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(bayt) < 1024 or birim == "PB":
            return f"{bayt:.1f} {birim}"
        bayt /= 1024


def _proc_stat_oku():
    """/proc/stat'tan toplam ve boşta geçen CPU jiffy sayılarını okur."""
    with open("/proc/stat") as f:
        sayilar = list(map(int, f.readline().split()[1:]))
    bosta = sayilar[3] + (sayilar[4] if len(sayilar) > 4 else 0)  # idle + iowait
    return sum(sayilar), bosta


def cpu_yuzde(aralik=0.5):
    """Verilen örnekleme aralığında ortalama CPU kullanımını (%) ölçer."""
    if PSUTIL:
        return psutil.cpu_percent(interval=aralik)
    t1, b1 = _proc_stat_oku()
    time.sleep(aralik)
    t2, b2 = _proc_stat_oku()
    toplam = t2 - t1
    return round(100 * (toplam - (b2 - b1)) / toplam, 1) if toplam else 0.0


def _meminfo():
    """/proc/meminfo içeriğini bayt cinsinden sözlük olarak döndürür."""
    degerler = {}
    with open("/proc/meminfo") as f:
        for satir in f:
            ad, deger = satir.split(":", 1)
            degerler[ad] = int(deger.strip().split()[0]) * 1024  # kB -> bayt
    return degerler


def _cgroup_oku(yollar, donustur):
    for yol in yollar:
        try:
            with open(yol) as f:
                return donustur(f.read().strip())
        except (OSError, ValueError):
            continue
    return None


def cgroup_limitleri():
    """Konteynerde çalışılıyorsa cgroup CPU/RAM limit ve kullanımını döndürür."""
    sonuc = {}
    limit = _cgroup_oku(
        ["/sys/fs/cgroup/memory.max",                      # cgroup v2
         "/sys/fs/cgroup/memory/memory.limit_in_bytes"],   # cgroup v1
        lambda ham: None if ham == "max" else int(ham))
    if limit and limit < 1 << 60:  # v1'de "limitsiz" çok büyük bir sayıdır
        sonuc["bellek_limiti"] = limit
        kullanim = _cgroup_oku(
            ["/sys/fs/cgroup/memory.current",
             "/sys/fs/cgroup/memory/memory.usage_in_bytes"], int)
        if kullanim is not None:
            sonuc["bellek_kullanimi"] = kullanim

    def _cpu_v2(ham):
        kota, periyot = ham.split()
        return None if kota == "max" else round(int(kota) / int(periyot), 2)

    cpu = _cgroup_oku(["/sys/fs/cgroup/cpu.max"], _cpu_v2)
    if cpu is None:
        kota = _cgroup_oku(["/sys/fs/cgroup/cpu/cpu.cfs_quota_us"], int)
        periyot = _cgroup_oku(["/sys/fs/cgroup/cpu/cpu.cfs_period_us"], int)
        if kota and periyot and kota > 0:
            cpu = round(kota / periyot, 2)
    if cpu:
        sonuc["cpu_limiti"] = cpu
    return sonuc


def _bu_surec_rss():
    """Bu notebook kernel'inin kullandığı fiziksel bellek (RSS)."""
    if PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss
    try:
        with open("/proc/self/status") as f:
            for satir in f:
                if satir.startswith("VmRSS:"):
                    return int(satir.split()[1]) * 1024
    except OSError:
        pass
    return None

# %% 2) Ölçüm fonksiyonları


def sistem_bilgisi():
    bilgi = {
        "Makine adı": platform.node(),
        "İşletim sistemi": f"{platform.system()} {platform.release()}",
        "Python sürümü": platform.python_version(),
        "Ölçüm zamanı": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        with open("/proc/uptime") as f:
            saniye = float(f.read().split()[0])
        bilgi["Makine açık kalma süresi"] = f"{saniye / 86400:.1f} gün"
    except OSError:
        pass
    try:  # DSS sürümü (notebook Dataiku içinde çalışıyorsa)
        import dataiku
        bilgi["DSS sürümü"] = (
            dataiku.api_client().get_instance_info().raw.get("dssVersion", "?"))
    except Exception:
        pass
    return pd.Series(bilgi)


def cpu_ozeti():
    """CPU özeti (Series) ve çekirdek bazlı anlık kullanım listesi döndürür."""
    bilgi = {
        "Mantıksal çekirdek": os.cpu_count(),
        "Fiziksel çekirdek": (psutil.cpu_count(logical=False) or "-") if PSUTIL else "-",
        "Anlık CPU kullanımı %": cpu_yuzde(1.0),
    }
    try:
        bilgi["Yük ortalaması (1/5/15 dk)"] = " / ".join(
            f"{yuk:.2f}" for yuk in os.getloadavg())
    except OSError:
        pass
    if PSUTIL:
        try:
            frekans = psutil.cpu_freq()
            if frekans:
                bilgi["CPU frekansı"] = f"{frekans.current:.0f} MHz"
        except Exception:
            pass
    limitler = cgroup_limitleri()
    if "cpu_limiti" in limitler:
        bilgi["Konteyner CPU limiti (çekirdek)"] = limitler["cpu_limiti"]
    cekirdekler = psutil.cpu_percent(interval=0.5, percpu=True) if PSUTIL else None
    return pd.Series(bilgi), cekirdekler


def bellek_ozeti():
    if PSUTIL:
        ram = psutil.virtual_memory()
        swap = psutil.swap_memory()
        toplam, uygun, yuzde = ram.total, ram.available, ram.percent
        swap_toplam, swap_kullanilan = swap.total, swap.used
    else:
        m = _meminfo()
        toplam = m["MemTotal"]
        uygun = m.get("MemAvailable", m.get("MemFree", 0))
        yuzde = round(100 * (toplam - uygun) / toplam, 1)
        swap_toplam = m.get("SwapTotal", 0)
        swap_kullanilan = swap_toplam - m.get("SwapFree", 0)
    bilgi = {
        "Toplam RAM": okunur(toplam),
        "Kullanılan RAM": f"{okunur(toplam - uygun)} (%{yuzde})",
        "Kullanılabilir RAM": okunur(uygun),
        "Swap (kullanılan / toplam)": f"{okunur(swap_kullanilan)} / {okunur(swap_toplam)}",
        "Bu notebook süreci (RSS)": okunur(_bu_surec_rss()),
    }
    limitler = cgroup_limitleri()
    if "bellek_limiti" in limitler:
        bilgi["Konteyner RAM limiti"] = okunur(limitler["bellek_limiti"])
        if "bellek_kullanimi" in limitler:
            bilgi["Konteyner RAM kullanımı"] = okunur(limitler["bellek_kullanimi"])
    return pd.Series(bilgi)


def disk_tablosu():
    yollar = {"/", tempfile.gettempdir(), os.getcwd()}
    dip_home = os.environ.get("DIP_HOME")  # DSS data dizini
    if dip_home:
        yollar.add(dip_home)
    kayitlar = []
    for yol in sorted(yollar):
        try:
            kullanim = shutil.disk_usage(yol)
        except OSError:
            continue
        kayitlar.append({
            "YOL": yol,
            "TOPLAM": okunur(kullanim.total),
            "KULLANILAN": okunur(kullanim.used),
            "BOŞ": okunur(kullanim.free),
            "KULLANIM %": round(100 * kullanim.used / kullanim.total, 1),
        })
    return pd.DataFrame(kayitlar)


def gpu_tablosu():
    """nvidia-smi ile GPU/VRAM bilgisi. GPU ya da sürücü yoksa None döner."""
    sorgu = ("index,name,driver_version,utilization.gpu,utilization.memory,"
             "memory.total,memory.used,memory.free,temperature.gpu,power.draw")
    try:
        cikti = subprocess.run(
            ["nvidia-smi", f"--query-gpu={sorgu}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if cikti.returncode != 0 or not cikti.stdout.strip():
        return None
    satirlar = [[alan.strip() for alan in satir.split(",")]
                for satir in cikti.stdout.strip().splitlines()]
    kolonlar = ["GPU", "MODEL", "SÜRÜCÜ", "GPU KULLANIM %", "VRAM MEŞGULİYET %",
                "VRAM TOPLAM (MiB)", "VRAM KULLANILAN (MiB)", "VRAM BOŞ (MiB)",
                "SICAKLIK (°C)", "GÜÇ (W)"]
    return pd.DataFrame(satirlar, columns=kolonlar)


def gpu_surecleri():
    """GPU belleği kullanan süreçler (pid, ad, VRAM)."""
    try:
        cikti = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if cikti.returncode != 0:
        return None
    satirlar = [[alan.strip() for alan in satir.split(",")]
                for satir in cikti.stdout.strip().splitlines() if satir.strip()]
    return pd.DataFrame(satirlar, columns=["PID", "SÜREÇ", "VRAM (MiB)"])


def surec_tablosu(adet=10):
    """RAM'e göre en çok tüketen süreçler (CPU %'si ~0.5 sn örneklemeyle)."""
    if PSUTIL:
        surecler = list(psutil.process_iter(["pid", "name", "username"]))
        for surec in surecler:
            try:
                surec.cpu_percent()  # ilk çağrı ölçüm penceresini başlatır
            except psutil.Error:
                pass
        time.sleep(0.5)
        satirlar = []
        for surec in surecler:
            try:
                satirlar.append({
                    "PID": surec.pid,
                    "SÜREÇ": surec.info["name"],
                    "KULLANICI": surec.info.get("username") or "-",
                    "_rss": surec.memory_info().rss,
                    "CPU %": surec.cpu_percent(),
                })
            except psutil.Error:
                continue
        if not satirlar:
            return pd.DataFrame()
        df = pd.DataFrame(satirlar).sort_values("_rss", ascending=False).head(adet)
        df["RAM"] = df["_rss"].map(okunur)
        return df[["PID", "SÜREÇ", "KULLANICI", "RAM", "CPU %"]].reset_index(drop=True)
    # psutil yoksa: ps ile
    cikti = subprocess.run(["ps", "aux", "--sort=-rss"],
                           capture_output=True, text=True, timeout=15)
    kayitlar = []
    for satir in cikti.stdout.strip().splitlines()[1:adet + 1]:
        parcalar = satir.split(None, 10)
        if len(parcalar) < 11:
            continue
        kayitlar.append({
            "PID": parcalar[1],
            "SÜREÇ": parcalar[10][:60],
            "KULLANICI": parcalar[0],
            "RAM": okunur(int(parcalar[5]) * 1024),
            "CPU %": parcalar[2],
        })
    return pd.DataFrame(kayitlar)

# %% 3) Anlık performans raporu


def basim(baslik):
    print("\n" + "=" * 72)
    print(baslik)
    print("=" * 72)


print("Not: Ölçümler notebook kernel'inin çalıştığı makineye aittir "
      "(varsayılan: DSS Designer node; containerized execution'da ilgili konteyner).")

basim("SİSTEM BİLGİSİ")
display(sistem_bilgisi().to_frame("DEĞER"))

basim("CPU")
cpu_bilgi, cekirdek_bazli = cpu_ozeti()
display(cpu_bilgi.to_frame("DEĞER"))
if cekirdek_bazli:
    display(pd.DataFrame([cekirdek_bazli],
                         columns=[f"C{i}" for i in range(len(cekirdek_bazli))],
                         index=["ÇEKİRDEK KULLANIM %"]))

basim("RAM / SWAP")
display(bellek_ozeti().to_frame("DEĞER"))

basim("DİSK")
display(disk_tablosu())

basim("GPU / VRAM")
gpular = gpu_tablosu()
if gpular is None or gpular.empty:
    print("GPU bulunamadı (nvidia-smi yok, sürücü kurulu değil ya da GPU erişilemiyor).")
else:
    display(gpular)
    gpu_prosesler = gpu_surecleri()
    if gpu_prosesler is not None and not gpu_prosesler.empty:
        print("GPU belleği kullanan süreçler:")
        display(gpu_prosesler)
    else:
        print("Şu anda GPU belleği kullanan süreç yok.")

basim(f"EN ÇOK RAM TÜKETEN {10} SÜREÇ")
display(surec_tablosu(10))

basim("ÖZET")
ozet = {"CPU KULLANIM %": cpu_yuzde(0.5)}
if PSUTIL:
    ram = psutil.virtual_memory()
    ozet["RAM"] = f"{okunur(ram.total - ram.available)} / {okunur(ram.total)} (%{ram.percent})"
else:
    m = _meminfo()
    kullanilan = m["MemTotal"] - m.get("MemAvailable", 0)
    ozet["RAM"] = (f"{okunur(kullanilan)} / {okunur(m['MemTotal'])} "
                   f"(%{100 * kullanilan / m['MemTotal']:.0f})")
if gpular is not None and not gpular.empty:
    ozet["GPU KULLANIM %"] = " | ".join(gpular["GPU KULLANIM %"])
    ozet["VRAM (MiB)"] = " | ".join(
        gpular["VRAM KULLANILAN (MiB)"] + " / " + gpular["VRAM TOPLAM (MiB)"])
display(pd.DataFrame([ozet], index=["DEĞER"]))

# %% 4) (Opsiyonel) Canlı izleme


def canli_izle(sure_sn=60, aralik_sn=2):
    """CPU / RAM / GPU kullanımını canlı izler; geçmişi DataFrame döndürür.

    Ayrı bir hücrede çalıştırın; Kernel > Interrupt ile erken durdurabilirsiniz
    (o ana kadarki geçmiş yine döner).
    """
    from IPython.display import clear_output
    gecmis = []
    bitis = time.time() + sure_sn
    try:
        while time.time() < bitis:
            kayit = {"ZAMAN": datetime.now().strftime("%H:%M:%S"),
                     "CPU %": cpu_yuzde(aralik_sn)}
            if PSUTIL:
                kayit["RAM %"] = psutil.virtual_memory().percent
            else:
                m = _meminfo()
                kayit["RAM %"] = round(
                    100 * (m["MemTotal"] - m.get("MemAvailable", 0)) / m["MemTotal"], 1)
            gpu_df = gpu_tablosu()
            if gpu_df is not None and not gpu_df.empty:
                kayit["GPU %"] = pd.to_numeric(
                    gpu_df["GPU KULLANIM %"], errors="coerce").max()
                kayit["VRAM (MiB)"] = pd.to_numeric(
                    gpu_df["VRAM KULLANILAN (MiB)"], errors="coerce").sum()
            gecmis.append(kayit)
            clear_output(wait=True)
            print(f"Canlı izleme — kalan süre: {max(0, bitis - time.time()):.0f} sn "
                  f"(durdurmak için Kernel > Interrupt)")
            display(pd.DataFrame(gecmis).tail(15))
    except KeyboardInterrupt:
        print("İzleme durduruldu.")
    return pd.DataFrame(gecmis)


# Kullanım örneği (yorumunu kaldırıp çalıştırın):
# gecmis = canli_izle(sure_sn=60, aralik_sn=2)
# gecmis.plot(x="ZAMAN", y=[k for k in ("CPU %", "RAM %", "GPU %") if k in gecmis],
#             figsize=(10, 4), grid=True)
