# rvc_trainer_gui_win2.py
# -*- coding: utf-8 -*-
#
# Windows GUI для подготовки датасета и обучения RVC v2.
# Особенности:
# - Авто-скачивание hubert/rmvpe и укладка в правильные пути.
# - Надёжный вызов скриптов WebUI с авто-детектом их сигнатур.
# - Прогресс F0/фич и обучение (epoch / total), можно продолжать обучение.
# - Безопасная работа на Windows (UTF-8, без обязательных симлинков/админа).
#
# Среда (пример):
#   conda create -n rvcwin python=3.10 -y
#   conda activate rvcwin
#   pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 torchaudio==2.5.1
#   pip install numpy soxr soundfile librosa tqdm faiss-cpu huggingface_hub gitpython pyyaml requests
#
# ВНИМАНИЕ: fairseq2/fairseq2n на Windows обычно без готовых колес; используем HuBERT.
#            (колеса fairseq2n официально только для Linux). см. ссылки в описании.
#
import os
import sys
import re
import shutil
import subprocess
import threading
import queue
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

# --- Константы/дефолты под ваш случай ---
DEFAULT_ROOT = r"C:\Users\admin\Documents\okrug"
DEFAULT_PROJECT = "okrug_rvc48"
DEFAULT_SR = "48k"  # 48k / 32000 в некоторых форках; для RVC v2 обычно 48k
DEFAULT_WAV = os.path.join(DEFAULT_ROOT, "input", "dataset.wav")

# Хранилище и WebUI
def webui_root(root: str) -> str:
    return os.path.join(root, "RVC", "Retrieval-based-Voice-Conversion-WebUI")

def assets_dir(root: str) -> str:
    return os.path.join(root, "RVC", "assets")

def datasets_dir(root: str, project: str) -> str:
    return os.path.join(webui_root(root), "datasets", project)

def logs_dir(root: str, project: str) -> str:
    return os.path.join(webui_root(root), "logs", project)

# Полезные пути с весами
def hubert_path(root: str) -> str:
    return os.path.join(assets_dir(root), "hubert_base.pt")  # классический путь

def rmvpe_model_dir(root: str) -> str:
    return os.path.join(assets_dir(root), "rmvpe")  # важно: именно assets/rmvpe

def rmvpe_path(root: str) -> str:
    return os.path.join(rmvpe_model_dir(root), "rmvpe.pt")

# Универсальный запуск подпроцессов с UTF-8 окружением
def run_stream(cmd, cwd=None, log_fn=None):
    # cmd: list[str] либо str; предпочтем list
    if not isinstance(cmd, (list, tuple)):
        cmd = cmd.split()

    env = os.environ.copy()
    # Принудительно включаем UTF-8 в дочерних процессах, чтобы не ловить UnicodeDecodeError
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    # Hugging Face: глушим ворнинг про симлинки, это норм на Windows
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    p = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        encoding="utf-8",
        env=env,
    )
    for line in p.stdout:
        if log_fn:
            log_fn(line.rstrip("\n"))
    p.wait()
    return p.returncode

# Тех. функции
def ensure_dirs(root, project):
    Path(root).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(root, "input")).mkdir(parents=True, exist_ok=True)
    Path(assets_dir(root)).mkdir(parents=True, exist_ok=True)
    Path(rmvpe_model_dir(root)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(assets_dir(root), "pretrained_v2", "48k", "Snowie")).mkdir(parents=True, exist_ok=True)
    Path(datasets_dir(root, project)).mkdir(parents=True, exist_ok=True)
    Path(logs_dir(root, project)).mkdir(parents=True, exist_ok=True)

def is_git_repo(path: str) -> bool:
    return Path(os.path.join(path, ".git")).exists()

def clone_or_update_webui(root: str, log):
    repo = webui_root(root)
    if not Path(repo).exists():
        Path(os.path.join(root, "RVC")).mkdir(parents=True, exist_ok=True)
        code = run_stream(["git", "clone", "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI", repo], log_fn=log)
        if code != 0:
            raise RuntimeError("Не удалось клонировать WebUI")
    else:
        log("Репозиторий существует — пробуем git pull …")
        if is_git_repo(repo):
            code = run_stream(["git", "pull"], cwd=repo, log_fn=log)
            if code != 0:
                log("[WARN] git pull завершился с кодом != 0, оставляем как есть.")
        else:
            log("[WARN] git pull не удался — папка не является git-репозиторием, оставляем как есть.")

# Скачивание весов (с корректными repo_type)
def download_assets(root: str, log):
    ensure_dirs(root, DEFAULT_PROJECT)
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        log(f"[ERR] huggingface_hub не установлен: {e}")
        raise

    # HuBERT (dataset)
    if not Path(hubert_path(root)).exists():
        p = hf_hub_download(repo_id="AI-C/rvc-models", filename="hubert_base.pt", repo_type="dataset")
        shutil.copy2(p, hubert_path(root))
        log("OK: hubert_base.pt")
    else:
        log("Есть: hubert_base.pt")

    # RMVPE (model) — кладём строго в assets/rmvpe/rmvpe.pt
    if not Path(rmvpe_path(root)).exists():
        p = hf_hub_download(repo_id="lj1995/VoiceConversionWebUI", filename="rmvpe.pt", repo_type="model")
        shutil.copy2(p, rmvpe_path(root))
        log("OK: rmvpe.pt")
    else:
        log("Есть: rmvpe.pt")

    # Предтрен Snowie v3.1 48k (G/D)
    for fn in ["G_SnowieV3.1_48k.pth", "D_SnowieV3.1_48k.pth"]:
        dst = os.path.join(assets_dir(root), "pretrained_v2", "48k", "Snowie", fn)
        if not Path(dst).exists():
            p = hf_hub_download(
                repo_id="Politrees/RVC_resources",
                filename=f"pretrained/v2/48k/Snowie/{fn}",
                repo_type="model",
            )
            Path(os.path.dirname(dst)).mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            log(f"OK: {fn}")
        else:
            log(f"Есть: {fn}")

# Поиск нужного препроцесс-скрипта (в разных ветках он может отличаться)
def find_preprocess_script(repo_dir: str):
    candidates = [
        "trainset_preprocess_pipeline_print.py",  # классический
        "trainset_preprocess_pipeline.py",        # альтернатива в других ветках
    ]
    for c in candidates:
        p = os.path.join(repo_dir, c)
        if Path(p).exists():
            return c
    return None

def _resample_audio(data, src_sr: int, dst_sr: int):
    """Пересэмплирует аудио-массив в целевую частоту."""
    import numpy as np

    arr = np.asarray(data, dtype="float32")
    if src_sr == dst_sr:
        return arr

    try:
        import soxr  # type: ignore

        return soxr.resample(arr, src_sr, dst_sr)
    except Exception:
        try:
            import librosa  # type: ignore

            return librosa.resample(arr, orig_sr=src_sr, target_sr=dst_sr)
        except Exception as exc:
            raise RuntimeError(
                "Для пересэмплинга в 16 кГц требуется пакет soxr или librosa. "
                "Установите один из них (pip install soxr)."
            ) from exc


def slice_and_copy_dataset(root: str, project: str, input_wav: str, sr: str, log, chunk_ms=8000):
    """Нарезаем входной WAV, готовим 0_gt_wavs и 1_16k_wavs."""
    repo = webui_root(root)
    if not Path(input_wav).exists():
        raise FileNotFoundError(f"Не найден входной WAV: {input_wav}")

    # Подготовим датасет: скопируем оригинал для совместимости
    dst_wav = os.path.join(datasets_dir(root, project), "dataset.wav")
    Path(os.path.dirname(dst_wav)).mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_wav, dst_wav)

    # Срезка на чанки (на Python — чтобы одинаково на Windows)
    import soundfile as sf
    import numpy as np

    data, in_sr = sf.read(input_wav)
    if data.ndim > 1:
        data = data.mean(axis=1)  # моно

    hop = int((chunk_ms / 1000.0) * in_sr)
    out_root = os.path.join(webui_root(root), "datasets", project)
    out_chunks = os.path.join(out_root, "chunks")
    Path(out_chunks).mkdir(parents=True, exist_ok=True)

    gt_dir = os.path.join(logs_dir(root, project), "0_gt_wavs")
    Path(gt_dir).mkdir(parents=True, exist_ok=True)

    wav16_dir = os.path.join(logs_dir(root, project), "1_16k_wavs")
    Path(wav16_dir).mkdir(parents=True, exist_ok=True)

    count = 0
    for i in range(0, len(data), hop):
        chunk = data[i:i+hop]
        if len(chunk) < hop // 3:  # коротыши не берём
            continue
        count += 1
        name = f"{project}_{count:05d}.wav"
        outp = os.path.join(out_chunks, name)
        sf.write(outp, chunk, in_sr)

        # 0_gt_wavs — оригинальная частота
        shutil.copy2(outp, os.path.join(gt_dir, name))

        # 1_16k_wavs — обязательно для rmvpe на Windows
        chunk16 = _resample_audio(chunk, in_sr, 16000).astype("float32")
        sf.write(os.path.join(wav16_dir, name), chunk16, 16000)
        if count % 50 == 0:
            log(f"Чанк {count}")

    log(f"Нарезка завершена, сегментов: {count}")
    log(f"Готово: скопировано в {gt_dir}")
    log(f"Готово: создано {wav16_dir}")

def preprocess_with_webui(root: str, project: str, sr: str, log):
    """Запуск trainset_preprocess_xxx.py (если есть). Иначе пропускаем (мы уже нарезали)."""
    repo = webui_root(root)
    script = find_preprocess_script(repo)
    if script is None:
        log("[WARN] Не найден trainset_preprocess_pipeline*.py в репозитории WebUI — пропустим (у нас уже есть 0_gt_wavs).")
        return
    # Некоторые форки ожидают: python script.py <datasets_dir> <sr> <n_threads> <logs_dir> <skip_norm_pitch>
    cmd = [sys.executable, script, datasets_dir(root, project), "48000", "8", logs_dir(root, project), "False"]
    log("Препроцесс WebUI…")
    code = run_stream(cmd, cwd=repo, log_fn=log)
    if code != 0:
        log("[WARN] Препроцесс в форке не завершился успешно — это не критично, если 0_gt_wavs уже готов.")

# --- F0: авто-детект сигнатуры extract_f0_rmvpe.py и вызов ---
def detect_f0_signature(repo_dir: str) -> str:
    """
    Возвращает 'parts' если скрипт принимает n_part, i_part, i_gpu, exp_dir, is_half
    Иначе 'simple' (если когда-то встретится иная вариация).
    """
    f0_script = os.path.join(repo_dir, "infer", "modules", "train", "extract", "extract_f0_rmvpe.py")
    if not Path(f0_script).exists():
        # В некоторых форках имя без поддиректории 'extract'
        f0_script = os.path.join(repo_dir, "infer", "modules", "train", "extract_f0_rmvpe.py")
    if not Path(f0_script).exists():
        raise FileNotFoundError("extract_f0_rmvpe.py не найден в вашем WebUI")

    text = Path(f0_script).read_text(encoding="utf-8", errors="ignore")
    # По мотивам распространённой реализации: sys.argv[1] = n_part, [2]=i_part, [3]=i_gpu, [4]=exp_dir, [5]=is_half
    if re.search(r"sys\.argv\[\s*1\s*\].*n_part", text) or "n_part = int(sys.argv[1])" in text:
        return "parts"
    return "simple"

def extract_f0_rmvpe(root: str, project: str, log, gpu_index="0", is_half="False"):
    repo = webui_root(root)
    sig = detect_f0_signature(repo)
    log(f"Извлечение F0 (скрипт: infer\\modules\\train\\extract\\extract_f0_rmvpe.py, сигнатура: {sig})…")

    # Скрипт ищет веса строго в assets/rmvpe/rmvpe.pt
    if not Path(rmvpe_path(root)).exists():
        raise FileNotFoundError(f"Не найден RMVPE: {rmvpe_path(root)}")

    if sig == "parts":
        # Разобьём на 1 часть для простоты. Формат: n_part i_part i_gpu exp_dir is_half
        cmd = [
            sys.executable,
            os.path.join("infer", "modules", "train", "extract", "extract_f0_rmvpe.py"),
            "1", "0", gpu_index, logs_dir(root, project), is_half
        ]
    else:
        # Редкие варианты — подправь при необходимости
        cmd = [
            sys.executable,
            os.path.join("infer", "modules", "train", "extract", "extract_f0_rmvpe.py"),
            logs_dir(root, project), gpu_index, is_half
        ]
    code = run_stream(cmd, cwd=repo, log_fn=log)
    if code != 0:
        raise RuntimeError("Извлечение F0 завершилось с ошибкой")

# --- Фичи: авто-детект сигнатуры extract_feature_print.py и вызов ---
def detect_feat_signature(repo_dir: str) -> str:
    """
    'parts'  -> с аргументами n_part, i_part, i_gpu, exp_dir, device, is_half, sr, [hubert?]
    'classic'-> use_gpu i_gpu exp_dir sr n_threads hubert
    """
    p = os.path.join(repo_dir, "infer", "modules", "train", "extract_feature_print.py")
    if not Path(p).exists():
        raise FileNotFoundError("extract_feature_print.py не найден в вашем WebUI")
    text = Path(p).read_text(encoding="utf-8", errors="ignore")

    # Признаки parted-варианта (как в rvc-playground):
    if re.search(r"i_part\s*=\s*int\(sys\.argv\[3\]\)", text) and ("n_part" in text):
        return "parts"

    # Признак «классики»: начало с use_gpu/i_gpu/exp_dir/sr/n_threads/... (встречается в старых ветках)
    if re.search(r"sys\.argv\[\s*1\s*\].*use_gpu", text) and "hubert" in text:
        return "classic"

    # fallback: попробуем classic
    return "classic"

def extract_features(root: str, project: str, sr: str, log, gpu_index="0"):
    repo = webui_root(root)
    hub = hubert_path(root)
    if not Path(hub).exists():
        raise FileNotFoundError(f"Не найден HuBERT: {hub}")

    sig = detect_feat_signature(repo)
    # На Windows fairseq2/fairseq2n почти всегда недоступен — используем HuBERT
    # (Даже если импорт fairseq2 пройдёт, в большинстве форков auto-switch уже внутри скрипта.)
    if sig == "parts":
        # Пример распространённой сигнатуры parted:
        # python extract_feature_print.py <device> <n_part> <i_part> <i_gpu> <exp_dir> <is_half> <sr> [hubert]
        cmd = [
            sys.executable, os.path.join("infer", "modules", "train", "extract_feature_print.py"),
            "cuda", "1", "0", gpu_index, logs_dir(root, project), "False", sr, hub
        ]
    else:
        # Классическая сигнатура:
        # python extract_feature_print.py <use_gpu> <i_gpu> <exp_dir> <sr> <n_threads> <hubert_path>
        cmd = [
            sys.executable, os.path.join("infer", "modules", "train", "extract_feature_print.py"),
            "True", gpu_index, logs_dir(root, project), sr, "0", hub
        ]
    log("Извлечение фичей…")
    code = run_stream(cmd, cwd=repo, log_fn=log)
    if code != 0:
        raise RuntimeError("Экстракция фичей завершилась с ошибкой.")

# --- Обучение ---
def pretrained_snowie_paths(root: str):
    g = os.path.join(assets_dir(root), "pretrained_v2", "48k", "Snowie", "G_SnowieV3.1_48k.pth")
    d = os.path.join(assets_dir(root), "pretrained_v2", "48k", "Snowie", "D_SnowieV3.1_48k.pth")
    return g, d

def start_training(root: str, project: str, log, sr="48k", f0_flag="1", bs="6", te="300", se="25"):
    repo = webui_root(root)
    g, d = pretrained_snowie_paths(root)

    # Наиболее совместимый тренер:
    # train_nsf_sim_cache_sid_load_pretrain.py
    trainer = os.path.join(repo, "train_nsf_sim_cache_sid_load_pretrain.py")
    if not Path(trainer).exists():
        # Некоторые ветки кладут тренер в корень infer/modules/train/...
        alt = os.path.join(repo, "infer", "modules", "train", "train_nsf_sim_cache_sid_load_pretrain.py")
        trainer = alt if Path(alt).exists() else None
    if trainer is None or not Path(trainer).exists():
        raise FileNotFoundError("Не найден train_nsf_sim_cache_sid_load_pretrain.py в вашем WebUI")

    if not Path(g).exists() or not Path(d).exists():
        log("[WARN] Предтрен Snowie не найден — обучение запустится с нуля.")

    # Формат параметров встречается одинаковый в разных форках:
    # -e <эксперимент>, -sr <48k>, -f0 <0/1>, -bs <batch>, -te <total_epochs>, -se <save_every>, -pg/-pd (pretrained)
    cmd = [
        sys.executable, trainer,
        "-e", project,
        "-sr", sr,
        "-f0", f0_flag,
        "-bs", bs,
        "-te", te,
        "-se", se
    ]
    if Path(g).exists():
        cmd += ["-pg", g]
    if Path(d).exists():
        cmd += ["-pd", d]

    # Чтобы показать прогресс эпох — будем парсить stdout
    epoch_re = re.compile(r"epoch[^0-9]*([0-9]+)\s*[/|]\s*([0-9]+)", re.I)
    total_epochs = None
    current_epoch = 0

    log("Старт обучения…")
    if not isinstance(cmd, (list, tuple)):
        cmd = cmd.split()

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    p = subprocess.Popen(
        cmd, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, encoding="utf-8", env=env,
    )
    for line in p.stdout:
        line = line.rstrip("\n")
        log(line)
        m = epoch_re.search(line)
        if m:
            current_epoch = int(m.group(1))
            try:
                total_epochs = int(m.group(2))
            except:
                pass
    p.wait()
    code = p.returncode
    if code != 0:
        raise RuntimeError("Обучение завершилось с ошибкой.")

# --- GUI ---
class App:
    def __init__(self, master):
        self.m = master
        self.m.title("RVC Trainer (Windows) — Politrees")
        self.root_var = tk.StringVar(value=DEFAULT_ROOT)
        self.project_var = tk.StringVar(value=DEFAULT_PROJECT)
        self.wav_var = tk.StringVar(value=DEFAULT_WAV)
        self.sr_var = tk.StringVar(value=DEFAULT_SR)
        self.bs_var = tk.StringVar(value="6")
        self.epochs_var = tk.StringVar(value="300")
        self.saveevery_var = tk.StringVar(value="25")
        self.gpu_var = tk.StringVar(value="0")

        frm = tk.Frame(self.m)
        frm.pack(fill="both", expand=True, padx=8, pady=8)

        row = 0
        tk.Label(frm, text="Рабочая папка").grid(row=row, column=0, sticky="w")
        tk.Entry(frm, textvariable=self.root_var, width=70).grid(row=row, column=1, sticky="we")
        tk.Button(frm, text="Обзор", command=self.pick_root).grid(row=row, column=2, padx=4)
        row += 1

        tk.Label(frm, text="Проект (имя)").grid(row=row, column=0, sticky="w")
        tk.Entry(frm, textvariable=self.project_var).grid(row=row, column=1, sticky="we")
        row += 1

        tk.Label(frm, text="Входной WAV").grid(row=row, column=0, sticky="w")
        tk.Entry(frm, textvariable=self.wav_var, width=70).grid(row=row, column=1, sticky="we")
        tk.Button(frm, text="Обзор", command=self.pick_wav).grid(row=row, column=2, padx=4)
        row += 1

        tk.Label(frm, text="Sample rate").grid(row=row, column=0, sticky="w")
        tk.Entry(frm, textvariable=self.sr_var, width=10).grid(row=row, column=1, sticky="w")
        row += 1

        tk.Label(frm, text="GPU index").grid(row=row, column=0, sticky="w")
        tk.Entry(frm, textvariable=self.gpu_var, width=6).grid(row=row, column=1, sticky="w")
        row += 1

        tk.Label(frm, text="Batch size").grid(row=row, column=0, sticky="w")
        tk.Entry(frm, textvariable=self.bs_var, width=6).grid(row=row, column=1, sticky="w")
        row += 1

        tk.Label(frm, text="Всего эпох").grid(row=row, column=0, sticky="w")
        tk.Entry(frm, textvariable=self.epochs_var, width=8).grid(row=row, column=1, sticky="w")
        row += 1

        tk.Label(frm, text="Сохранять каждые N эпох").grid(row=row, column=0, sticky="w")
        tk.Entry(frm, textvariable=self.saveevery_var, width=8).grid(row=row, column=1, sticky="w")
        row += 1

        btns = tk.Frame(frm)
        btns.grid(row=row, column=0, columnspan=3, sticky="we", pady=(8, 4))
        tk.Button(btns, text="1) Скачать/обновить WebUI + веса", command=self.step_1).pack(side="left", padx=4)
        tk.Button(btns, text="2) Нарезать WAV и подготовить датасет", command=self.step_2).pack(side="left", padx=4)
        tk.Button(btns, text="3) Препроцесс WebUI (если есть)", command=self.step_3).pack(side="left", padx=4)
        tk.Button(btns, text="4) Извлечь F0", command=self.step_4).pack(side="left", padx=4)
        tk.Button(btns, text="5) Извлечь фичи", command=self.step_5).pack(side="left", padx=4)
        tk.Button(btns, text="6) Обучение (start/continue)", command=self.step_6).pack(side="left", padx=4)
        row += 1

        self.pbar = ttk.Progressbar(frm, orient=tk.HORIZONTAL, mode='indeterminate', length=400)
        self.pbar.grid(row=row, column=0, columnspan=3, sticky="we", pady=(4, 4))
        row += 1

        self.log = scrolledtext.ScrolledText(frm, height=22)
        self.log.grid(row=row, column=0, columnspan=3, sticky="nsew")
        frm.rowconfigure(row, weight=1)
        frm.columnconfigure(1, weight=1)

    def pick_root(self):
        d = filedialog.askdirectory(initialdir=self.root_var.get() or "C:\\")
        if d:
            self.root_var.set(d)

    def pick_wav(self):
        f = filedialog.askopenfilename(filetypes=[("WAV", "*.wav"), ("All", "*.*")])
        if f:
            self.wav_var.set(f)

    def write_log(self, s: str):
        self.log.insert("end", s + "\n")
        self.log.see("end")
        self.m.update_idletasks()

    def run_in_thread(self, target):
        def _task():
            try:
                self.pbar.start(10)
                target()
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))
                self.write_log(f"[ERR] {e}")
            finally:
                self.pbar.stop()
        threading.Thread(target=_task, daemon=True).start()

    # Шаги
    def step_1(self):
        def job():
            root = self.root_var.get().strip()
            clone_or_update_webui(root, self.write_log)
            download_assets(root, self.write_log)
            self.write_log("Готово.")
        self.run_in_thread(job)

    def step_2(self):
        def job():
            root = self.root_var.get().strip()
            project = self.project_var.get().strip()
            wav = self.wav_var.get().strip()
            ensure_dirs(root, project)
            self.write_log(f"Нарезка входного WAV → {datasets_dir(root, project)}")
            slice_and_copy_dataset(root, project, wav, self.sr_var.get().strip(), self.write_log)
        self.run_in_thread(job)

    def step_3(self):
        def job():
            root = self.root_var.get().strip()
            project = self.project_var.get().strip()
            preprocess_with_webui(root, project, self.sr_var.get().strip(), self.write_log)
        self.run_in_thread(job)

    def step_4(self):
        def job():
            root = self.root_var.get().strip()
            project = self.project_var.get().strip()
            extract_f0_rmvpe(root, project, self.write_log, gpu_index=self.gpu_var.get().strip(), is_half="False")
        self.run_in_thread(job)

    def step_5(self):
        def job():
            root = self.root_var.get().strip()
            project = self.project_var.get().strip()
            extract_features(root, project, self.sr_var.get().strip(), self.write_log, gpu_index=self.gpu_var.get().strip())
        self.run_in_thread(job)

    def step_6(self):
        def job():
            root = self.root_var.get().strip()
            project = self.project_var.get().strip()
            start_training(
                root, project, self.write_log,
                sr=self.sr_var.get().strip(),
                f0_flag="1",
                bs=self.bs_var.get().strip(),
                te=self.epochs_var.get().strip(),
                se=self.saveevery_var.get().strip(),
            )
        self.run_in_thread(job)

def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
