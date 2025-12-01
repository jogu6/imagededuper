# -*- coding: utf-8 -*-
"""
imagededuper.py — 独自 log/safe 統合・最終版

主な仕様:
- フォルダ配下の画像から重複画像を検出し、duplicates フォルダへ移動
- 拡張子前処理（HEIC→JPG, JFIF→JPG, 拡張子統一）
- pHash で候補を絞り、SSIM で最終判定
- 重複は SSIM >= 0.85、解像度（幅×高さ）が小さい方を duplicates へ
- 画像読み込みフェーズと比較フェーズの両方に進捗バー表示
- ETA は「比較元画像が変わったとき」にだけ更新
- ログと進捗バーは独自 log/safe により衝突しない
"""

import os
import sys
import shutil
import hashlib
import time
import traceback
import threading
import random
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
import pillow_heif
from skimage.metrics import structural_similarity as ssim
import psutil
from scipy.fftpack import dct as dct_1d  # pHash 用 DCT
import json
import msvcrt

pillow_heif.register_heif_opener()

# ============================================
# デバッグ設定
# ============================================
DEBUG_LOG_SSIM = False  # SSIM 計算時の詳細ログを出すかどうか

# ============================================
# パス関連
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "log")
os.makedirs(LOG_DIR, exist_ok=True)

# ============================================
# ログ用グローバル
# ============================================
_log_lock = threading.Lock()
_last_log_time = None  # 差分表示用
_last_log_line = None  # トレースバックログ用

# ============================================
# 進捗用グローバル
# ============================================
CURRENT_PROGRESS = 0
TOTAL_PROGRESS = 1
CURRENT_ETA_STR = "計測中"

LOAD_START_TIME = 0.0
LOAD_LAST_PERCENT = -1

PROGRESS_BAR_WIDTH = 30

PROGRESS_MODE = "none"  # "none" | "load" | "compare"
LOAD_DONE = 0
LOAD_TOTAL = 1

BASE_START_DONE = 0
BASE_START_TIME = 0.0
MOVE_START_TIME = 0.0  # 全体開始時間

TOTAL_SOURCE_IMAGES = 0
PROCESSED_BASE_COUNT = 0

RESUME_FILE_NAME = "resume.json"

def quit_requested():
    return msvcrt.kbhit() and msvcrt.getch() in (b"q", b"Q")

def save_resume(resume_path: str, i: int, j: int,
                moved: set[str], progress: int):
    data = {
        "i": i,
        "j": j,
        "moved": list(moved),
        "current_progress": progress,
    }
    with open(resume_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_resume(resume_path: str):
    if not os.path.exists(resume_path):
        return None
    with open(resume_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ============================================
# 低レベル：ログの整形と書き込み
# ============================================
def _format_timestamp_with_delta() -> str:
    """[YYYYMMDD HHMMSS ΔΔΔΔΔΔ] 形式のタイムスタンプを返す"""
    global _last_log_time
    now = datetime.now()
    ts = now.strftime("%Y%m%d %H%M%S")

    if _last_log_time:
        diff = (now - _last_log_time).total_seconds()
        if diff < 1.0:
            delta_str = "------"
        else:
            h = int(diff // 3600)
            m = int((diff % 3600) // 60)
            s = int(diff % 60)
            delta_str = f"{h:02d}{m:02d}{s:02d}"
    else:
        delta_str = "------"

    _last_log_time = now
    return f"[{ts} {delta_str}]"


def _base_log(msg: str):
    """
    純粋なログ出力。
    進捗バーとの連携は行わない（synced_log が面倒を見る）。
    """
    global _last_log_line

    # ログファイル名
    script_name = "imagededuper"
    log_name = f"{script_name}_{datetime.now().strftime('%Y%m')}.log"
    logfile = os.path.join(LOG_DIR, log_name)

    # 複数行を1行ずつタイムスタンプ付きに
    lines = []
    for raw in msg.splitlines():
        ts = _format_timestamp_with_delta()
        lines.append(f"{ts} {raw}")
    full_msg = "\n".join(lines)

    with _log_lock:
        # コンソール
        print(full_msg, flush=True)
        # ファイル
        try:
            with open(logfile, "a", encoding="utf-8") as f:
                f.write(full_msg + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            print(f"[log-error] ログ書き込み失敗: {e}", flush=True)

        if lines:
            _last_log_line = lines[-1]


def _write_traceback_log(desc: str, exc: Exception):
    """エラー用トレースバックログ"""
    today = datetime.now().strftime("%Y%m%d")
    tb_file = os.path.join(LOG_DIR, f"error_traceback_{today}.log")
    ts = _format_timestamp_with_delta()
    tb_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    with _log_lock:
        with open(tb_file, "a", encoding="utf-8") as f:
            f.write("\n" + "─" * 50 + "\n")
            f.write(f"{ts} imagededuper - {desc}\n")
            if _last_log_line:
                f.write(_last_log_line + "\n")
            for line in tb_text.strip().splitlines():
                f.write(f"{ts} {line}\n")
            f.write("─" * 50 + "\n")


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def log_processing_stats(label: str):
    elapsed = time.time() - MOVE_START_TIME if MOVE_START_TIME else 0.0
    log(f"[{label}] 経過時間: {format_duration(elapsed)} / 処理済み元画像数: {PROCESSED_BASE_COUNT}/{TOTAL_SOURCE_IMAGES}")


# ============================================
# 進捗バー
# ============================================
def _clear_progress_line():
    """進捗行を完全に消去し、表示位置を先頭に戻す"""
    sys.stdout.write("\r")
    sys.stdout.write("\033[K")  # ← ANSIエスケープで行クリア
    sys.stdout.flush()


def print_loading_progress(done: int, total: int, width: int = PROGRESS_BAR_WIDTH):
    global PROGRESS_MODE, LOAD_DONE, LOAD_TOTAL, LOAD_LAST_PERCENT, CURRENT_ETA_STR
    PROGRESS_MODE = "load"
    LOAD_DONE = done
    LOAD_TOTAL = total

    if total <= 0:
        return
        
    ratio = max(0.0, min(1.0, done / total))
    percent = int(ratio * 100)

    eta_str = ""
    # 10枚ごとに ETA を更新
    if done % 10 == 0:
        CURRENT_ETA_STR = compute_load_eta(done, total)

    # 常時表示（更新がなくてもこの文字列を使う）
    eta_str = f" 終了予定 {CURRENT_ETA_STR}"


    filled = int(width * ratio)
    bar = "#" * filled + "." * (width - filled)

    sys.stdout.write(
        f"\r[読込] [{bar}] {percent:3d}% ({done}/{total}){eta_str}  [q:⛔ 中断]"
    )
    sys.stdout.flush()

def compute_load_eta(done: int, total: int) -> str:
    if done == 0:
        return "計測中"

    elapsed = time.time() - LOAD_START_TIME
    speed = elapsed / done
    remain = total - done
    eta = datetime.now() + timedelta(seconds=remain * speed)
    return eta.strftime("%Y-%m-%d %H:%M:%S")


def print_compare_progress(done: int, total: int, width: int = PROGRESS_BAR_WIDTH):
    """比較フェーズ用進捗バー（ETA付き）"""
    global CURRENT_PROGRESS, TOTAL_PROGRESS, PROGRESS_MODE
    CURRENT_PROGRESS = done
    TOTAL_PROGRESS = total
    PROGRESS_MODE = "compare"

    if total <= 0:
        return

    ratio = max(0.0, min(1.0, done / total))
    filled = int(width * ratio)
    bar = "#" * filled + "." * (width - filled)
    sys.stdout.write(
        f"\r[進捗] [{bar}] {ratio*100:6.2f}% ({done}/{total}) 終了予定: {CURRENT_ETA_STR}  [q:⛔ 中断]"
    )
    sys.stdout.flush()


def redraw_progress():
    """ログ出力後に現在の進捗バーを再描画"""
    if PROGRESS_MODE == "load":
        print_loading_progress(LOAD_DONE, LOAD_TOTAL)
    elif PROGRESS_MODE == "compare":
        print_compare_progress(CURRENT_PROGRESS, TOTAL_PROGRESS)


# ============================================
# 公開ログ関数（進捗と連携）
# ============================================
def log(msg: str):
    """進捗バーと衝突しないログ出力"""

    # 1) まず進捗バーをクリアする
    _clear_progress_line()

    # 2) ログ出力
    _base_log(msg)

    # ※ 改行を入れない（← 空白行の原因を消す）
    # sys.stdout.write("\n") ← 削除

    # 3) 進捗バーを再描画
    redraw_progress()


# ============================================
# safe: 例外安全な関数実行
# ============================================
def _backoff(attempt: int):
    delay = min(30, (2 ** attempt) + random.uniform(0, 1))
    time.sleep(delay)

# ============================================
# KeyboardInterrupt 用の静かな excepthook
# ============================================
def install_silent_keyboardinterrupt_hook():
    """
    KeyboardInterrupt のときだけトレースバックを出さない。
    それ以外の例外は通常通り表示。
    """
    old_hook = sys.excepthook

    def _hook(exc_type, exc, tb):
        if exc_type is KeyboardInterrupt:
            return  # 完全サイレント
        old_hook(exc_type, exc, tb)

    sys.excepthook = _hook

def safe(func, *args, desc="処理", retries=0, **kwargs):
    for attempt in range(retries + 1):
        try:
            return func(*args, **kwargs)

        except KeyboardInterrupt:
            # ← Ctrl+C は絶対に safe() で処理しない
            raise

        except Exception as e:
            is_final = (attempt >= retries)
            if is_final:
                msg = f"⚠️ {desc} 失敗 (試行 {attempt+1}/{retries+1}) [traceback]"
                log(msg)
                _write_traceback_log(desc, e)
                log(f"❌ {desc} 完全失敗")
                return None
            else:
                msg = f"⚠️ {desc} 失敗 (試行 {attempt+1}/{retries+1})"
                log(msg)
                _backoff(attempt)

# ============================================
# CPU worker 数
# ============================================
def get_optimal_workers():
    phys = psutil.cpu_count(logical=False)
    logi = psutil.cpu_count(logical=True)

    # 物理が2なら、ほぼ確実に 2c/4t のCPU
    if phys == 2 and logi == 4:
        return 4  # ← i3-4160 の最適値

    # その他は conservative に
    return max(2, min(logi, int(phys * 1.3)))


# ============================================
# SHA-1
# ============================================
def compute_file_sha1(path: str, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            c = f.read(chunk)
            if not c:
                break
            h.update(c)
    return h.hexdigest()


# ============================================
# rename（衝突 → SHA1 比較）
# ============================================
def safe_rename_with_hash(src: str, dst: str, desc: str) -> bool:
    def _try(a, b):
        os.rename(a, b)
        return True

    result = safe(_try, src, dst, desc=desc, retries=2)
    if result:
        return True

    # rename失敗 & dstが存在しない → よく分からないので諦める
    if not os.path.exists(dst):
        return False

    log(f"[衝突検知] rename失敗 → SHA-1比較へ {os.path.basename(src)}")

    try:
        h_src = compute_file_sha1(src)
        h_dst = compute_file_sha1(dst)
    except Exception as e:
        log(f"[ハッシュ比較失敗] {src} / {dst}: {e}")
        return False

    if h_src == h_dst:
        log(f"[同一判定] {src} と {dst} は内容一致と判定")

        m_src = os.path.getmtime(src)
        m_dst = os.path.getmtime(dst)
        if m_src < m_dst:
            rem = src
            surv = dst
        else:
            rem = dst
            surv = src

        log(f"[削除] 内容一致 → 古い方を削除: {rem}")
        try:
            os.remove(rem)
        except Exception as e:
            log(f"[削除失敗] {rem}: {e}")
            return False

        if surv == src:
            try:
                os.rename(src, dst)
            except Exception as e:
                log(f"[再rename失敗] {src} → {dst}: {e}")
                return False

        log(f"[統一完了] rename成立: {dst}")
        return True

    log(f"[統一スキップ] SHA-1不一致 {src} / {dst}")
    return False


# ============================================
# HEIC → JPG
# ============================================
def convert_heic_to_jpg(path: str, dup_dir: str) -> str | None:
    try:
        with Image.open(path) as img:
            new_path = os.path.splitext(path)[0] + ".jpg"
            img.convert("RGB").save(new_path, "JPEG", quality=95)
        dst = os.path.join(dup_dir, os.path.basename(path))
        shutil.move(path, dst)
        log(f"[🔄 HEIC→JPG] {path} → {new_path}")
        return new_path
    except Exception as e:
        log(f"[❌ HEIC変換失敗] {path}: {e}")
        return None


# ============================================
# JFIF → JPG
# ============================================
def rename_jfif_to_jpg(path: str) -> str:
    try:
        new_path = os.path.splitext(path)[0] + ".jpg"
        os.rename(path, new_path)
        log(f"[🔄 JFIF→JPG] {path} → {new_path}")
        return new_path
    except Exception as e:
        log(f"[❌ JFIF→JPG失敗] {path}: {e}")
        return path


# ============================================
# 拡張子修正
# ============================================
def fix_wrong_extension(path: str) -> str:
    try:
        with Image.open(path) as img:
            fmt = (img.format or "").upper()
        current_ext = os.path.splitext(path)[1].lower()

        if fmt in ("JPEG", "JFIF"):
            new_path = os.path.splitext(path)[0] + ".jpg"
            if current_ext == ".jpg":
                return path
            ok = safe_rename_with_hash(path, new_path, desc="拡張子統一(JPEG)")
            return new_path if ok else path

        ext_map = {
            "PNG": ".png",
            "GIF": ".gif",
            "WEBP": ".webp",
            "TIFF": ".tiff",
            "BMP": ".bmp",
        }
        if fmt in ext_map:
            correct = ext_map[fmt]
            if current_ext == correct:
                return path
            new_path = os.path.splitext(path)[0] + correct
            ok = safe_rename_with_hash(path, new_path, desc=f"[🔧 拡張子修正]({fmt})")
            return new_path if ok else path

        return path

    except Exception as e:
        log(f"[❌ 拡張子判定失敗] {path}: {e}")
        return path


# ============================================
# pHash関連
# ============================================
def dct2(a: np.ndarray) -> np.ndarray:
    return dct_1d(dct_1d(a, axis=0, norm="ortho"), axis=1, norm="ortho")


def calc_phash(img_arr: np.ndarray) -> int:
    """
        グレースケール配列（SSIM用に224x224へ縮小したもの）→ 32x32 → DCT → 上位8x8 → 64bit ハッシュ
    """
    img = Image.fromarray(img_arr).resize((32, 32), Image.LANCZOS)
    mat = np.asarray(img, dtype=np.float32)
    d = dct2(mat)
    d_low = d[:8, :8]
    med = np.median(d_low)
    bits = (d_low > med).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(bool(b))
    return v


def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


PHASH_THRESHOLD = 40  # これより大きいと SSIM を実行しない
DEFAULT_SSIM_THRESHOLD = 0.85  # SSIM 判定のデフォルト値


# ============================================
# 画像キャッシュ（224x224 + pHash + size）
# ============================================
def cache_all_images(paths: list[str]):
    global LOAD_START_TIME, LOAD_LAST_PERCENT
    LOAD_START_TIME = time.time()
    LOAD_LAST_PERCENT = -1

    imgs = []
    sizes = []
    phashes = []
    valid_paths = []
    resolutions = []   # ★ 追加：ここで必ず初期化

    total = len(paths)
    log(f"[📥 読込開始] {total} 枚")
    CURRENT_ETA_STR = "計測中"

    for idx, p in enumerate(paths, start=1):
        # ★ q 中断チェック（最小）
        if quit_requested():
            log("読み込み中に中断操作を検出したわ。")
            return None

        print_loading_progress(idx, total)

        if not os.path.exists(p):
            continue

        try:
            with Image.open(p) as img:
                width, height = img.size        # ★ 元解像度を取得                         
                g = img.convert("L")
                arr = np.array(g.resize((224, 224)))
        except Exception as e:
            log(f"[⚠️ 読込失敗] {p}: {e}")
            continue

        imgs.append(arr)
        sizes.append(os.path.getsize(p))
        resolutions.append((width, height))   # ★ 解像度保存
        phashes.append(calc_phash(arr))
        valid_paths.append(p)

    _clear_progress_line()
    log(f"[✅ 読込完了] 有効画像数: {len(valid_paths)}")
    return valid_paths, imgs, sizes, phashes, resolutions


# ============================================
# ワーカー初期化
# ============================================
W_IMAGES = None
W_SIZES = None
W_PATHS = None
W_PHASHES = None
W_RESOLUTIONS = None  # ★ 追加

def compute_next_pair(i: int, j: int, n: int) -> tuple[int, int]:
    """
    (i,j) が「最後に処理済みのペア」のとき、
    次に処理すべき (i,j) を返す。
    n は画像枚数。
    """
    j = j + 1
    if j <= i:
        j = i + 1

    if j >= n:
        i += 1
        if i >= n - 1:
            # もう処理すべきペアが無い場合の終端的な値
            return n - 1, n
        j = i + 1

    return i, j

# ============================================
# ★ pHash ハミング距離キャッシュ（グローバル）
# ============================================
hamming_cache = {}

def fast_hamming(a, b):
    key = (a << 64) | b
    if key in hamming_cache:
        return hamming_cache[key]
    d = hamming64(a, b)
    hamming_cache[key] = d
    return d

# ============================================
# SSIMタスク
# ============================================
def ssim_task(pair):
    try:
        i, j = pair
        # pHash スキップ
        if fast_hamming(W_PHASHES[i], W_PHASHES[j]) > PHASH_THRESHOLD:
            return None

        img1 = W_IMAGES[i]
        img2 = W_IMAGES[j]
        score = float(ssim(img1, img2, full=False))

        if DEBUG_LOG_SSIM:
            log(f"[DEBUG SSIM] {W_PATHS[i]} vs {W_PATHS[j]} → SSIM={score:.4f}")

        return (
            i, j,
            W_PATHS[i], W_PATHS[j],
            score,
            W_SIZES[i], W_SIZES[j],
            W_RESOLUTIONS[i], W_RESOLUTIONS[j]   # ★追加
        )

    except KeyboardInterrupt:
        return "INTERRUPT"

    except Exception:
        return None

# ============================================
# メイン処理
# ============================================
def move_duplicates(folder_path: str, threshold: float = DEFAULT_SSIM_THRESHOLD):
    global CURRENT_PROGRESS, TOTAL_PROGRESS, CURRENT_ETA_STR
    global PROGRESS_MODE, BASE_START_DONE, BASE_START_TIME, MOVE_START_TIME
    global W_IMAGES, W_SIZES, W_PATHS, W_PHASHES, W_RESOLUTIONS
    global TOTAL_SOURCE_IMAGES, PROCESSED_BASE_COUNT

    resume_path = os.path.join(folder_path, RESUME_FILE_NAME)
    resume = load_resume(resume_path)
    is_resume = resume is not None

    if is_resume:
        start_i = int(resume["i"])
        start_j = int(resume["j"])
        moved = set(resume["moved"])
        CURRENT_PROGRESS = int(resume.get("current_progress", 0))
        PROCESSED_BASE_COUNT = start_i
        log(f"[⏸️→▶️ 再開] i={start_i}, j={start_j} から再開するわ。")
    else:
        start_i = 0
        start_j = 1  # 最初のペアは (0,1)
        moved = set()
        CURRENT_PROGRESS = 0
        PROCESSED_BASE_COUNT = 0

    moved_before = set(moved)   # ★ 今回実行前の moved を保存


    log(f"=== 開始: 重複画像チェック {folder_path} ===")
    MOVE_START_TIME = time.time()
    TOTAL_SOURCE_IMAGES = 0
    PROCESSED_BASE_COUNT = 0

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif",
            ".tiff", ".webp", ".jfif", ".heic", ".heif")

    dup_dir = os.path.join(folder_path, "duplicates")
    os.makedirs(dup_dir, exist_ok=True)

    # 画像収集（duplicates 以下は除外）
    all_files: list[str] = []
    for root, dirs, filenames in os.walk(folder_path):
        dirs[:] = [d for d in dirs if d.lower() != "duplicates"]
        for f in filenames:
            if f.lower().endswith(exts):
                all_files.append(os.path.join(root, f))

    if not all_files:
        log("画像が1枚もないので終了するわ。")
        log_processing_stats("完了")
        return

    # 収集後すぐ
    log(f"[収集] {len(all_files)} 枚")

    # 予定メモリ消費量の表示（画像キャッシュ用）
    estimated_mem_mb = len(all_files) * 0.05  # 約 50KB/枚（224x224 グレースケールキャッシュ）
    log(f"[予定メモリ消費] 約 {estimated_mem_mb:.2f} MB（224x224 グレースケールキャッシュ）")

    # 前処理：HEIC → JPG
    after = []
    for f in all_files:
        if f.lower().endswith((".heic", ".heif")):
            new = safe(convert_heic_to_jpg, f, dup_dir, desc="HEIC変換", retries=2)
            if new:
                after.append(new)
        else:
            after.append(f)

    # 前処理：JFIF → JPG
    tmp = []
    for f in after:
        if f.lower().endswith(".jfif"):
            new = safe(rename_jfif_to_jpg, f, desc="JFIF→JPG", retries=2)
            tmp.append(new if new else f)
        else:
            tmp.append(f)

    # 前処理：拡張子修正
    final = []
    for f in tmp:
        fixed = safe(fix_wrong_extension, f, desc="拡張子修正", retries=2)
        final.append(fixed if fixed else f)

    # キャッシュ
    cached = cache_all_images(final)
    if cached is None:
        log("読み込みが中断されたから、比較処理には進まずに終了するわ。")
        log_processing_stats("中断")
        return

    cached_paths, cached_images, cached_sizes, cached_phashes, cached_resolutions = cached
    n = len(cached_paths)
    TOTAL_SOURCE_IMAGES = n
    PROCESSED_BASE_COUNT = 0

    # ============================================
    # ★ 追加1：pHash順ソートで比較順最適化
    # ============================================
    order = sorted(range(n), key=lambda x: cached_phashes[x])

    cached_paths       = [cached_paths[i] for i in order]
    cached_images      = [cached_images[i] for i in order]
    cached_sizes       = [cached_sizes[i] for i in order]
    cached_phashes     = [cached_phashes[i] for i in order]
    cached_resolutions = [cached_resolutions[i] for i in order]

    # 再ソート後の枚数 n は変わらない
    if n < 2:
        log("比較対象が1枚しかないから処理することがないわ。")
        PROCESSED_BASE_COUNT = n
        log_processing_stats("完了")
        return

    total_pairs = n * (n - 1) // 2
    workers = get_optimal_workers()
    log(f"[比較設定] 画像数={n}, 組み合わせ={total_pairs}, workers={workers}")
    log("[🔍 比較] pHash で候補を絞り、その中だけ SSIM で最終判定するわ。")

    # 進捗初期化
    if not is_resume:
        CURRENT_PROGRESS = 0  # 初回のみ 0 にする    
    TOTAL_PROGRESS = total_pairs
    CURRENT_ETA_STR = "計測中"
    PROGRESS_MODE = "compare"
    BASE_START_DONE = 0
    BASE_START_TIME = time.time()
    print_compare_progress(0, total_pairs)

    # 比較元が変わったときに呼ばれる
    def on_new_base(i: int):
        global BASE_START_DONE, BASE_START_TIME, CURRENT_ETA_STR, PROCESSED_BASE_COUNT
        BASE_START_DONE = CURRENT_PROGRESS
        BASE_START_TIME = time.time()
        PROCESSED_BASE_COUNT = i

        # ETA をここで一度だけ再計算
        total_pairs = TOTAL_PROGRESS
        done_pairs = CURRENT_PROGRESS
        remaining_pairs = total_pairs - done_pairs

        # 進捗が少なすぎる間は無理に予測しない
        if done_pairs <= 0 or done_pairs < total_pairs * 0.01:
            CURRENT_ETA_STR = "計測中"
        else:
            elapsed = time.time() - MOVE_START_TIME  # 全体開始からの経過秒数
            avg_speed = done_pairs / elapsed         # 平均ペア/秒

            # 後半ほど比較対象が減って速くなるのを、残り割合で補正
            # rem_ratio が 1.0 → 0.0 に近づくにつれ accel も小さくなる
            rem_ratio = remaining_pairs / total_pairs
            accel = rem_ratio ** 0.5  # 0.0〜1.0（後半になるほど小さい）

            est_sec = (remaining_pairs / avg_speed) * accel

            eta_dt = datetime.now() + timedelta(seconds=max(0.0, est_sec))
            CURRENT_ETA_STR = eta_dt.strftime("%Y-%m-%d %H:%M:%S")

        save_resume(resume_path, i, 0, moved, CURRENT_PROGRESS)
        log(f"[比較] 基準画像 {i+1}/{n}: {os.path.basename(cached_paths[i])}")

    # 比較ペア生成
    def pair_gen():
        current_i = None

        # 開始位置は resume の時だけ反映。それ以外は 0,1 から確実に開始。
        if is_resume:
            i = start_i
            j = start_j
        else:
            i = 0
            j = 1

        while i < n:
            # 基準画像が変わったらログなど更新
            if i != current_i:
                current_i = i
                on_new_base(i)

            # j が末尾まで行ったら次の i へ
            if j >= n:
                i += 1
                if i >= n:
                    break
                j = i + 1
                continue

            # i < j の全ペアを順番に返す
            yield (i, j)
            j += 1

    pairs = pair_gen()
    max_pending = workers * 2

    # 「最後に処理したペア」を覚えておく（Ctrl+C 用）
    last_i = start_i
    last_j = start_i  # compute_next_pair() が (start_i, start_i+1) を返すようにしている

    # ThreadPool は initializer が不要なので、グローバルに渡す
    W_IMAGES       = cached_images
    W_SIZES        = cached_sizes
    W_PATHS        = cached_paths
    W_PHASHES      = cached_phashes
    W_RESOLUTIONS  = cached_resolutions

    exe = ThreadPoolExecutor(
        max_workers=workers
    )

    try:
        pending = set()

        # 初期投入
        try:
            while len(pending) < max_pending:
                pair = next(pairs)
                fut = exe.submit(ssim_task, pair)
                pending.add((pair, fut))
        except StopIteration:
            pass

        while pending:
            # q 中断
            if quit_requested():
                raise KeyboardInterrupt

            done_futs = []
            for pair, fut in list(pending):
                if fut.done():
                    done_futs.append((pair, fut))
                    pending.remove((pair, fut))

            if not done_futs:
                if quit_requested():
                    raise KeyboardInterrupt
                time.sleep(0.01)
                continue

            for pair, fut in done_futs:
                i, j = pair
                last_i, last_j = i, j

                res = fut.result()

                if res == "INTERRUPT":
                    raise KeyboardInterrupt

                if not res:
                    CURRENT_PROGRESS += 1
                    print_compare_progress(CURRENT_PROGRESS, TOTAL_PROGRESS)
                    continue

                # ★ 9 要素すべてを unpack（解像度も受け取る）
                i, j, a, b, score, sa, sb, ra, rb = res
                CURRENT_PROGRESS += 1
                print_compare_progress(CURRENT_PROGRESS, TOTAL_PROGRESS)

                if CURRENT_PROGRESS % 10 == 0:
                    ni, nj = compute_next_pair(i, j, n)
                    save_resume(resume_path, ni, nj, moved, CURRENT_PROGRESS)

                if a in moved or b in moved:
                    continue

                if score >= threshold:
                    # ★ ra / rb には (width, height) が入っている
                    (width_a, height_a) = ra
                    (width_b, height_b) = rb

                    res_a = width_a * height_a
                    res_b = width_b * height_b

                    # 解像度が小さい方 → 削除側（= duplicates へ移動）
                    smaller = a if res_a < res_b else b

                    dst = os.path.join(dup_dir, os.path.basename(smaller))
                    moved.add(smaller)
                    log(f"[🧩 重複検出] SSIM={score:.4f} → {smaller} を移動")
                    safe(
                        shutil.move,
                        smaller,
                        dst,
                        desc="重複移動",
                        retries=2,
                    )

            # 補充投入
            try:
                while len(pending) < max_pending:
                    pair = next(pairs)
                    fut = exe.submit(ssim_task, pair)
                    pending.add((pair, fut))
            except StopIteration:
                pass

    except (Exception, KeyboardInterrupt) as e:
        exe.shutdown(wait=False, cancel_futures=True)

        ni, nj = compute_next_pair(last_i, last_j, n)
        save_resume(resume_path, ni, nj, moved, CURRENT_PROGRESS)
        log_processing_stats("中断")
        log("中断操作を検知したから、中断位置を保存して終了するわ。")
        return
    finally:
        exe.shutdown(wait=False, cancel_futures=True)
    # --- Executor 版ここまで ---

    if os.path.exists(resume_path):
        os.remove(resume_path)
        log("[🗑 再開データ削除] 正常終了したから resume.json を削除したわ。")

    PROCESSED_BASE_COUNT = TOTAL_SOURCE_IMAGES
    _clear_progress_line()
    log_processing_stats("完了")
    log("=== 🎉 完了 ===")

    new_moved_count = len(moved) - len(moved_before)  # ★ 今回分だけ
    log(f"今回移動した重複画像枚数: {new_moved_count}")


# ============================================
# エントリポイント
# ============================================
if __name__ == "__main__":

    # メインプロセスでも KeyboardInterrupt のトレースバックを封印
    install_silent_keyboardinterrupt_hook()

    folder = input("対象フォルダを入力してね: ").strip().strip('"')
    if not os.path.isdir(folder):
        log(f"[エラー] フォルダが存在しないわ: {folder}")
        sys.exit(1)

    resume_path = os.path.join(folder, RESUME_FILE_NAME)

    # --- ここから追加 ---
    if os.path.exists(resume_path):
        print("\n=== 中断データが見つかったわ ===")
        print("  [1] 続きから再開")
        print("  [2] 最初からやり直す")
        print("  [3] キャンセルして終了")
        choice = input("番号を選んでね: ").strip()

        if choice == "1":
            log("[選択] 続きから再開するわね。")

        elif choice == "2":
            log("[選択] 最初から再処理するわ。")
            try:
                os.remove(resume_path)
                log("[再開データ削除] 古い再開情報を削除したわ。")
            except:
                log("[エラー] 再開データを削除できなかったわ。")
                sys.exit(1)

        elif choice == "3":
            log("[選択] キャンセルするわ。")
            sys.exit(0)

        else:
            log("[エラー] 無効な番号よ。処理を停止するわ。")
            sys.exit(1)
    # --- 追加ここまで ---

    threshold_value = DEFAULT_SSIM_THRESHOLD
    log(f"[設定] SSIM 閾値: {threshold_value:.2f} (変更したい場合は DEFAULT_SSIM_THRESHOLD を編集してね)")
    safe(
        move_duplicates,
        folder,
        threshold=threshold_value,
        desc="重複削除処理",
        retries=2,
    )
