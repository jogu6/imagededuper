"""重複検出ロジック全体（ログ・進捗・比較・前処理）をまとめたモジュール。"""

import hashlib
import json
import os
import random
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import msvcrt
import numpy as np
import pillow_heif
import psutil
from PIL import Image
from scipy.fftpack import dct as dct_1d
from skimage.metrics import structural_similarity as ssim

from config import (
    DEBUG_LOG_SSIM,
    DEFAULT_SSIM_THRESHOLD,
    LOG_DIR,
    PHASH_THRESHOLD,
    PROGRESS_BAR_WIDTH,
    RESUME_FILE_NAME,
)

pillow_heif.register_heif_opener()

_log_lock = threading.Lock()
_last_log_time = None
_last_log_line = None

CURRENT_PROGRESS = 0
TOTAL_PROGRESS = 1
CURRENT_ETA_STR = "計測中"
LOAD_START_TIME = 0.0
LOAD_LAST_PERCENT = -1
PROGRESS_MODE = "none"
LOAD_DONE = 0
LOAD_TOTAL = 1
BASE_START_DONE = 0
BASE_START_TIME = 0.0
MOVE_START_TIME = 0.0
TOTAL_SOURCE_IMAGES = 0
PROCESSED_BASE_COUNT = 0

W_IMAGES = None
W_SIZES = None
W_PATHS = None
W_PHASHES = None
W_RESOLUTIONS = None
W_HASHES = None


def _format_timestamp_with_delta():
    """ログラベル用のタイムスタンプ（直近出力との差分付き）を生成する。"""
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
    """進捗バーを気にせずに素のログを出力し、ファイルにも書き込む。"""
    global _last_log_line
    log_name = f"imagededuper_{datetime.now().strftime('%Y%m')}.log"
    logfile = os.path.join(LOG_DIR, log_name)

    lines = []
    for raw in msg.splitlines():
        ts = _format_timestamp_with_delta()
        lines.append(f"{ts} {raw}")
    full_msg = "\n".join(lines)

    with _log_lock:
        print(full_msg, flush=True)
        try:
            with open(logfile, "a", encoding="utf-8") as f:
                f.write(full_msg + "\n")
        except Exception as exc:
            print(f"[log-error] ログ書き込み失敗: {exc}", flush=True)
        if lines:
            _last_log_line = lines[-1]


def _write_traceback_log(desc: str, exc: Exception):
    """例外発生時の詳細トレースバックを専用ファイルへ追記する。"""
    today = datetime.now().strftime("%Y%m%d")
    tb_file = os.path.join(LOG_DIR, f"error_traceback_{today}.log")
    ts = _format_timestamp_with_delta()
    tb_text = "".join(
        traceback for traceback in __import__("traceback").format_exception(type(exc), exc, exc.__traceback__)
    )

    with _log_lock:
        with open(tb_file, "a", encoding="utf-8") as f:
            f.write("\n" + "─" * 50 + "\n")
            f.write(f"{ts} imagededuper - {desc}\n")
            if _last_log_line:
                f.write(_last_log_line + "\n")
            for line in tb_text.strip().splitlines():
                f.write(f"{ts} {line}\n")
            f.write("─" * 50 + "\n")


def log(msg: str):
    """進捗バーを一時的に隠してからログ出力し、終わったら進捗を再描画する。"""
    _clear_progress_line()
    _base_log(msg)
    redraw_progress()


def log_processing_stats(label: str):
    """経過時間と処理済み基準画像数をまとめてログに出す。"""
    elapsed = time.time() - MOVE_START_TIME if MOVE_START_TIME else 0.0
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)
    log(
        f"[{label}] 経過時間: {h:02d}:{m:02d}:{s:02d} / "
        f"処理済み元画像数: {PROCESSED_BASE_COUNT}/{TOTAL_SOURCE_IMAGES}"
    )


def save_resume(resume_path: str, i: int, j: int, moved: set[str], progress: int):
    """再開に必要な情報を JSON 形式で保存する。"""
    data = {
        "i": i,
        "j": j,
        "moved": list(moved),
        "current_progress": progress,
    }
    with open(resume_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_resume(resume_path: str):
    """保存済みの再開データを読み取って返す。"""
    if not os.path.exists(resume_path):
        return None
    with open(resume_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _clear_progress_line():
    """進捗バー用の行を消去し、カーソルを先頭へ戻す。"""
    sys.stdout.write("\r")
    sys.stdout.write("\033[K")
    sys.stdout.flush()


def print_loading_progress(done: int, total: int):
    """読み込みフェーズの進捗バーを描画する。"""
    global PROGRESS_MODE, LOAD_DONE, LOAD_TOTAL, CURRENT_ETA_STR, LOAD_LAST_PERCENT
    PROGRESS_MODE = "load"
    LOAD_DONE = done
    LOAD_TOTAL = total

    if total <= 0:
        return

    ratio = max(0.0, min(1.0, done / total))
    percent = int(ratio * 100)

    if done % 10 == 0:
        CURRENT_ETA_STR = compute_load_eta(done, total)

    filled = int(PROGRESS_BAR_WIDTH * ratio)
    bar = "#" * filled + "." * (PROGRESS_BAR_WIDTH - filled)

    sys.stdout.write(
        f"\r[読込] [{bar}] {percent:3d}% ({done}/{total}) 終了予定 {CURRENT_ETA_STR}  [q:⛔ 中断]"
    )
    sys.stdout.flush()
    LOAD_LAST_PERCENT = percent


def compute_load_eta(done: int, total: int) -> str:
    """現在の読み込み枚数から終了予定時刻を推定する。"""
    if done == 0:
        return "計測中"

    elapsed = time.time() - LOAD_START_TIME
    speed = elapsed / done
    remain = total - done
    eta = datetime.now() + timedelta(seconds=remain * speed)
    return eta.strftime("%Y-%m-%d %H:%M:%S")


def print_compare_progress(done: int, total: int):
    """比較フェーズの進捗バーを描画する。"""
    global CURRENT_PROGRESS, TOTAL_PROGRESS, PROGRESS_MODE
    CURRENT_PROGRESS = done
    TOTAL_PROGRESS = total
    PROGRESS_MODE = "compare"

    if total <= 0:
        return

    ratio = max(0.0, min(1.0, done / total))
    filled = int(PROGRESS_BAR_WIDTH * ratio)
    bar = "#" * filled + "." * (PROGRESS_BAR_WIDTH - filled)
    sys.stdout.write(
        f"\r[進捗] [{bar}] {ratio*100:6.2f}% ({done}/{total}) 終了予定: {CURRENT_ETA_STR}  [q:⛔ 中断]"
    )
    sys.stdout.flush()


def redraw_progress():
    """直前のログ出力で消えた進捗表示を復元する。"""
    if PROGRESS_MODE == "load":
        print_loading_progress(LOAD_DONE, LOAD_TOTAL)
    elif PROGRESS_MODE == "compare":
        print_compare_progress(CURRENT_PROGRESS, TOTAL_PROGRESS)


def update_move_start_time():
    """比較処理の開始時刻を記録し直す。"""
    global MOVE_START_TIME
    MOVE_START_TIME = time.time()


def set_total_source_images(count: int):
    """総画像枚数カウンタを更新する。"""
    global TOTAL_SOURCE_IMAGES
    TOTAL_SOURCE_IMAGES = count


def set_processed_base_count(count: int):
    """現在までに扱った基準画像数を記録する。"""
    global PROCESSED_BASE_COUNT
    PROCESSED_BASE_COUNT = count


def increment_progress(delta: int = 1):
    """比較済みペア数を加算する。"""
    global CURRENT_PROGRESS
    CURRENT_PROGRESS += delta


def set_progress(value: int):
    """比較済みペア数を直接設定する。"""
    global CURRENT_PROGRESS
    CURRENT_PROGRESS = value


def _backoff(attempt: int):
    """指数バックオフでリトライ前の待機時間を調整する。"""
    delay = min(30, (2**attempt) + random.uniform(0, 1))
    time.sleep(delay)


def install_silent_keyboardinterrupt_hook():
    """KeyboardInterrupt のときだけトレースバックを抑制する excepthook を仕込む。"""
    old_hook = sys.excepthook

    def _hook(exc_type, exc, tb):
        if exc_type is KeyboardInterrupt:
            return
        old_hook(exc_type, exc, tb)

    sys.excepthook = _hook


def safe(func, *args, desc="処理", retries=0, **kwargs):
    """例外に強いリトライ付きラッパーで任意の関数を実行する。"""
    for attempt in range(retries + 1):
        try:
            return func(*args, **kwargs)

        except KeyboardInterrupt:
            raise

        except Exception as exc:
            is_final = attempt >= retries
            msg = f"⚠ {desc} 失敗 (試行 {attempt+1}/{retries+1})"
            log(msg if not is_final else f"{msg} [traceback]")
            if is_final:
                _write_traceback_log(desc, exc)
                log(f"❌ {desc} 完全失敗")
                return None
            _backoff(attempt)


def quit_requested():
    """コンソールで q/Q が押されていないかを調べる。"""
    return msvcrt.kbhit() and msvcrt.getch() in (b"q", b"Q")


def compute_file_sha1(path: str, chunk: int = 1024 * 1024) -> str:
    """ファイルの SHA-1 ハッシュ値を計算する。"""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            c = f.read(chunk)
            if not c:
                break
            h.update(c)
    return h.hexdigest()


def safe_rename_with_hash(src: str, dst: str, desc: str) -> bool:
    """rename に失敗した場合でも SHA-1 照合で安全に統一する。"""
    def _try(a, b):
        os.rename(a, b)
        return True

    result = safe(_try, src, dst, desc=desc, retries=2)
    if result:
        return True

    if not os.path.exists(dst):
        return False

    log(f"[衝突検知] rename失敗 → SHA-1比較へ {os.path.basename(src)}")

    try:
        h_src = compute_file_sha1(src)
        h_dst = compute_file_sha1(dst)
    except Exception as exc:
        log(f"[ハッシュ比較失敗] {src} / {dst}: {exc}")
        return False

    if h_src == h_dst:
        log(f"[同一判定] {src} と {dst} は内容一致と判定")

        m_src = os.path.getmtime(src)
        m_dst = os.path.getmtime(dst)
        rem, surv = (src, dst) if m_src < m_dst else (dst, src)

        log(f"[削除] 内容一致 → 古い方を削除: {rem}")
        try:
            os.remove(rem)
        except Exception as exc:
            log(f"[削除失敗] {rem}: {exc}")
            return False

        if surv == src:
            try:
                os.rename(src, dst)
            except Exception as exc:
                log(f"[再rename失敗] {src} → {dst}: {exc}")
                return False

        log(f"[統一完了] rename成功 {dst}")
        return True

    log(f"[統一スキップ] SHA-1不一致 {src} / {dst}")
    return False


def convert_heic_to_jpg(path: str, dup_dir: str) -> str | None:
    """HEIC/HEIF を JPEG に変換し、元ファイルを duplicates へ退避する。"""
    try:
        with Image.open(path) as img:
            new_path = os.path.splitext(path)[0] + ".jpg"
            img.convert("RGB").save(new_path, "JPEG", quality=95)
        dst = os.path.join(dup_dir, os.path.basename(path))
        shutil.move(path, dst)
        log(f"[🔄 HEIC→JPG] {path} → {new_path}")
        return new_path
    except Exception as exc:
        log(f"[❌ HEIC変換失敗] {path}: {exc}")
        return None


def rename_jfif_to_jpg(path: str) -> str:
    """拡張子 .jfif を .jpg に変更する。"""
    try:
        new_path = os.path.splitext(path)[0] + ".jpg"
        os.rename(path, new_path)
        log(f"[🔄 JFIF→JPG] {path} → {new_path}")
        return new_path
    except Exception as exc:
        log(f"[❌ JFIF→JPG失敗] {path}: {exc}")
        return path


def fix_wrong_extension(path: str) -> str:
    """実際の画像フォーマットに合わせて拡張子を補正する。"""
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

        ext_map = {"PNG": ".png", "GIF": ".gif", "WEBP": ".webp", "TIFF": ".tiff", "BMP": ".bmp"}
        if fmt in ext_map:
            correct = ext_map[fmt]
            if current_ext == correct:
                return path
            new_path = os.path.splitext(path)[0] + correct
            ok = safe_rename_with_hash(path, new_path, desc=f"[🔧 拡張子修正]({fmt})")
            return new_path if ok else path

        return path

    except Exception as exc:
        log(f"[❌ 拡張子判定失敗] {path}: {exc}")
        return path


def dct2(a: np.ndarray) -> np.ndarray:
    """2 次元 DCT を計算するヘルパー。"""
    return dct_1d(dct_1d(a, axis=0, norm="ortho"), axis=1, norm="ortho")


def calc_phash(img_arr: np.ndarray) -> int:
    """224x224 グレースケール画像から 64bit の pHash を計算する。"""
    img = Image.fromarray(img_arr).resize((32, 32), Image.LANCZOS)
    mat = np.asarray(img, dtype=np.float32)
    d = dct2(mat)
    d_low = d[:8, :8]
    med = np.median(d_low)
    bits = (d_low > med).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return value


def hamming64(a: int, b: int) -> int:
    """64bit 整数同士のハミング距離を求める。"""
    return (a ^ b).bit_count()


def cache_all_images(paths: list[str]):
    """対象ファイルを読み込み、SSIM 用のキャッシュ一式を作って返す。"""
    global LOAD_START_TIME, LOAD_LAST_PERCENT, CURRENT_ETA_STR
    LOAD_START_TIME = time.time()
    LOAD_LAST_PERCENT = -1
    imgs = []
    sizes = []
    phashes = []
    valid_paths = []
    resolutions = []
    hashes = []

    total = len(paths)
    log(f"[📥 読込開始] {total} 枚")
    CURRENT_ETA_STR = "計測中"

    for idx, path in enumerate(paths, start=1):
        if quit_requested():
            log("読み込み中に中断操作を検知したわ。")
            return None

        print_loading_progress(idx, total)

        if not os.path.exists(path):
            continue

        try:
            with Image.open(path) as img:
                width, height = img.size
                g = img.convert("L")
                arr = np.array(g.resize((224, 224)))
        except Exception as exc:
            log(f"[⚠ 読込失敗] {path}: {exc}")
            continue

        imgs.append(arr)
        sizes.append(os.path.getsize(path))
        resolutions.append((width, height))
        phashes.append(calc_phash(arr))
        valid_paths.append(path)
        try:
            hashes.append(compute_file_sha1(path))
        except Exception as exc:
            log(f"[⚠ SHA-1計算失敗] {path}: {exc}")
            hashes.append("")

    _clear_progress_line()
    log(f"[✅ 読込完了] 有効画像数: {len(valid_paths)}")
    return valid_paths, imgs, sizes, phashes, resolutions, hashes


def get_optimal_workers():
    """CPU コア数からスレッドプールの適切な worker 数を推定する。"""
    phys = psutil.cpu_count(logical=False)
    logi = psutil.cpu_count(logical=True)

    if phys == 2 and logi == 4:
        return 4
    return max(2, min(logi, int(phys * 1.3)))


hamming_cache = {}


def fast_hamming(a, b):
    """pHash のハミング距離をキャッシュ付きで高速に求める。"""
    key = (a << 64) | b
    if key in hamming_cache:
        return hamming_cache[key]
    distance = hamming64(a, b)
    hamming_cache[key] = distance
    return distance


def set_worker_data(images, sizes, paths, phashes, resolutions, hashes):
    """比較ワーカーが参照するグローバルデータを設定する。"""
    global W_IMAGES, W_SIZES, W_PATHS, W_PHASHES, W_RESOLUTIONS, W_HASHES
    W_IMAGES = images
    W_SIZES = sizes
    W_PATHS = paths
    W_PHASHES = phashes
    W_RESOLUTIONS = resolutions
    W_HASHES = hashes


def clear_worker_data():
    """比較ワーカー用のグローバルデータをクリアする。"""
    set_worker_data(None, None, None, None, None, None)


def ssim_task(pair):
    """1 ペアの SHA-1 / pHash / SSIM 判定を行い、結果を返す。"""
    try:
        i, j = pair
        sha_match = False

        if W_HASHES[i] and W_HASHES[i] == W_HASHES[j]:
            sha_match = True
        else:
            if fast_hamming(W_PHASHES[i], W_PHASHES[j]) > PHASH_THRESHOLD:
                return None

        if sha_match:
            score = 1.0
        else:
            img1 = W_IMAGES[i]
            img2 = W_IMAGES[j]
            score = float(ssim(img1, img2, full=False))

            if DEBUG_LOG_SSIM:
                log(f"[DEBUG SSIM] {W_PATHS[i]} vs {W_PATHS[j]} -> SSIM={score:.4f}")

        return (
            i,
            j,
            W_PATHS[i],
            W_PATHS[j],
            score,
            W_SIZES[i],
            W_SIZES[j],
            W_RESOLUTIONS[i],
            W_RESOLUTIONS[j],
            sha_match,
        )

    except KeyboardInterrupt:
        return "INTERRUPT"

    except Exception:
        return None


def compute_next_pair(i: int, j: int, n: int) -> tuple[int, int]:
    """現在の (i, j) に続く比較ペアを計算する。"""
    j = j + 1
    if j <= i:
        j = i + 1

    if j >= n:
        i += 1
        if i >= n - 1:
            return n - 1, n
        j = i + 1

    return i, j


def move_duplicates(folder_path: str, threshold: float = DEFAULT_SSIM_THRESHOLD):
    """指定フォルダ内の重複画像を検出し、duplicates へ移動するメイン処理。"""
    resume_path = os.path.join(folder_path, RESUME_FILE_NAME)
    resume = load_resume(resume_path)
    is_resume = resume is not None

    if is_resume:
        start_i = int(resume["i"])
        start_j = int(resume["j"])
        moved = set(resume["moved"])
        current_progress = int(resume.get("current_progress", 0))
        set_progress(current_progress)
        set_processed_base_count(start_i)
        log(f"[⏸️→▶️ 再開] i={start_i}, j={start_j} から再開するわ。")
    else:
        start_i = 0
        start_j = 1
        moved = set()
        current_progress = 0
        set_progress(0)
        set_processed_base_count(0)

    moved_before = set(moved)

    log(f"=== 開始: 重複画像チェック {folder_path} ===")
    update_move_start_time()
    set_total_source_images(0)
    set_processed_base_count(0)

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp", ".jfif", ".heic", ".heif")

    dup_dir = os.path.join(folder_path, "duplicates")
    os.makedirs(dup_dir, exist_ok=True)

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

    log(f"[収集] {len(all_files)} 枚")

    estimated_mem_mb = len(all_files) * 0.05
    log(f"[予定メモリ消費] 約 {estimated_mem_mb:.2f} MB（224x224 グレースケールキャッシュ）")

    after = []
    for f in all_files:
        if f.lower().endswith((".heic", ".heif")):
            new = safe(convert_heic_to_jpg, f, dup_dir, desc="HEIC変換", retries=2)
            if new:
                after.append(new)
            continue
        after.append(f)

    tmp = []
    for f in after:
        if f.lower().endswith(".jfif"):
            new = safe(rename_jfif_to_jpg, f, desc="JFIF→JPG", retries=2)
            tmp.append(new if new else f)
            continue
        tmp.append(f)

    final = []
    for f in tmp:
        fixed = safe(fix_wrong_extension, f, desc="拡張子修正", retries=2)
        final.append(fixed if fixed else f)

    cached = cache_all_images(final)
    if cached is None:
        log("読み込みが中断されたから、比較処理には進まずに終了するわ。")
        log_processing_stats("中断")
        return

    cached_paths, cached_images, cached_sizes, cached_phashes, cached_resolutions, cached_hashes = cached
    n = len(cached_paths)
    set_total_source_images(n)
    set_processed_base_count(0)

    order = sorted(range(n), key=lambda x: cached_phashes[x])
    cached_paths = [cached_paths[i] for i in order]
    cached_images = [cached_images[i] for i in order]
    cached_sizes = [cached_sizes[i] for i in order]
    cached_phashes = [cached_phashes[i] for i in order]
    cached_resolutions = [cached_resolutions[i] for i in order]
    cached_hashes = [cached_hashes[i] for i in order]

    if n < 2:
        log("比較対象が1枚しかないから処理することがないわ。")
        set_processed_base_count(n)
        log_processing_stats("完了")
        return

    total_pairs = n * (n - 1) // 2
    workers = get_optimal_workers()
    log(f"[比較設定] 画像数={n}, 組み合わせ={total_pairs}, workers={workers}")
    log("[🔍 比較] pHash で候補を絞り、その中だけ SSIM で最終判定するわ。")

    if not is_resume:
        current_progress = 0
        set_progress(0)
    global TOTAL_PROGRESS
    TOTAL_PROGRESS = total_pairs
    print_compare_progress(current_progress, total_pairs)

    def on_new_base(i: int):
        global CURRENT_ETA_STR
        set_processed_base_count(i)
        if current_progress <= 0 or current_progress < total_pairs * 0.01:
            CURRENT_ETA_STR = "計測中"
        else:
            elapsed = time.time() - MOVE_START_TIME
            avg_speed = current_progress / max(elapsed, 1e-6)
            remaining_pairs = total_pairs - current_progress
            rem_ratio = remaining_pairs / total_pairs
            accel = rem_ratio ** 0.5
            est_sec = (remaining_pairs / max(avg_speed, 1e-6)) * accel
            eta_dt = datetime.now() + timedelta(seconds=max(0.0, est_sec))
            CURRENT_ETA_STR = eta_dt.strftime("%Y-%m-%d %H:%M:%S")

        save_resume(resume_path, i, 0, moved, current_progress)
        log(f"[比較] 基準画像 {i+1}/{n}: {os.path.basename(cached_paths[i])}")

    def pair_gen():
        current_i = None
        i = start_i if is_resume else 0
        j = start_j if is_resume else 1

        while i < n:
            if i != current_i:
                current_i = i
                on_new_base(i)

            if j >= n:
                i += 1
                if i >= n:
                    break
                j = i + 1
                continue

            yield (i, j)
            j += 1

    pairs = pair_gen()
    max_pending = workers * 2
    last_i = start_i
    last_j = start_i

    set_worker_data(cached_images, cached_sizes, cached_paths, cached_phashes, cached_resolutions, cached_hashes)
    exe = ThreadPoolExecutor(max_workers=workers)

    try:
        pending = set()
        try:
            while len(pending) < max_pending:
                pair = next(pairs)
                fut = exe.submit(ssim_task, pair)
                pending.add((pair, fut))
        except StopIteration:
            pass

        while pending:
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
                    current_progress += 1
                    set_progress(current_progress)
                    print_compare_progress(current_progress, total_pairs)
                    continue

                i, j, a, b, score, sa, sb, ra, rb, sha_match = res
                current_progress += 1
                set_progress(current_progress)
                print_compare_progress(current_progress, total_pairs)

                if current_progress % 10 == 0:
                    ni, nj = compute_next_pair(i, j, n)
                    save_resume(resume_path, ni, nj, moved, current_progress)

                if a in moved or b in moved:
                    continue

                if sha_match or score >= threshold:
                    (width_a, height_a) = ra
                    (width_b, height_b) = rb

                    res_a = width_a * height_a
                    res_b = width_b * height_b

                    smaller = a if res_a < res_b else b

                    dst = os.path.join(dup_dir, os.path.basename(smaller))
                    moved.add(smaller)
                    if sha_match:
                        log(f"[♻ SHA-1一致] {smaller} を移動するわ。")
                    else:
                        log(f"[🧩 重複検出] SSIM={score:.4f} → {smaller} を移動")
                    safe(shutil.move, smaller, dst, desc="重複移動", retries=2)

            try:
                while len(pending) < max_pending:
                    pair = next(pairs)
                    fut = exe.submit(ssim_task, pair)
                    pending.add((pair, fut))
            except StopIteration:
                pass

    except (Exception, KeyboardInterrupt):
        exe.shutdown(wait=False, cancel_futures=True)

        ni, nj = compute_next_pair(last_i, last_j, n)
        save_resume(resume_path, ni, nj, moved, current_progress)
        log_processing_stats("中断")
        log("中断操作を検知したから、中断位置を保存して終了するわ。")
        return

    finally:
        exe.shutdown(wait=False, cancel_futures=True)
        clear_worker_data()

    if os.path.exists(resume_path):
        os.remove(resume_path)
        log("[🗑 再開データ削除] 正常終了したから resume.json を削除したわ。")

    set_processed_base_count(n)
    log_processing_stats("完了")
    log("=== 🎉 完了 ===")

    new_moved_count = len(moved) - len(moved_before)
    log(f"今回移動した重複画像枚数: {new_moved_count}")
