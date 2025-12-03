"""Duplicate detection module that bundles logging, progress, comparison, and preprocessing."""

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
from i18n import t

pillow_heif.register_heif_opener()

_log_lock = threading.Lock()
_last_log_time = None
_last_log_line = None

CURRENT_PROGRESS = 0
TOTAL_PROGRESS = 1
CURRENT_ETA_STR = t("status.estimating")
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
    """Return a timestamp string for log labels, including the delta since the last log."""
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
    """Write a message to stdout and the rotating log file without breaking progress bars."""
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
            print(t("log.error_write", exc=exc), flush=True)
        if lines:
            _last_log_line = lines[-1]


def _write_traceback_log(desc: str, exc: Exception):
    """Append a detailed traceback entry to a dedicated log file."""
    today = datetime.now().strftime("%Y%m%d")
    tb_file = os.path.join(LOG_DIR, f"error_traceback_{today}.log")
    ts = _format_timestamp_with_delta()
    tb_text = "".join(
        traceback for traceback in __import__("traceback").format_exception(type(exc), exc, exc.__traceback__)
    )

    with _log_lock:
        with open(tb_file, "a", encoding="utf-8") as f:
            f.write("\n" + "-" * 50 + "\n")
            f.write(f"{ts} {t('log.traceback_header', desc=desc)}\n")
            if _last_log_line:
                f.write(_last_log_line + "\n")
            for line in tb_text.strip().splitlines():
                f.write(f"{ts} {line}\n")
            f.write("-" * 50 + "\n")


def log(msg: str):
    """Hide the live progress line, log the provided message, then redraw the progress display."""
    _clear_progress_line()
    _base_log(msg)
    redraw_progress()


def log_processing_stats(label_key: str):
    """Log the total elapsed move time and how many base images finished under the given label."""
    elapsed = time.time() - MOVE_START_TIME if MOVE_START_TIME else 0.0
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)
    log(
        t(
            "log.processing_stats",
            label=t(label_key),
            h=h,
            m=m,
            s=s,
            processed=PROCESSED_BASE_COUNT,
            total=TOTAL_SOURCE_IMAGES,
        )
    )


def save_resume(resume_path: str, i: int, j: int, moved: set[str], progress: int):
    """Persist everything needed to resume a run."""
    data = {
        "i": i,
        "j": j,
        "moved": list(moved),
        "current_progress": progress,
    }
    with open(resume_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_resume(resume_path: str):
    """Load resume data if it exists, otherwise return None."""
    if not os.path.exists(resume_path):
        return None
    with open(resume_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _clear_progress_line():
    """Erase the progress bar line and reset the cursor."""
    sys.stdout.write("\r")
    sys.stdout.write("\033[K")
    sys.stdout.flush()


def print_loading_progress(done: int, total: int):
    """Update the progress bar while images are being loaded."""
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
        t(
            "progress.load_line",
            bar=bar,
            percent=percent,
            done=done,
            total=total,
            eta=CURRENT_ETA_STR,
        )
    )
    sys.stdout.flush()
    LOAD_LAST_PERCENT = percent


def compute_load_eta(done: int, total: int) -> str:
    """Estimate the finish time based on the number of files loaded so far."""
    if done == 0:
        return t("status.estimating")

    elapsed = time.time() - LOAD_START_TIME
    speed = elapsed / done
    remain = total - done
    eta = datetime.now() + timedelta(seconds=remain * speed)
    return eta.strftime("%Y-%m-%d %H:%M:%S")


def print_compare_progress(done: int, total: int):
    """Update the progress bar while performing comparisons."""
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
        t(
            "progress.compare_line",
            bar=bar,
            percent=ratio * 100,
            done=done,
            total=total,
            eta=CURRENT_ETA_STR,
        )
    )
    sys.stdout.flush()


def redraw_progress():
    """Redraw the most recent progress bar after a log line hides it."""
    if PROGRESS_MODE == "load":
        print_loading_progress(LOAD_DONE, LOAD_TOTAL)
    elif PROGRESS_MODE == "compare":
        print_compare_progress(CURRENT_PROGRESS, TOTAL_PROGRESS)


def update_move_start_time():
    """Record (or reset) the start time for the comparison / move phase."""
    global MOVE_START_TIME
    MOVE_START_TIME = time.time()


def set_total_source_images(count: int):
    """Store the total number of source images being processed."""
    global TOTAL_SOURCE_IMAGES
    TOTAL_SOURCE_IMAGES = count


def set_processed_base_count(count: int):
    """Record how many base images have been processed so far."""
    global PROCESSED_BASE_COUNT
    PROCESSED_BASE_COUNT = count


def increment_progress(delta: int = 1):
    """Increment the processed pair counter."""
    global CURRENT_PROGRESS
    CURRENT_PROGRESS += delta


def set_progress(value: int):
    """Set the processed pair counter to the provided value."""
    global CURRENT_PROGRESS
    CURRENT_PROGRESS = value


def _backoff(attempt: int):
    """Sleep with exponential backoff (plus jitter) before the next retry."""
    delay = min(30, (2**attempt) + random.uniform(0, 1))
    time.sleep(delay)


def install_silent_keyboardinterrupt_hook():
    """Install an excepthook that suppresses tracebacks for KeyboardInterrupt."""
    old_hook = sys.excepthook

    def _hook(exc_type, exc, tb):
        """Handle Ctrl+C quietly and delegate everything else to the original hook."""
        if exc_type is KeyboardInterrupt:
            return
        old_hook(exc_type, exc, tb)

    sys.excepthook = _hook


def safe(func, *args, desc=None, retries=0, **kwargs):
    """Call `func` with retries and log failures in a consistent format."""
    for attempt in range(retries + 1):
        try:
            return func(*args, **kwargs)

        except KeyboardInterrupt:
            raise

        except Exception as exc:
            is_final = attempt >= retries
            desc_text = desc or t("desc.generic_operation")
            msg = t("log.warn_retry", desc=desc_text, current=attempt + 1, total=retries + 1)
            log(msg if not is_final else f"{msg} [traceback]")
            if is_final:
                _write_traceback_log(desc_text, exc)
                log(t("log.error_permanent", desc=desc_text))
                return None
            _backoff(attempt)


def quit_requested():
    """Return True when q/Q was pressed in the console."""
    return msvcrt.kbhit() and msvcrt.getch() in (b"q", b"Q")


def compute_file_sha1(path: str, chunk: int = 1024 * 1024) -> str:
    """Compute the SHA-1 hash of a file and return it as a hex string."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            c = f.read(chunk)
            if not c:
                break
            h.update(c)
    return h.hexdigest()


def safe_rename_with_hash(src: str, dst: str, desc: str) -> bool:
    """Attempt to rename `src` into `dst`, falling back to SHA-1 validation if needed."""
    def _try(a, b):
        """Wrapper for os.rename so it can be retried via `safe`."""
        os.rename(a, b)
        return True

    result = safe(_try, src, dst, desc=desc, retries=2)
    if result:
        return True

    if not os.path.exists(dst):
        return False

    log(t("log.conflict_rename", name=os.path.basename(src)))

    try:
        h_src = compute_file_sha1(src)
        h_dst = compute_file_sha1(dst)
    except Exception as exc:
        log(t("log.hash_compare_error", src=src, dst=dst, exc=exc))
        return False

    if h_src == h_dst:
        log(t("log.match_same", src=src, dst=dst))

        m_src = os.path.getmtime(src)
        m_dst = os.path.getmtime(dst)
        rem, surv = (src, dst) if m_src < m_dst else (dst, src)

        log(t("log.delete_old", path=rem))
        try:
            os.remove(rem)
        except Exception as exc:
            log(t("log.delete_error", path=rem, exc=exc))
            return False

        if surv == src:
            try:
                os.rename(src, dst)
            except Exception as exc:
                log(t("log.rename_error", src=src, dst=dst, exc=exc))
                return False

        log(t("log.rename_success", path=dst))
        return True

    log(t("log.sha_mismatch", src=src, dst=dst))
    return False


def convert_heic_to_jpg(path: str, dup_dir: str) -> str | None:
    """Convert HEIC/HEIF images to JPEG and move the original into the duplicates folder."""
    try:
        with Image.open(path) as img:
            new_path = os.path.splitext(path)[0] + ".jpg"
            img.convert("RGB").save(new_path, "JPEG", quality=95)
        dst = os.path.join(dup_dir, os.path.basename(path))
        shutil.move(path, dst)
        log(t("log.heic_success", src=path, dst=new_path))
        return new_path
    except Exception as exc:
        log(t("log.heic_fail", path=path, exc=exc))
        return None


def rename_jfif_to_jpg(path: str) -> str:
    """Rename `.jfif` files to `.jpg`."""
    try:
        new_path = os.path.splitext(path)[0] + ".jpg"
        os.rename(path, new_path)
        log(t("log.jfif_success", src=path, dst=new_path))
        return new_path
    except Exception as exc:
        log(t("log.jfif_fail", path=path, exc=exc))
        return path


def fix_wrong_extension(path: str) -> str:
    """Correct file extensions so they match the actual image format."""
    try:
        with Image.open(path) as img:
            fmt = (img.format or "").upper()
        current_ext = os.path.splitext(path)[1].lower()

        if fmt in ("JPEG", "JFIF"):
            new_path = os.path.splitext(path)[0] + ".jpg"
            if current_ext == ".jpg":
                return path
            ok = safe_rename_with_hash(path, new_path, desc=t("desc.extension_normalize_jpeg"))
            return new_path if ok else path

        ext_map = {"PNG": ".png", "GIF": ".gif", "WEBP": ".webp", "TIFF": ".tiff", "BMP": ".bmp"}
        if fmt in ext_map:
            correct = ext_map[fmt]
            if current_ext == correct:
                return path
            new_path = os.path.splitext(path)[0] + correct
            ok = safe_rename_with_hash(path, new_path, desc=t("desc.extension_fix_format", format=fmt))
            return new_path if ok else path

        return path

    except Exception as exc:
        log(t("log.extension_detect_fail", path=path, exc=exc))
        return path


def dct2(a: np.ndarray) -> np.ndarray:
    """Compute a 2D DCT using scipy's 1D helper twice."""
    return dct_1d(dct_1d(a, axis=0, norm="ortho"), axis=1, norm="ortho")


def calc_phash(img_arr: np.ndarray) -> int:
    """Calculate a 64-bit perceptual hash from a 224x224 grayscale image array."""
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
    """Return the Hamming distance between two 64-bit integers."""
    return (a ^ b).bit_count()


def cache_all_images(paths: list[str]):
    """Load image data and build the SSIM/pHash/SHA-1 caches."""
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
    log(t("log.load_start", total=total))
    CURRENT_ETA_STR = t("status.estimating")

    for idx, path in enumerate(paths, start=1):
        if quit_requested():
            log(t("log.load_quit"))
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
            log(t("log.load_error", path=path, exc=exc))
            continue

        imgs.append(arr)
        sizes.append(os.path.getsize(path))
        resolutions.append((width, height))
        phashes.append(calc_phash(arr))
        valid_paths.append(path)
        try:
            hashes.append(compute_file_sha1(path))
        except Exception as exc:
            log(t("log.sha1_error", path=path, exc=exc))
            hashes.append("")

    _clear_progress_line()
    log(t("log.load_done", count=len(valid_paths)))
    return valid_paths, imgs, sizes, phashes, resolutions, hashes


def get_optimal_workers():
    """Estimate a reasonable ThreadPool worker count based on CPU topology."""
    phys = psutil.cpu_count(logical=False)
    logi = psutil.cpu_count(logical=True)

    if phys == 2 and logi == 4:
        return 4
    return max(2, min(logi, int(phys * 1.3)))


hamming_cache = {}


def fast_hamming(a, b):
    """Return the cached pHash distance when available, otherwise compute and cache it."""
    key = (a << 64) | b
    if key in hamming_cache:
        return hamming_cache[key]
    distance = hamming64(a, b)
    hamming_cache[key] = distance
    return distance


def set_worker_data(images, sizes, paths, phashes, resolutions, hashes):
    """Populate global references so worker threads can access cached data."""
    global W_IMAGES, W_SIZES, W_PATHS, W_PHASHES, W_RESOLUTIONS, W_HASHES
    W_IMAGES = images
    W_SIZES = sizes
    W_PATHS = paths
    W_PHASHES = phashes
    W_RESOLUTIONS = resolutions
    W_HASHES = hashes


def clear_worker_data():
    """Reset the worker globals to release references."""
    set_worker_data(None, None, None, None, None, None)


def ssim_task(pair):
    """Evaluate SHA-1/pHash/SSIM for a pair and return the comparison result."""
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
                log(t("log.debug_ssim", a=W_PATHS[i], b=W_PATHS[j], score=score))

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
    """Return the next (i, j) pair to process given the current indices."""
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
    """Main entry point that finds duplicates in `folder_path` and moves them into `duplicates/`."""
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
        log(t("log.resume_continue", start_i=start_i, start_j=start_j))
    else:
        start_i = 0
        start_j = 1
        moved = set()
        current_progress = 0
        set_progress(0)
        set_processed_base_count(0)

    moved_before = set(moved)

    log(t("log.duplicate_start", folder=folder_path))
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
        log(t("log.no_images"))
        log_processing_stats("status.done")
        return

    log(t("log.collect_count", count=len(all_files)))

    estimated_mem_mb = len(all_files) * 0.05
    log(t("log.memory_estimate", mb=estimated_mem_mb))

    after = []
    for f in all_files:
        if f.lower().endswith((".heic", ".heif")):
            new = safe(convert_heic_to_jpg, f, dup_dir, desc=t("desc.heic_conversion"), retries=2)
            if new:
                after.append(new)
            continue
        after.append(f)

    tmp = []
    for f in after:
        if f.lower().endswith(".jfif"):
            new = safe(rename_jfif_to_jpg, f, desc=t("desc.jfif_conversion"), retries=2)
            tmp.append(new if new else f)
            continue
        tmp.append(f)

    final = []
    for f in tmp:
        fixed = safe(fix_wrong_extension, f, desc=t("desc.extension_fix"), retries=2)
        final.append(fixed if fixed else f)

    cached = cache_all_images(final)
    if cached is None:
        log(t("log.loading_interrupted"))
        log_processing_stats("status.interrupted")
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
        log(t("log.single_image"))
        set_processed_base_count(n)
        log_processing_stats("status.done")
        return

    total_pairs = n * (n - 1) // 2
    workers = get_optimal_workers()
    log(t("log.compare_config", count=n, pairs=total_pairs, workers=workers))
    log(t("log.compare_strategy"))

    if not is_resume:
        current_progress = 0
        set_progress(0)
    global TOTAL_PROGRESS
    TOTAL_PROGRESS = total_pairs
    print_compare_progress(current_progress, total_pairs)

    def on_new_base(i: int):
        """Update ETA, resume data, and logs whenever the base image index changes."""
        global CURRENT_ETA_STR
        set_processed_base_count(i)
        if current_progress <= 0 or current_progress < total_pairs * 0.01:
            CURRENT_ETA_STR = t("status.estimating")
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
        log(t("log.compare_base", index=i + 1, total=n, name=os.path.basename(cached_paths[i])))

    def pair_gen():
        """Yield (i, j) pairs sequentially based on the outer scope state."""
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
                        log(t("log.move_sha", path=smaller))
                    else:
                        log(t("log.move_ssim", score=score, path=smaller))
                    safe(shutil.move, smaller, dst, desc=t("desc.move_duplicate"), retries=2)

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
        log_processing_stats("status.interrupted")
        log(t("log.quit_saved"))
        return

    finally:
        exe.shutdown(wait=False, cancel_futures=True)
        clear_worker_data()

    if os.path.exists(resume_path):
        os.remove(resume_path)
        log(t("log.resume_cleanup_success"))

    set_processed_base_count(n)
    log_processing_stats("status.done")
    log(t("log.complete_banner"))

    new_moved_count = len(moved) - len(moved_before)
    log(t("log.moved_summary", count=new_moved_count))
