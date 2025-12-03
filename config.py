"""Shared configuration values and common path definitions for the app."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "log"
LOG_DIR.mkdir(exist_ok=True)

# --- Settings the user may safely customize ---
LANGUAGE = "en"  # "en" for English, "ja" for Japanese
DEBUG_LOG_SSIM = False  # Enable verbose SSIM logging when True
DEFAULT_SSIM_THRESHOLD = 0.85  # Images at or above this SSIM score count as duplicates
PHASH_THRESHOLD = 40  # Run SSIM when the pHash distance is at or below this value
PROGRESS_BAR_WIDTH = 30  # Width of the progress bars during load and compare phases
RESUME_FILE_NAME = "resume.json"  # Filename used to store resume / checkpoint data
