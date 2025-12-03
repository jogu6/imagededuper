# -*- coding: utf-8 -*-
"""
imagededuper.py - CLI entry point for the duplicate image detection workflow.

Highlights:
- Convert HEIC/JFIF files, normalize extensions, and run SHA-1 -> pHash -> SSIM checks.
- Move the lower resolution duplicate into duplicates/ while showing progress bars.
- Support saving and resuming interrupted runs (including ETA tracking).
- Keep tracebacks in a separate log file without breaking the progress display.
"""

import os
import sys

from config import DEFAULT_SSIM_THRESHOLD, RESUME_FILE_NAME
from core import install_silent_keyboardinterrupt_hook, log, move_duplicates, safe
from i18n import t


def handle_resume_choice(resume_path: str) -> bool:
    """Ask the user how to handle an existing resume file."""
    if not os.path.exists(resume_path):
        return True

    print(t("prompt.resume.detected"))
    print(t("prompt.resume.option_continue"))
    print(t("prompt.resume.option_restart"))
    print(t("prompt.resume.option_cancel"))
    choice = input(t("prompt.resume.choice")).strip()

    if choice == "1":
        log(t("log.resume_choice_continue"))
        return True
    if choice == "2":
        log(t("log.resume_choice_restart"))
        try:
            os.remove(resume_path)
            log(t("log.resume_cleanup_stale"))
        except OSError:
            log(t("log.error_remove_resume"))
            return False
        return True
    if choice == "3":
        log(t("log.resume_choice_cancel"))
        return False

    log(t("log.error_invalid_menu"))
    return False


def main():
    """Prompt for the target folder and kick off the duplicate detection workflow."""
    install_silent_keyboardinterrupt_hook()

    folder = input(t("prompt.folder_path")).strip().strip('"')
    if not os.path.isdir(folder):
        log(t("log.error_folder_missing", folder=folder))
        sys.exit(1)

    resume_path = os.path.join(folder, RESUME_FILE_NAME)
    if not handle_resume_choice(resume_path):
        sys.exit(1)

    log(t("log.settings_threshold", value=DEFAULT_SSIM_THRESHOLD))
    safe(
        move_duplicates,
        folder,
        threshold=DEFAULT_SSIM_THRESHOLD,
        desc=t("desc.duplicate_cleanup"),
        retries=2,
    )


if __name__ == "__main__":
    main()
