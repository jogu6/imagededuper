# -*- coding: utf-8 -*-
"""
imagededuper.py — 画像重複検出・整理ツール

機能概要:
- HEIC/JFIF の変換や拡張子補正を行い、SHA-1 → pHash → SSIM の順に重複を判定
- 解像度の低い画像を duplicates/ へ移動し、読み込み・比較フェーズの進捗を表示
- 中断位置の保存と復元（進捗率・ETA を含む）に対応
- ログと進捗バーの競合を避けつつ、トレースバックを別ファイルへ記録
"""

import os
import sys

from config import DEFAULT_SSIM_THRESHOLD, RESUME_FILE_NAME
from core import install_silent_keyboardinterrupt_hook, log, move_duplicates, safe


def handle_resume_choice(resume_path: str) -> bool:
    """用途: resume ファイルの有無に応じて再開・再実行・キャンセルを選択させる。
    引数:
        resume_path: 再開情報ファイルのパス。
    戻り値: 続行すべきなら True。キャンセルやエラーの場合は False。"""
    if not os.path.exists(resume_path):
        return True

    print("\n=== 中断データが見つかったわ ===")
    print("  [1] 続きから再開")
    print("  [2] 最初からやり直す")
    print("  [3] キャンセルして終了")
    choice = input("番号を選んでね: ").strip()

    if choice == "1":
        log("[選択] 続きから再開するわね。")
        return True
    if choice == "2":
        log("[選択] 最初から再処理するわ。")
        try:
            os.remove(resume_path)
            log("[再開データ削除] 古い再開情報を削除したわ。")
        except OSError:
            log("[エラー] 再開データを削除できなかったわ。")
            return False
        return True
    if choice == "3":
        log("[選択] キャンセルするわ。")
        return False

    log("[エラー] 無効な番号よ。処理を停止するわ。")
    return False


def main():
    """用途: CLI からフォルダパスを受け取り、重複検出処理を起動する。
    引数: なし（標準入力を利用）。
    戻り値: なし。"""
    install_silent_keyboardinterrupt_hook()

    folder = input("対象フォルダを入力してね: ").strip().strip('"')
    if not os.path.isdir(folder):
        log(f"[エラー] フォルダが存在しないわ: {folder}")
        sys.exit(1)

    resume_path = os.path.join(folder, RESUME_FILE_NAME)
    if not handle_resume_choice(resume_path):
        sys.exit(1)

    log(f"[設定] SSIM 閾値: {DEFAULT_SSIM_THRESHOLD:.2f} (変更したい場合は config.py を編集してね)")
    safe(
        move_duplicates,
        folder,
        threshold=DEFAULT_SSIM_THRESHOLD,
        desc="重複削除処理",
        retries=2,
    )


if __name__ == "__main__":
    main()
