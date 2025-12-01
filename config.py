"""アプリ全体で共有する設定値やパス定義。"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "log"
LOG_DIR.mkdir(exist_ok=True)

# --- 利用者が変更してもよい設定 ---
DEBUG_LOG_SSIM = False  # True にすると SSIM の詳細ログを出力
DEFAULT_SSIM_THRESHOLD = 0.85  # SSIM がこの値以上なら重複とみなす
PHASH_THRESHOLD = 40  # pHash の距離がこの値以下なら SSIM を計算
PROGRESS_BAR_WIDTH = 30  # 読み込み・比較フェーズで表示する進捗バーの幅
RESUME_FILE_NAME = "resume.json"  # 中断・再開情報の保存ファイル名
