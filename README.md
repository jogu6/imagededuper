# imagededuper

- 日本語でのご案内のあとに英語版を続けています。  
- Japanese description is followed by an English translation.
- GitHub Pages 版（ドキュメントサイト）: https://jogu6.github.io/imagededuper/

---

## 概要 / Overview

**日本語**: imagededuper は、フォルダ配下の画像を再帰的に走査し、拡張子の統一や HEIC/JFIF 変換を経て、SHA-1 → pHash → SSIM の三段階で重複検出を行う Python 製ツールです。重複と判定した画像は、解像度の低い方を `duplicates/` ディレクトリへ自動的に移動します。
この重複判定はファイル名・更新日時・ファイルサイズといったメタ情報には一切依存せず、画像データそのものを比較しています。

**English**: imagededuper is a Python tool that recursively scans image folders, normalizes extensions (including HEIC/JFIF conversion), and detects duplicates via a three-step process: SHA-1 hashing, perceptual hashing (pHash), and SSIM. Lower-resolution duplicates are automatically moved to the `duplicates/` directory.
This duplicate detection ignores metadata such as file names, timestamps, and file sizes, comparing only the actual image content.

---

## 特徴 / Features

1. **拡張子・フォーマットの自動補正 / Automatic format normalization**  
   日本語: HEIC/HEIF → JPEG 変換、JFIF→JPG 変換、拡張子誤りの修正を自動で実行します。  
   English: Automatically converts HEIC/HEIF to JPEG, renames JFIF to JPG, and fixes mismatched extensions.

2. **段階的な判定フロー / Layered duplicate detection**  
   日本語: SHA-1 で完全一致を除去し、pHash で候補を絞り、SSIM で最終判定することで速度と精度を両立しています。  
   English: Combines SHA-1 (exact matches), pHash (candidate filtering), and SSIM (final decision) for a balance of speed and accuracy.

3. **重複画像の自動整理 / Automatic duplicate relocation**  
   日本語: 解像度の低い方の画像のみ `duplicates/` へ移動し、ログへ記録します。  
   English: Moves only the lower-resolution file to `duplicates/` and logs the action.

4. **進捗表示と中断再開 / Progress bars & resumable runs**  
   日本語: 読み込み・比較フェーズそれぞれで進捗バーと ETA を表示し、`q` で中断すると再開用 `resume.json` を保存します。  
   English: Displays progress bars with ETA for both loading and comparison phases; pressing `q` saves `resume.json` for resuming later.

5. **設定の一元管理 / Centralized configuration**  
   日本語: `config.py` で SSIM 閾値や進捗バー幅などをまとめて調整できます。  
   English: All tunable settings (SSIM threshold, progress bar width, etc.) live in `config.py`.

---

## 対応フォーマット / Supported Formats

- 日本語: imagededuper は以下のフォーマットを再帰的に走査し、重複判定の対象とします。
- English: imagededuper scans and deduplicates the image formats listed below.

| フォーマット / Format | 説明 (日本語) / Description (English) |
| --- | --- |
| `JPEG / JPG` | もっとも一般的な JPEG 画像。JPEG/JPG のどちらの拡張子でもそのまま扱います。 / Standard JPEG images; both `.jpeg` and `.jpg` extensions are handled directly. |
| `PNG` | 透過などを含む PNG 画像もそのまま比較対象に含めます。 / PNG files (including transparency) are compared as-is. |
| `BMP` | 可逆ビットマップ形式。読み込み後に他形式と同様のフローで処理します。 / Lossless BMP files are loaded and processed through the same pipeline. |
| `GIF` | 静止画 GIF（先頭フレーム）を比較に利用します。 / Still GIFs (first frame) are analyzed for duplicates. |
| `TIFF` | TIFF/TIF 画像も拡張子を補正しつつ処理します。 / TIFF/TIF files are supported with extension correction when needed. |
| `WebP` | `webp` 拡張子の画像を直接読み込み、ほかの形式と比較します。 / WebP images are loaded directly and compared with other formats. |
| `JFIF` | `.jfif` が見つかった場合は `.jpg` へ自動リネームしたうえで比較に回します。 / `.jfif` files are automatically renamed to `.jpg` before comparison. |
| `HEIC / HEIF` | HEIC/HEIF は JPEG へ自動変換し、元ファイルは `duplicates/` に退避します。 / HEIC/HEIF images are converted to JPEG automatically, with originals moved to `duplicates/`. |

---


## 必要要件 / Requirements

- Python 3.10+ を推奨 / Python 3.10+ recommended  
- 主要依存ライブラリ / Key dependencies:  
  `numpy`, `Pillow`, `pillow-heif`, `scikit-image`, `scipy`, `psutil`

---

## セットアップ / Setup

```bash
python -m venv .venv
. .venv/Scripts/activate   # Windows PowerShell 例 / example
pip install -r requirements.txt
# または individually / or install individually:
# pip install numpy Pillow pillow-heif scikit-image scipy psutil
```

---

## 使い方 / Usage

```bash
python imagededuper.py
```

1. **フォルダ入力 / Enter target folder**  
   実行すると対象フォルダを尋ねられます。 / You will be prompted for a folder path.

2. **処理と出力 / Processing & output**  
   `duplicates/` に重複が移動され、`log/imagededuper_YYYYMM.log` に詳細が追記されます。  
   Duplicates are moved to `duplicates/`, and logs are appended under `log/`.

3. **中断と再開 / Interruption & resume**  
   途中で `q` を押すと中断し、次回起動時に「再開／やり直し／キャンセル」を選べます。  
   Press `q` to interrupt; next run will offer resume/reset/cancel options.

---

## 設定変更 / Configuration

`config.py` で以下を調整できます / You can edit the following in `config.py`:

| 変数 / Variable | 説明 (日本語) | Description (English) |
| --- | --- | --- |
| `DEFAULT_SSIM_THRESHOLD` | SSIM がこの値以上で重複判定（既定 0.85） | SSIM threshold for duplicates (default 0.85) |
| `PHASH_THRESHOLD` | pHash の距離がこの値以下なら SSIM を計算 | pHash Hamming distance threshold to run SSIM |
| `DEBUG_LOG_SSIM` | True で SSIM の詳細ログを出力 | Enable verbose SSIM logs |
| `PROGRESS_BAR_WIDTH` | 進捗バーの幅（文字数） | Width of progress bars |
| `RESUME_FILE_NAME` | 中断情報ファイル名 | Filename used for resume data |
| `LANGUAGE` | `"ja"` / `"en"` を選択してログ・CLI 表示を切り替え | Switch CLI/log language between `"ja"` and `"en"` |

### 言語設定 / Localization

- `config.py` の `LANGUAGE` 変数で既定言語を切り替えられます（既定は `"en"`）。  
- 表示テキストは `locales/en.json` と `locales/ja.json` に定義されています。必要に応じてこれらの JSON を編集・追加してカスタム翻訳を利用できます。  
- 新しい言語を追加する場合は `locales/<lang>.json` を作成し、`LANGUAGE` にその `<lang>` コードを指定してください。

---

## ログと再開データ / Logs & Resume Data

- **ログ / Logs**: `log/imagededuper_YYYYMM.log` に通常ログ、`log/error_traceback_YYYYMMDD.log` に例外トレースを記録します。  
  Normal logs go to `log/imagededuper_YYYYMM.log`; tracebacks to `log/error_traceback_YYYYMMDD.log`.
- **再開データ / Resume data**: `resume.json` が存在すると起動時に再開／リセット／キャンセルを選択できます。完了後は自動削除されます。  
  If `resume.json` exists, the tool prompts you to resume, restart, or cancel on launch; the file is deleted after a successful run.

---

## サポートと言語について / Support & Language

日本語:
- 個人の趣味開発のため、Issues 等への対応は可能な範囲で行います。
- このリポジトリは日本語をベースに運用しているため、英語での質問には翻訳を介して対応します。

English:
- This is a hobby project; support (Issues/PRs) will be handled on a best-effort basis.
- The repository is maintained primarily in Japanese, so English inquiries will be answered through translation tools.

---

## ライセンス / License

- 本ソフトウェアは MIT License で配布しています。詳細は `LICENSE.md` を参照してください。  
- Third-party dependencies and their licenses are summarized in `THIRD_PARTY_LICENSES.md`.

This project is distributed under the MIT License (see `LICENSE.md`). Dependencies are listed in `THIRD_PARTY_LICENSES.md`.
