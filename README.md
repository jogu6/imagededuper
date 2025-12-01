# imagededuper

Python 製の重複画像検出ツールです。フォルダを再帰的に走査し、拡張子統一 → SHA-1 → pHash → SSIM を段階的に適用して重複判定を行い、解像度の低い画像を `duplicates/` フォルダへ自動的に移動します。読み込み・比較の各フェーズでは進捗バーと ETA を表示し、中断操作を検知すると再開用データを保存します。

## 主な特徴

- HEIC/HEIF から JPEG への変換、および JFIF/誤った拡張子の補正を自動化
- SHA-1 の完全一致、pHash による候補絞り込み、SSIM による最終判定で精度と速度を両立
- 解像度の低い画像だけを `duplicates/` へ移動し、ログへ詳細を記録
- 読み込み・比較の両フェーズで進捗バーと ETA を表示
- 処理中断時に `resume.json` を生成し、次回起動時に再開／リセットを選択可能
- `config.py` で SSIM 閾値や進捗バー幅などの設定を一括管理

## 必要要件

- Python 3.10 以上を推奨
- 主要依存パッケージ  
  `numpy`, `Pillow`, `pillow-heif`, `scikit-image`, `scipy`, `psutil`

※ 仮想環境での利用を推奨します。既存環境にインストール済みであれば追加作業は不要です。

## セットアップ

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell の例
pip install -r requirements.txt  # まだ無い場合は poetry/pip などで各パッケージを導入
```

`requirements.txt` が無い場合は `pip install numpy Pillow pillow-heif scikit-image scipy psutil` のように個別に導入してください。

## 使い方

```bash
python imagededuper.py
```

1. 実行すると対象フォルダを尋ねられるので、重複検出したいフォルダパスを入力します。
2. `duplicates/` フォルダ（存在しない場合は自動作成）に重複と判断された画像が移動され、詳細は `log/imagededuper_YYYYMM.log` に記録されます。
3. 処理途中に `q` を押すと中断し、次回起動時に「続きから再開／最初からやり直す／キャンセル」を選択できます。

### 設定の変更

`config.py` に以下のような設定値があります。編集後は再実行するだけで反映されます。

| 変数名 | 説明 |
| --- | --- |
| `DEFAULT_SSIM_THRESHOLD` | SSIM がこの値以上のとき重複とみなす（既定 0.85） |
| `PHASH_THRESHOLD` | pHash の距離がこの値以下なら SSIM を計算（既定 40） |
| `DEBUG_LOG_SSIM` | `True` にすると SSIM 計算の詳細ログを出力 |
| `PROGRESS_BAR_WIDTH` | 進捗バーの幅（文字数） |
| `RESUME_FILE_NAME` | 中断情報ファイル名（既定 `resume.json`） |

## ログと再開データ

- ログ: `log/` フォルダに `imagededuper_YYYYMM.log` 形式で追記され、重大なエラーは `error_traceback_YYYYMMDD.log` に記録されます。
- 再開データ: `resume.json` が存在する場合、起動時に再開／リセット／キャンセルを選択できます。完了後は自動的に削除されます。

## サポートとご留意点

個人の趣味開発プロジェクトのため、Issues への対応は可能な範囲になります。また、このリポジトリは日本語をベースに運用しているため、英語での質問には翻訳を介して対応する点をご理解ください。

## ライセンス

詳細は `LICENSE.md` を参照してください（MIT License）。
