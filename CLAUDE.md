# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

SongEchoは、ICLR 2026のカバーソング生成研究プロジェクト。ACE-Stepの音楽生成基盤モデル上に構築され、Instance-Adaptive Element-wise Linear Modulation (IA-EiLM) により、元のボーカルメロディとテキストプロンプトを条件として新しいボーカルと伴奏を同時合成する。ベースラインと比較して訓練可能パラメータが30%未満で済む点が特徴。

## 環境セットアップ

```bash
conda create -n songecho python=3.10 -y
conda activate songecho
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

## 主要コマンド

### データ前処理パイプライン（順序実行）
```bash
python 0_download_audio.py                    # 音声ダウンロード
python 1_process_caption.py                   # キャプション・歌詞処理
python 2_extract_f0.py                        # F0（ピッチ）特徴量抽出
python 3_convert2hf_dataset_split.py --data_dir ./suno70k/audio --repeat_count 1 --output_name suno70k
```

### 学習
```bash
sh train.sh
# または直接:
CUDA_VISIBLE_DEVICES=0,1,2 python train.py \
    --sample_size 90 --batch_size 1 --exp_name test \
    --train_dataset_path /path/to/train_dataset \
    --val_dataset_path /path/to/val_dataset \
    --devices 3 --warmup_steps 1000 --max_steps 30000
```

### 推論
```bash
python inference.py \
    --output_dir ./val_results \
    --pt_path ./checkpoints/melody.pt \
    --dataset_path /path/to/val_dataset
```

## アーキテクチャ

### コアパイプライン

- **`acestep/pipeline_ace_step.py`** — ACEStepPipeline本体（1730行）。ベーストランスフォーマー、テキストエンコーダ(UMT5)、音楽DCAE デコーダのロードと推論スケジューリングを管理
- **`train.py`** — `Pipeline(LightningModule)`: ACEStepPipelineの基盤モデルをフリーズし、LoRAアダプタを追加してflow matching拡散過程で学習
- **`inference.py`** — `Pipeline(nn.Module)`: 学習済みメロディ/テキスト条件付けを読み込み拡散サンプリングで音声生成

### MelodyEncoder

`train.py` と `inference.py` の両方に定義されたConv1Dベースのエンコーダ。F0時系列(B, T, 1)を隠れ表現(B, 256, T)に変換する。

### IA-EiLM（コア手法）

トランスフォーマーFFN内の学習可能な線形射影層で実装:
- `proj_c`: 条件射影 (256→256)
- `proj_x`: 特徴射影 (2560→256)
- `cond_project`: 変調ネットワーク (256→5120)

### 条件付けメカニズム

1. **テキスト条件**: UMT5テキストエンコーダ → 768次元埋め込み
2. **メロディ条件**: F0特徴量 → MelodyEncoder → 隠れ表現
3. **歌詞条件**: BPEトークン化 → Conformerエンコーダ（`acestep/models/lyrics_utils/`）
4. **話者条件**: 話者埋め込み（512次元、オプション）

### データセットクラス

`acestep/text2music_dataset.py` の `Text2MusicDataset` が音声ロード、リサンプリング、無音検出、歌詞トークン化（16言語対応）、F0ロードを担当。

## ディレクトリ構成の要点

| パス | 役割 |
|------|------|
| `acestep/` | ACE-Stepコアモジュール（モデル、スケジューラ、データセット、GUI） |
| `acestep/models/` | トランスフォーマー、アテンション、歌詞エンコーダ |
| `acestep/music_dcae/` | DCAE ボコーダ（メルスペクトログラム→波形） |
| `acestep/schedulers/` | Flow matchingスケジューラ（Euler/Heun/Pingpong） |
| `config/suno_config.json` | LoRA設定（r=256, target_modules等） |
| `metrics/` | F0ピアソン相関メトリクス（RMVPE/parselmouth） |
| `checkpoints/` | モデルチェックポイント格納先（要ダウンロード） |
| `examples/` | サンプルデータ（音声、F0、メタデータ） |
| `suno70k_prompt_val/` | 公式バリデーション分割（HuggingFace arrow形式） |
| `GTsinger_tech_recognition/` | GTsingerピッチ推定ユーティリティ |

## 重要な技術的注意点

- 音声サンプルレート: 16kHz（標準）、ホップサイズ160サンプル
- F0範囲: 50-900Hz、RMVPEモデル使用（`checkpoints/rmvpe_model.pt`）
- 学習はPyTorch Lightningベースで分散学習対応（DistributedSampler）
- 勾配チェックポインティング有効でメモリ効率化
- Classifier-Free Guidance (CFG) を推論時に使用
- LoRAはPEFTライブラリ経由で適用、`config/suno_config.json` で設定
