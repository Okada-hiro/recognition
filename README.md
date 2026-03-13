# recognition

`recognition` は、オフィス受付を想定したリアルタイム人物検出・顔検出・顔認識 POC のための実装フォルダです。

このフォルダの役割は次のとおりです。

- カメラ映像から人物を検出する
- 人物が近づいてきているかを簡易的に判定する
- 顔を検出する
- 顔画像を社員データベースと照合する
- 検出結果や照合結果を保存する

現時点では、会話機能そのものは含めていません。会話に使うための前段として、誰が来たか、いつ来たか、どのフレームで検出されたかを残すための構成です。

## 想定している技術構成

- 人物検出: `ultralytics` の YOLO
- 顔検出と顔認識: `insightface`
- 保存: `recognition/logs/events.jsonl`

CUDA が使える環境では、人物検出の YOLO と `insightface` の ONNX Runtime が GPU を優先して使う想定です。
また、動画ファイルを入力として処理し、注釈付き動画を保存することもできます。

## フォルダ構成

`recognition` 配下の主なファイルは以下です。

- [main.py](/Users/okadahiroaki/Downloads/AI/infodeliver/face_recognition/recognition/main.py)
  エントリーポイントです。`python -m recognition.main` で起動します。

- [cli.py](/Users/okadahiroaki/Downloads/AI/infodeliver/face_recognition/recognition/cli.py)
  コマンドライン引数を処理します。カメラ番号、入力動画、出力動画、デバイス、人物検出モデルなどをここで受け取ります。

- [config.py](/Users/okadahiroaki/Downloads/AI/infodeliver/face_recognition/recognition/config.py)
  アプリ全体の設定をまとめています。データベースの場所、ログの保存先、閾値などを定義しています。

- [pipeline.py](/Users/okadahiroaki/Downloads/AI/infodeliver/face_recognition/recognition/pipeline.py)
  POC の中心です。1フレームごとに人物検出、追跡、顔検出、顔照合、保存を実行します。

- [detectors.py](/Users/okadahiroaki/Downloads/AI/infodeliver/face_recognition/recognition/detectors.py)
  人物検出器のラッパーです。現在は YOLO を使います。

- [tracker.py](/Users/okadahiroaki/Downloads/AI/infodeliver/face_recognition/recognition/tracker.py)
  人物追跡と、接近判定のための簡易ロジックです。

- [face_recognition.py](/Users/okadahiroaki/Downloads/AI/infodeliver/face_recognition/recognition/face_recognition.py)
  `insightface` の `FaceAnalysis` を使って顔検出と顔埋め込み生成を行い、顔データベースとの距離比較もここで扱います。

- [database.py](/Users/okadahiroaki/Downloads/AI/infodeliver/face_recognition/recognition/database.py)
  `data_base` フォルダの顔画像を読み込み、照合用埋め込みを準備します。

- [models.py](/Users/okadahiroaki/Downloads/AI/infodeliver/face_recognition/recognition/models.py)
  検出結果、照合結果、保存イベントのデータ構造を定義しています。

- [storage.py](/Users/okadahiroaki/Downloads/AI/infodeliver/face_recognition/recognition/storage.py)
  実行結果を JSONL とスナップショット画像として保存します。

- [runpod_recognition_browser.py](/Users/okadahiroaki/Downloads/AI/infodeliver/face_recognition/recognition/runpod_recognition_browser.py)
  `recognition` フォルダをブラウザで閲覧するためのサーバです。RunPod の port `8000` で使う想定です。ファイル閲覧に加えて、Safari からカメラを使ってライブ認識ページも開けます。

## データベースの作り方

顔認識用のデータベースは、プロジェクト直下の `data_base` フォルダを使います。

ディレクトリ構成は次のようにしてください。

```text
data_base/
  alice/
    001.jpg
    002.jpg
  bob/
    001.jpg
    002.png
```

- フォルダ名が `person_id` として扱われます
- その人物の顔画像を複数枚置けます
- 画像形式は `.jpg`, `.jpeg`, `.png` を想定しています

## 環境構築

環境構築はプロジェクト直下の [environment.sh](/Users/okadahiroaki/Downloads/AI/infodeliver/face_recognition/environment.sh) を使います。現在の `recognition` は `insightface + onnxruntime + YOLO` 前提です。

```bash
bash environment.sh
```

CPU のみで環境を作りたい場合:

```bash
FORCE_CPU=1 bash environment.sh
```

Colab などで `venv` 作成が失敗する場合:

```bash
SKIP_VENV=1 bash environment.sh
```

仮想環境を有効化する場合:

```bash
source .venv/bin/activate
```

## 使い方

### 1. 標準設定でカメラ起動

```bash
source .venv/bin/activate
python -m recognition.main --device auto
```

### 2. カメラ番号を指定して起動

```bash
python -m recognition.main --camera-index 1 --device auto
```

### 3. 人物検出モデルを指定して起動

```bash
python -m recognition.main --person-model yolo11s.pt --device 0
```

### 4. 顔データベースの場所を指定して起動

```bash
python -m recognition.main \
  --database-dir /workspace/face_recognition/data_base \
  --device auto
```

### 5. スナップショットも保存する

```bash
python -m recognition.main --device auto --save-snapshots
```

### 6. 動画ファイルを処理する

```bash
python -m recognition.main \
  --input-video movie.mp4 \
  --output-video movie_annotated.mp4 \
  --device auto \
  --no-display
```

`--output-video` を省略した場合は、入力動画と同じ場所に `*_annotated.mp4` を自動で作ります。

### 7. 動画ファイルを表示しながら処理する

```bash
python -m recognition.main \
  --input-video movie.mp4 \
  --device auto
```

RunPod や Colab のように GUI 表示が難しい環境では `--no-display` を付けるのを前提にしてください。

### 8. RunPod で `recognition` フォルダをブラウザ閲覧する

RunPod 側で次を実行します。

```bash
cd /workspace
source .venv/bin/activate
python recognition/runpod_recognition_browser.py
```

その後、Safari などで RunPod の `8000` 番ポートを開きます。

- ルート一覧: `/`
- ディレクトリ閲覧: `/browse/logs`
- ファイル直リンク: `/files/test_annotated.mp4`

環境変数 `RECOGNITION_BROWSE_ROOT` を指定すると、公開するルートを変更できます。

### 9. Safari から Mac のカメラを使ってライブ認識する

同じサーバを起動したまま、Safari で次を開きます。

- `/live`

このページでは、
- Safari が Mac のカメラにアクセス
- 一定間隔でフレームを RunPod に送信
- RunPod 側で人物検出、顔検出、顔認識
- 注釈付き画像をブラウザに返して表示

という流れで動きます。

最初は 1 秒ごとの送信から始める簡易版です。重ければ送信間隔を上げてください。

## 保存されるデータ

イベントは次のファイルに追記されます。

- [logs/events.jsonl](/Users/okadahiroaki/Downloads/AI/infodeliver/face_recognition/recognition/logs/events.jsonl)

1行に1イベントの JSON を保存します。内容には次の情報が含まれます。

- タイムスタンプ
- フレーム番号
- 人物の検出結果
- 顔の検出結果
- 照合に成功した社員ID
- 補足メモ

`--save-snapshots` を付けると、接近判定や照合成功が発生したフレームを次に保存します。

- `recognition/snapshots/frame_000001.jpg`

動画入力時に `--output-video` を付けると、人物検出・顔検出・顔認識の結果が描画された動画も保存されます。

## 実装上の注意

- 現在の人物追跡は簡易的な IoU ベースです
- 接近判定もバウンディングボックス面積の増加で見ているだけです
- 服装から職業を推定する処理はまだ実装していません
- 顔認識は `insightface` の埋め込みとコサイン距離比較を使う前提です
- 顔検出と顔認識は同じ `insightface` 系に寄せており、以前の `retinaface + deepface` 構成より依存関係は単純です

## 今後の改善候補

- YOLO の組み込み tracker を使った安定した追跡
- Colab / RunPod 向けに `cv2.imshow` を使わない実行モードの追加
- 顔照合結果のキャッシュ
- 再入室・滞在時間の記録
- 会話システムに渡すためのセッション管理
- 服装や持ち物の属性認識
