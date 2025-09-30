# サッカー仮想戦術ボード (Soccer Virtual Tactics Board)

## 概要
YOLOv8 を用いてサッカーの試合映像から選手・ボールを検出し，
射影変換によって仮想戦術ボード上にプロットするシステムです。

従来の戦術ボードと異なり，実際の試合映像とリンクした戦術の「見える化」が可能です。

## 特徴
- 選手とボールを自動検出（YOLOv8）
- チームごとの色分け表示
- 仮想フィールドへの射影変換
- ハーフライン，ゴールエリア，ペナルティエリア，センターサークルを描画
- 選手アイコンを自由に操作可能



例：
![結果](/images/結果.png)

## 環境
- Python 3.10+
- OpenCV
- Ultralytics YOLOv8
- NumPy

## インストール
```bash
git clone https://github.com/username/repo-name.git
cd repo-name
pip install -r requirements.txt
