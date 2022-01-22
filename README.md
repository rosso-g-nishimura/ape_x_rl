## 分散深層強化学習(Ape-X)によるソニック・ザ・ヘッジホッグの攻略
- RayとPyTorchを使用した、Ape-XによるSteam版ソニック・ザ・ヘッジホッグの攻略。
- 論文の技法等を極力再現したつもりですが、一部実装・処理が異なる可能性があります。

## 学習後のサンプル(400,000回Learner更新後)
https://user-images.githubusercontent.com/96220400/150657888-1a30b595-24bd-47e0-959a-9a9f43b6fb6d.mp4

## 実行ファイル
- main.py: 学習用実行ファイル
- demo.py: main.py 実行により学習した.chkptファイルを読み込み、テスト実行＆動画(mp4)を生成

## 元論文
- https://arxiv.org/abs/1803.00933

## 参考
- https://g.co/kgs/Hh1Sku
- https://horomary.hatenablog.com/entry/2021/03/02/235512
