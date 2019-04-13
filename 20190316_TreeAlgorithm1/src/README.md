## 実験用スクリプトの実行


### 実行方法

1. データの取得   
  https://www.kaggle.com/c/titanic よりデータを取得しdata/orgディレクトリへ格納する

2. データの加工
  createDataset.pyを以下の通り実行することでdata/basedataディレクトリに試験用（加工済み）のデータセットが作成される   

```
python createDataset.py
```

  すでにdata/basedataディレクトリが存在している場合data/(日付)ディレクトリが生成される。   
  ここで作成したデータセットを使用する場合、学習スクリプト実行時にディレクトリ名を指定する。  

3. 実行

```
python train_model.py --dataset 20190304_172115
```

引数一覧
```
--modelname 使用するアルゴリズム。以下の中から選択（デフォルト：DecisionTree）
           【DecisionTree, Bagging, RandomForest, Adaboost, GradientBoosting】
--train_filename 学習に使用するデータセット
--testsize  学習時の訓練・評価データの分割サイズ（デフォルト: 0.2）
--n_estimators 生成する木の数（Random ForestかGradient Boostingのみ／デフォルト: 10）
--oob_score Out-of-Bagスコアの算出要否（デフォルト: False）
--learning_rate 学習率（デフォルト: 0.1）
--max_depth 木の深さの最大値（デフォルト: None）
--min_samples_split あるノードから分岐させるために最低限必要なデータ数（デフォルト: 2）
--randomseed 乱数のシード
```

RandomForest以外は引数を取らない設定にしてあるので、それらのモデルでパラメータをいじりたい場合はmodels.py内で各モデルをインスタンス化している部分を修正のこと。
