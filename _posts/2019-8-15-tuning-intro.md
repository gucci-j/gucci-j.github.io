---
layout: post
title: 実装とともに学ぶハイパーパラメータチューニングのお話
description: ハイパーパラメータチューニングの基礎について、Keras Tunerを活用した実装例とともに学びます。
lang: ja_JP
custom_css: post
tags:
- Tips
- 機械学習全般
- 「ゼロからKeras」シリーズ
---

## はじめに
[前回](https://gucci-j.github.io/cv-intro/)は交差検証について紹介をしました。今回は、ゼロからKerasシリーズの総まとめとして、ハイパーパラメータチューニングについて紹介します。実装例としては、[Keras Tuner](https://github.com/keras-team/keras-tuner) と呼ばれる、Keras用のハイパーパラメータ自動最適化ツールを活用した実装を紹介します。


## 1. ハイパーパラメータとは
ハイパーパラメータとは、最適化アルゴリズムによって最適化できないパラメータのことを指します。例えば、学習エポック数やバッチサイズ・隠れ層の次元数、学習率などがあたります。

最適化アルゴリズムでは最適化できないハイパーパラメータを最適化するには、人手ではなく、専用のツールを使うのが便利です。専用のツールはたくさん種類があるので、フレームワーク等の状況に応じたものを取捨選択すると良いです。


## 2. 実装
例によって、画像データセット: CIFAR10 の分類実験を題材にハイパーパラメータチューニングを実施してみます。基にするソースコードは前回の交差検証で用いた、`cnn.py` とします。冒頭で述べたように、Keras Tuner を活用して実装を行います。

> <i class="fas fa-link" style="padding: 0 2px 0 0;"></i>参照: [（GitHub: cnn.py）](https://github.com/gucci-j/intro-deep-learning-keras/tree/master/cross-validation/src)

### 2.1 Keras Tunerについて
Keras Tuner は、その名の通り Keras 用に開発中のハイパーパラメータ最適化ツールです。現時点で対応している最適化手法は、「ランダムサーチ」と `Hyperband` になります。今回はランダムサーチを適用してチューニングを行います。ランダムサーチは決められた範囲内からパラメータをランダムに選択し、試していく手法です。

なお、Keras Tuner は、TensorFlow 2.0 以降のTensorFlowに統合されたKeras (`tf.keras`) に対応していることが明記されているので、これまで本ブログで扱ってきたいわゆる無印Keras とは少し異なります。そのため、本記事では、無印Kerasユーザの方でも、`tf.keras` を使って最低限動かせるような構成にしてあります。

> <i class="fas fa-link" style="padding: 0 2px 0 0;"></i>参照: [（GitHub: Keras Tuner）](https://github.com/keras-team/keras-tuner)

### 2.2 Keras Tunerのインストール
まずは、Keras Tunerをインストールします。Python 3.6〜 と TensorFlow 2.0 が Requirementsとして指定されています。

```
git clone https://github.com/keras-team/keras-tuner.git
cd keras-tuner
pip install .
```

### 2.3 CNNモデルの実装
これまで活用してきた、`cnn.py` は `tf.keras` に互換性がないので、`tf.keras` 向けに書き換えます。

#### 2.3.1 ライブラリ読み出し
ライブラリ読み出し部分は至って簡単に移植できます。今までのソースコードに、`tensorflow.` を追加するだけです。

```python
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np
```

Keras Tuner のライブラリも読み込ませます。
```python
from kerastuner.tuners import RandomSearch
```

#### 2.3.2 モデルの定義
モデルの定義部分は、Keras Tuner 向けに書き換える必要が出てきます。具体的には、次の2点について改造を施します。
* 引数に `hp` : ハイパーパラメータ を取らせる  
* チューニングしたいハイパーパラメータの部分を、`hp.Range` に置き換える。

`hp.Range` の使い方は次の表の通りです。  

| **引数** | **説明** |
|:--:|:--|
| `min_value` | チューニングしたいパラメータの最小値を指定します。 |
| `max_value` | チューニングしたいパラメータの最大値を指定します。 |
| `step` | インクリメントしていく値(幅)を指定します。 |

以下が、Keras Tunerに対応させたソースコードとなります。

```python
def build_model(hp) -> Model:
    # モデル定義
    _input = Input(shape=(32, 32, 3))
    _hidden = Conv2D(filters=hp.Range('filters', min_value=10, 
                    max_value=40, step=10), 
                    kernel_size=hp.Range('kernel_size', min_value=2,
                    max_value=5, step=1), 
                    strides=(1, 1), padding='valid', activation='relu')(_input)
    _hidden = MaxPooling2D(pool_size=(2, 2))(_hidden)
    _hidden = Flatten()(_hidden)
    _hidden = Dense(units=hp.Range('units', min_value=50,
                    max_value=200, step=50), 
                    activation='relu')(_hidden)
    _output = Dense(10, activation='softmax')(_hidden)
    model = Model(_input, _output)
    model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        loss='categorical_crossentropy', metrics=['accuracy'])

    return model
```

#### 2.3.3 データの読み込み
データセットの読み込み部は一切変更する必要はありません。無印のKeras用に実装したソースコードをそのまま活用できます。

```python
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float') / 255.
x_test = x_test.astype('float') / 255.
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
```

#### 2.3.4 RandomSearchクラスのインスタンス生成
ここからが本題になります。まずは、ランダムサーチを行うためのインスタンス `tuner` を生成します。

```python
tuner = RandomSearch(build_model, objective='val_accuracy',
        max_trials=5, executions_per_trial=1, directory='tuning', project_name='log')
```

RandomSearch クラスの主な引数は次の表の通りです。

| **引数** | **説明** |
|:--:|:--|
| `hypermodel` | `HyperModel` クラスのインスタンスか、ハイパーパラメータを引数にとる、`Model` インスタンスを返す関数を与えます。 |
| `objective` | 何を基準に最適化を行うかを指定します。 |
| `max_trials` | 最大何回探索を行うかを指定します。 |
| `executions_per_trial` | 一回の探索で何回学習を繰り返すかを指定できます。<br />複数回指定すると、結果を安定させる効果があります。 |
| `directory` | ログの保存先ディレクトリを指定します。 |
| `project_name` | ログの保存先ディレクトリ2を指定します。<br />つまり、`directory`/`project_name`下にログファイルが保存されます。 |

#### 2.3.4 ランダムサーチの実行
RandomSearch クラスのインスタンスを生成したら、ランダムサーチを実行できます。その前に、`search_space_summary()`メソッドを使って、探索候補を確認することができます。

```python
tuner.search_space_summary()
```

実行結果：
```
[Search space summary]
 |-Default search space size: 4
 > filters (Range)
 |-default: None
 |-max_value: 40
 |-min_value: 10
 |-step: 10
 > kernel_size (Range)
 |-default: None
 |-max_value: 5
 |-min_value: 2
 |-step: 1
 > units (Range)
 |-default: None
 |-max_value: 200
 |-min_value: 50
 |-step: 50
 > learning_rate (Choice)
 |-default: 0.01
 |-values: [0.01, 0.001, 0.0001]
```
探索候補の範囲が定義した通りに表示されていますね。

では、本題の探索に移っていきましょう。探索は、`search()` メソッドを使います。このメソッドは、`model.fit()`に対応しています。
```python
tuner.search(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
```

#### 2.3.5 チューニング結果の確認
チューニングの結果は、`results_summary()` で確認できます。

```python
tuner.results_summary()
```

実行結果：
```
[Results summary]
 |-Results in tuning/log
 |-Ran 5 trials
 |-Ran 5 executions (1 per trial)
 |-Best val_accuracy: 0.6094
```

実行結果から、最高精度を記録したパラメータの設定は下記の設定のときでした。  
今回は5回しか探索をしていないので、最適なパラメータが得られたとは言い難いです。実データでパラメータチューニングを行う際は、探索空間の大きさに応じて探索回数を適度に増やすのが無難です。
```
"values": {"filters": 10, "kernel_size": 3, "units": 50, "learning_rate": 0.001}
```

> 今回は使いませんでしたが、`get_best_models()` で最良の結果を残したモデルをピックアップできます。（このメソッドを使わなくても、重みファイルは自動で保存されます。）

## まとめ
今回は、Keras Tunerを活用したハイパーパラメータチューニングの方法について紹介をしました。簡単に使えるので便利ですね。

他にも便利なチューニングツール（例: Optuna）が数多く公開されているので、確認してみると良いかもしれません。

## ソースコード
ソースコードは、[GitHub](https://github.com/gucci-j/intro-deep-learning-keras/)にて公開してあります。