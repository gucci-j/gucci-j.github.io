---
layout: post
title: torchtextでk-分割交差検証をする話
description: PyTorchの自然言語処理（NLP）系データ処理ライブラリである，torchtextでどうしても交差検証をしたい人のためのTipsです．scikit-learnライクな，交差検証をしやすいライブラリとして有名である，skorchは使いません．
lang: ja_JP
custom_css: post
tags:
- Tips
- PyTorch
---

## はじめに
[torchtext](https://github.com/pytorch/text)は，PyTorchで自然言語処理（NLP）系のデータを比較的簡単に読み込むことができるライブラリとして有名です．しかし，とっつきやすい性質を持つ分，細かいところで苦戦する場合があります．その一例として，交差検証をやりにくいという点が挙げられます．

正確には，torchtextで処理したデータを用いて交差検証をした例がネット上に少ないことに加え，torchtextのドキュメントにそれに関する記述がないことも災いしていると思われます．

通常なら，torchtextで交差検証をするのは諦めて，[skorch](https://github.com/skorch-dev/skorch)などの他のライブラリを使うと思いますが，ここではあえて「torchtext」と「sklearn」の `KFold` を使うことで交差検証を適用する方法を紹介したいと思います．

> <i class="fas fa-link" style="padding: 0 2px 0 0;"></i>参考: [Use torchtext to Load NLP Datasets — Part II](https://towardsdatascience.com/use-torchtext-to-load-nlp-datasets-part-ii-f146c8b9a496)

## 1. タスク設定
映画レビュー文データセットである，IMDBデータセットを用いたネガティブ・ポジティブの2値分類タスクを解くモデルを，k分割交差検証にかけてみます．

ベースにするモデルは，GRUとSelf-Attentionで構成されたモデルです．この実装は，[GitHub](https://github.com/gucci-j/imdb-classification-gru)にて公開してあります．

## 2. 実装
それでは，torchtextで読み込んだデータを交差検証にかけられるようにしていきましょう．

### 2.1 データローダ側
#### 2.1.1 初期設定 ＆ コンストラクタ

コンストラクタ内では，通常のtorchtextの用法と同じく，`datasets.IMDB.splits()` でIMDBデータセットを呼び出すようにします．

返り値は，self.train_data, self.test_data として保持しておきます．

```python
import torch
from torchtext import data, datasets
import random
from sklearn.model_selection import KFold
import numpy as np

class load_data(object):
    def __init__(self, SEED=1234):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True

        TEXT = data.Field(tokenize='spacy')
        LABEL = data.LabelField(dtype=torch.float)

        self.train_data, self.test_data = datasets.IMDB.splits(TEXT, LABEL)
        self.SEED = SEED
```

#### 2.1.2 学習データ読み込み

学習データを読み込む際は，`get_fold_data()` を使うようにします．

scikit-learnの `model_selection.KFold` クラスを使うことで，データセットを交差分割用に分割します．scikit-learnを普段から使っている人なら，おなじみかもしれません．

`KFold` のメソッドである，`split` は，引数にNumPy配列を渡す必要があるので，torchtextから生成されたデータセットでは型エラーとなってしまいます．そこで，データをNumPy配列に変換して渡してあげると型エラーにならずに動作してくれます．

しかしながら，無理やりNumPy配列に変換したことによる弊害も生じます．というのも，そのまま，`torchtext.data.Iterator` にデータを渡すと，再び型エラーになってしまいます．学習をラクして回すためにイテレータは欲しいところです．

そこで，`torchtext.data.Dataset` にNumPy配列に変換されてしまった学習データを渡して，イテレータを生成できる状態に戻してあげます．

以上が，学習データの読み込み部分になります．

```python
def get_fold_data(self, num_folds=10):

    TEXT = data.Field(tokenize='spacy')
    LABEL = data.LabelField(dtype=torch.float)
    fields = [('text', TEXT), ('label', LABEL)]

    kf = KFold(n_splits=num_folds, random_state=self.SEED)
    train_data_arr = np.array(self.train_data.examples)

    for train_index, val_index in kf.split(train_data_arr):
        yield(
            TEXT,
            LABEL,
            data.Dataset(train_data_arr[train_index], fields=fields),
            data.Dataset(train_data_arr[val_index], fields=fields),
        )
```

#### 2.1.3 テストデータ読み込み

テストデータの読み込みは，NumPy配列に変換する必要もないので，メソッドが呼び出されたら，そのままデータを渡してあげるだけで大丈夫です．

```python
def get_test_data(self):
    return self.test_data
```


### 2.2 呼び出し側

呼び出し側は基本的には，交差検証無しの[ベースモデル](https://github.com/gucci-j/imdb-classification-gru)と同じです．

追加されている点としては，`data.Iterator` でイテレータを生成する作業が追加されていることです．また，各foldでの結果を保存するために，リスト: `_history` を用意してあります．

細かい点は，[GitHub](https://github.com/gucci-j/pytorch-imdb-cv)にて実装を公開しているので，そちらを参照いただければと思います．

```python
def main():
    data_generator = load_data()
    _history = []
    device = None
    model = None
    criterion = None

    for TEXT, LABEL, train_data, val_data in data_generator.get_fold_data():

        TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.300d")
        LABEL.build_vocab(train_data)

        model = Model(len(TEXT.vocab), args['embedding_dim'], args['hidden_dim'],
            args['output_dim'], args['num_layers'], args['dropout'])
        
        optimizer = optim.Adam(model.parameters())
        criterion = nn.BCEWithLogitsLoss()

        if args['gpu'] is True and args['gpu_number'] is not None:
            torch.cuda.set_device(args['gpu_number'])
            device = torch.device('cuda')
            model = model.to(device)
            criterion = criterion.to(device)
        else:
            device = torch.device('cpu')
            model = model.to(device)
            criterion = criterion.to(device)
        
        train_iterator = data.Iterator(train_data, batch_size=args['batch_size'], sort_key=lambda x: len(x.text), device=device)
        val_iterator = data.Iterator(val_data, batch_size=args['batch_size'], sort_key=lambda x: len(x.text), device=device)

        for epoch in range(args['epochs']):
            train_loss, train_acc = train_run(model, train_iterator, optimizer, criterion)
            print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        val_loss, val_acc = eval_run(model, val_iterator, criterion)
        print(f'Val. Loss: {val_loss:.3f} | Val. Acc: {val_acc*100:.2f}% |')

        _history.append([val_loss, val_acc])
    
    _history = np.asarray(_history)
    loss = np.mean(_history[:, 0])
    acc = np.mean(_history[:, 1])
    
    print(f'LOSS: {loss}, ACC: {acc}')
```

## まとめ

やや駆け足の解説となりましたが，一回NumPy配列に変換してあげることで交差検証が可能になるので，どうしてもtorchtextでデータセットを読み込みたい人には使えるテクニックだと思います．

実際のところtorchtextのレポジトリを見ると，交差検証に関するissueが出ているので，この機能を設けて欲しい人はそれなりにいるみたいですね．（ですが，今の所はこの投稿のような形で無理やり対処するしかないでしょう...）

## ソースコード

ソースコードは，[GitHub](https://github.com/gucci-j/pytorch-imdb-cv)にて公開しています．