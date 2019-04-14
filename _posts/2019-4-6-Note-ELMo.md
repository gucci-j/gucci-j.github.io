---
layout: post
title: 論文メモ：Deep contextualized word representations
description: NAACL 2018でベストペーパーアワードに輝いた，Deep contextualized word representations（通称ELMo）の論文メモ書きを共有・紹介します．
mathjax: true
lang: ja_JP
---

## 文献情報  
著者: M. Peters et al.  
所属: Allen Institute for Artificial Intelligence / Paul G. Allen School of Computer Science & Engineering, University of Washington  
出典: NAACL 2018 [(https://aclweb.org/anthology/papers/N/N18/N18-1202/)](https://aclweb.org/anthology/papers/N/N18/N18-1202/)

## どんな論文か？
ELMo = Embeddings from Language Modelsの略であり，言語モデルを活用した文脈に応じた単語分散表現: ELMoを提唱した論文．

## 先行研究と比べてどこが凄い？
既存のNLPタスクのモデルの埋め込み層にELMoを追加するだけで，様々なタスクで当時のSoTAを達成した．

## 技術や手法のキモはどこ？
双方向言語モデルを活用し，隠れ層の重みを重み付き線形和で圧縮する点．  
→ 従来手法では，単に双方向言語モデルの出力層だけを取ってきていた．

## どうやって有効だと検証した？
* 各種NLPタスクに適用して，そのスコアにより評価．

## 議論はある？
* ELMoの中間層が捉えている特徴素について  
→ 著者らによると，ELMoの中間層は低層であればあるほど，**文法的(syntactic)**な情報を含み，高層の方が，**意味的(semantic)**な情報を含むとのこと．  
→ 基本的にどのタスクもsyntacticな情報を好む傾向にあるらしい．

## 論文の詳細メモ
### 1. 順方向言語モデル
トークン数: $N$の単語列: $(t_1, t_2, \dots, t_N)$が与えられたとき，順方向の言語モデルは，単語: $t_k$と単語列: $(t_1, \dots, t_{k-1})$の条件付き確率をモデリングすることで，次のように表される.順方向の言語モデルでは，次の単語: $t_{k+1}$を予測することを目的とする．

$$p(t_1, t_2, ..., t_N) = \prod_{k=1}^N p(t_k | t_1, t_2, \dots, t_{k-1})$$

* 最近の言語モデルでは，文脈非依存な単語表現: $\mathbf{x}_k^{LM}$を単語埋め込みやcharacter-based CNNにより算出してから，L層のLSTMに入力することが多い．

* 単語列の位置: $k$において，各LSTM層は，文脈依存な単語表現: $\overrightarrow{\mathbf{h}}_{k, j}^{LM}$（ただし，$1 \leq j \leq L$）を出力する．

* $$\overrightarrow{\mathbf{h}}_{k, L}^{LM}$$は，次の単語: $t_{k+1}$をsoftmax層で予測するのに用いられる．

### 2. 逆方向言語モデル
逆方向言語モデルは順方向言語モデルと類似しているが，単語列を逆に処理していく点で異なる．なお，逆方向の言語モデルでは，未来の単語列から一つ前の単語: $t_{k-1}$を予測することを目的とする．つまり，逆方向の言語モデルは次のように定義される．

$$p(t_1, t_2, ..., t_N) = \prod_{k=1}^N p(t_k | t_{k+1}, t_{k+2}, \dots, t_N)$$

* 最終的には，$$\overleftarrow{\mathbf{h}}_{k, L}^{LM}$$ を求めることで，$t_{k-1}$ をsoftmax層で予測する．

### 3. 双方向言語モデル
* 通常言語モデルは対数尤度を最大化することで最適化を行う．  
→ 文脈中で出現してほしい単語の確率を最大化するため．

* ELMoで使われる双方向言語モデルは次の式の対数尤度を最大化することで学習する．  
→ 文脈非依存な単語表現とソフトマックス層のパラメータ: $\Theta_{x}, \Theta_s$は共有．LSTMのパラメータについてのみ独立．  
→ 従来手法ではパラメータはすべて独立

$$
\sum_{k=1}^{N}(\log p(t_k | t_1, t_2, \dots, t_{k-1}; \Theta_x, \overrightarrow{\Theta}_{LSTM}, \Theta_s) \\+ \log p(t_k | t_{k+1}, t_{k+2}, \dots, t_N; \Theta_x, \overleftarrow{\Theta}_{LSTM}, \Theta_s))
$$

### 4. ELMoのパラメータ算出
* 全てのトークン: $t_k$はL層の双方向言語モデルに対し，2L+1個の特徴量を持つ．  
→ 文脈依存しない単語ベクトル: 1個  
→ 文脈依存のする順方向と逆方向のLSTM: 2L個

$$
\begin{equation*}
\begin{split}
R_k &= \{x_k^{LM}, \overrightarrow{\mathbf{h}}_{k, j}^{LM}, \overleftarrow{\mathbf{h}}_{k, j}^{LM} | j = 1, \dots, L \}\\
    &= \{\mathbf{h}_{k, j}^{LM} | j=0, \dots, L \}
\end{split}
\end{equation*}
$$

* ELMoを他のタスクに応用するには，Rのすべての要素を一つのベクトル表現に変換する．  
→ 簡単な例だと，一番上の層だけを取ってくるものがある．CoVeやTagLMはこれを採用している．

$$
\mathbf{ELMo}_k^{task} = E(R_k; \Theta_e) = \mathbf{h}_{k, L}^{LM}
$$

<div style="text-align: center">ただし，$\mathbf{h}_{k, L}^{LM} = \left[\overrightarrow{\mathbf{h}}_{k, j}^{LM}; \overleftarrow{\mathbf{h}}_{k, j}^{LM}\right]$</div>

* ELMoではより便宜をはかり，重み付き線形和の形で変換する．  
→ $s_j^{task}$はsoftmaxで正規化された重み  
→ $\gamma^{task}$はELMoのベクトル全体をスケーリングするため．チューニングの最適化の観点から必要．

$$
\mathbf{ELMo}_k^{task} = \gamma^{task} \sum_{j=0}^L s_j^{task}\mathbf{h}_{k, j}^{LM}
$$

### 5. ELMoモデルのNLPタスクへの適用
単に$\mathbf{ELMo}_k^{task}$を入力の埋め込みベクトルとconcatすれば良い．

## まとめスライド
<div style="text-align: center"><iframe src="//www.slideshare.net/slideshow/embed_code/key/hvw0gfJhsc8aWL" width="510" height="420" frameborder="0" marginwidth="0" marginheight="0" scrolling="no" style="border:1px solid #CCC; border-width:1px; margin-bottom:5px; max-width: 100%;" allowfullscreen> </iframe></div><div style="margin-bottom:5px"></div>

## 実装
試しにKerasでELMo + BiLSTMを使ってIMDBの分類を行ったので，GitHubにあげました．  
→ [https://github.com/gucci-j/elmo-imdb](https://github.com/gucci-j/elmo-imdb)

