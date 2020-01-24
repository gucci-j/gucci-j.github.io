---
layout: post
title: 論文メモ：Grounding Strategic Conversation
description: Grounding Strategic Conversation Using negotiation dialogues to predict trades in a win-lose game の論文メモ書きを共有・紹介します．
lang: ja_JP
custom_css: post
tags:
- 論文メモ
---

## 文献情報  
著者: Cadilhac et al.  
所属: IRIT Univ. Toulouse  
出典: EMNLP 2013 ([https://www.aclweb.org/anthology/D13-1035/](https://www.aclweb.org/anthology/D13-1035/)  

<div class="crowdfunding_ad">
    <div class="pc" style="text-align: center;">
        <a href="https://anchor.fm/melancholy">
            <img src="{{ site.baseurl }}/resources/ads/mefm_banner_large.png" alt="めらんこりーFM!"/>
        </a>
        <br />
    </div>
    <div class="sp" style="text-align: center;">
        <a href="https://anchor.fm/melancholy">
            <img src="{{ site.baseurl }}/resources/ads/mefm_banner_small.png" alt="めらんこりーFM!"/>
        </a>
        <br />
    </div>
</div>

## どんなもの？  
交渉ゲームにおいてプレイヤーの行動を予測する手法を提案した論文であり，「Settlers of Catan ゲーム」の交渉ログをアノテーションしたデータを構築して検証を行った．


## 先行研究と比べてどこがすごい？  
* **アノテーション粒度が既存研究よりも細かい**  
    単に Accept や Reject だけをアノテーションするのではなく，dialogue act にどのような物品をやりとりしたかを示す属性： Resource を設けている．

* **交渉対話のアノテーションについておそらく初めて扱った論文**  
    なお，論文からだけではデータが公開されているかどうかは不明．


## 技術や手法のキモはどこ？  
### データセット  
3つのフェーズに分けてアノテーションした．コーパスの規模は 511 dialogues と各交渉ログの長さを勘案すると若干少なめ．　　 

1. **交渉ダイアログを「ターン：EDU」に分割**  
    各ターンを Elementary Discourse Unit（EDU）と呼ばれる単位に分割する．EDU には発言者が予め付与される．


2. **Dialogue Act Annotation**  
    交渉対話なので，Dialogue Act は "offer, counter offer, accept, refusal" に加えて， "other" からなる．各EDUごとに Dialogue Act のアノテーションがされている．

    > other は 交渉とはあまり関係のない行動について付与するもの．  
    > Dialogue Act については，[SLP3](https://web.stanford.edu/~jurafsky/slp3/) の26章を参照するとよい．  
    
3. **Resource Type Annotation**  
    各EDUに付与した Dialogue Act の具体的な内容をアノテーションしており，交渉でやり取りする物品についての内容と取引の属性をまとめている．具体的には，Givable・Not Givable・Receivable・Not Receivable の4つがある．

    カタンゲームが多者間交渉であることから，取引先の関係性を明示するため（照応解析）に Anaphora Link という属性もアノテーション対象に含まれている．  

以上のフェーズを考慮してアノテーションを行った結果が以下の表となっている．  

<div style="text-align: center;">
    <img src="{{ site.baseurl }}/resources/2020-01-24/table1.png" alt="アノテーション済みの対話データ例" style="width: 700px;"/><br />
</div>

> <i class="fas fa-image" style="padding: 0 2px 0 0;"></i> 図引用: [https://www.aclweb.org/anthology/D13-1035/](https://www.aclweb.org/anthology/D13-1035/)


### 手法（Dialogue Act Prediction & Resource Prediction）
Dialogue act と resource の予測は3つのフェーズごとにモデリングした．

#### 第1フェーズ  
Dialogue Act の特定を行う．  

あるEDUはそれより前のEDUと依存関係がある：（例）Accept や Reject は Offer や Counter Offer のあとに続くことが多い．  
→ 系列ラベリングとして考えられるので， Conditional Random Field (CRF) が Dialogue Act の特定に使えるという仮説．

#### 第2フェーズ  
Resource の範囲を特定する．

交渉の最中にやりとりした物品の内容を特定するために必要なフェーズ．単一カテゴリの交渉であるためやり取りする内容が決まっていることから，予め決めた辞書に語句が含まれているかどうかだけを検知する．

#### 第3フェーズ  
Resource の属性を特定する．（つまり前述の Givable・Not Givable・Receivable・Not Receivable を CRF を使って推定．） 


### 手法（Predicting Player's Strategic Actions）　　
プレーヤーの行動の予測には CP-net を活用した．CP-net はグラフィカルモデルの一種．


## どうやって有効だと検証した？  
F値（マクロ平均）と精度が主に使われている．CP-netについては，混同行列の値も求めている．


## 議論はある？
基本的に各手法はベースラインとして比較されている手法を上回っている．