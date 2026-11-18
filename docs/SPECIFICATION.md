# 最小実験の凍結仕様 v1

2026-11-14。結果生成前に固定。研究番号なし。FEASIBILITY.mdの未固定項目を本書で具体化する。成立性評価のための均一背景・既知位置モデルであり、患者乳房や実機SoftVueの再現ではない。

## 先行研究との差分とゲート

当初の116例USCT研究以外に、Hardtの2012年学位論文の石灰化USCT計算、Bahramian/Insanaの2014年検出理論＋ファントム研究、Thonらの2022年点散乱体検出評価を確認した。したがって「サブ波長石灰化のUSCTシミュレーション」「画像処理による検出改善」「検出の理論化」は新規性としない。

今回確認する候補差分は、径・開口・帯域・物性を明示した計測領域の条件付き理想観測者評価と、弾性散乱を無視した場合の差である。独立した物理検証・未知背景評価がない最小モデルだけでは、研究全体の論文成立と判定しない。

## 物性と採用根拠

- 背景fat：c=1440.2 m/s、rho=911 kg/m³、alpha0=4.3578 Np/m/MHz^y。
- 背景glandular：c=1540.0 m/s、rho=1041 kg/m³、alpha0=8.635 Np/m/MHz^y。
- 上記はLiらの数値乳房ファントム論文Table IIの平均値。分布の全幅を検証済み生体範囲とは扱わない。指数y=1.2は簡略化した設計値。主結果はこの指数に条件付ける。
- 主石灰化HAp：rho=3180 kg/m³、E=6 GPa、nu=0.27。Paulら2022 Table 2のシミュレーション材料値を採用する。同表の1374 m/sはsqrt(E/rho)とほぼ一致する棒の速度であり、等方3D固体の縦波速度と組み合わせて独立入力しない。cL=sqrt[E(1−nu)/(rho(1+nu)(1−2nu))]、cT=sqrt[E/(2rho(1+nu))]から一貫して導出する。この解釈は本計画での整合性点検であり著者の誤りと断定しない。
- 材料感度：calcium oxalate、rho=2200 kg/m³、E=50.38 GPa、nu=0.32（同表）。両材料を生体石灰化物性の上下限とは呼ばない。HApの硬さ・孔隙・組成とMHz減衰の生体実測根拠は未確立。
- 内部固体は損失なし・等方弾性、背景は流体近似。HApをcL・rhoが同じ流体に置き換える反例を代表条件で評価する。これを固体の妥当な代替とはしない。

## Forward model

2D無限円柱1個、中心位置固定、径50/100/150/200/300/500 µm。均一な既知背景を用いる。水槽・皮膚境界、組織テクスチャ、multiple scattering、形態や位置分布、クラスタは含めない。

弾性円柱内に縦波・横波ポテンシャル、外部にBessel/Hankel展開を置き、法線変位連続、法線応力=−圧力、せん断応力0をモードごとに解く。Faran型分離変数法を境界条件から実装する。開放境界の外向Hankel関数を使いPML・空間格子を不要とする。単一円柱の損失なし散乱は厳密なモデル解、伝搬減衰は以下の近似である。

背景の経路長Lに対しexp(−alpha0 f_MHz^1.2 L)を無損失Green関数に掛ける。分散と界面の減衰は解かない。これは因果的な全波損失モデルではなく、伝搬損失係数を付加した周波数領域近似。結果を完全な乳房forward simulationの検証とは呼ばない。

## 仮想装置・パルス・ノイズ

- 半径110 mmのリング、32送信位置・32受信位置、full 360°。limitedは同じ位置集合の連続16位置（180°の半開区間）。同素子送受信は除く。全組合せのtrace数は992／240、発射回数32／16。双方で送信係数と各traceの独立ノイズを同じにする。同一曝露比較ではない。
- 各素子は幅0.3 mmの円弧上で一様に送受信を平均。中央円柱に対する各角モードの係数はsinc(n width/(2R))²で厳密平均する。指向性・整合層・電気応答・elevationは含めない。32素子は計算用の疎な仮想配置であり実機の2048素子仕様と混同しない。
- 周波数1/3 MHz。実数Gaussian-modulated cosine、t0=5 µs、Gaussian振幅スペクトルの−6 dB全幅=0.8 fc（半振幅の定義、sigma_f=0.4 fc/sqrt(2 ln 2)）。各fcで送信時間波形の最大値を1に正規化する。Green関数iH0/4に掛ける圧力係数1 Paであり、実機素子表面圧・MI・安全性の校正ではない。
- fs=20 MHz、N=4096、T=204.8 µs。DC／Nyquistは0、0.1–8 MHzを使用。20 MHzの帯域外を無視するモデル定義。観測は全T固定。
- 校正参照はfat・1 MHz・円柱なし・対向素子間220 mmの1本のRF。全TのRMSを固定し、sigma=RMS×10^(−SNR/20)、SNR=20/40 dB。全径・材料・背景・fc・開口で共通の絶対sigmaを使う。径・fcごとの再正規化をしない。参照は中心への経路に対して素子幅方向の位相差が無視できる点素子近似。実測電子雑音の根拠はなく、この人工noise条件のみに結論を限定する。

## 評価・確認・規模

- 主96条件：2背景×2fc×6径×2view×2noise。SKE/BKE・共通白色Gaussian共分散下でd′=||ΔRF||/sigma、AUC=Phi(d′/sqrt(2))、FPR 0.05での感度=Phi(d′−Phi^-1(0.95))。
- 各条件につき各クラス1,000独立反復、しきい値校正用H0も別seedで1,000反復。多次元ノイズの固定templateへの射影は厳密に標準正規となるため、十分統計量を直接標本化する。これは独立forward検証ではなくROC実装の照合。AUCはMann–Whitney、95% CIはクラス別非パラメトリックbootstrap 200回、感度・FPRはWilson区間。多数条件の区間を同時区間と呼ばない。
- 収束：主モード次数16→24、同じTでfs 20→40 MHz（増分noise sigmaはsqrt(2)倍で同じPSDを保つ）、T 204.8→409.6 µsで最初の204.8 µsを比較。代表は両背景・両fcの50/500 µm、full view。ΔRFノルムとd′変化5%以内。空間格子細分化は解析境界表現に置換する。実行前の方式具体化であり、収束基準は弱めない。
- 材料／流体置換感度：100/300 µm、両背景、両fc、両view、両noise。主条件を見た後の材料選択・範囲変更をしない。
- 減衰感度：HAp 100/300 µm、両背景・両fc、alpha0×0.9/1.1。10%は文献phantom分布の設定規模に沿う感度係数であり実測confidence boundではない。
- optionalの正則化画像再構成は、この物理モデルのゲートが通った場合のみ予算内で実施する。今回は画像再構成の優劣を必須成立条件とせず、物理・新規性で不成立なら画像やAIへ進まない。未実施なら明記する。
- チェック：固体モードの境界残差、損失なし受動性|1+2A_n|=1、fluidの同物性時ゼロ散乱、剛体極限、source/receiver相反性、Parseval、ノイズ／送信数に対するd′則、解析ROCとMonte Carlo。これらは数理実装の確認であり実測・独立ソルバーによる物理検証ではない。

## ライセンス・環境・送信

新規Python実装＋既存NumPy/SciPyを使用、図のみMatplotlib。ライブラリをコピー・vendorせず、インストール済み版のmetadataと同梱licenseを記録する。NumPy/SciPyの主ライセンスはBSD-3-Clause、MatplotlibはPSF由来の独自ライセンス。wheel内の他ライセンスを主ライセンスに丸めない。

usct-breast-phantomは論文でGPL-2.0、公開phantomはCC0と記載されるが、今回取得・利用しない。後日使う際は配布元の該当versionの条件を確認する。本pilotには公開画像・患者データ・学習済みモデル・外部コード片を使わない。論文の図表や全文を再配布しない。新規コードの公開ライセンス付与・公開は今回行わない。

確認環境：Windows、Python3.14.3、NumPy2.5.1、SciPy1.17.1、Matplotlib3.10.8。RTX3080 10GBを確認したが本最小系はCPUのみ、BLASは最大1thread。CPU wall時間をdevice-hourとして保守的に記録。自動停止は1run最大1時間、総計24device-hours／作業18時間／期限2026-11-18 18:00 JSTを超えない。

外部AI送信は本計画で承認されていないため実行しない。独立Codex/Claudeレビューを未実施のままPASSとしない。文献の公開情報検索のみを行う。

## 出典

- [Li et al., acoustic phantom、物性Table IIと仮想装置](https://arxiv.org/html/2106.02744)、最終論文 DOI [10.1109/TUFFC.2021.3112544](https://doi.org/10.1109/TUFFC.2021.3112544)。
- [Paul et al., mineral properties Table 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC8938815/)、DOI [10.3390/tomography8020047](https://pubmed.ncbi.nlm.nih.gov/35314624/)。
- [Hardt 2012, §6.5](https://publikationen.bibliothek.kit.edu/1000028779)。全文の検索索引で石灰化節を確認、本文PDFの完全取得は未了。独立性／精度の詳細は未確認。
- [Bahramian/Insana 2014](https://experts.illinois.edu/en/publications/improved-microcalcification-detection-in-breast-ultrasound-phanto/)、[Thon et al. 2022](https://pubmed.ncbi.nlm.nih.gov/34797764/)。前者は大学掲載抄録、後者はPubMed抄録まで確認。
- [Faranの原著](https://msvlab.ntou.edu.tw/grades/bem%282008%29/Sound%20scattering%20by%20solid%20cylinders%20and%20spheres.pdf)、[Mamouの学位論文 §2.1](https://www.brl.uiuc.edu/Publications/2005/JonathanMamouPhDThesis_2005.pdf)。コードの転用なし。
- [SoftVue一次研究：装置条件](https://pmc.ncbi.nlm.nih.gov/articles/PMC10079278/)、[NumPy](https://numpy.org/about/)、[SciPy](https://scipy.org/faq/)。
