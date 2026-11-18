# 全例再構成・3経路AI比較の資源再配分

2026-11-14。利用者がGPU内枠2時間の見直しと全例処理を明示承認。総上限24 device-hours、期限2026-11-18 18:00 JST、物理条件と数値許容5%は維持する。研究番号なし。既存の実験snapshotと結果を遡及変更しない。

## 固定条件

v2の480 phantom（各クラス160）、各クラスtrain100/validation30/test30、seed20261115、径100 µm、石灰化なし／5個の等総量・等RMS距離の線状／環状配置、背景glandularと既定のnuisance範囲を維持。全例で相互散乱を計算し、許容を超えた3 MHz線状を重ね合わせで代替しない。2fc×2view×2noiseの計3,840画像。test患者相当phantomとその派生はtrainから分離する。

全周波数1,618 bins、±16角モード、3点素子幅積分、128×128 grid、lambda1e−3を維持したGPU逆投影を使う。2noiseの独立RF雑音を生成し、Full/Limitedは同じRFの対応channelを使う。4条件の内積を同じtemplate走査で求め、各viewのtemplate normと各SNRのsigmaで個別に正規化する。

## 実測と予算

4条件preflightは1fc約55.25秒、既存Full20との差8.34×10^-7、Limited20とCPU参照との差5.89×10^-7。学習計時はrandom dataの100stepのみで重みを破棄し、研究データやtestを使わない。2つの計時processが短時間重なった可能性があるため、専有時の最速値とはせず、再構成に25%の余裕を持たせる。

- 再構成：960回×55.25秒＝約14.73時間、25%余裕で約18.42時間。配分18.5 GPU-job hours。
- 相互散乱RF生成：陽性320×2fc×約8.13秒＝約1.45時間、配分2 CPU-job hours。陰性は既知背景差分0なので追加の波動計算をしない。
- 学習：3モデル×30epoch。計時では合計約167秒のtrain-step相当。validation／データ読込等を含め45分を配分する。全モデル250k parameter以下、Adam lr1e−3、batch8、30epoch、単一seed。最良validation loss checkpointだけを採用する。
- 評価：推論・1000回のphantom単位bootstrap・局在／偽点評価に15分を配分する。
- 既実施分と今回preflightに保守的0.5時間を予約。以上22.0時間、残り2.0時間を誤差・実行検査・評価の予備とする。

device-hourの会計単位は、独立したCPU計算段階のwall時間＋GPUを確保している計算段階のwall時間。GPU段階のhost処理・driver待ちはそのGPU-job wallに既に含むため重複加算しない。host process CPU active時間とCUDA event時間は別途情報として記録する。過去のGPU/CPU保守値は0.5時間の予約内に繰り越し、時計をリセットしない。課金API・cloud・追加GPUは使わない。

実測累積をJSONL台帳に記録し、各phase/block/epochで総86400秒と暦期限を確認する。未完了phaseはcheckpoint可能な最小単位ごとに記録し、異常終了も消費時間を残す。予備を使う際も総上限を超えない。再構成中は残余見積りに学習・評価の枠を残し、上限近くまで画像を作って評価不能とならないよう停止する。

## 学習実装の固定

3経路はUS直接、bicubic2倍＋unsharp、US由来のX-ray風SR map＋2channel分類。classifierはConv16/32/64（stride2）＋global average pooling＋3class linear。SR branchはConv8/8＋bilinear2倍＋1channel logits。3経路とも同系列で、パラメータ数を報告する。SR lossはCE＋occupancy BCE（重み1）、他はCE。

通常強調は256grid上でsigma2 output pixels（入力pixelのsigma1）、13×13正規化Gaussian、reflect padding、amount1。高解像度教師は32 µm gridへの石灰化円の面積占有率。各画素16×16部分面積samplingを用い、円の総面積との一致を5%で確認する。実MGの画素を教師として重ねない。

正規化は全train画像だけからmedian/IQRを一度計算、全splitへ固定適用し[−5,5]でclip。各クラスのphantom単位で分割を先に固定。学習の比較終了までtest画像・test指標をモデル選択へ使わない。事前計時の開発phantomと本testを共用しない。

## 評価の固定

testの各撮像条件に存在検出AUC、validationの陰性例からFPR5%となるthresholdを設定した感度／実test FPR、陽性例の線状対環状balanced accuracy、confusion matrixを報告。phantom単位bootstrap1000回で区間と3経路のpaired差を計算する。差が弱くてもmodelやnoise範囲を追加しない。

SR mapは閾値0.5で150 µm以内の一対一病変対応、感度・FP/画像を評価。閾値0.1/0.2/.../0.9も固定してFROC operating pointsを補助報告し、見た目・PSNRを主成功条件としない。実患者の良悪性分類・臨床代替性とはしない。

公開データ調査は今回ユーザーが示した5候補に絞り、画像全量取得・追加臨床annotation・外部AI送信は行わない。合成比較へ臨床MG画像を後から混ぜない。
