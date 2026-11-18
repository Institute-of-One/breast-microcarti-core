# 論文化の主題再設定と限定追加解析（実行前固定）

固定日2026-11-15。最終判定をSR優越の有無だけで決めない。研究の独立性は問い・物理系・対象・評価方法・再現可能な知見の組合せで判断する。既存結果を見た後の解析であることを明示し、確認的未使用testとは呼ばない。実行前のこの文書を保存し、既存freezeと評価結果を変更しない。

## 問いと主張候補

既知背景の2D弾性散乱モデルで100 µm石灰化クラスターの存在判別が可能な条件でも、US由来の高解像度候補mapによる粒子局在が成立するとは限らない。その乖離を、同一phantomの検出・配置分類・真値対応・偽陽性の対応関係として評価する。原理一般の初発見、情報論的不能限界、臨床USCT・実X-ray生成・臨床代替性は主張しない。

仮題：**Detection–Localization Dissociation in Physics-Simulated Ultrasound Imaging of Microcalcification Clusters**。

## 近い一次研究と差分（2026-11-15確認）

1. Zhangら2021, *Impact of deep learning-based image super-resolution on binary signal detection*, [DOI](https://doi.org/10.1117/1.JMI.8.6.065501)。SRの見た目の改善と二値検出の利益が一致しないというタスク評価は既存。本研究はその原理の初報ではなく、音響の微小石灰化クラスターに対象を定め、存在・配置・粒子局在・FPの対応を評価する。
2. Bhadraら2021, *On Hallucinations in Tomographic Image Reconstruction*, [一次全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC8673588/)。線形撮像系のmeasurement/null componentsに基づくhallucinationの定式化は既存。本研究のGT非対応成分は操作的な局在FPであり、この理論上のhallucination mapを算出したものではない。用語はfalse localizations / spurious candidatesを基本とする。
3. Huangら2013, *Detecting breast microcalcifications using super-resolution ultrasound imaging: a clinical study*, [DOI](https://doi.org/10.1117/12.2007653)、[著者公開原稿](https://hansonhub.com/kmh-lanl/publications/medim13.pdf)。PC-MUSICによる微小石灰化の超解像US臨床検討は既存。「USで微小石灰化をSR表示」が新規ではない。本研究は学習型mapと分類branchのタスク乖離・真値対応FPに焦点を置く。彼らの臨床結果を否定しない。
4. 2025 *Breast cancer detection from ultrasound computed tomography imaging using radiomic analysis: in silico trial*, [DOI](https://doi.org/10.1002/mp.17710)。116 virtual phantomでFWI再構成SOSから病変判別を評価。本文の対象は5–10 mmのspiculated masses。本研究の100 µm・複数弾性散乱体の配置／粒子局在とは対象とendpointsが異なる。今回の逆投影をFWIと呼ばない。
5. Liuら2025, *Deep-Learning-Driven High Spatial Resolution Attenuation Imaging for Ultrasound Tomography (AI-UT)*, [一次全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12455677/)。SOS情報を用いる高空間分解能減衰画像の生成・検証が対象。本研究のoccupancy教師mapは減衰画像や実マンモグラムではなく、3経路のタスク評価を主題とする。
6. Gharamalekiら *Evaluating Detection Thresholds...*, [arXiv v4](https://arxiv.org/abs/2411.07426)（2026-01改訂、プレプリントとして確認）。ULMのmicrobubbleのFP/FNとthresholdの影響を扱う。粒子局在のthreshold依存性自体は既存。本研究は静的石灰化・クラスター分類を対象に、固定全閾値を提示する。

今回の検索範囲で、この対象と対応endpointを同じ設計で扱う研究は確認できなかった。ただし網羅的systematic reviewや世界初の証明ではない。上記に被る一般原理を新規性として数えず、対象系と問いの異なる独立したoriginal computational studyとして継続を検討する。

## 追加解析：既存保存出力のみ

新規forward、再構成、学習、GPU推論、症例追加、モデル探索は行わない。保存済み3経路のtest予測、SRの各phantomの局在件数、事前固定9閾値（0.1–0.9）の件数だけを使う。

- 閾値0.5で、検出true-positive例のうち粒子が1個以上対応した割合／全5粒対応した割合を集計。
- 局在FPを陰性30例／陽性60例に分け、配置別にも提示。FP/全画像とFP/陽性画像を混同しない。
- 事前固定全9閾値で同じ区分を提示。test最良の閾値を採用・学習へ還流しない。非単調な曲線もそのまま保存する。
- phantom単位層別bootstrap1000回で、閾値0.5の局在感度・FP/画像・検出TP条件付き乖離の区間を補助提示。条件間に同一phantomの派生があることを明記し、720画像を720独立例と呼ばない。
- 分類はUSとSR-mapの2channelを使うため、分類の成功をmap単独の情報や局在の成功と解釈しない。追加ablationを実施していない以上、US channelの因果的寄与は断定しない。

解析区間は既存testを見た後の探索的記述。元の固定評価に遡って事前規定だったと装わない。追加の学習seedや公平な全経路局在器の比較は今回の必須項目としない。主張を特定pipelineの乖離に限れば、SR一般の優劣を決める実験は不要。

## 期限・資源

暦期限は2026-11-18 18:00 JSTを維持。追加はCPU実行5分以内・GPU 0時間。各phaseのwall時間を台帳へ追記する。文献確認と文書整理も今回作業内で終了し、大規模原稿・追加計算を始めない。

既知の計上17.16845h、未記録の中断phaseをU時間とすると、追加C時間後の総量は **17.16845 + C + U**。Uは不明で0と置かない。C=5/60hなら24h適合条件はU≤6.7482h。したがって既知台帳の余裕6.83hを、そのまま保証された追加GPU予算にしない。本解析は最大0.0834hの既存出力集計に限定し、無条件の24h適合とは報告しない。未記録時間だけで科学的結果を無効とはしないが、完全な資源会計とは区別する。

## 判断基準

- **継続**：固定された同一例の存在／局在乖離が保存出力で再現し、全閾値・陰性／陽性内訳を示しても、対象系のタスク依存性という解釈が成立する。SRの利益、有意な改善、臨床データは必須としない。original computational/methodological paperの準備へ進める。
- **主張限定**：FPが陽性内だけなら「陰性乳房に石灰化を生成」を捨て、局在誤り・分裂／変形候補として報告。単一threshold依存ならそのoperating pointの報告に限定。これは論文自体の中止理由とはしない。
- **中止／保留**：件数・座標・split等の実装不整合で現象を再現できない、同一対象・問い・主要endpointまで実質的に同一の先行研究が見つかり差分を説明できない、または必須検証が上限内に収まらない場合。先行する一般原理の存在やSR改善なしだけでは中止しない。

既存の「検出限界の証明」題名は今回の主題と切り離し、結果に合う題名・結論へ更新する。アクセプトは保証しないが、一般的な既存原理との重複だけで論文化を弱く評価しない。
