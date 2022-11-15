# Basic-Transformer

## モデル一覧
- archi0(basic_transformer)：最もベーシックなTransformer
- archi1(sim_transformer)：１バッチ内にある文について異なる類似文を使った全ての事例が含まれる
- *archi2(sim_transformer2)：archi1から派生。出力のlogitを全て足す
- archi3: basic_transformerから分岐。Encoderの出力文長をsrcの文長のみにする
- archi4: sim_transformer2から分岐。Encoderの出力文長をsrcの文長のみにする。Refernceも含まれる
- *archi5: archi4から分岐。Encoderの出力文長をsrcの文長のみにする+KL div
- archi6: maskはせず、lossもそのまま。refは追加。
- archi7: archi5から派生。cos * cross_entropy (実装中)
- archi8: archi7から派生。SentWightedLoss
- *archi9: archi8から派生。Load dataを改良
- archi10: archi8から派生。Encoder出力をConcat
