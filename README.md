# Basic-Transformer

## モデル一覧
- basic_transformer：最もベーシックなTransformer
- sim_transformer(archi1)：１バッチ内にある文について異なる類似文を使った全ての事例が含まれる
- sim_transformer2(archi2)：出力のlogitを全て足す
- archi3: basic_transformerから分岐。Encoderの出力文長をsrcの文長のみにする
- archi4: sim_transformer2から分岐。Encoderの出力文長をsrcの文長のみにする。Refernceも含まれる
- archi5: archi4から分岐。Encoderの出力文長をsrcの文長のみにする+KL div