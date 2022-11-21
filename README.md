# Basic-Transformer

## モデル一覧
- archi0：最もベーシックなTransformer
- archi1：Encoderの出力を固定長にする。Src+Refの事例を追加する。
- archi2：Encoderの出力を固定長にする。Src+Refの事例を追加しない。Encoderの出力を文長方向にConcatする手法
    - 参考：https://aclanthology.org/2021.eacl-main.74/
- archi3(予定)：Encoderの出力を固定長にする。Src+Refの事例を追加する。Archi1から派生。Decodingを改良。
- archi4(予定)：Encoderの出力を固定長にする。Src+Refの事例を追加しない。Encoderの出力を深さ方向にConcatする手法。Archi2から派生。
    - 参考：https://aclanthology.org/2022.acl-long.438/