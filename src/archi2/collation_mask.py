import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List


class CollationAndMask:
    def __init__(self, vocab):
        self.vocab = vocab
        # self.num_of_simを後で定義

    ######################################################################
    # Collation
    # ---------
    #
    # As seen in the ``Data Sourcing and Processing`` section, our data iterator yields a pair of raw strings.
    # We need to convert these string pairs into the batched tensors that can be processed by our ``Seq2Seq`` network
    # defined previously. Below we define our collate function that convert batch of raw strings into batch tensors that
    # can be fed directly into our model.
    #
    def sequential_transforms(self, *transforms):
        # helper function to club together sequential operations
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input
        return func


    def tensor_transform(self, token_ids: List[int]):
        # function to add BOS/EOS and create tensor for input sequence indices
        return torch.cat((torch.tensor([self.vocab.BOS_IDX], dtype=torch.int64),
                        torch.tensor(token_ids, dtype=torch.int64),
                        torch.tensor([self.vocab.EOS_IDX], dtype=torch.int64)))


    def collate_fn(self, batch):
        # batch[0]
        # {'src': 'These results indicated that the insulin deficiency causes change of the protein metabolism and collapse of the muscle protein . | これら の 結果 は ， Ｉ 型 および ＩＩＩ 型 コラーゲン の 合成 と 分解 が 蛋白質 欠乏 によって 影響 さ れる こと を 示し た 。 | 運動@@ 不足 は 筋肉 で の インスリン抵抗性 を 誘導 し ， 糖質 代謝 障害 に 導く 。 | これら の 結果 は 運動 や エタノール の 急性 投与 が 生体 内 に 脂質 過 酸化 を 誘発 する こと を 示唆 し た 。', 'tgt': 'インシュリン 不足 が 蛋白 代謝 の 変化 および 筋 蛋白 の 崩壊 を 引き起こす こと を 示唆 し た 。', 'sent_idx': 74949}
        
        # function to collate data samples into batch tesors
        src_batch, tgt_batch, sent_idx_batch = [], [], []
        for x in batch:
            splited=x['src'].split('|')
            self.num_of_sim = len(splited) - 1
            for i in range(self.num_of_sim):
                src_batch.append(self.vocab.text_transform['src'](('|'.join([splited[0], splited[i+1]]).split())))
                tgt_batch.append(self.vocab.text_transform['tgt'](x['tgt'].split()))
                sent_idx_batch.append(x['sent_idx'])
            
                # print('|'.join([splited[0], splited[i+1]]))
                # print(x['tgt'])
                # print(x['sent_idx'])

        src_batch = pad_sequence(src_batch, padding_value=self.vocab.PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.vocab.PAD_IDX)
        return src_batch, tgt_batch, sent_idx_batch


    ######################################################################
    # During training, we need a subsequent word mask that will prevent model to look into
    # the future words when making predictions. We will also need masks to hide
    # source and target padding tokens. Below, let's define a function that will take care of both.
    #
    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def create_mask(self, src, tgt, device):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len, device)
        src_mask = torch.zeros((src_seq_len, src_seq_len),
                            device=device).type(torch.bool)

        src_padding_mask = (src == self.vocab.PAD_IDX).transpose(0, 1)
        tgt_padding_mask = (tgt == self.vocab.PAD_IDX).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
