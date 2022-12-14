import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List


class CollationAndMask:
    def __init__(self, vocab, num_sim):
        self.vocab = vocab
        self.num_sim = num_sim

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

    
    def collate_fn_orig(self, batch):
        # function to collate data samples into batch tesors
        src_batch, tgt_batch = [], []
        for x in batch:
            src_batch.append(self.vocab.text_transform['src'](x['src'].replace('|', '<sep>').split()))
            tgt_batch.append(self.vocab.text_transform['tgt'](x['tgt'].split()))

        src_batch = pad_sequence(src_batch, padding_value=self.vocab.PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.vocab.PAD_IDX)
        return src_batch, tgt_batch
        
    
    def collate_fn(self, batch):
        # function to collate data samples into batch tesors
        src_batch = []                  # srcのテンソル
        tgt_batch = []                  # tgtのテンソル

        for x in batch:
            src_and_sims_list = x['src'].split('|') # 0:orig_src 1~:sims

            i=1
            for i in range(1,len(src_and_sims_list)):
                tmp = ' <sep> '.join([src_and_sims_list[0], src_and_sims_list[i]])
                src_batch.append(self.vocab.text_transform['src'](tmp.split()))
                tgt_batch.append(self.vocab.text_transform['tgt'](x['tgt'].split()))
            else:
                while i < self.num_sim:
                    tmp = ' <sep> '.join([src_and_sims_list[0], ''])
                    src_batch.append(self.vocab.text_transform['src'](tmp.split()))
                    tgt_batch.append(self.vocab.text_transform['tgt'](x['tgt'].split()))
                    i+=1

        src_batch = pad_sequence(src_batch, padding_value=self.vocab.PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.vocab.PAD_IDX)

        return src_batch, tgt_batch


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
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

        src_padding_mask = (src == self.vocab.PAD_IDX).transpose(0, 1)
        tgt_padding_mask = (tgt == self.vocab.PAD_IDX).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
