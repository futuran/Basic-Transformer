import numpy as np
import os
from omegaconf import DictConfig, OmegaConf
import hydra
from random import shuffle
from timeit import default_timer as timer
from torch.nn.utils.rnn import pad_sequence
import math
from torch.nn import Transformer
import torch.nn as nn
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List

import logs
logger = logs.set_logger('log_predict.log')



class Vocab:
    def __init__(self):
        # Place-holders
        self.token_transform = {}
        self.vocab_transform = {}
        self.text_transform = {}

        # Define special symbols and indices
        self.UNK_IDX = 0
        self.PAD_IDX = 1
        self.BOS_IDX = 2
        self.EOS_IDX = 3
        # Make sure the tokens are in order of their indices to properly insert them in vocab
        self.special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


vocab = Vocab()


class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, src_transform=None, tgt_transform=None, max_length=-1) -> None:
        logger.info('loading dataset')
        logger.info('src: {}'.format(src_file))
        logger.info('tgt: {}'.format(tgt_file))
        logger.info('max length: {}'.format(max_length))

        self.src, self.tgt = [], []
        eliminate_count = 0

        with open(src_file, 'r') as fs:
            with open(tgt_file, 'r') as ft:
                if max_length == -1:
                    for ls, lt in zip(fs, ft):
                        self.src.append(ls.strip())
                        self.tgt.append(lt.strip())
                else:
                    for ls, lt in zip(fs, ft):
                        if len(ls.strip().split()) <= max_length and len(lt.strip().split()) <= max_length:
                            self.src.append(ls.strip())
                            self.tgt.append(lt.strip())

        assert len(self.src) == len(self.tgt)

        logger.info('num of dataset: {}'.format(len(self.src)))
        logger.info('eliminated sents: {}'.format(eliminate_count))

        self.src_transform = src_transform
        self.tgt_transform = tgt_transform

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src = self.src[idx]
        if self.src_transform:
            src = self.src_transform(src)
        tgt = self.tgt[idx]
        if self.tgt_transform:
            tgt = self.tgt_transform(tgt)
        sample = {"src": src, "tgt": tgt}
        return sample


def load_vocab_data(cfg: DictConfig):
    vocab_data = TranslationDataset(
        src_file=cfg.ex.vocab.src,
        tgt_file=cfg.ex.vocab.tgt,
    )
    return vocab_data


def load_train_data(cfg: DictConfig):
    train_data = TranslationDataset(
        src_file=cfg.ex.dataset.train.src,
        tgt_file=cfg.ex.dataset.train.tgt,
        max_length=256
    )
    dev_data = TranslationDataset(
        src_file=cfg.ex.dataset.dev.src,
        tgt_file=cfg.ex.dataset.dev.tgt,
        max_length=256
    )
    test_data = TranslationDataset(
        src_file=cfg.ex.dataset.test.src,
        tgt_file=cfg.ex.dataset.test.tgt,
    )

    return train_data, dev_data, test_data


def yield_tokens_share(data_iter: Iterable, languages: List[str]) -> List[str]:
    for data_samples in data_iter:
        for lang in languages:
            for data_sample in data_samples[lang]:
                yield data_sample.split()


class PositionalEncoding(nn.Module):
    # helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)
                        * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class TokenEmbedding(nn.Module):
    # helper Module to convert tensor of input indices into corresponding tensor of token embeddings
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Seq2SeqTransformer(nn.Module):
    """
    ---------------------------------
    Seq2Seq Network using Transformer
    ---------------------------------

    Transformer is a Seq2Seq model introduced in `“Attention is all you
    need” <https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf>`__
    paper for solving machine translation tasks.
    Below, we will create a Seq2Seq network that uses Transformer. The network
    consists of three parts. First part is the embedding layer. This layer converts tensor of input indices
    into corresponding tensor of input embeddings. These embedding are further augmented with positional
    encodings to provide position information of input tokens to the model. The second part is the
    actual `Transformer <https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html>`__ model.
    Finally, the output of Transformer model is passed through linear layer
    that give un-normalized probabilities for each token in the target language.
    """

    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       layer_norm_eps=1e-6,
                                       norm_first=True)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        # self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        # for share embedding
        self.tgt_tok_emb = self.src_tok_emb

        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

        self.softmax = nn.Softmax()

    def forward(self,
                src: Tensor,
                tgt: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.src_tok_emb(tgt))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
            self.tgt_tok_emb(tgt)), memory,
            tgt_mask)


######################################################################
# During training, we need a subsequent word mask that will prevent model to look into
# the future words when making predictions. We will also need masks to hide
# source and target padding tokens. Below, let's define a function that will take care of both.
#
def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, device):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len),
                           device=device).type(torch.bool)

    src_padding_mask = (src == vocab.PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == vocab.PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


class NoamOpt:
    # for Optimizer Source : https://stackoverflow.com/questions/65343377/adam-optimizer-with-warmup-on-pytorch
    "Optim wrapper that implements rate."

    def __init__(self, model_size, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.model_size = model_size
        self._rate = 0

    def state_dict(self):
        """Returns the state of the warmup scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the warmup scheduler's state.
        Arguments:
            state_dict (dict): warmup scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


######################################################################
# Collation
# ---------
#
# As seen in the ``Data Sourcing and Processing`` section, our data iterator yields a pair of raw strings.
# We need to convert these string pairs into the batched tensors that can be processed by our ``Seq2Seq`` network
# defined previously. Below we define our collate function that convert batch of raw strings into batch tensors that
# can be fed directly into our model.
#
def sequential_transforms(*transforms):
    # helper function to club together sequential operations
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func


def tensor_transform(token_ids: List[int]):
    # function to add BOS/EOS and create tensor for input sequence indices
    return torch.cat((torch.tensor([vocab.BOS_IDX], dtype=torch.int64),
                      torch.tensor(token_ids, dtype=torch.int64),
                      torch.tensor([vocab.EOS_IDX], dtype=torch.int64)))


def collate_fn(batch):
    # function to collate data samples into batch tesors
    src_batch, tgt_batch = [], []
    for x in batch:
        src_batch.append(vocab.text_transform['src'](x['src'].split()))
        tgt_batch.append(vocab.text_transform['tgt'](x['tgt'].split()))

    src_batch = pad_sequence(src_batch, padding_value=vocab.PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=vocab.PAD_IDX)
    return src_batch, tgt_batch


######################################################################
# Let's define training and evaluation loop that will be called for each
#
def train_epoch(train_data, model, optimizer, loss_fn, cfg: DictConfig, device):

    model.train()
    losses = 0
    train_dataloader = DataLoader(
        train_data, batch_size=cfg.ex.model.batch_size, shuffle=True, collate_fn=collate_fn)

    for src, tgt in train_dataloader:
        # print(" ".join(vocab_transform['src'].lookup_tokens(src.transpose(1,0)[0].numpy())).replace("<pad>", ""))
        # print(" ".join(vocab_transform['tgt'].lookup_tokens(tgt.transpose(1,0)[0].numpy())).replace("<pad>", ""))

        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input, device)

        logits = model(src, tgt_input, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)

        # optimizer.zero_grad() # for Original Adam
        optimizer.optimizer.zero_grad()  # for w/ Noam

        tgt_out = tgt[1:, :]
        loss = loss_fn(
            logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        # print(optimizer)
        losses += loss.item()

    return losses / len(train_dataloader)


def evaluate(dev_data, model, loss_fn, cfg: DictConfig, device):
    model.eval()
    losses = 0

    dev_dataloader = DataLoader(
        dev_data, batch_size=cfg.ex.model.batch_size, shuffle=True, collate_fn=collate_fn)

    for src, tgt in dev_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input, device)

        logits = model(src, tgt_input, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(
            logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(dev_dataloader)


def greedy_decode(model, src, src_mask, max_len, start_symbol, device):
    # function to generate output sequence using greedy algorithm
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)

    q_mt = 1.0
    q_mts = []

    for i in range(max_len-1):
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0), device)
                    .type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(model.softmax(prob), dim=1)

        next_word = next_word.item()
        q_mt *= _.to('cpu').detach().numpy().copy()
        _ = _.to('cpu').detach().numpy().copy()
        # print(_)
        _ = np.log2(_[0])

        q_mts.append('{:.10f}'.format(_))

        ys = torch.cat([ys, torch.ones(1, 1).type_as(
            src.data).fill_(next_word)], dim=0)
        if next_word == vocab.EOS_IDX:
            break

    return ys, q_mts


def translate(model: torch.nn.Module, src_sentence: str, device):
    # actual function to translate input sentence into target language
    model.eval()
    src = vocab.text_transform['src'](src_sentence.split()).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens, q_mts = greedy_decode(
        model,  src, src_mask, max_len=128, start_symbol=vocab.BOS_IDX, device=device)
    tgt_tokens = tgt_tokens.flatten()
    return " ".join(vocab.vocab_transform['tgt'].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", ""), q_mts


@hydra.main(version_base=None, config_path="conf", config_name="config_bestp")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))

    torch.manual_seed(8128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Building Vocabulary
    if os.path.isfile(cfg.ex.vocab.save) == True:
        logger.info('load exsiting vocab file...')
        vocab.vocab_transform = torch.load(cfg.ex.vocab.save)
    else:
        vocab_data = load_vocab_data(cfg)
        vocab_dataloader = DataLoader(vocab_data, batch_size=128, shuffle=True)

        logger.info('compose new vocab file...')
        for ln in ['src', 'tgt']:
            # Create torchtext's Vocab object
            vocab.vocab_transform[ln] = \
                build_vocab_from_iterator(yield_tokens_share(vocab_dataloader, ['src', 'tgt']),
                                          min_freq=1,
                                          specials=vocab.special_symbols,
                                          special_first=True)

            # Set UNK_IDX as the default index. This index is returned when the token is not found.
            # If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
            vocab.vocab_transform[ln].set_default_index(vocab.UNK_IDX)
        torch.save(vocab.vocab_transform, cfg.ex.vocab.save)

    ######################################################################
    # Let's now define the parameters of our model and instantiate the same. Below, we also
    # define our loss function which is the cross-entropy loss and the optmizer used for training.
    #
    train_data, dev_data, test_data = load_train_data(cfg)

    # src and tgt language text transforms to convert raw strings into tensors indices
    for ln in ['src', 'tgt']:
        vocab.text_transform[ln] = sequential_transforms(
            vocab.vocab_transform[ln], tensor_transform)  # Add BOS/EOS and create tensor

    # Training
    if cfg.do_train:
        model = Seq2SeqTransformer(num_encoder_layers=cfg.ex.model.num_encoder_layers,
                                   num_decoder_layers=cfg.ex.model.num_decoder_layers,
                                   emb_size=cfg.ex.model.emb_size,
                                   nhead=cfg.ex.model.nhead,
                                   src_vocab_size=len(
                                       vocab.vocab_transform['src']),
                                   tgt_vocab_size=len(
                                       vocab.vocab_transform['tgt']),
                                   dim_feedforward=cfg.ex.model.ffn_hid_dim)
        logger.debug(model)

        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        model = model.to(device)

        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=vocab.PAD_IDX)

        # optimizer = torch.optim.AdamW(
        #     transformer.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

        optimizer = NoamOpt(512, cfg.ex.model.warmup, torch.optim.Adam(
            model.parameters(), lr=0, betas=(0.9, 0.998), eps=1e-9))

        ######################################################################
        # Now we have all the ingredients to train our model. Let's do it!
        for epoch in range(1, cfg.ex.model.num_epochs + 1):
            start_time = timer()
            train_loss = train_epoch(
                train_data, model, optimizer, loss_fn, cfg, device)
            end_time = timer()
            val_loss = evaluate(dev_data, model, loss_fn, cfg, device)
            torch.save(model.state_dict(),
                       '{}/model_{}.pt'.format(cfg.ex.model.save_dir, epoch))
            logger.info(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Train ppl: {math.exp(train_loss):.3f}, Val loss: {val_loss:.3f}, Val ppl: {math.exp(val_loss):.3f}, "f"Epoch time = {(end_time - start_time):.3f}s")

    # Predict
    if cfg.do_predict:
        model = Seq2SeqTransformer(num_encoder_layers=cfg.ex.model.num_encoder_layers,
                                   num_decoder_layers=cfg.ex.model.num_decoder_layers,
                                   emb_size=cfg.ex.model.emb_size,
                                   nhead=cfg.ex.model.nhead,
                                   src_vocab_size=len(
                                       vocab.vocab_transform['src']),
                                   tgt_vocab_size=len(
                                       vocab.vocab_transform['tgt']),
                                   dim_feedforward=cfg.ex.model.ffn_hid_dim)
        model.load_state_dict(torch.load(cfg.ex.predict.model))
        model.to(device)
        model.eval()

        test_dataloader = DataLoader(test_data, shuffle=False)

        out_list = []
        out_lqmt_list = []  # log q_mt

        for i, batch in enumerate(test_dataloader):
            print('No.{}:{}'.format(i, batch['src']))
            tmp, q_mts = translate(model, batch['src'][0], device)
            out_list.append(tmp + '\n')
            out_lqmt_list.append(q_mts)

        with open(cfg.ex.predict.out, 'w') as f:
            f.writelines(out_list)


        import csv
        with open(cfg.ex.predict.out_lqmt + '.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(out_lqmt_list)


if __name__ == "__main__":
    main()

######################################################################
# References
# ----------
#
# 1. Attention is all you need paper.
#    https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
# 2. The annotated transformer. https://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding
# 3. Pytorch Tutorial. https://pytorch.org/tutorials/beginner/translation_transformer.html#references
