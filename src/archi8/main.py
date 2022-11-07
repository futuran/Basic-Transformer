import os, sys
import random 
from pathlib import Path
from typing import Iterable, List, Final
from timeit import default_timer as timer

import math
import numpy as np
import csv
from tqdm import tqdm
import wandb
wandb.login()

from src.archi8.collation_mask import *
from src.archi8.optimizer import *
from src.archi8.vocabs import *
from src.archi8.load_data import *
from src.archi8.transformer import *
from src.archi8.decode import *
from src.archi8.loss import *

# log関連
from src.util.logger import *
logger = setup_logger(__name__)
logger.debug(sys.argv)

# hydra関連
import hydra
from omegaconf import DictConfig, OmegaConf

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
# from torch.nn.utils.rnn import pad_sequence


def train_epoch(collation_mask: CollationAndMask, train_data, model, optimizer, loss_fn, loss_fn_for_sim, loss_sentweight_fn, cfg: DictConfig, device):

    model.train()
    losses = 0
    train_dataloader = DataLoader(
        train_data, batch_size=cfg.ex.model.batch_size, shuffle=True, collate_fn=collation_mask.collate_fn)

    for src, tgt, src_type_label, src_length_mask, matches in tqdm(train_dataloader):
        # print(" ".join(vocab_transform['src'].lookup_tokens(src.transpose(1,0)[0].numpy())).replace("<pad>", ""))
        # print(" ".join(vocab_transform['tgt'].lookup_tokens(tgt.transpose(1,0)[0].numpy())).replace("<pad>", ""))

        num_sim = int(len(src_type_label) / cfg.ex.model.batch_size)

        src = src.to(device)
        tgt = tgt.to(device)
        src_length_mask = src_length_mask.to(device)
        matches = matches.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = collation_mask.create_mask(
            src, tgt_input, device)

        logits, encoder_outs = model(src, tgt_input, src_type_label, src_length_mask, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)

        # optimizer.zero_grad() # for Original Adam
        optimizer.optimizer.zero_grad()  # for w/ Noam

        tgt_out = tgt[1:, :]

        # src+simのEncoderの出力とsrc+refのCos類似度をCrossEntropyLossの各文に対する損失の重みとする。
        # src+ref→src+sim1→src+sim2の順序で来ることを前提としている
        # TODO encoder_outsはmaskしただけなので0埋めされているので、src文長で切り出すべき
        cos = nn.CosineSimilarity(dim=0)
        pdist = nn.PairwiseDistance(p=2)
        wight_for_each_sent_loss = torch.ones(len(src_type_label), dtype=torch.float32).to(device)
        for i in range(len(src_type_label)):
            if src_type_label[i] == 0:
                ref_dist = encoder_outs[:,i,:]  # 97*512
                wight_for_each_sent_loss[i] = 1
            else:
                current_dist = encoder_outs[:,i,:]  # 97*512
                wight_for_each_sent_loss[i] = cos(ref_dist.reshape(-1), current_dist.reshape(-1)).detach()

        
        softmax = nn.LogSoftmax(dim=0)
        new_wight_for_each_sent_loss = -softmax(wight_for_each_sent_loss)

        # various losses
        loss_orig = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss_sentweight = loss_sentweight_fn(logits, tgt_out, matches)

        loss = loss_sentweight

        loss.backward()
        optimizer.step()
        losses += loss.item()

        wandb.log({
                'Orig Train loss': loss_orig,
                'New Train loss': loss_sentweight,
                })

        # wandb.log({
        #         'Train Orig loss': loss_orig,
        #         'Train Cosw loss': loss,
        #         'Cos of Ref and Sim-1': torch.mean(wight_for_each_sent_loss[1::num_sim]),
        #         'Cos of Ref and Sim-2': torch.mean(wight_for_each_sent_loss[2::num_sim]),
        #         'Cos of Ref and Sim-K': torch.mean(wight_for_each_sent_loss[num_sim-1::num_sim]),
        #         })

    return losses / len(train_dataloader)


def evaluate(collation_mask: CollationAndMask, dev_data, model, loss_fn, loss_fn_for_sim, loss_sentweight_fn, cfg: DictConfig, device):
    model.eval()
    losses = 0

    dev_dataloader = DataLoader(
        dev_data, batch_size=cfg.ex.model.batch_size, shuffle=True, collate_fn=collation_mask.collate_fn)

    for src, tgt, src_type_label, src_length_mask, matches in dev_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)
        src_length_mask = src_length_mask.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = collation_mask.create_mask(
            src, tgt_input, device)

        logits, encoder_outs = model(src, tgt_input, src_type_label, src_length_mask, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(
            logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        # src+simのEncoderの出力をsrc+refの出力に近づける
        # src+ref→src+sim1→src+sim2の順序で来ることを前提としている
        # loss_fn_for_sim = nn.KLDivLoss(reduction="batchmean")
        # loss_for_sim = 0
        # for i in range(len(src_type_label)):
        #     if src_type_label[i] == 0:
        #         ref_dist = torch.squeeze(encoder_outs[:,i,:])
        #     else:
        #         current_dist = torch.squeeze(encoder_outs[:,i,:])
        #         loss_for_sim += loss_fn_for_sim(current_dist, ref_dist)
        new_loss = loss

        losses += new_loss.item()

    return losses / len(dev_dataloader)


def translate(collation_mask: CollationAndMask, vocab: Vocab, model: torch.nn.Module, src_sentence: str, device):
    # actual function to translate input sentence into target language
    model.eval()
    src = vocab.text_transform['src'](src_sentence.split()).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)

    # src_length_mask
    # collation_mask.pyに合わせたコード。
    src_length_mask_batch = []
    src_length = len(src_sentence.split('|')[0].split())
    src_length_mask_batch.append(torch.ones(src_length))
    src_length_mask_batch.append(torch.ones(num_tokens))
    src_length_mask_batch = pad_sequence(src_length_mask_batch, padding_value=0)[:,:-1]

    tgt_tokens, q_mts = greedy_decode(
        collation_mask, 
        vocab, 
        model,  
        src, 
        src_mask,
        src_length_mask_batch,
        max_len=128, 
        start_symbol=vocab.BOS_IDX, device=device)
    tgt_tokens = tgt_tokens.flatten()
    return " ".join(vocab.vocab_transform['tgt'].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", ""), q_mts


@hydra.main(version_base=None, config_path="../../conf",config_name="config")
def main(cfg: DictConfig):

    # os.environ["WANDB_DISABLED"] = "true"
    cwd: Final[Path] = Path(hydra.utils.get_original_cwd())
    Path.mkdir(cwd / Path(cfg.ex.checkpoint), exist_ok=True, parents=True)
    logger.info('\n' + OmegaConf.to_yaml(cfg))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info('project name of wandb: ' + cfg.ex.project_name)
    wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    run = wandb.init(project=cfg.ex.project_name)


    # Building Vocabulary
    vocab = Vocab()
    collation_mask = CollationAndMask(vocab)
    if os.path.isfile(cfg.ex.vocab.save) == True:
        logger.info('load exsiting vocab file...')
        logger.info(cfg.ex.vocab.save)
        vocab.vocab_transform = torch.load(cfg.ex.vocab.save)
    else:
        vocab_data = load_vocab_data(cfg, logger)
        vocab_dataloader = DataLoader(vocab_data, batch_size=128, shuffle=True)

        def yield_tokens_share(data_iter: Iterable, languages: List[str]) -> List[str]:
            for data_samples in data_iter:
                for lang in languages:
                    for data_sample in data_samples[lang]:
                        yield data_sample.split()

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
    train_data, dev_data, test_data = load_train_data(cfg, logger)

    # src and tgt language text transforms to convert raw strings into tensors indices
    for ln in ['src', 'tgt']:
        vocab.text_transform[ln] = collation_mask.sequential_transforms(
            vocab.vocab_transform[ln], collation_mask.tensor_transform)  # Add BOS/EOS and create tensor

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
        logger.info('\n' + str(model))

        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        model = model.to(device)

        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=vocab.PAD_IDX)
        loss_fn_for_sim = nn.KLDivLoss(reduction="batchmean", log_target=True)
        loss_sentweight_fn = SentWeightedCrossEntropyLoss(ignore_index=vocab.PAD_IDX)

        # optimizer = torch.optim.AdamW(
        #     transformer.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

        optimizer = NoamOpt(512, cfg.ex.model.warmup, torch.optim.Adam(
            model.parameters(), lr=0, betas=(0.9, 0.998), eps=1e-9))

        ######################################################################
        # Now we have all the ingredients to train our model. Let's do it!
        wandb.watch(model, loss_fn, log="all", log_freq=10)

        for epoch in range(1, cfg.ex.model.num_epochs + 1):
            start_time = timer()
            train_loss = train_epoch(
                collation_mask, train_data, model, optimizer, loss_fn, loss_fn_for_sim, loss_sentweight_fn, cfg, device)
            end_time = timer()
            val_loss = evaluate(collation_mask, dev_data, model, loss_fn, loss_fn_for_sim, loss_sentweight_fn, cfg, device)
            torch.save(model.state_dict(),
                       Path( cwd / '{}/model_{}.pt'.format(cfg.ex.checkpoint, epoch)))
            logger.info(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Train ppl: {math.exp(train_loss):.3f}, Val loss: {val_loss:.3f}, Val ppl: {math.exp(val_loss):.3f}, "f"Epoch time = {(end_time - start_time):.3f}s")
            wandb.log({
                'Train loss': train_loss,
                'Train ppl': math.exp(train_loss),
                'Val loss': val_loss,
                'Val ppl': math.exp(val_loss)})

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
        if cfg.ex.load_checkpoint != '':
            model.load_state_dict(torch.load(cwd / Path(cfg.ex.load_checkpoint)))
        model.to(device)
        model.eval()

        test_dataloader = DataLoader(test_data, shuffle=False)

        out_list = []
        out_lqmt_list = []  # log q_mt

        for i, batch in enumerate(test_dataloader):
            print('No.{}:{}'.format(i, batch['src']))
            tmp, q_mts = translate(collation_mask, vocab, model, batch['src'][0], device)
            print('     :{}'.format(tmp))
            out_list.append(tmp + '\n')
            out_lqmt_list.append(q_mts)

        with open(cfg.ex.out_txt, 'w') as f:
            f.writelines(out_list)

        with open(cfg.ex.out_lqmt + '.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(out_lqmt_list)


if __name__ == "__main__":
    # if os.getcwd()[-3:] == 'rer':
    #     os.chdir("..")
    
    # Fix seed
    seed: Final[int] = 8128
    random.seed(seed)   # python
    np.random.seed(seed)    # numpy
    torch.manual_seed(seed)  # pytorch
    torch.cuda.manual_seed(seed)    # pytorch, cuda

    main()


######################################################################
# References
# ----------
#
# 1. Attention is all you need paper.
#    https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
# 2. The annotated transformer. https://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding
# 3. Pytorch Tutorial. https://pytorch.org/tutorials/beginner/translation_transformer.html#references
