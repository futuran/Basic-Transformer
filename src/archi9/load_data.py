from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
import random


class TranslationDataset(Dataset):
    def __init__(self, logger, src_file, tgt_file, match_file, src_transform=None, tgt_transform=None, max_length=-1, shuffle=False) -> None:
        logger.info('loading dataset')
        logger.info('src: {}'.format(src_file))
        logger.info('tgt: {}'.format(tgt_file))
        logger.info('max length: {}'.format(max_length))

        self.src, self.tgt = [], []
        self.match = []
        self.sim_rank = []
        eliminate_count = 0

        with open(src_file, 'r') as fs, open(tgt_file, 'r') as ft, open(match_file, 'r') as fm:
            if max_length == -1:
                for ls, lt, lm in zip(fs, ft, fm):
                    lss = ls.strip().split(' | ')
                    lt = lt.strip()
                    lm = [x.split()[1] for x in lm.strip().split(' ||| ')]

                    # src+ref
                    self.src.append(' | '.join([lss[0], lt]))
                    self.sim_rank.append(0)
                    self.tgt.append(lt)
                    self.match.append(float(1))

                    # src+sim
                    for i in range(1, len(lss)):
                        self.src.append(' | '.join([lss[0], lss[i]]))
                        self.sim_rank.append(0)
                        self.tgt.append(lt)
                        self.match.append(float(lm[i]))
            else:
                for ls, lt, lm in zip(fs, ft, fm):
                    if len(ls.strip().split()) <= max_length and len(lt.strip().split()) <= max_length:
                        eliminate_count += 1
                        lss = ls.strip().split(' | ')
                        lt = lt.strip()
                        lm = [x.split()[1] for x in lm.strip().split(' ||| ')]

                        # src+ref
                        self.src.append(' | '.join([lss[0], lt]))
                        self.sim_rank.append(0)
                        self.tgt.append(lt)
                        self.match.append(float(1))

                        # src+sim
                        for i in range(1, len(lss)):
                            self.src.append(' | '.join([lss[0], lss[i]]))
                            self.sim_rank.append(i)
                            self.tgt.append(lt)
                            self.match.append(float(lm[i]))

        assert len(self.src) == len(self.tgt)

        logger.info('num of dataset: {}'.format(len(self.src)))
        logger.info('eliminated sents: {}'.format(eliminate_count))

        self.src_transform = src_transform
        self.tgt_transform = tgt_transform

        if shuffle == True:

            random.seed(8128)
            zipped = list(zip(self.src, self.tgt, self.sim_rank, self.match))
            random.shuffle(zipped)
            self.src, self.tgt, self.sim_rank, self.match = zip(*zipped)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src = self.src[idx]
        if self.src_transform:
            src = self.src_transform(src)
        sim_rank = self.sim_rank[idx]
        tgt = self.tgt[idx]
        if self.tgt_transform:
            tgt = self.tgt_transform(tgt)
        match = self.match[idx]
        sample = {"src": src, "tgt": tgt, "sim_rank": sim_rank, "match": match}
        return sample


def load_vocab_data(cfg: DictConfig, logger):
    vocab_data = TranslationDataset(
        logger=logger,
        src_file=cfg.ex.vocab.src,
        tgt_file=cfg.ex.vocab.tgt,
    )
    return vocab_data


def load_train_data(cfg: DictConfig, logger):
    train_data = TranslationDataset(
        logger=logger,
        src_file=cfg.ex.dataset.train.src,
        tgt_file=cfg.ex.dataset.train.tgt,
        match_file=cfg.ex.dataset.train.match,
        max_length=256,
        shuffle=True
    )
    dev_data = TranslationDataset(
        logger=logger,
        src_file=cfg.ex.dataset.dev.src,
        tgt_file=cfg.ex.dataset.dev.tgt,
        match_file=cfg.ex.dataset.train.match,
        max_length=256
    )
    test_data = TranslationDataset(
        logger=logger,
        src_file=cfg.ex.dataset.test.src,
        tgt_file=cfg.ex.dataset.test.tgt,
        match_file=cfg.ex.dataset.train.match,
    )

    return train_data, dev_data, test_data
