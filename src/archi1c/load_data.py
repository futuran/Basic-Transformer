from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, logger, src_file, tgt_file, match_file, src_transform=None, tgt_transform=None, max_length=-1) -> None:
        logger.info('loading dataset')
        logger.info('src: {}'.format(src_file))
        logger.info('tgt: {}'.format(tgt_file))
        logger.info('max length: {}'.format(max_length))    # src, tgt, simの各最大長を指定。

        self.src, self.tgt, self.match = [], [], []

        with open(src_file, 'r') as fs:
            with open(tgt_file, 'r') as ft:
                with open(match_file, 'r') as fm:
                    for ls, lt, lm in zip(fs, ft, fm):
                        self.src.append(' '.join(ls.strip().split()[:max_length]))
                        self.tgt.append(' '.join(lt.strip().split()[:max_length]))
                        self.match.append(lm.strip())

        assert len(self.src) == len(self.tgt)

        logger.info('num of dataset: {}'.format(len(self.src)))

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
        match = self.match[idx]
        sample = {"src": src, "tgt": tgt, "match": match}
        return sample


def load_vocab_data(cfg: DictConfig, logger):
    vocab_data = TranslationDataset(
        logger=logger,
        src_file=cfg.ex.vocab.src,
        tgt_file=cfg.ex.vocab.tgt,
        match_file=cfg.ex.vocab.match,
    )
    return vocab_data


def load_train_data(cfg: DictConfig, logger):
    train_data = TranslationDataset(
        logger=logger,
        src_file=cfg.ex.dataset.train.src,
        tgt_file=cfg.ex.dataset.train.tgt,
        match_file=cfg.ex.dataset.train.match,
        max_length=256
    )
    dev_data = TranslationDataset(
        logger=logger,
        src_file=cfg.ex.dataset.dev.src,
        tgt_file=cfg.ex.dataset.dev.tgt,
        match_file=cfg.ex.dataset.dev.match,
        max_length=256
    )
    test_data = TranslationDataset(
        logger=logger,
        src_file=cfg.ex.dataset.test.src,
        tgt_file=cfg.ex.dataset.test.tgt,
        match_file=cfg.ex.dataset.test.match,
    )

    return train_data, dev_data, test_data
