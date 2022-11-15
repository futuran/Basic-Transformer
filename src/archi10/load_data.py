from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, logger, src_file, tgt_file, match_file, src_transform=None, tgt_transform=None, max_length=-1) -> None:
        logger.info('loading dataset')
        logger.info('src: {}'.format(src_file))
        logger.info('tgt: {}'.format(tgt_file))
        logger.info('max length: {}'.format(max_length))

        self.src, self.tgt = [], []
        self.match = []
        eliminate_count = 0

        with open(src_file, 'r') as fs:
            with open(tgt_file, 'r') as ft:
                with open(match_file, 'r') as fm:
                    if max_length == -1:
                        for ls, lt, lm in zip(fs, ft, fm):
                            self.src.append(ls.strip())
                            self.tgt.append(lt.strip())
                            self.match.append(lm.strip())
                    else:
                        for ls, lt, lm in zip(fs, ft, fm):
                            if len(ls.strip().split()) <= max_length and len(lt.strip().split()) <= max_length:
                                self.src.append(ls.strip())
                                self.tgt.append(lt.strip())
                                self.match.append(lm.strip())

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
        match = self.match[idx]
        sample = {"src": src, "tgt": tgt, "match": match}
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
        max_length=256
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
