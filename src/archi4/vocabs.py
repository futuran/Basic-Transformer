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
        self.SEP_IDX = 4
        # Make sure the tokens are in order of their indices to properly insert them in vocab
        self.special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>', '<sep>']

