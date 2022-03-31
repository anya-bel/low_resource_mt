import torch
from nltk import word_tokenize
from torch.utils.data import Dataset


class MTDataset(Dataset):
    def __init__(self, dataset, maxlen, device, token2int_src=None, token2int_tgt=None):
        """
        dataset (pandas.DataFrame) : a dataframe with two columns "original"
                                     (source) and "translation" (target)
        maxlen (int) : maximal length of a sentence
        device : device to use for training
        """
        self.dataset = dataset.dropna().reset_index()
        self.device = device
        self.maxlen = maxlen
        # takes each row of the "original" column, tokenizes and puts all words in the lower case
        self.src_dataset = [list(map(str.lower, word_tokenize(item))) for item in self.dataset['original']]
        self.tgt_dataset = [list(map(str.lower, word_tokenize(item))) for item in self.dataset['translation']]
        if not token2int_src:
            self.token2int_src = self._get_vocab(self.src_dataset)
        else:
            self.token2int_src = token2int_src
        if not token2int_tgt:
            self.token2int_tgt = self._get_vocab(self.tgt_dataset)
        else:
            self.token2int_tgt = token2int_tgt
        self.int2token_src = {digit: word for word, digit in self.token2int_src.items()}
        self.int2token_tgt = {digit: word for word, digit in self.token2int_tgt.items()}

    def __getitem__(self, item):
        src_text = self.src_dataset[item]
        tgt_text = self.tgt_dataset[item]
        transformed_text_src = [self.token2int_src.get(word, 1) for word in src_text][:self.maxlen]
        transformed_text_src = torch.tensor(
            transformed_text_src + [self.token2int_tgt['PAD'] for _ in range(self.maxlen - len(transformed_text_src))],
            dtype=torch.long, device=self.device)

        transformed_text_tgt = [self.token2int_tgt.get(word, 1) for word in tgt_text][:self.maxlen]
        transformed_text_tgt = torch.tensor(
            transformed_text_tgt + [self.token2int_tgt['PAD'] for _ in range(self.maxlen - len(transformed_text_tgt))],
            dtype=torch.long, device=self.device)

        return transformed_text_src, transformed_text_tgt

    def __len__(self):
        return self.dataset.shape[0]

    def _get_vocab(self, sentences):
        token2int = {'PAD': 0, 'UNK': 1, 'SOS': 2}
        for sentence in sentences:
            for word in sentence:
                if word not in token2int:
                    token2int[word] = len(token2int)
        return token2int

    def get_sentence_src(self, idx_sentence):
        return [self.int2token_src[index.item()] for index in idx_sentence]

    def get_sentence_tgt(self, idx_sentence):
        return [self.int2token_tgt[index.item()] for index in idx_sentence]