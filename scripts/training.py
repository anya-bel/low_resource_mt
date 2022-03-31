from itertools import product

import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from dataset import MTDataset
from model import EncoderRNN, DecoderRNN, Seq2Seq
from utils import training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = pd.read_csv('../datasets/ludic.csv')

train_ludic_dataset = MTDataset(dataset[:1000], 40, device)
token2int_src = train_ludic_dataset.token2int_src
token2int_tgt = train_ludic_dataset.token2int_tgt
test_ludic_dataset = MTDataset(dataset[1000:1190], 40, device, token2int_src, token2int_tgt)
val_ludic_dataset = MTDataset(dataset[1190:-1], 40, device, token2int_src, token2int_tgt)
len(train_ludic_dataset), len(test_ludic_dataset), len(val_ludic_dataset)

train_ludic_dataloader = DataLoader(train_ludic_dataset, batch_size=4)
test_ludic_dataloader = DataLoader(test_ludic_dataset, batch_size=4)
val_ludic_dataloader = DataLoader(val_ludic_dataset, batch_size=4)

lr_range = [0.001, 0.0001, 0.00001]
hid_range = [50, 100, 150, 200]
drop_range = [0.2, 0.4, 0.6]
emb_range = [30, 50, 100]
with open('../models/log.txt', 'w') as logfile:
    for lr_val, hid_val, drop_val, emb_val in product(lr_range, hid_range, drop_range, emb_range):
        print('Params:', lr_val, hid_val, drop_val, emb_val)
        src_vocab_size = len(train_ludic_dataset.token2int_src)
        tgt_vocab_size = len(train_ludic_dataset.token2int_tgt)
        emb_hidden_size = emb_val
        hidden_size = hid_val
        dropout_rate = drop_val
        lr = lr_val
        encoder = EncoderRNN(src_vocab_size, hidden_size, emb_hidden_size, dropout_rate)
        decoder = DecoderRNN(tgt_vocab_size, hidden_size, emb_hidden_size, dropout_rate)
        seq2seq = Seq2Seq(encoder, decoder)
        optimizer = optim.Adam(seq2seq.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        trained_model, losstotal, best_bleu, best_epoch = training(seq2seq, train_ludic_dataloader, val_ludic_dataloader,
                                                                   train_ludic_dataset, 10, loss_fn, optimizer)
        print(best_bleu, best_epoch)
        logfile.write(f'Params: {lr_val}, {hid_val}, {drop_val}, {emb_val}\n')
        logfile.write(f'BLEU: {best_bleu}, EPOCH: {best_epoch}\n\n')