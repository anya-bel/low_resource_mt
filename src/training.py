import argparse
import nltk
import pandas as pd
import torch
from dataset import MTDataset
from itertools import product
from model import EncoderRNN, DecoderRNN, Seq2Seq
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from utils import training

nltk.download('punkt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(datasetname):
    dataset = pd.read_csv(f'../datasets/{datasetname}.csv')
    test_size = int(dataset.shape[0] * 0.1)
    val_size = int(dataset.shape[0] * 0.1)
    train_size = dataset.shape[0] - test_size - val_size
    train_dataset = MTDataset(dataset[:train_size], 40, device)
    token2int_src = train_dataset.token2int_src
    token2int_tgt = train_dataset.token2int_tgt
    test_dataset = MTDataset(dataset[train_size:train_size+test_size], 40, device, token2int_src, token2int_tgt)
    val_dataset = MTDataset(dataset[train_size+test_size:], 40, device, token2int_src, token2int_tgt)
    print(len(train_dataset), len(test_dataset), len(val_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=4)
    test_dataloader = DataLoader(test_dataset, batch_size=4)
    val_dataloader = DataLoader(val_dataset, batch_size=4)

    lr_range = [0.001, 0.0001, 0.00001]
    hid_range = [50, 100, 150, 200]
    drop_range = [0.2, 0.4, 0.6]
    emb_range = [30, 50, 100]


    with open(f'../models/{datasetname}_log.txt', 'w') as logfile:
        for lr_val, hid_val, drop_val, emb_val in product(lr_range, hid_range, drop_range, emb_range):
            print('Params:', lr_val, hid_val, drop_val, emb_val)
            src_vocab_size = len(train_dataset.token2int_src)
            tgt_vocab_size = len(train_dataset.token2int_tgt)
            emb_hidden_size = emb_val
            hidden_size = hid_val
            dropout_rate = drop_val
            lr = lr_val
            encoder = EncoderRNN(src_vocab_size, hidden_size, emb_hidden_size, dropout_rate)
            decoder = DecoderRNN(tgt_vocab_size, hidden_size, emb_hidden_size, dropout_rate, device=device)
            seq2seq = Seq2Seq(encoder, decoder)
            optimizer = optim.Adam(seq2seq.parameters(), lr=lr)
            loss_fn = nn.CrossEntropyLoss(ignore_index=0)
            trained_model, losstotal, best_bleu, best_epoch = training(seq2seq, train_dataloader, val_dataloader,
                                                                       train_dataset, 1, loss_fn, optimizer, device)
            print(best_bleu, best_epoch)
            logfile.write(f'Params: {lr_val}, {hid_val}, {drop_val}, {emb_val}, BESTBLEU: {best_bleu}, BESTEPOCH: {best_epoch}\n\n')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trainer")
    parser.add_argument("--dataset", type=str, default='ludic', help="name of a dataset to use")
    args = parser.parse_args()
    main(args.dataset)
