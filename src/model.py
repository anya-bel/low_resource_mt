import torch
import torch.nn as nn


class EncoderGRU(nn.Module):
    def __init__(self, src_vocab_size, hidden_size, emb_hidden_size, dropout_rate):
        super(EncoderRNN, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.hidden_size = hidden_size
        self.emb_hidden_size = emb_hidden_size
        self.dropout_rate = dropout_rate
        self.embedding = nn.Embedding(self.src_vocab_size, self.emb_hidden_size)
        self.gru = nn.GRU(self.emb_hidden_size, self.hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, sentence):
        sent_emb = self.dropout(self.embedding(sentence))
        output, hidden = self.gru(sent_emb)
        return output, hidden


class DecoderGRU(nn.Module):
    def __init__(self, tgt_vocab_size, hidden_size, emb_hidden_size, dropout_rate, device):
        super(DecoderRNN, self).__init__()
        self.tgt_vocab_size = tgt_vocab_size
        self.hidden_size = hidden_size
        self.emb_hidden_size = emb_hidden_size
        self.dropout_rate = dropout_rate
        self.device = device
        self.embedding = nn.Embedding(self.tgt_vocab_size, self.emb_hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.gru = nn.GRU(self.emb_hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.tgt_vocab_size)

    def forward(self, word, hidden):
        batch_size = word.shape[0]
        word = word.to(self.device)
        word_emb = self.embedding(word).view(batch_size, 1, -1).to(self.device)
        word_emb = self.dropout(word_emb).to(self.device)
        output, hidden = self.gru(word_emb, hidden)
        output = self.out(output.view(batch_size, -1))
        return output, hidden


class Seq2SeqGRU(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.trg_vocab_size = decoder.tgt_vocab_size
        self.device = decoder.device

    def forward(self, src_sentence, tgt_len):
        batch_size = src_sentence.shape[0]
        pred_probas = torch.zeros(tgt_len, batch_size, self.trg_vocab_size).to(self.device)
        pred_probas[0, :, 2] = 1
        prev_word = torch.ones(batch_size).int() * 2
        prev_word.to(self.device)

        output, hidden = self.encoder(src_sentence)

        for t in range(1, tgt_len):
            pred_proba_t, hidden = self.decoder(prev_word, hidden)
            pred_probas[t, :, :] = pred_proba_t
            prev_word = pred_proba_t.argmax(-1)

        return pred_probas

class EncoderLSTM(nn.Module):
    def __init__(self, src_vocab_size, hidden_size, emb_hidden_size, dropout_rate, n_layers=1):
        super(EncoderLSTM, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.hidden_size = hidden_size
        self.emb_hidden_size = emb_hidden_size
        self.dropout_rate = dropout_rate
        self.n_layers = n_layers
        self.embedding = nn.Embedding(self.src_vocab_size, self.emb_hidden_size)
        self.gru = nn.LSTM(self.emb_hidden_size, self.hidden_size, self.n_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, sentence):
        sent_emb = self.dropout(self.embedding(sentence))
        output, (hidden, cell) = self.gru(sent_emb)
        return output, hidden, cell


class DecoderLSTM(nn.Module):
    def __init__(self, tgt_vocab_size, hidden_size, emb_hidden_size, dropout_rate, device, n_layers=1):
        super(DecoderLSTM, self).__init__()
        self.tgt_vocab_size = tgt_vocab_size
        self.hidden_size = hidden_size
        self.emb_hidden_size = emb_hidden_size
        self.dropout_rate = dropout_rate
        self.n_layers = n_layers
        self.embedding = nn.Embedding(self.tgt_vocab_size, self.emb_hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.device = device
        self.gru = nn.LSTM(self.emb_hidden_size, self.hidden_size, self.n_layers, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.tgt_vocab_size)

    def forward(self, word, hidden, cell):
        batch_size = word.shape[0]
        word = word.to(self.device)
        word_emb = self.embedding(word).view(batch_size, 1, -1).to(self.device)
        word_emb = self.dropout(word_emb).to(self.device)
        output, (hidden, cell) = self.gru(word_emb, (hidden, cell))
        output = self.out(output.view(batch_size, -1))
        return output, hidden, cell


class Seq2SeqLSTM(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2SeqLSTM, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = decoder.device
        self.trg_vocab_size = decoder.tgt_vocab_size

    def forward(self, src_sentence, tgt_len):
        batch_size = src_sentence.shape[0]
        pred_probas = torch.zeros(tgt_len, batch_size, self.trg_vocab_size).to(self.device)
        pred_probas[0, :, 2] = 1
        prev_word = torch.ones(batch_size).int() * 2

        output, hidden, cell = self.encoder(src_sentence)

        for t in range(1, tgt_len):
            pred_proba_t, hidden, cell = self.decoder(prev_word, hidden, cell)
            pred_probas[t, :, :] = pred_proba_t
            prev_word = pred_proba_t.argmax(-1)

        return pred_probas