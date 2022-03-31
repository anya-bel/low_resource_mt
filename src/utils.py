import torch
from nltk.translate import bleu_score
from tqdm.auto import tqdm


def evaluate_val(model, dataset, val_dataloader):
    references = []
    hypotheses = []
    for el in val_dataloader:
        for tgt_sent, hyp_sent in zip(el[1], model(el[0], 41)[1:].argmax(0)):
            reference = dataset.get_sentence_tgt(tgt_sent)
            hypothesis = dataset.get_sentence_tgt(hyp_sent)
            references.append([reference])
            hypotheses.append(hypothesis)
    return bleu_score.corpus_bleu(references, hypotheses)


def training(model, train_dataloader, val_dataloader, dataset, num_epochs, loss_fn, optimizer, device,
             verbose=True):
    model.train()
    model.to(device)
    loss_total = []
    best_bleu = 0
    best_epoch = 0

    for epoch in range(num_epochs):

        loss_current_epoch = 0
        t = tqdm(train_dataloader)
        for i, (src_sentences, tgt_sentences) in enumerate(t):
            src, tgt = src_sentences.to(device), tgt_sentences.to(device)
            tgt_len = tgt.shape[1]

            optimizer.zero_grad()

            pred_probas = model(src, tgt_len + 1)
            output_dim = pred_probas.shape[-1]
            pred_probas = pred_probas[1:].view(-1, tgt_len, output_dim)
            pred_probas = pred_probas.view(-1, output_dim)
            tgt = tgt.view(-1)

            loss = loss_fn(pred_probas, tgt)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_current_epoch += loss.item()
            t.set_description(f'loss: {round(float(loss), 3)}')
            t.refresh()

        loss_total.append(loss_current_epoch)
        if verbose:
            with torch.no_grad():
                model.eval()
                try:
                    bleu = evaluate_val(model, dataset, val_dataloader)
                except ZeroDivisionError:
                    bleu = 0
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss_current_epoch), 'BLEU:', bleu)
                if bleu > best_bleu:
                    best_bleu = bleu
                    best_epoch = epoch
                model.train()
        torch.save(model.state_dict(), '../models/seq2seq_model.pt')

    return model, loss_total, best_bleu, best_epoch
