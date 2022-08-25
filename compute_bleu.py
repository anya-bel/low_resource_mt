from nltk.translate import bleu_score

def compute_4_bleu(refs, hyps):
    bleu1 = bleu_score.corpus_bleu(refs, hyps, weights=(1,))
    bleu2 = bleu_score.corpus_bleu(refs, hyps, weights=(0.5, 0.5))
    bleu3 = bleu_score.corpus_bleu(refs, hyps, weights=(1/3, 1/3, 1/3))
    bleu4 = bleu_score.corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25))
    for num, bleu in enumerate((bleu1, bleu2, bleu3, bleu4)):
        print(f'BLEU{num+1} = {round(bleu*100, 1)}/100')
    return bleu1*100, bleu2*100, bleu3*100, bleu4*100
