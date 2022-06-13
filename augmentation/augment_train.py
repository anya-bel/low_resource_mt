import argparse
import os
import requests
import re
from bs4 import BeautifulSoup
from pymorphy2 import MorphAnalyzer
from rnnmorph.predictor import RNNMorphPredictor

predictor = RNNMorphPredictor(language="ru")
morph = MorphAnalyzer()

transforms = {'nom': 'nomn',
              'gen': 'gent',
              'dat': 'datv',
              'acc': 'accs',
              'ins': 'ablt',
              'loc': 'loct',
              'masc': 'masc',
              'fem': 'femn',
              'neut': 'neut'}

def find_synonym(normal_form, gender=None):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    if os.path.exists('synonyms.txt'):
        with open('synonyms.txt') as synonyms_cache:
            synonyms_cache = [x.strip().split() for x in synonyms_cache]
            synonyms_cache = {(form, gen):syn for form, gen, syn in synonyms_cache}
        if (normal_form, str(gender)) in synonyms_cache:
            #print('yes')
            return synonyms_cache[(normal_form, str(gender))]
    try:
        content = requests.get(f'https://kartaslov.ru/синонимы-к-слову/{normal_form}', headers=headers).content
    except ConnectionError:
        print('Sleeping')
        sleep(5)
        try:
            content = requests.get(f'https://kartaslov.ru/синонимы-к-слову/{normal_form}', headers=headers).content
        except ConnectionError:
            print('no')
            return None
    soup = BeautifulSoup(content)
    #print(normal_form)
    if soup.findAll('div', class_="v3-none-text"):
        print(normal_form)
        return None
    synonyms_page = soup.findAll('ul')
    if len(synonyms_page) > 1:
        synonyms = sum([x.text.strip().split(', ') for x in synonyms_page[1].findAll('li')], [])
        synonyms = [x for x in synonyms if ' ' not in x and '\xa0' not in x]
    else:
        return None
    
    # ADD CHECK WITH W2V
    #print(synonyms)
    if not synonyms:
        return None
    if gender:
        try:
            synonym = list(filter(lambda x: gender in x.tag, predictor.predict(synonyms)))[0].word
        except IndexError:
            synonym = None
    else:
        synonym = synonyms[0]
    #print(synonym)
    if synonym is not None:
        with open('synonyms.txt', 'a') as synonyms_cache:
            synonyms_cache.write(f'{normal_form} {gender} {synonym}'+'\n')
    return synonym

def replace_nouns(source_sentence, num=1):
    target_sentence = source_sentence.copy()

    candidate_forms = list(filter(lambda x: x.pos == 'NOUN', predictor.predict(source_sentence)))[:num]
    for form in candidate_forms:
        case_num = re.search('case=(.+)\|.+number=(.+)', form.tag.lower())
        gender = re.search('Gender=(.+)\|', form.tag)
        if gender:
            gender = gender.group(1)
        else:
            continue
        tags = {transforms[case_num.group(1)], case_num.group(2)}
        
        synonym = find_synonym(form.normal_form, gender)
        if synonym is None:
            continue
        inflected_word = list(filter(lambda x : x.tag.POS in ['NOUN'], morph.parse(synonym)))
        if inflected_word:
            #print(inflected_word[0], tags, inflected_word[0].inflect(tags))
            inflected_word = inflected_word[0].inflect(tags)
            if inflected_word:
                inflected_word = inflected_word.word
            else:
                continue
        else:
            continue
        target_sentence[target_sentence.index(form.word)] = inflected_word
    return target_sentence

def replace_adjectives(source_sentence, num=1):
    target_sentence = source_sentence.copy()

    candidate_forms = list(filter(lambda x: x.pos == 'ADJ', predictor.predict(source_sentence)))[:num]
    for form in candidate_forms:
        case = re.search('case=(.{3})\|.+', form.tag.lower())
        num = re.search('number=(.{4})', form.tag.lower())
        #print(form)
        if not num:
            continue
        if num.group(1) == 'plur':
            gender = None
            if case:
                tags = {transforms[case.group(1)], num.group(1)}
            else:
                print(form)
                tags = {num.group(1)}
        else:
            gender = re.search('gender=(.{3,4})\|.+', form.tag.lower()).group(1)
            if not case:
                #print(12, num.group(1), case_num)
                tags = {num.group(1), transforms[gender]}
            else:
                tags = {transforms[case.group(1)], num.group(1), transforms[gender]}
        
        #print(tags)
        synonym = find_synonym(form.normal_form)
        if synonym is None:
            continue
        inflected_word = list(filter(lambda x : x.tag.POS in ['ADJF'], morph.parse(synonym)))
        if inflected_word:
            inflected_word = inflected_word[0].inflect(tags).word
        else:
            continue
        target_sentence[target_sentence.index(form.word)] = inflected_word
    return target_sentence

def replace_adverbs(source_sentence, num=1):
    target_sentence = source_sentence.copy()

    candidate_forms = list(filter(lambda x: x.pos == 'ADV', predictor.predict(source_sentence)))[:num]
    for form in candidate_forms:
        synonym = find_synonym(form.normal_form)
        if synonym is None:
            continue
        else:
            target_sentence[target_sentence.index(form.word)] = synonym
    return target_sentence

def modify_sentences(src, tgt):
    modified_src = []
    modified_tgt = []
    for num, (src_sentence, sentence) in enumerate(zip(src, tgt)):
        #print(sentence)
        modified_sentence = replace_nouns(sentence)
        if modified_sentence == sentence:
            modified_sentence = replace_adverbs(sentence)
            if modified_sentence == sentence:
                modified_sentence = replace_adjectives(sentence)
                if modified_sentence == sentence:
                    continue
        modified_src.append(src_sentence)
        modified_tgt.append(modified_sentence)
        if num % 50 == 0:
            print(f'{num}/{len(src)}')
    return modified_src, modified_tgt


def augment(trainsetpath):
    if not os.path.exists(f'{trainsetpath}_aug'):
        os.mkdir(f'{trainsetpath}_aug')
    with open(f'{trainsetpath}/src-train.txt') as srctrain, open(f'{trainsetpath}/tgt-train.txt') as tgttrain, open(f'{trainsetpath}_aug/src-train.txt', 'w') as aug_srctrain, open(f'{trainsetpath}_aug/tgt-train.txt', 'w') as aug_tgttrain:
            srctrain = [x.strip().split() for x in srctrain]
            tgttrain = [x.strip().split() for x in tgttrain]
            aug_src, aug_tgt = modify_sentences(srctrain, tgttrain)
            for src_line, tgt_line in zip(srctrain, tgttrain):
                aug_srctrain.write(' '.join(src_line)+'\n')
                aug_tgttrain.write(' '.join(tgt_line)+'\n')
            for src_line, tgt_line in zip(aug_src, aug_tgt):
                aug_srctrain.write(' '.join(src_line)+'\n')
                aug_tgttrain.write(' '.join(tgt_line)+'\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Augment target train set')
    parser.add_argument("--trainset_path", type=str, help='path to the train set to augment')
    args = parser.parse_args()
    
    augment(args.trainset_path)

