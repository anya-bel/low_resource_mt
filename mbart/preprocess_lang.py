import argparse
import os
import shutil
import sentencepiece as spm


def sp_process(data_path, part, lang_code, s_model):
    with open(f'{data_path}/{part}.spm.{lang_code}', 'w') as out, open(f'{data_path}/{part}.{lang_code}') as inp:
        for sent in inp.readlines():
            conv_sent = s_model.encode(sent, out_type=str)
            out.write(' '.join(conv_sent)+'\n')

def preprocess_data(folder_path, language, src, tgt, sp_model='mbart.cc25/sentence.bpe.model'):
    s_model = spm.SentencePieceProcessor(model_file=sp_model)
    
    data_path = f'preprocessed_{language}'

    files = [x for x in os.listdir(folder_path) if x.endswith('txt')]
    srctrain_path = [x for x in files if x.startswith('src-train')][0]
    tgttrain_path = [x for x in files if x.startswith('tgt-train')][0]
    srcval_path = [x for x in files if x.startswith('src-val')][0]
    tgtval_path = [x for x in files if x.startswith('tgt-val')][0]
    srctest_path = [x for x in files if x.startswith('src-test')][0]
    tgttest_path = [x for x in files if x.startswith('tgt-test')][0]
    print('Identified files: ', srctrain_path, tgttrain_path, srcval_path, tgtval_path, srctest_path, tgttest_path)
    
    

    file_mapping = {
        f'train.{src}_XX': srctrain_path,
        f'train.{tgt}_XX': tgttrain_path,
        f'valid.{src}_XX': srcval_path,
        f'valid.{tgt}_XX': tgtval_path,
        f'test.{src}_XX': srctest_path,
        f'test.{tgt}_XX': tgttest_path,

    }
    if not os.path.isdir(data_path):
        os.mkdir(f'preprocessed_{language}')
    for k, v in file_mapping.items():
        shutil.copyfile(os.path.join(folder_path, v), f'{data_path}/{k}')
    print('Finished copy')
    
    sp_process(data_path, 'train', f'{src}_XX', s_model)
    print(f'Sentencepiece for src-train created')
    sp_process(data_path, 'train', f'{tgt}_XX', s_model)
    print(f'Sentencepiece for tgt-train created')
    sp_process(data_path, 'valid', f'{src}_XX', s_model)
    print(f'Sentencepiece for src-val created')
    sp_process(data_path, 'valid', f'{tgt}_XX', s_model)
    print(f'Sentencepiece for tgt-val created')
    sp_process(data_path, 'test', f'{src}_XX', s_model)
    print(f'Sentencepiece for src-test created')
    sp_process(data_path, 'test', f'{tgt}_XX', s_model)
    print(f'Sentencepiece for tgt-test created')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess files with sentencepiece model')
    parser.add_argument("--folder_path", type=str, help='path to the test set')
    parser.add_argument("--language", type=str, help='src language for which the dataset is created')
    parser.add_argument("--src", type=str, help='src language abbreviation')
    parser.add_argument("--tgt", type=str, help='tgt language abbreviation (default: ru)', default='ru')
    parser.add_argument("--sp_model", type=str, help='path to the sentencepiece model to use (default: mbart.cc25/sentence.bpe.model)', 
                       default='mbart.cc25/sentence.bpe.model')
    
    args = parser.parse_args()
    
    preprocess_data(args.folder_path, args.language, args.src, args.tgt, args.sp_model)

    
