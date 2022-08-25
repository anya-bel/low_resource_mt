# Machine Translation for low-resource languages
This repository is dedicated to the machine translation from 7 low-resource language of Russia to Russian. 

Each folder gathers all experiments made with one of the models (smt, seq2seq, mbart, gpt) or all experiments related to some kind of data augmentation

#### General prerequisites:

`python3 -m venv mtenv`
`git clone https://github.com/anya-bel/low_resource_mt.git`
`cd low_resource_mt`
`pip3 install -r requirements.txt`

#### Repostiory's structure:

```bash
├── datasets # datasets of low-resource languages and their variations
├── gpt # gpt models used for translation and augmentation
├── manual_augmentation # manual annotation and addition of the parallel sentences to the Evenki dataset
├── mbart # mBART experiments
├── rulebased_augmentation # synonyms replacement augmenation
├── smt # phrase-based model
├── seq2seq # experiments with sequence-to-sequence
├── torch_models # old part of the repository 
├── torch_scripts # old part of the repository
├── torch_seq2seq # old part of the repository
├── app.py # application for the manual annotation
├── compute_bleu.py # module with the compute_4_bleu function which is used for BLEU calculation
```

#### Checklists for models:

### Neural MT

##### Simple:

- [x] Evenki
- [x] Ludic
- [x] Karelian
- [x] Selkup
- [x] Ket
- [x] Chukchi
- [x] Veps

##### TfIdf:

- [x] Evenki
- [x] Ludic
- [x] Karelian
- [x] Selkup
- [x] Ket
- [x] Chukchi
- [x] Veps

##### Word2Vec:

- [x] Ludic
- [x] Karelian
- [x] Veps

##### GloVe:

- [x] Ludic
- [x] Karelian
- [x] Veps

##### Multilingual:

- [x] Evenki
- [x] Ludic
- [x] Karelian
- [x] Selkup
- [x] Ket
- [x] Chukchi
- [x] Veps

##### Character-based:

- [x] Evenki

### Statistical MT

- [x] Evenki
- [x] Ludic
- [x] Karelian
- [x] Selkup
- [x] Ket
- [x] Chukchi
- [x] Veps

### Augmentation

##### Target augmentation with Russian synonyms:

- [x] Evenki
- [x] Ludic
- [x] Karelian
- [x] Selkup
- [x] Ket
- [x] Chukchi
- [x] Veps

##### Manual augmentation:

- [x] Evenki

##### Training automatic declension model for the source language:

- [ ] Evenki

### mBART

- [x] Evenki
- [x] Ludic
- [x] Karelian
- [x] Selkup
- [x] Ket
- [x] Chukchi
- [x] Veps

### mT5

- [x] Evenki

### ruT5

- [x] Evenki

### mGPT

- [x] Evenki

### ruGPT

- [x] Evenki
- [x] Ludic
- [x] Karelian
- [x] Selkup
- [x] Ket
- [x] Chukchi
- [x] Veps

### Finnish GPT

- [x] Ludic
- [x] Karelian
- [x] Veps

### SeqGAN

- [x] Evenki

### mBART + ruGPT augmentation

- [x] Evenki
- [x] Ludic
- [x] Karelian
- [x] Selkup
- [x] Ket
- [x] Chukchi
- [x] Veps



To start the scripts in the old torch part of the repository: 

```shell
cd scripts
bash hyper_params.sh
```


### 
