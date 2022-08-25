# Rule-based augmentation with synonyms 

Files:

```bash
├── synonyms augmentation.ipynb # examples of the augmented sentences
├── grammem_prediction.ipynb # work in progress, an attempt to predict grammatical labels for suffixes of words
├── synonyms.txt # cache for words and synonyms with their grammatical tags
├── augment_train.py # augmentation script
```



To start the augmentation run:

`python3 augment_train.py --trainset_path low_resource_mt/LANGUAGE`