### Sequence-to-Sequence

To start the preprocessing and training:

`onmt_build_vocab -config path/to/the/yaml/file.yaml`

`onmt_train -config  path/to/the/yaml/file.yaml > >(tee -a[LOG FILE NAME].txt) 2> >(tee -a [ERROR LOG FILE NAME].txt >&2)`

> Notes:
> to start the training of the models that use word2vec, glove or tfidf, you need to have the embeddings files. the examples of training are given in the ```embeddings_training``` folder