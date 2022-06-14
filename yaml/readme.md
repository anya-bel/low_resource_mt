### open-nmt

to start the preprocessing and training:

`onmt_build_vocab -config [YAML FILE NAME].yaml`

`onmt_train -config  [YAML FILE NAME].yaml > >(tee -a[LOG FILE NAME].txt) 2> >(tee -a [ERROR LOG FILE NAME].txt >&2)`