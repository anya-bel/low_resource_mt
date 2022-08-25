# Statistical machine translation

To start the experiments with the phrase-based model you firstly need to install Moses (http://www2.statmt.org/moses/)

After this, in the main Moses folder (which will be created during the tutorial), move the folder with the languages of interest, for example, tazke the folder `evenki` from `datasets/evenki`

Then you can copy the `training.sh` (or `first_training.sh`) script to this folder and start the training:

`sh first_training.sh LANGUAGE PRE_SRC` if this is the first time you train or:

`sh training.sh LANGUAGE PRE_SRC PRE_LANGUAGE` if you already have trained models in the folder

`LANGUAGE` is the current language,  `PRE_SRC` is the short name for the source language, `PRE_LANGUAGE` is the previous language on which the model was trained

##### Example:

`sh training.sh ludic lu chukchi`