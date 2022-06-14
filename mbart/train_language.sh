#source mtenv/bin/activate

#ex. LANGUAGE=ludic PRE_SRC AND FOLDER_PATH
LANGUAGE="$1"

#ex. PRE_SRC=lu
PRE_SRC="$2"

#ex. FOLDER_PATH=../datasets/$LANGUAGE/ or evenki/
FOLDER_PATH="$3"

PRE_TGT=ru
SRC=${PRE_SRC}_XX
TGT=ru_XX
NAME=$PRE_SRC-$PRE_TGT
DEST=postprocessed_$LANGUAGE
DICT=mbart.cc25/dict.txt
DATA=preprocessed_$LANGUAGE
FAIRSEQ=fairseq/fairseq_cli
TRAIN=train
VALID=valid
TEST=test

python3 preprocess_lang.py --folder_path $FOLDER_PATH --language $LANGUAGE --src $PRE_SRC

python3 ${FAIRSEQ}/preprocess.py \
--source-lang ${SRC} \
--target-lang ${TGT} \
--trainpref ${DATA}/${TRAIN}.spm \
--validpref ${DATA}/${VALID}.spm \
--testpref ${DATA}/${TEST}.spm  \
--destdir ${DEST}/${NAME} \
--thresholdtgt 0 \
--thresholdsrc 0 \
--srcdict ${DICT} \
--tgtdict ${DICT} \
--workers 70

FAIRSEQ=fairseq
PRETRAIN=mbart.cc25/model.pt
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

DATADIR=postprocessed_$LANGUAGE/$NAME
SAVEDIR=checkpoint_$LANGUAGE

python3 ${FAIRSEQ}/train.py ${DATADIR}  --encoder-normalize-before --decoder-normalize-before  --arch mbart_large --task translation_from_pretrained_bart  --source-lang ${SRC} --target-lang ${TGT} --criterion label_smoothed_cross_entropy --label-smoothing 0.2  --dataset-impl mmap --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' --lr-scheduler polynomial_decay --lr 3e-05 --warmup-updates 2500 --total-num-update 40000 --dropout 0.3 --attention-dropout 0.1  --weight-decay 0.0 --max-tokens 768 --update-freq 2 --save-interval 1 --save-interval-updates 8000 --keep-interval-updates 10 --no-epoch-checkpoints --seed 222 --log-format simple --log-interval 2 --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler --restore-file $PRETRAIN --langs $langs --layernorm-embedding  --ddp-backend no_c10d --save-dir ${SAVEDIR}


