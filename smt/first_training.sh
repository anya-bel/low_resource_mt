#ex. sh training.sh ludic lu


LANGUAGE="$1"
PRE_SRC="$2"


cd ${LANGUAGE}
mv src-train.txt ${LANGUAGE}.${PRE_SRC}-ru.${PRE_SRC}
mv tgt-train.txt ${LANGUAGE}.${PRE_SRC}-ru.ru
mv src-val.txt ${LANGUAGE}.dev.${PRE_SRC}-ru.${PRE_SRC}
mv tgt-val.txt ${LANGUAGE}.dev.${PRE_SRC}-ru.ru
cd ..

mkdir lm 

cd lm 

 
 ../mosesdecoder/bin/lmplz -o 3 <../${LANGUAGE}/${LANGUAGE}.${PRE_SRC}-ru.ru > ${LANGUAGE}.${PRE_SRC}-ru.arpa.ru


../mosesdecoder/bin/build_binary \
   ${LANGUAGE}.${PRE_SRC}-ru.arpa.ru \
   ${LANGUAGE}.${PRE_SRC}-ru.blm.ru

head -n 1  ../${LANGUAGE}/src-test.txt | ../mosesdecoder/bin/query ${LANGUAGE}.${PRE_SRC}-ru.blm.ru

cd ..        

cd ${LANGUAGE}      

sudo chmod 755 *.*

cd ..    

nohup nice mosesdecoder/scripts/training/train-model.perl -root-dir train \
 -corpus ${LANGUAGE}/${LANGUAGE}.${PRE_SRC}-ru                             \
 -f ${PRE_SRC} -e ru -alignment grow-diag-final-and -reordering msd-bidirectional-fe \
 -lm 0:3:/Users/anna/Documents/GitHub/moses/lm/${LANGUAGE}.${PRE_SRC}-ru.blm.ru:8                          \
 -external-bin-dir giza-pp/GIZA++-v2 >& training_${LANGUAGE}.out 

echo "moses model trained"

nohup nice mosesdecoder/scripts/training/mert-moses.pl \
  ${LANGUAGE}/${LANGUAGE}.dev.${PRE_SRC}-ru.${PRE_SRC}  ${LANGUAGE}/${LANGUAGE}.dev.${PRE_SRC}-ru.ru \
  mosesdecoder/bin/moses train/model/moses.ini --mertdir /Users/anna/Documents/GitHub/moses/mosesdecoder/bin/ \
  &> mert_${LANGUAGE}.out 

echo "moses model fine-tuned"


 nohup nice mosesdecoder/bin/moses            \
   -f train/model/moses.ini   \
   < ${LANGUAGE}/src-test.txt                \
   > ${LANGUAGE}.test.translated.ru         \
   2> ${LANGUAGE}test.out

echo "translations predicted"

rsync --progress -r  ${LANGUAGE}.test.translated.ru anmosolova@access.grid5000.fr:nancy/     

