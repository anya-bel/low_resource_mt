echo $PWD 
cd ..
echo $PWD
source mtenv/bin/activate
echo $PWD
cd low_resource_mt/src
python3 -u  training_lstm.py --dataset ket > >(tee -a logs/lstm_ket_log.txt) 2> >(tee -a logs/lstm_ket_log_err.txt >&2)
python3 -u  training_lstm.py --dataset chukchi > >(tee -a logs/lstm_chukchi_log.txt) 2> >(tee -a logs/lstm_chukchi_log_err.txt >&2)
python3 -u  training_lstm.py --dataset ludic > >(tee -a logs/lstm_ludic_log.txt) 2> >(tee -a logs/lstm_ludic_log_err.txt >&2)
python3 -u  training_lstm.py --dataset selkup > >(tee -a logs/lstm_selkup_log.txt) 2> >(tee -a logs/lstm_selkup_log_err.txt >&2)
python3 -u  training_lstm.py --dataset karelian > >(tee -a logs/lstm_karelian_log.txt) 2> >(tee -a logs/lstm_karelian_log_err.txt >&2)
python3 -u  training_lstm.py --dataset evenki > >(tee -a logs/lstm_evenki_log.txt) 2> >(tee -a logs/lstm_evenki_log_err.txt >&2)
python3 -u  training_lstm.py --dataset yeps > >(tee -a logs/lstm_yeps_log.txt) 2> >(tee -a logs/lstm_yeps_log_err.txt >&2)
python3 -u  training_lstm.py --dataset chuvash > >(tee -a logs/lstm_chuvash_log.txt) 2> >(tee -a logs/lstm_chuvash_log_err.txt >&2)
