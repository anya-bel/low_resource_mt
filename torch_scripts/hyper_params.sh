cd ../..
source mtenv/bin/activate
cd low_resource_mt/src
python3 -u  training.py --dataset ket > >(tee -a logs/ket_log.txt) 2> >(tee -a logs/ket_log_err.txt >&2)
python3 -u  training.py --dataset chukchi > >(tee -a logs/chukchi_log.txt) 2> >(tee -a logs/chukchi_log_err.txt >&2)
python3 -u  training.py --dataset ludic > >(tee -a logs/ludic_log.txt) 2> >(tee -a logs/ludic_log_err.txt >&2)
python3 -u  training.py --dataset selkup > >(tee -a logs/selkup_log.txt) 2> >(tee -a logs/selkup_log_err.txt >&2)
python3 -u  training.py --dataset karelian > >(tee -a logs/karelian_log.txt) 2> >(tee -a logs/karelian_log_err.txt >&2)
python3 -u  training.py --dataset evenki > >(tee -a logs/evenki_log.txt) 2> >(tee -a logs/evenki_log_err.txt >&2)
python3 -u  training.py --dataset yeps > >(tee -a logs/yeps_log.txt) 2> >(tee -a logs/yeps_log_err.txt >&2)
python3 -u  training.py --dataset chuvash > >(tee -a logs/chuvash_log.txt) 2> >(tee -a logs/chuvash_log_err.txt >&2)