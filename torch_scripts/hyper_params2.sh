cd ../..
source mtenv/bin/activate
cd low_resource_mt/src
python3 -u  training.py --dataset karelian > >(tee -a logs/karelian_log.txt) 2> >(tee -a logs/karelian_log_err.txt >&2)
python3 -u  training.py --dataset evenki > >(tee -a logs/evenki_log.txt) 2> >(tee -a logs/evenki_log_err.txt >&2)
python3 -u  training.py --dataset yeps > >(tee -a logs/yeps_log.txt) 2> >(tee -a logs/yeps_log_err.txt >&2)
python3 -u  training.py --dataset chuvash > >(tee -a logs/chuvash_log.txt) 2> >(tee -a logs/chuvash_log_err.txt >&2)
