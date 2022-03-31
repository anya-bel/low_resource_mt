cd ../..
source env/bin/activate
cd low_resource_mt/scripts
python3 -u  training.py > >(tee -a logs/log.txt) 2> >(tee -a logs/log_err.txt >&2)
