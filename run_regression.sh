export PYTHONPATH=$(pwd)
cd encoder

# AMASS / BUFF
python3 process_amass_sequence.py
python3 encode_amass_poses.py
cd ..
python3 predict_amass_sequences.py

# Reconstructed / our own dataset
# python3 process_amass_sequence.py
# python3 encode_amass_poses.py
# cd ..
# python3 predict_amass_sequences.py
