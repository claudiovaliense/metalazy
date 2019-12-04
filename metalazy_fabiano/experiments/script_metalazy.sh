
for index in $(seq 0 1); do
    nohup python3.6 experiment_tunning_time.py -train ../../dataset/metalazy/split0/train0_metalazy_scaler -test ../../dataset/metalazy/split0/test0.part.$index > ../../dataset/metalazy/split0/nohup/$index.txt &
done