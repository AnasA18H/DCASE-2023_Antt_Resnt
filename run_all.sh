#!/bin/bash

echo "Starting full training and evaluation for Task 3 (Attn-ResAE)"
source /home/x/Proton/Assignments/DeepLearning_Assignment#2_3/Task_2/DCASE_2023/dcase2023_task2_baseline_ae/venv/bin/activate

MACHINES="DCASE2023T2bearing DCASE2023T2fan DCASE2023T2gearbox DCASE2023T2slider DCASE2023T2ToyCar DCASE2023T2ToyTrain DCASE2023T2valve"

# 1. Train all models (100 epochs)
for machine_type in $MACHINES; do
    echo "============ Training: ${machine_type} ============"
    python3 train.py --dataset=${machine_type} --dev --use_ids 0 --use_cuda=False --train_only -tag="id(0_)" --mono=True --epochs 100
done

# 2. Test all models (MSE and Mahalanobis)
for machine_type in $MACHINES; do
    echo "============ Testing MSE: ${machine_type} ============"
    python3 train.py --dataset=${machine_type} --dev --use_ids 0 --use_cuda=False --test_only --score=MSE -tag="id(0_)" --mono=True
    
    echo "============ Testing MAHALA: ${machine_type} ============"
    python3 train.py --dataset=${machine_type} --dev --use_ids 0 --use_cuda=False --test_only --score=MAHALA -tag="id(0_)" --mono=True
done

# 3. Summarize results
echo "============ Summarizing Results ============"
bash 03_summarize_results.sh DCASE2023T2 -d

echo "All Done! Results are in results/dev_data/baseline/summarize/DCASE2023T2/"
