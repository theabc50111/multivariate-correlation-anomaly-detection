#!/bin/bash

for ((lag_period=2;lag_period<200;lag_period++))
do
    echo '=============== Start execute ==============='
    echo "utils/gen_synthetic_data.py --data_type t_shift_collection multi_collections  --n_collections 2 --lag_period $lag_period --dim 5 --noise_scale 0"
    python /workspace/multivariate-correlation-anomaly-detection/utils/gen_synthetic_data.py --data_type t_shift_collection multi_collections  --n_collections 2 --lag_period $lag_period --dim 5 --noise_scale 0 --save_data
    echo '===============      End      ==============='
done
