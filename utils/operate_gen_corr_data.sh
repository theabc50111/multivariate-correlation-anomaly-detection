#!/bin/bash


# Define the list of items


dataset_list=("--data_implement SP500_20112015_PCA_CLUSTER_MIX_GIVEN_ITEMS")
train_items_setting_list=("--train_items_setting train_train")  # ("--train_items_setting train_train" "--train_items_setting train_all")
corr_type_list=("--corr_type pearson")  # ("--corr_type pearson" "--corr_type cross_corr")
corr_win_list=("--corr_window 50")  # ("--corr_window 10" "--corr_window 30" "--corr_window 50")
corr_str_list=("--corr_stride 1")
custom_discrete_bins_list=("--custom_discrete_bins -1 -0.3 0.3 1")  # ("--custom_discrete_bins -1 0 1" "--custom_discrete_bins -1 -0.25 0.25 1" "--custom_discrete_bins -1 -0.5 0 0.5 1")

# Loop through the list

for dataset in "${dataset_list[@]}"
do
    for train_items_setting in "${train_items_setting_list[@]}"
    do
        for corr_type in "${corr_type_list[@]}"
        do
            for corr_win in "${corr_win_list[@]}"
            do
                for corr_str in "${corr_str_list[@]}"
                do
                    for custom_discrete_bins in "${custom_discrete_bins_list[@]}"
                    do
                        echo "start generate data with $dataset $train_items_setting $corr_type $corr_win $corr_str $custom_discrete_bins"
                        python ./gen_corr_data.py $dataset $train_items_setting $corr_type $corr_win $corr_str $custom_discrete_bins --save_corr_data
                    done
                done
            done
        done
    done
done
