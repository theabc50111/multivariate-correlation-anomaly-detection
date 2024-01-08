#!/bin/bash

#EXPORT TZ=Asia/Taipei

ARGUMENT_LIST=(
  "log_suffix"
  "data_implement"
  "batch_size"
  "n_folds"
  "tr_epochs"
  "train_model"
  "save_model"
  "seq_len"
  "corr_type"
  "corr_window"
  "corr_stride"
  "model_input_cus_bins"
  "target_mats_path"
  "cuda_device"
  "learning_rate"
  "weight_decay"
  "use_optim_scheduler"
  "drop_pos"
  "drop_p"
  "gru_l"
  "gru_h"
  "input_idx"
  "kernel_size"
  "kernel_pad"
  "attn_num_heads"
  "use_weighted_loss"
  "tol_edge_acc_loss_atol"
  "custom_indices_loss_idx"
  "tol_edge_acc_metric_atol"
  "custom_indices_metric_idx"
  "output_type"
)

# Default empty values of arguments
sh_script_err_log_file="$HOME/Documents/codes/multivariate-correlation-anomaly-detection/models/crontab_main_sh_err.log"
data_implement=""
log_suffix=""
batch_size=""
n_folds=""
tr_epochs=""
seq_len=""
corr_type=""
corr_window=""
corr_stride=""
model_input_cus_bins=""
target_mats_path=""
cuda_device=""
learning_rate=""
weight_decay=""
use_optim_scheduler=""
drop_pos=()
drop_p=""
gru_l=""
gru_h=""
input_idx=()
kernel_size=""
kernel_pad=""
attn_num_heads=""
use_weighted_loss=""
tol_edge_acc_loss_atol=""
custom_indices_loss_idx=()
tol_edge_acc_metric_atol=""
custom_indices_metric_idx=()
output_type=""
save_model=""

# read arguments
opts=$(getopt \
  --longoptions "$(printf "%s:," "${ARGUMENT_LIST[@]}")" \
  --name "$(basename "$0")" \
  --options "" \
  -- "$@" 2>> $sh_script_err_log_file)
#  -- "$@")

# if sending invalid option, stop script
if [ $? -ne 0 ]; then
  echo "========================== Error:Invalid option provided to crontab_main.sh at $(/usr/bin/date) ================================" >> $sh_script_err_log_file
  exit 1
fi

# The eval in eval set --$opts is required as arguments returned by getopt are quoted.
eval set --$opts


while [[ $# -gt 0 ]]; do
  case "$1" in
    --log_suffix)
      log_suffix="$2" # Note: In order to handle the argument containing space, the quotes around '$2': they are essential!
      log_file="$HOME/Documents/codes/multivariate-correlation-anomaly-detection/models/crontab_main_${log_suffix}.log"
      shift 2 # The 'shift' eats a commandline argument, i.e. converts $1=a, $2=b, $3=c, $4=d into $1=b, $2=c, $3=d. shift 2 moves it all the way to $1=c, $2=d. It's done since that particular branch uses an argument, so it has to remove two things from the list (the -r and the argument following it) not just one.
      ;;

    --data_implement)
      data_implement="--data_implement $2"
      shift 2
      ;;

    --batch_size)
      batch_size="--batch_size $2"
      shift 2
      ;;

    --n_folds)
      n_folds="--n_folds $2"
      shift 2
      ;;

    --tr_epochs)
      tr_epochs="--tr_epochs $2"
      shift 2
      ;;

    --train_model)
      train_model_args+=("$2")
      train_model="--train_model ${train_model_args[@]}"
      shift 2
      ;;

    --seq_len)
      seq_len="--seq_len $2"
      shift 2
      ;;

    --corr_type)
      corr_type="--corr_type $2"
      shift 2
      ;;

    --corr_window)
      corr_window="--corr_window $2"
      shift 2
      ;;

    --corr_stride)
      corr_stride="--corr_stride $2"
      shift 2
      ;;

    --model_input_cus_bins)
      model_input_cus_bins_args+=("$2")
      model_input_cus_bins="--model_input_cus_bins ${model_input_cus_bins_args[@]}"
      shift 2
      ;;

    --target_mats_path)
      target_mats_path="--target_mats_path $2"
      shift 2
      ;;

    --cuda_device)
      cuda_device="--cuda_device $2"
      shift 2
      ;;

    --weight_decay)
      weight_decay="--weight_decay $2"
      shift 2
      ;;

    --use_optim_scheduler)
      use_optim_scheduler="--use_optim_scheduler"
      shift 2
      ;;

    --learning_rate)
      learning_rate="--learning_rate $2"
      shift 2
      ;;

    --drop_pos)
      drop_pos_args+=("$2")
      drop_pos="--drop_pos ${drop_pos_args[@]}"
      shift 2
      ;;

    --drop_p)
      drop_p="--drop_p $2"
      shift 2
      ;;

    --gru_l)
      gru_l="--gru_l $2"
      shift 2
      ;;

    --gru_h)
      gru_h="--gru_h $2"
      shift 2
      ;;

    --input_idx)
      gru_input_feature_idx_args+=("$2")
      gru_input_feature_idx="--gru_input_feature_idx ${gru_input_feature_idx_args[@]}"
      shift 2
      ;;

    --kernel_size)
      kernel_size="--kernel_size $2"
      shift 2
      ;;

    --kernel_pad)
      kernel_pad="--kernel_pad $2"
      shift 2
      ;;

    --attn_num_heads)
      attn_num_heads="--attn_num_heads $2"
      shift 2
      ;;

    --use_weighted_loss)
      use_weighted_loss="--use_weighted_loss"
      shift 2
      ;;

    --tol_edge_acc_loss_atol)
      tol_edge_acc_loss_atol="--tol_edge_acc_loss_atol $2"
      shift 2
      ;;

    --custom_indices_loss_idx)
      custom_indices_loss_idx_args+=("$2")
      custom_indices_loss_idx="--custom_indices_loss_indices ${custom_indices_loss_idx_args[@]}"
      shift 2
      ;;

    --tol_edge_acc_metric_atol)
      tol_edge_acc_metric_atol="--tol_edge_acc_metric_atol $2"
      shift 2
      ;;

    --custom_indices_metric_idx)
      custom_indices_metric_idx_args+=("$2")
      custom_indices_metric_idx="--custom_indices_metric_indices ${custom_indices_metric_idx_args[@]}"
      shift 2
      ;;

    --output_type)
      output_type="--output_type $2"
      shift 2
      ;;

    --save_model)
      save_model="--save_model"
      shift 2
      ;;

    --)
      # if getopt reached the end of options, exit loop
      shift
      break
      ;;

    *)
      # if sending invalid option, stop script
      echo "========================== Error:Unrecognized option: $1 provided to crontab_main.sh at $(/usr/bin/date) ================================" >> $log_file
      exit 1
      ;;

  esac
done

echo "========================== Start training at $(/usr/bin/date) ==========================" >> $log_file
/usr/bin/docker container exec ywt-pytorch python /workspace/multivariate-correlation-anomaly-detection/main.py $data_implement $batch_size $n_folds $tr_epochs $train_model $seq_len $corr_type $corr_window $corr_stride $model_input_cus_bins $target_mats_path $cuda_device $learning_rate $weight_decay $use_optim_scheduler ${drop_pos[@]} $drop_p $gru_l $gru_h $gru_input_feature_idx $kernel_size $kernel_pad $attn_num_heads $use_weighted_loss $tol_edge_acc_loss_atol $custom_indices_loss_idx $tol_edge_acc_metric_atol $custom_indices_metric_idx $output_type $save_model >> "$log_file" 2>&1

echo "========================== End training at $(/usr/bin/date) ================================" >> $log_file
