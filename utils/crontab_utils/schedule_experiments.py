import argparse
import os
from datetime import datetime, timedelta
from itertools import chain, product, repeat
from pprint import pprint

data_implement_list = ["--data_implement SP500_20112015_CORR_MEAN_POSITIVE_NEGATIVE_KEEP"]  # "--data_implement LINEAR_REG_ONE_CLUSTER_DIM_30_BKPS_0_NOISE_STD_30"
batch_size_list = ["--tr_epochs 200"]
tr_epochs_list = [""]
train_models_list = ["--train_models GRUCORRCLASSCUSTOMFEATURES"]  # ["", "--train_models GRUCORRCLASS", "--train_models GRUCORRCLASS --train_models GRUCORRCLASSCUSTOMFEATURES --train_models GRUCORRCOEFPRED"]
corr_type_list = ["--corr_type pearson"]  # ["--corr_type pearson", "--corr_type cross_corr"]
seq_len_list = ["--seq_len 30"]  # ["--seq_len 5", "--seq_len 10"]
model_input_cus_bins_list = [""]  # ["", "--model_input_cus_bins -1 --model_input_cus_bins 0 --model_input_cus_bins 1", "--model_input_cus_bins -1 --model_input_cus_bins -0.25 --model_input_cus_bins 0.25 --model_input_cus_bins 1", "--model_input_cus_bins -1 --model_input_cus_bins -0.5 --model_input_cus_bins 0 --model_input_cus_bins 0.5 --model_input_cus_bins 1"]
target_mats_path_list = ["--target_mats_path pearson/custom_discretize_corr_data/bins_-10_-03_03_10"]  # ["", "--target_mats_path pearson/custom_discretize_corr_data/bins_-10_-025_025_10", "--target_mats_path pearson/quan_discretize_corr_data/bin3"]
learning_rate_list = [""]  # ["--learn_rate 0.0001", "--learn_rate 0.0005", "--learn_rate 0.001", "--learn_rate 0.005", "--learn_rate 0.01", "--learn_rate 0.05", "--learn_rate 0.1"]
weight_decay_list = [""]  # ["--weight_decay 0.0001", "--weight_decay 0.0005", "--weight_decay 0.001", "--weight_decay 0.005", "--weight_decay 0.01", "--weight_decay 0.05", "--weight_decay 0.1"]
use_optim_scheduler_list = [""]  # ["", "--use_optim_scheduler true"]
drop_pos_list = [""]  # ["", "--drop_pos gru", "--drop_pos decoder --drop_pos gru", "--drop_pos gru --drop_pos decoder"]
drop_p_list = [""]  # ["--drop_p 0.33", "--drop_p 0.5", "--drop_p 0.66"]
gru_l_list = [""]  # ["--gru_l 1", "--gru_l 2", "--gru_l 3", "--gru_l 4", "--gru_l 5"]
gru_h_list = [""]  # ["--gru_h 40", "--gru_h 80", "--gru_h 100", "--gru_h 320", "--gru_h 640"]
gru_input_feature_idx_list = ["--gru_input_feature_idx 2 --gru_input_feature_idx 4 --gru_input_feature_idx 5 --gru_input_feature_idx 7 --gru_input_feature_idx 9 --gru_input_feature_idx 10 --gru_input_feature_idx 11 --gru_input_feature_idx 13 --gru_input_feature_idx 14 --gru_input_feature_idx 15 --gru_input_feature_idx 18 --gru_input_feature_idx 19"]  # ["--gru_input_feature_idx 0", "--gru_input_feature_idx 1", "--gru_input_feature_idx 2", "--gru_input_feature_idx 0 --gru_input_feature_idx 1 "]
###gru_input_feature_idx_list = ["--gru_input_feature_idx 0", "--gru_input_feature_idx 1", "--gru_input_feature_idx 2", "--gru_input_feature_idx 3", "--gru_input_feature_idx 4",
###                              "--gru_input_feature_idx 5", "--gru_input_feature_idx 6", "--gru_input_feature_idx 7", "--gru_input_feature_idx 8", "--gru_input_feature_idx 9",
###                              "--gru_input_feature_idx 10", "--gru_input_feature_idx 11", "--gru_input_feature_idx 12", "--gru_input_feature_idx 13", "--gru_input_feature_idx 14",
###                              "--gru_input_feature_idx 15", "--gru_input_feature_idx 16", "--gru_input_feature_idx 17", "--gru_input_feature_idx 18", "--gru_input_feature_idx 19",
###                              "--gru_input_feature_idx 20", "--gru_input_feature_idx 21"]  # ["--gru_input_feature_idx 0", "--gru_input_feature_idx 1", "--gru_input_feature_idx 2", "--gru_input_feature_idx 0 --gru_input_feature_idx 1 "]
use_weighted_loss_list = ["", "--use_weighted_loss true"]  # ["", "--use_weighted_loss true"]
tol_edge_acc_loss_atol_list = [""]  # ["", "--tol_edge_acc_loss_atol 0.05", "--tol_edge_acc_loss_atol 0.1", "--tol_edge_acc_loss_atol 0.33"]
output_type_list = ["--output_type class_probability"]  # ["--output_type discretize", "--output_type class_probability"]

args_values = list(product(data_implement_list, batch_size_list, tr_epochs_list, train_models_list, corr_type_list, seq_len_list, model_input_cus_bins_list,
                           target_mats_path_list, learning_rate_list, weight_decay_list, use_optim_scheduler_list,
                           drop_pos_list, drop_p_list, gru_l_list, gru_h_list,
                           gru_input_feature_idx_list, use_weighted_loss_list, tol_edge_acc_loss_atol_list, output_type_list))
args_keys = ["data_implement", "batch_size", "tr_epochs", "train_models", "corr_type", "seq_len", "model_input_cus_bins", "target_mats_path",
             "learning_rate", "weight_decay", "use_optim_scheduler", "drop_pos", "drop_p", "gru_l", "gru_h", "gru_input_feature_idx", "use_weighted_loss",
             "tol_edge_acc_loss_atol", "output_type"]
args_list = []
for args_value in args_values:
    args_dict = dict(zip(args_keys, args_value))
    args_list.append(args_dict)

args_list = list(filter(lambda x: not ((not x["drop_pos"] and x["drop_p"]) or (x["drop_pos"] and not x["drop_p"])), args_list))  # Eliminate cases where either one of {drop_p, drop_pos} is null.

if set(map(lambda x: x['gru_l'], args_list)) != {""}:
    gru_l_values_set = set(map(lambda x: x['gru_l'], args_list))
    gru_l_values_set.discard("")
    gru_l_pop_value = gru_l_values_set.pop()
    num_models = sum(1 for x in args_list if x['gru_l'] == gru_l_pop_value)
    model_timedelta_list = [timedelta(hours=1, minutes=20), timedelta(hours=1, minutes=20), timedelta(hours=1, minutes=25)]  # The order of elements of model_timedelta_list should comply with the order of elements of args_list
else:
    num_models = len(args_list)
    model_timedelta_list = [timedelta(hours=0, minutes=50)]

model_timedelta_list = list(chain.from_iterable(repeat(x, num_models) for x in model_timedelta_list))
model_timedelta_list = [0] + model_timedelta_list
model_timedelta_list.pop()
assert len(args_list) == len(model_timedelta_list), f"The order of elements of model_timedelta_list〔length: {len(model_timedelta_list)}〕 should comply with the order of args_list: 〔length: {len(args_list)}〕.\nps. model_timedelta_list is based on num_modelsm num_models: {num_models}"
print(f"# len of experiments: {len(args_list)}")

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--script", type=str, nargs='?', default="crontab_main.sh",
                             help="Input the name of operating script")
    args_parser.add_argument("--operating_time", type=str, nargs='?', default="+ 0:03",
                             help=(f"Input the operating time, the format of time: +/- hours:minutes.\n"
                                   f"For example:\n"
                                   f"    - postpone 1 hour and 3 minutes: \"+ 1:03\"\n"
                                   f"    - in advance 11 hour and 5 minutes: \"- 11:05\""))
    args_parser.add_argument("--cuda_device", type=int, nargs='?', default=0,
                             help="Input the gpu id")
    args_parser.add_argument("--log_suffix", type=str, nargs='?', default="",
                             help="Input the suffix of log file")
    ARGS = args_parser.parse_args()
    pprint(f"\n{vars(ARGS)}", indent=1, width=40, compact=True)
    operating_time_status = "postpone" if ARGS.operating_time.split(" ")[0] == "+" else "advance"
    operating_hours = int(ARGS.operating_time.split(" ")[1].split(":")[0])
    operating_minutes = int(ARGS.operating_time.split(" ")[1].split(":")[1].lstrip("0"))
    if operating_time_status == "postpone":
        experiments_start_t = datetime.now()+timedelta(hours=operating_hours, minutes=operating_minutes)
    elif operating_time_status == "advance":
        experiments_start_t = datetime.now()-timedelta(hours=operating_hours, minutes=operating_minutes)
    for i, (prev_model_time_len, model_args) in enumerate(zip(model_timedelta_list, args_list)):
        # print({"operate time length of previous model": prev_model_time_len, "model argumets": model_args})
        model_start_t = experiments_start_t if i == 0 else model_start_t + prev_model_time_len
        home_directory = os.path.expanduser("~")
        cron_args = [model_start_t.strftime("%M %H %d %m")+" *", home_directory, ARGS.script, f"--log_suffix {ARGS.log_suffix}", f"--cuda_device {ARGS.cuda_device}"] + list(model_args.values())
        args_cloze = " ".join(repeat("{}", len(model_args.values())))
        print(("{} {}/Documents/codes/multivariate-correlation-anomaly-detection/utils/crontab_utils/{} {} {} "+args_cloze+" --save_model true").format(*cron_args))