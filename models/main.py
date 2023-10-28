#!/usr/bin/env python
# coding: utf-8
import argparse
import logging
import sys
import traceback
import warnings
from enum import Enum, auto
from math import ceil
from pathlib import Path
from pprint import pformat

import dynamic_yaml
import numpy as np
import pandas as pd
import torch
import yaml
from torch.nn import CrossEntropyLoss, MSELoss

sys.path.append("/workspace/multivariate-correlation-anomaly-detection/utils/")
from metrics_utils import CustomIndicesEdgeAccuracy, TolEdgeAccuracyLoss
from plot_utils import plot_heatmap
from utils import split_and_norm_data

from gru_models import GRUCorrClass, GRUCorrCoefPred

current_dir = Path(__file__).parent
data_config_path = current_dir / "../config/data_config.yaml"
with open(data_config_path) as f:
    data_cfg_yaml = dynamic_yaml.load(f)
    data_cfg = yaml.full_load(dynamic_yaml.dump(data_cfg_yaml))

logger = logging.getLogger(__name__)
logger_console = logging.StreamHandler()
logger_formatter = logging.Formatter('%(levelname)-8s [%(filename)s.%(funcName)s] %(message)s')
logger_console.setFormatter(logger_formatter)
logger.addHandler(logger_console)
metrics_logger = logging.getLogger("metrics")
utils_logger = logging.getLogger("utils")
gru_model_logger = logging.getLogger("gru_models")
logger.setLevel(logging.INFO)
metrics_logger.setLevel(logging.INFO)
utils_logger.setLevel(logging.INFO)
gru_model_logger.setLevel(logging.INFO)
warnings.simplefilter("ignore")


class ModelType(Enum):
    GRUCORRCOEFPRED = auto()
    GRUCORRCLASS = auto()

    def set_model(self, basic_model_cfg, args):
        gru_corr_coef_cfg = basic_model_cfg.copy()
        gru_corr_coef_cfg["gru_in_dim"] = basic_model_cfg["num_pairs"]
        gru_corr_class_cfg = gru_corr_coef_cfg.copy()
        gru_corr_class_cfg["num_labels_classes"] = basic_model_cfg["target_data_bins"].replace("bins_", "").count("_") if basic_model_cfg["target_data_bins"] else None
        ###baseline_gru_one_feature_cfg = gru_corr_coef_cfg.copy()
        ###baseline_gru_one_feature_cfg["gru_in_dim"] = 1
        ###baseline_gru_one_feature_cfg["input_feature_idx"] = args.gru_input_feature_idx
        ###baseline_gru_custom_feature_cfg = gru_corr_coef_cfg.copy()
        ###baseline_gru_custom_feature_cfg["gru_in_dim"] = len(args.gru_input_feature_idx) if args.gru_input_feature_idx else 1
        ###baseline_gru_custom_feature_cfg["input_feature_idx"] = args.gru_input_feature_idx
        ###assert ((basic_model_cfg["num_nodes"]-1)/2*(1+basic_model_cfg["num_nodes"]-1)).is_integer(), "baseline_gru_without_self_corr_cfg[gru_in_dim] is not an integer"
        model_dict = {"GRUCORRCOEFPRED": GRUCorrCoefPred(gru_corr_coef_cfg),
                      "GRUCORRCLASS": GRUCorrClass(gru_corr_class_cfg)}
                      ###"CLASSBASELINE": ClassBaselineGRU(gru_corr_coef_cfg),
                      ###"CLASSBASELINEONEFEATURE": ClassBaselineGRUOneFeature(baseline_gru_one_feature_cfg),
                      ###"CLASSBASELINECUSTOMFEATURE": ClassBaselineGRUCustomFeatures(baseline_gru_custom_feature_cfg),
        model = model_dict[self.name]
        assert ModelType.__members__.keys() == model_dict.keys(), f"ModelType members and model_dict must be the same keys, ModelType.__members__.keys(): {ModelType.__members__.keys()}, model_dict.keys(): {model_dict.keys()}"

        return model

    def set_save_model_dir(self, current_dir, output_file_name, corr_type, s_l, w_l):
        save_model_dir_base_dict = {"GRUCORRCOEFPRED": "gru_corr_coef_pred",
                                    "GRUCORRCLASS": "gru_corr_class"}
                                    ###"CLASSBASELINE": "class_baseline_gru",
                                    ###"CLASSBASELINEONEFEATURE": "class_baseline_gru_one_feature",
                                    ###"CLASSBASELINECUSTOMFEATURE": "class_baseline_gru_custom_feature",
        assert ModelType.__members__.keys() == save_model_dir_base_dict.keys(), f"ModelType members and save_model_dir_base_dict must be the same keys, ModelType.__members__.keys(): {ModelType.__members__.keys()}, save_model_dir_base_dict.keys(): {save_model_dir_base_dict.keys()}"
        model_dir = current_dir/f'save_models/{save_model_dir_base_dict[self.name]}/{output_file_name}/{corr_type}/corr_s{s_l}_w{w_l}'
        model_log_dir = current_dir/f'save_models/{save_model_dir_base_dict[self.name]}/{output_file_name}/{corr_type}/corr_s{s_l}_w{w_l}/train_logs/'
        model_dir.mkdir(parents=True, exist_ok=True)
        model_log_dir.mkdir(parents=True, exist_ok=True)

        return model_dir, model_log_dir


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--data_implement", type=str, nargs='?', default="PW_CONST_DIM_70_BKPS_0_NOISE_STD_10",
                             help="input the data implement name, watch options by operate: logger.info(data_cfg['DATASETS'].keys())")
    args_parser.add_argument("--batch_size", type=int, nargs='?', default=64,
                             help="input the number of batch size")
    args_parser.add_argument("--tr_epochs", type=int, nargs='?', default=1500,
                             help="input the number of training epochs")
    args_parser.add_argument("--seq_len", type=int, nargs='?', default=10,
                             help="input the number of sequence length")
    args_parser.add_argument("--corr_type", type=str, nargs='?', default="pearson",
                             choices=["pearson", "cross_corr"],
                             help="input the type of correlation computing, the choices are [pearson, cross_corr]")
    args_parser.add_argument("--corr_stride", type=int, nargs='?', default=1,
                             help="input the number of stride length of correlation computing")
    args_parser.add_argument("--corr_window", type=int, nargs='?', default=50,
                             help="input the number of window length of correlation computing")
    args_parser.add_argument("--model_input_cus_bins", type=float, nargs='*', default=None,
                             help="input the custom discrete boundaries(bins) of model input data")
    args_parser.add_argument("--target_mats_path", type=str, nargs='?', default=None,
                             help="input the relative path of target matrices, the base directory of path is data_cfg[DIR][PIPELINE_DATA_DIR])/data_cfg[DATASETS][data_implement][OUTPUT_FILE_NAME_BASIS] + train_items_setting")
    args_parser.add_argument("--cuda_device", type=int, nargs='?', default=0,
                             help="input the number of cuda device")
    args_parser.add_argument("--train_models", type=str, nargs='+', default=[],
                             choices=["GRUCORRCOEFPRED", "GRUCORRCLASS"],
                             help="input to decide which models to train, the choices are [GRUCORRCOEFPRED, GRUCORRCLASS]")
    args_parser.add_argument("--learning_rate", type=float, nargs='?', default=0.001,
                             help="input the learning rate of training")
    args_parser.add_argument("--weight_decay", type=float, nargs='?', default=0,
                             help="input the weight decay of training")
    args_parser.add_argument("--use_optim_scheduler", type=bool, default=False, action=argparse.BooleanOptionalAction,
                             help="input --use_optim_scheduler to use optimizer scheduler")
    args_parser.add_argument("--drop_pos", type=str, nargs='*', default=[],
                             choices=["gru", "fc_decoder", "class_fc"],
                             help="input to decide the position of drop layers, the choices are [gru, fc_decoder, class_fc]")
    args_parser.add_argument("--drop_p", type=float, default=0,
                             help="input 0~1 to decide the probality of drop layers")
    args_parser.add_argument("--gru_l", type=int, nargs='?', default=2,  # range:1~n, for gru
                             help="input the number of stacked-layers of gru")
    args_parser.add_argument("--gru_h", type=int, nargs='?', default=80,
                             help="input the number of gru hidden size")
    args_parser.add_argument("--gru_input_feature_idx", type=int, nargs='*', default=None,
                             help="input the order of input features of gru, the order is from 0 to combination(num_nodes, 2)-1")
    args_parser.add_argument("--tol_edge_acc_loss_atol", type=float, nargs='?', default=None,
                             help="input the absolute tolerance of TolEdgeAccuracyLoss")
    args_parser.add_argument("--use_weighted_loss", type=bool, default=False, action=argparse.BooleanOptionalAction,
                             help="input --use_weighted_loss to use CrossEntropyLoss weight")
    args_parser.add_argument("--custom_indices_edge_acc_metric_indices", type=int, nargs='*', default=[],
                             help="input the indices of CustomIndicesEdgeAccuracy")
    args_parser.add_argument("--output_type", type=str, nargs='?', default=None,
                             choices=["corr_coef", "class_probability"],
                             help="input the type of output, the choices are [class_probability]")
    args_parser.add_argument("--save_model", type=bool, default=False, action=argparse.BooleanOptionalAction,  # setting of output files
                             help="input --save_model to save model weight and model info")
    args_parser.add_argument("--inference_models", type=str, nargs='+', default=[],
                             choices=["GRUCORRCOEFPRED", "GRUCORRCLASS"],
                             help="input to decide which models to inference, the choices are [GRUCORRCOEFPRED, GRUCORRCLASS]")
    args_parser.add_argument("--inference_model_paths", type=str, nargs='+', default=[],
                             help="input the path of inference model weight")
    args_parser.add_argument("--inference_data_split", type=str, nargs='?', default="val",
                             help="input the data split of inference data, the choices are [train, val, test]")
    ARGS = args_parser.parse_args()
    assert bool(ARGS.train_models) != bool(ARGS.inference_models), "train_models and inference_models must be input one of them"
    assert bool(ARGS.drop_pos) == bool(ARGS.drop_p), "drop_pos and drop_p must be both input or not input"
    assert ("GRUCORRCOEFPRED" not in ARGS.train_models+ARGS.inference_models) or (ARGS.output_type == "corr_coef"), "output_type must be corr_coef when train_models|inferene_models is GRUCORRCOEFPRED"
    assert ("GRUCORRCLASS" not in ARGS.train_models+ARGS.inference_models) or ARGS.output_type == "class_probability", "output_type must be class_probability when train_models|inferene_models is GRUCORRCLASS"
    ###assert ("CLASSBASELINEONEFEATURE" not in ARGS.train_models+ARGS.inference_models) or ARGS.output_type == "class_probability", "output_type must be class_probability when train_models|inferene_models is ClassBaselineOneFeature"
    ###assert ("CLASSBASELINECUSTOMFEATURE" not in ARGS.train_models+ARGS.inference_models) or ARGS.output_type == "class_probability", "output_type must be class_probability when train_models|inferene_models is ClassBaselineCustomFeature"
    assert "class_fc" not in ARGS.drop_pos or ARGS.output_type == "class_probability", "output_type must be class_probability when class_fc in drop_pos"
    ###assert ("CLASSBASELINEONEFEATURE" not in ARGS.train_models+ARGS.inference_models) or (ARGS.gru_input_feature_idx is not None and len(ARGS.gru_input_feature_idx) == 1), "gru_input_feature_idx must be input when train_models|inferene_models is ClassBaselineOneFeature and len(gru_input_feature_idx) must be 1"
    ###assert ("CLASSBASELINECUSTOMFEATURE" not in ARGS.train_models+ARGS.inference_models) or (ARGS.gru_input_feature_idx is not None and len(ARGS.gru_input_feature_idx) > 1), "gru_input_feature_idx must be input when train_models|inferene_models is ClassBaselineCustomFeature and len(gru_input_feature_idx) must be greater than 1"
    logger.info(pformat(f"\n{vars(ARGS)}", indent=1, width=100, compact=True))

    # Data implement & output setting & testset setting
    # data implement setting
    data_implement = ARGS.data_implement
    # train set setting
    train_items_setting = "-train_train"  # -train_train|-train_all
    # setting of name of output files and pictures title
    output_file_name = data_cfg["DATASETS"][data_implement]['OUTPUT_FILE_NAME_BASIS'] + train_items_setting
    # setting of output files
    save_model_info = ARGS.save_model
    # set devide of pytorch
    device = torch.device(f'cuda:{ARGS.cuda_device}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float64)
    torch.autograd.set_detect_anomaly(True)  # for debug grad

    s_l, w_l = ARGS.corr_stride, ARGS.corr_window
    if ARGS.model_input_cus_bins:
        corr_data_mode_dir = f"custom_discretize_corr_data/bins_{'_'.join((str(f) for f in ARGS.model_input_cus_bins)).replace('.', '')}"
    else:
        corr_data_mode_dir = "corr_data"
    corr_df_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}/{ARGS.corr_type}/{corr_data_mode_dir}"
    target_df_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}/{ARGS.target_mats_path}"

    # model configuration
    corr_df = pd.read_csv(corr_df_dir/f"corr_s{s_l}_w{w_l}.csv", index_col=["item_pair"])
    target_df = pd.read_csv(target_df_dir/f"corr_s{s_l}_w{w_l}.csv", index_col=["item_pair"]) if ARGS.target_mats_path else None
    train_dataset, val_dataset, test_dataset = split_and_norm_data(model_input_df=corr_df, target_df=target_df, batch_size=ARGS.batch_size)
    basic_model_cfg = {"tr_epochs": ARGS.tr_epochs,
                       "batch_size": ARGS.batch_size,
                       "num_batches": {"train": ceil((train_dataset["model_input"].shape[1]-ARGS.seq_len)/ARGS.batch_size),
                                       "val": ceil((val_dataset["model_input"].shape[1]-ARGS.seq_len)/ARGS.batch_size),
                                       "test": ceil((test_dataset["model_input"].shape[1]-ARGS.seq_len)/ARGS.batch_size)},
                       "seq_len": ARGS.seq_len,
                       "num_pairs": train_dataset["model_input"].shape[0],
                       "model_input_cus_bins": '_'.join((str(f) for f in ARGS.model_input_cus_bins)).replace('.', '') if ARGS.model_input_cus_bins else None,
                       "learning_rate": ARGS.learning_rate,
                       "weight_decay": ARGS.weight_decay,
                       "can_use_optim_scheduler": ARGS.use_optim_scheduler,
                       "drop_pos": ARGS.drop_pos,
                       "drop_p": ARGS.drop_p,
                       "gru_l": ARGS.gru_l,
                       "gru_h": ARGS.gru_h if ARGS.gru_h else ARGS.gra_enc_l*ARGS.gra_enc_h,
                       "output_type": ARGS.output_type,
                       "target_data_bins": ARGS.target_mats_path.split("/")[-1] if ARGS.target_mats_path else None,
                       "tol_edge_acc_loss_atol": ARGS.tol_edge_acc_loss_atol}

    # setting of loss function of model
    loss_fns_dict = {"fns": [MSELoss()],
                     "fn_args": {"MSELoss()": {}}}
    if ARGS.output_type == "class_probability":
        loss_fns_dict["fns"].clear(); loss_fns_dict["fn_args"].clear()
        if ARGS.use_weighted_loss:
            tr_labels, tr_labels_freq_counts = np.unique(train_dataset['target'], return_counts=True)
            weight = torch.tensor(np.reciprocal(tr_labels_freq_counts/tr_labels_freq_counts.sum()))
        loss_fns_dict["fns"].append(CrossEntropyLoss(weight if ARGS.use_weighted_loss else None))
        loss_fns_dict["fn_args"].update({"CrossEntropyLoss()": {}})
    elif ARGS.tol_edge_acc_loss_atol is not None:
        loss_fns_dict["fns"].append(TolEdgeAccuracyLoss())
        loss_fns_dict["fn_args"].update({"TolEdgeAccuracyLoss()": {"atol": ARGS.tol_edge_acc_loss_atol}})

    # setting of metric function of edge_accuracy of model
    if ARGS.custom_indices_edge_acc_metric_indices:
        num_labels_classes = ARGS.target_mats_path.split("/")[-1].replace("bins_", "").count("_")
        basic_model_cfg["edge_acc_metric_fn"] = CustomIndicesEdgeAccuracy(selected_indices=ARGS.custom_indices_edge_acc_metric_indices, num_classes=num_labels_classes)

    # show info
    logger.info(f"===== file_name basis:{output_file_name} =====")
    logger.info(f"===== pytorch running on:{device} =====")
    logger.info(f"corr_df.shape:{corr_df.shape}, target_df.shape:{target_df.shape if target_df is not None else None}")
    logger.info(f"corr_df.max:{corr_df.max().max()}, corr_df.min:{corr_df.min().min()}")
    logger.info(f"train_dataset['model_input'].max:{train_dataset['model_input'].max()}, train_dataset['model_input'].min:{train_dataset['model_input'].min()}")
    logger.info(f"val_dataset['model_input'].max:{val_dataset['model_input'].max()}, val_dataset['model_input'].min:{val_dataset['model_input'].min()}")
    logger.info(f"test_dataset['model_input'].max:{test_dataset['model_input'].max()}, test_dataset['model_input'].min:{test_dataset['model_input'].min()}")
    logger.info(f'Training set   = {train_dataset["model_input"].shape[1]} timesteps')
    logger.info(f'Validation set = {val_dataset["model_input"].shape[1]} timesteps')
    logger.info(f'Test set       = {test_dataset["model_input"].shape[1]} timesteps')
    logger.info("="*80)

    if len(ARGS.train_models) > 0:
        assert list(filter(lambda x: x in ModelType.__members__.keys(), ARGS.train_models)), f"train_models must be input one of {ModelType.__members__.keys()}"
        for model_type in ModelType:
            is_training, train_count = True, 0
            while (model_type.name in ARGS.train_models) and (is_training is True) and (train_count < 100):
                try:
                    logger.info(f"===== train model:{model_type.name} =====")
                    train_count += 1
                    model_dir, model_log_dir = model_type.set_save_model_dir(current_dir, output_file_name, ARGS.corr_type, s_l, w_l)
                    model = model_type.set_model(basic_model_cfg, ARGS)
                    best_model, best_model_info = model.train(train_data=train_dataset, val_data=val_dataset, loss_fns=loss_fns_dict, epochs=ARGS.tr_epochs, show_model_info=True)
                except AssertionError as e:
                    logger.error(f"\n{e}")
                except Exception as e:
                    error_class = e.__class__.__name__  # 取得錯誤類型
                    detail = e.args[0]  # 取得詳細內容
                    cl, exc, tb = sys.exc_info()  # 取得Call Stack
                    last_call_stack = traceback.extract_tb(tb)[-1]  # 取得Call Stack的最後一筆資料
                    file_name = last_call_stack[0]  # 取得發生的檔案名稱
                    line_num = last_call_stack[1]  # 取得發生的行號
                    func_name = last_call_stack[2]  # 取得發生的函數名稱
                    err_msg = "File \"{}\", line {}, in {}: [{}] {}".format(file_name, line_num, func_name, error_class, detail)
                    logger.error(f"===\n{err_msg}")
                    logger.error(f"===\n{traceback.extract_tb(tb)}")
                else:
                    is_training = False
                    if save_model_info:
                        model_dir, model_log_dir = model_type.set_save_model_dir(current_dir, output_file_name, ARGS.corr_type, s_l, w_l)
                        model.save_model(best_model, best_model_info, model_dir=model_dir, model_log_dir=model_log_dir)
    elif len(ARGS.inference_models) > 0:
        logger.info(f"===== inference model:[{ARGS.inference_models}] on {ARGS.inference_data_split} data =====")
        logger.info(f"===== if inference_models is more than one, the inference result is ensemble result =====")
        assert list(filter(lambda x: x in ModelType.__members__.keys(), ARGS.inference_models)), f"inference_models must be input one of {ModelType.__members__.keys()}"
        if ARGS.inference_data_split == "train":
            inference_data = train_dataset
        elif ARGS.inference_data_split == "val":
            inference_data = val_dataset
        elif ARGS.inference_data_split == "test":
            inference_data = test_dataset
        loss = None
        edge_acc = None
        if len(ARGS.inference_models) == 1:
            model_type = ModelType[ARGS.inference_models[0]]
            model = model_type.set_model(basic_model_cfg, ARGS)
            model_dir, _ = model_type.set_save_model_dir(current_dir, output_file_name, ARGS.corr_type, s_l, w_l)
            model_param_path = model_dir.parents[2].joinpath(ARGS.inference_model_paths[0])
            assert model_param_path.exists(), f"{model_param_path} not exists"
            model.load_state_dict(torch.load(model_param_path, map_location=device))
            model.eval()
            loss, edge_acc, preds, y_labels = model.test(inference_data, loss_fns=loss_fns_dict)
            conf_mat_save_fig_dir = current_dir/f"exploration_model_result/model_result_figs/{ARGS.inference_models[0]}/{model_param_path.stem}"
            conf_mat_save_fig_name = f'confusion_matrix-{ARGS.inference_data_split}.png'
    ###    elif len(ARGS.inference_models) > 1:
    ###        assert sorted(ARGS.inference_models) == ARGS.inference_models, f"inference_models must be input in order, but the input order is {ARGS.inference_models}"
    ###        ensemble_pred_prob = 0
    ###        ensemble_weights = [10, 12]
    ###        for model_type, infer_model_path, weight in zip_longest(ARGS.inference_models, ARGS.inference_model_paths, ensemble_weights, fillvalue=None):
    ###            model = ModelType[model_type].set_model(basic_model_cfg, ARGS)
    ###            model_dir, _ = ModelType[model_type].set_save_model_dir(current_dir, output_file_name, ARGS.corr_type, s_l, w_l)
    ###            model_param_path = model_dir.parents[2].joinpath(infer_model_path)
    ###            assert model_param_path.exists(), f"{model_param_path} not exists"
    ###            model.load_state_dict(torch.load(model_param_path, map_location=device))
    ###            model.eval()
    ###            if "MTSCORRAD" in model_type:
    ###                if ARGS.inference_data_split in ["val", "test"]:
    ###                    model.model_cfg["batch_size"] = inference_data["edges"].shape[0]-1-ARGS.seq_len
    ###                test_loader = model.create_pyg_data_loaders(graph_adj_mats=inference_data["edges"],  graph_nodes_mats=inference_data["nodes"], target_mats=inference_data["target"], loader_seq_len=model.model_cfg["seq_len"], show_log=True)
    ###            elif "BASELINE" in model_type:
    ###                test_loader = model.yield_batch_data(graph_adj_mats=inference_data['edges'], target_mats=inference_data['target'], batch_size=model.model_cfg['batch_size'], seq_len=model.model_cfg['seq_len'])
    ###            for batch_idx, batch_data in enumerate(test_loader):
    ###                infer_res = model.infer_batch_data(batch_data)
    ###                batch_pred_prob, batch_preds, batch_y_labels = infer_res[0], infer_res[1], infer_res[2]
    ###                all_batch_pred_prob = batch_pred_prob if batch_idx == 0 else torch.cat((all_batch_pred_prob, batch_pred_prob), dim=0)
    ###                y_labels = batch_y_labels if batch_idx == 0 else torch.cat((y_labels, batch_y_labels), dim=0)
    ###            ensemble_pred_prob += all_batch_pred_prob*weight
    ###        preds = torch.argmax(ensemble_pred_prob, dim=1)
    ###        if "edge_acc_metric_fn" in basic_model_cfg.keys():
    ###            edge_acc = basic_model_cfg["edge_acc_metric_fn"](preds, y_labels)
    ###        else:
    ###            edge_acc = preds.eq(y_labels).to(torch.float).mean()
    ###        model_param_paths = [Path(model_path).stem for model_path in ARGS.inference_model_paths]
    ###        conf_mat_save_fig_dir = current_dir/f"exploration_model_result/model_result_figs/ensemble_{'_'.join(ARGS.inference_models)}"/'-'.join(model_param_paths)
    ###        conf_mat_save_fig_name = 'confusion_matrix-ensemble_rate_'+'_'.join([str(w) for w in ensemble_weights])+f'-{ARGS.inference_data_split}.png'

    ###    assert preds.shape == y_labels.shape, f"preds.shape:{preds.shape} != y_labels.shape:{y_labels.shape}"
    ###    preds, y_labels = preds.cpu().numpy(), y_labels.cpu().numpy()
    ###    conf_mat_save_fig_path = conf_mat_save_fig_dir/conf_mat_save_fig_name
    ###    conf_mat_save_fig_dir.mkdir(parents=True, exist_ok=True)
    ###    if ARGS.use_upper_tri_edge_acc_metric:
    ###        assert sqrt(y_labels.shape[1]).is_integer(), f"y_labels.shape[1]:{y_labels.shape[1]} is not square number"
    ###        num_edges = int(sqrt(y_labels.shape[1]))
    ###        idx_upper_tri = np.triu_indices(num_edges, k=1)
    ###        preds, y_labels = preds.reshape(-1, num_edges, num_edges), y_labels.reshape(-1, num_edges, num_edges)
    ###        preds = preds[:, idx_upper_tri[0], idx_upper_tri[1]]
    ###        y_labels = y_labels[:, idx_upper_tri[0], idx_upper_tri[1]]
    ###    plot_heatmap(preds, y_labels, can_show_conf_mat=True, save_fig_path=conf_mat_save_fig_path)
    ###    loss = loss.item() if isinstance(loss, torch.Tensor) else loss
    ###    edge_acc = edge_acc.item() if isinstance(edge_acc, torch.Tensor) else edge_acc
    ###    logger.info(f"loss_fns:{loss_fns_dict['fns']}")
    ###    logger.info(f"metric_fn:{basic_model_cfg['edge_acc_metric_fn'] if 'edge_acc_metric_fn' in basic_model_cfg.keys() else None}")
    ###    logger.info(f"Special args of loss_fns: {[(loss_fn, loss_args) for loss_fn, loss_args in loss_fns_dict['fn_args'].items() for arg in loss_args if arg not in ['input', 'target']]}")
    ###    logger.info(f"loss:{loss}, edge_acc:{edge_acc}")
