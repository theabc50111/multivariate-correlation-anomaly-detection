#!/usr/bin/env python
# coding: utf-8
import argparse
import sys
import traceback
import warnings
from enum import Enum, auto
from math import ceil
from pathlib import Path
from pprint import pformat

import numpy as np
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss, MSELoss

from models.gru_models import (GRUCorrClass, GRUCorrClassCustomFeatures,
                               GRUCorrCoefPred)
from utils.assorted_utils import load_data_cfg, split_and_norm_data
from utils.log_utils import Log
from utils.metrics_utils import (CustomIndicesCrossEntropyLoss,
                                 CustomIndicesEdgeAccuracy,
                                 TolEdgeAccuracyLoss)
from utils.plot_utils import plot_heatmap

THIS_FILE_DIR = Path(__file__).resolve().parent
DATA_CFG = load_data_cfg()

LOGGER = Log().init_logger(logger_name=__name__)
warnings.simplefilter("ignore")


class ModelType(Enum):
    GRUCORRCOEFPRED = auto()
    GRUCORRCLASS = auto()
    GRUCORRCLASSCUSTOMFEATURES = auto()

    def set_model(self, basic_model_cfg, args):
        gru_corr_coef_cfg = basic_model_cfg.copy()
        gru_corr_coef_cfg["gru_in_dim"] = basic_model_cfg["num_pairs"]
        gru_corr_class_cfg = gru_corr_coef_cfg.copy()
        gru_corr_class_cfg["num_labels_classes"] = basic_model_cfg["target_data_bins"].replace("bins_", "").count("_") if basic_model_cfg["target_data_bins"] else None
        gru_corr_class_custom_feature_cfg = gru_corr_class_cfg.copy()
        gru_corr_class_custom_feature_cfg["gru_in_dim"] = len(args.gru_input_feature_idx) if args.gru_input_feature_idx else 1
        gru_corr_class_custom_feature_cfg["input_feature_idx"] = args.gru_input_feature_idx
        model_dict = {"GRUCORRCOEFPRED": GRUCorrCoefPred(gru_corr_coef_cfg),
                      "GRUCORRCLASS": GRUCorrClass(gru_corr_class_cfg),
                      "GRUCORRCLASSCUSTOMFEATURES": GRUCorrClassCustomFeatures(gru_corr_class_custom_feature_cfg)}
        model = model_dict[self.name]
        assert ModelType.__members__.keys() == model_dict.keys(), f"ModelType members and model_dict must be the same keys, ModelType.__members__.keys(): {ModelType.__members__.keys()}, model_dict.keys(): {model_dict.keys()}"

        return model

    def set_save_model_dir(self, save_model_base_dir, output_file_name, corr_type, s_l, w_l):
        save_model_dir_base_dict = {"GRUCORRCOEFPRED": "gru_corr_coef_pred",
                                    "GRUCORRCLASS": "gru_corr_class",
                                    "GRUCORRCLASSCUSTOMFEATURES": "gru_corr_class_custom_features"}
        assert ModelType.__members__.keys() == save_model_dir_base_dict.keys(), f"ModelType members and save_model_dir_base_dict must be the same keys, ModelType.__members__.keys(): {ModelType.__members__.keys()}, save_model_dir_base_dict.keys(): {save_model_dir_base_dict.keys()}"
        model_dir = save_model_base_dir/f'models/save_models/{save_model_dir_base_dict[self.name]}/{output_file_name}/{corr_type}/corr_s{s_l}_w{w_l}'
        model_log_dir = save_model_base_dir/f'models/save_models/{save_model_dir_base_dict[self.name]}/{output_file_name}/{corr_type}/corr_s{s_l}_w{w_l}/train_logs/'
        model_dir.mkdir(parents=True, exist_ok=True)
        model_log_dir.mkdir(parents=True, exist_ok=True)

        return model_dir, model_log_dir


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--data_implement", type=str, nargs='?', default="PW_CONST_DIM_70_BKPS_0_NOISE_STD_10",
                             help="input the data implement name, watch options by operate: logger.info(DATA_CFG['DATASETS'].keys())")
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
    args_parser.add_argument("--model_input_items", type=bool, default=False, action=argparse.BooleanOptionalAction,
                             help="input --model_input_items to use items as model input, if not input, use corr_pair as model input")
    args_parser.add_argument("--model_input_cus_bins", type=float, nargs='*', default=None,
                             help="input the custom discrete boundaries(bins) of model input data")
    args_parser.add_argument("--target_mats_path", type=str, nargs='?', default=None,
                             help="input the relative path of target matrices, the base directory of path is DATA_CFG[DIR][PIPELINE_DATA_DIR])/DATA_CFG[DATASETS][data_implement][OUTPUT_FILE_NAME_BASIS] + train_items_setting")
    args_parser.add_argument("--cuda_device", type=int, nargs='?', default=0,
                             help="input the number of cuda device")
    args_parser.add_argument("--train_models", type=str, nargs='+', default=[],
                             choices=["GRUCORRCOEFPRED", "GRUCORRCLASS", "GRUCORRCLASSCUSTOMFEATURES"],
                             help="input to decide which models to train, the choices are [GRUCORRCOEFPRED, GRUCORRCLASS, GRUCORRCLASSCUSTOMFEATURES]")
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
    args_parser.add_argument("--custom_indices_loss_indices", type=int, nargs='*', default=[],
                             help="input the indices of CustomIndicesCrossEntropyLoss")
    args_parser.add_argument("--use_weighted_loss", type=bool, default=False, action=argparse.BooleanOptionalAction,
                             help="input --use_weighted_loss to use CrossEntropyLoss weight")
    args_parser.add_argument("--custom_indices_metric_indices", type=int, nargs='*', default=[],
                             help="input the indices of CustomIndicesEdgeAccuracy")
    args_parser.add_argument("--output_type", type=str, nargs='?', default=None,
                             choices=["corr_coef", "class_probability"],
                             help="input the type of output, the choices are [class_probability]")
    args_parser.add_argument("--save_model", type=bool, default=False, action=argparse.BooleanOptionalAction,  # setting of output files
                             help="input --save_model to save model weight and model info")
    args_parser.add_argument("--inference_models", type=str, nargs='+', default=[],
                             choices=["GRUCORRCOEFPRED", "GRUCORRCLASS", "GRUCORRCLASSCUSTOMFEATURES"],
                             help="input to decide which models to inference, the choices are [GRUCORRCOEFPRED, GRUCORRCLASS, GRUCORRCLASSCUSTOMFEATURES]")
    args_parser.add_argument("--inference_model_paths", type=str, nargs='+', default=[],
                             help="input the path of inference model weight")
    args_parser.add_argument("--inference_data_split", type=str, nargs='?', default="val",
                             help="input the data split of inference data, the choices are [train, val, test]")
    ARGS = args_parser.parse_args()
    assert bool(ARGS.train_models) != bool(ARGS.inference_models), "train_models and inference_models must be input one of them"
    assert bool(ARGS.drop_pos) == bool(ARGS.drop_p), "drop_pos and drop_p must be both input or not input"
    assert ("GRUCORRCOEFPRED" not in ARGS.train_models+ARGS.inference_models) or (ARGS.output_type == "corr_coef"), "output_type must be corr_coef when train_models|inferene_models is GRUCORRCOEFPRED"
    assert ("GRUCORRCLASS" not in ARGS.train_models+ARGS.inference_models) or ARGS.output_type == "class_probability", "output_type must be class_probability when train_models|inferene_models is GRUCORRCLASS"
    assert ("GRUCORRCLASSCUSTOMFEATURES" not in ARGS.train_models+ARGS.inference_models) or ARGS.output_type == "class_probability", "output_type must be class_probability when train_models|inferene_models is GRUCORRCLASSCUSTOMFEATURES"
    assert "class_fc" not in ARGS.drop_pos or ARGS.output_type == "class_probability", "output_type must be class_probability when class_fc in drop_pos"
    assert ("GRUCORRCLASS" not in ARGS.train_models+ARGS.inference_models) or ARGS.gru_input_feature_idx is None, "gru_input_feature_idx must be None when train_models|inferene_models is GRUCORRCLASS"
    assert ("GRUCORRCLASSCUSTOMFEATURES" not in ARGS.train_models+ARGS.inference_models) or (ARGS.gru_input_feature_idx is not None and len(ARGS.gru_input_feature_idx) >= 1), "gru_input_feature_idx must be input when train_models|inferene_models is GRUCORRCLASSCUSTOMFEATURES and len(gru_input_feature_idx) must be greater equal to 1"
    LOGGER.info(pformat(f"\n{vars(ARGS)}", indent=1, width=100, compact=True))

    # Data implement & output setting & testset setting
    # data implement setting
    data_implement = ARGS.data_implement
    # train set setting
    train_items_setting = "-train_train"  # -train_train|-train_all
    # setting of name of output files and pictures title
    output_file_name = DATA_CFG["DATASETS"][data_implement]['OUTPUT_FILE_NAME_BASIS'] + train_items_setting
    # setting of output files
    save_model_info = ARGS.save_model
    # set devide of pytorch
    device = torch.device(f'cuda:{ARGS.cuda_device}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float64)
    torch.autograd.set_detect_anomaly(True)  # for debug grad

    # setting of data
    s_l, w_l = ARGS.corr_stride, ARGS.corr_window
    if ARGS.model_input_cus_bins:
        corr_data_mode_dir = f"custom_discretize_corr_data/bins_{'_'.join((str(f) for f in ARGS.model_input_cus_bins)).replace('.', '')}"
    else:
        corr_data_mode_dir = "corr_data"
    corr_df_dir = Path(DATA_CFG["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}/{ARGS.corr_type}/{corr_data_mode_dir}"
    target_df_dir = Path(DATA_CFG["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}/{ARGS.target_mats_path}"
    corr_df = pd.read_csv(corr_df_dir/f"corr_s{s_l}_w{w_l}.csv", index_col=["item_pair"])
    target_df = pd.read_csv(target_df_dir/f"corr_s{s_l}_w{w_l}.csv", index_col=["item_pair"]) if ARGS.target_mats_path else None
    if ARGS.model_input_items:
        # model_input_df = items_df
        pass
    else:
        model_input_df = corr_df
    train_dataset, val_dataset, test_dataset = split_and_norm_data(model_input_df=corr_df, target_df=target_df, batch_size=ARGS.batch_size)

    # model configuration
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
    if ARGS.use_weighted_loss:
        tr_labels, tr_labels_freq_counts = np.unique(train_dataset['target'], return_counts=True)
        loss_weight = torch.tensor(np.reciprocal(tr_labels_freq_counts/tr_labels_freq_counts.sum()))
    loss_fns_dict = {"fns": [MSELoss()],
                     "fn_args": {"MSELoss()": {}}}
    if ARGS.output_type == "class_probability":
        loss_fns_dict["fns"].clear(); loss_fns_dict["fn_args"].clear()
        if ARGS.custom_indices_loss_indices:
            num_labels_classes = ARGS.target_mats_path.split("/")[-1].replace("bins_", "").count("_") if ARGS.target_mats_path else None
            loss_fns_dict["fns"].append(CustomIndicesCrossEntropyLoss(selected_indices=ARGS.custom_indices_loss_indices, num_classes=num_labels_classes, weight=loss_weight if ARGS.use_weighted_loss else None))
            loss_fns_dict["fn_args"].update({"CustomIndicesCrossEntropyLoss()": {}})
            basic_model_cfg["custom_indices_loss_indices"] = ARGS.custom_indices_loss_indices
        else:
            loss_fns_dict["fns"].append(CrossEntropyLoss(loss_weight if ARGS.use_weighted_loss else None))
            loss_fns_dict["fn_args"].update({"CrossEntropyLoss()": {}})
    elif ARGS.tol_edge_acc_loss_atol is not None:
        loss_fns_dict["fns"].append(TolEdgeAccuracyLoss())
        loss_fns_dict["fn_args"].update({"TolEdgeAccuracyLoss()": {"atol": ARGS.tol_edge_acc_loss_atol}})
    basic_model_cfg["loss_fns"] = loss_fns_dict

    # setting of metric function of edge_accuracy of model
    if ARGS.custom_indices_metric_indices:
        num_labels_classes = ARGS.target_mats_path.split("/")[-1].replace("bins_", "").count("_") if ARGS.target_mats_path else None
        basic_model_cfg["metric_fn"] = CustomIndicesEdgeAccuracy(selected_indices=ARGS.custom_indices_metric_indices, num_classes=num_labels_classes)
        basic_model_cfg["custom_indices_metric_indices"] = ARGS.custom_indices_metric_indices

    # show info
    LOGGER.info(f"===== file_name basis:{output_file_name} =====")
    LOGGER.info(f"===== pytorch running on:{device} =====")
    LOGGER.info(f"corr_df.shape:{corr_df.shape}, target_df.shape:{target_df.shape if target_df is not None else None}")
    LOGGER.info(f"corr_df.max:{corr_df.max().max()}, corr_df.min:{corr_df.min().min()}")
    LOGGER.info(f"train_dataset['model_input'].max:{train_dataset['model_input'].max()}, train_dataset['model_input'].min:{train_dataset['model_input'].min()}")
    LOGGER.info(f"val_dataset['model_input'].max:{val_dataset['model_input'].max()}, val_dataset['model_input'].min:{val_dataset['model_input'].min()}")
    LOGGER.info(f"test_dataset['model_input'].max:{test_dataset['model_input'].max()}, test_dataset['model_input'].min:{test_dataset['model_input'].min()}")
    LOGGER.info(f'Training set   = {train_dataset["model_input"].shape[1]} timesteps')
    LOGGER.info(f'Validation set = {val_dataset["model_input"].shape[1]} timesteps')
    LOGGER.info(f'Test set       = {test_dataset["model_input"].shape[1]} timesteps')
    LOGGER.info("="*80)

    if len(ARGS.train_models) > 0:
        assert list(filter(lambda x: x in ModelType.__members__.keys(), ARGS.train_models)), f"train_models must be input one of {ModelType.__members__.keys()}"
        for model_type in ModelType:
            is_training, train_count = True, 0
            while (model_type.name in ARGS.train_models) and (is_training is True) and (train_count < 10):
                try:
                    LOGGER.info(f"===== train model:{model_type.name} =====")
                    train_count += 1
                    model_dir, model_log_dir = model_type.set_save_model_dir(THIS_FILE_DIR, output_file_name, ARGS.corr_type, s_l, w_l)
                    model = model_type.set_model(basic_model_cfg, ARGS)
                    best_model, best_model_info = model.train(train_data=train_dataset, val_data=val_dataset, loss_fns=loss_fns_dict, epochs=ARGS.tr_epochs, show_model_info=True)
                except AssertionError as e:
                    LOGGER.error(f"\n{e}")
                except Exception as e:
                    error_class = e.__class__.__name__  # 取得錯誤類型
                    detail = e.args[0]  # 取得詳細內容
                    cl, exc, tb = sys.exc_info()  # 取得Call Stack
                    last_call_stack = traceback.extract_tb(tb)[-1]  # 取得Call Stack的最後一筆資料
                    file_name = last_call_stack[0]  # 取得發生的檔案名稱
                    line_num = last_call_stack[1]  # 取得發生的行號
                    func_name = last_call_stack[2]  # 取得發生的函數名稱
                    err_msg = "File \"{}\", line {}, in {}: [{}] {}".format(file_name, line_num, func_name, error_class, detail)
                    LOGGER.error(f"===\n{err_msg}")
                    LOGGER.error(f"===\n{traceback.extract_tb(tb)}")
                else:
                    is_training = False
                    if save_model_info:
                        model_dir, model_log_dir = model_type.set_save_model_dir(THIS_FILE_DIR, output_file_name, ARGS.corr_type, s_l, w_l)
                        model.save_model(best_model, best_model_info, model_dir=model_dir, model_log_dir=model_log_dir)
    elif len(ARGS.inference_models) > 0:
        LOGGER.info(f"===== inference model:[{ARGS.inference_models}] on {ARGS.inference_data_split} data =====")
        LOGGER.info("===== if inference_models is more than one, the inference result is ensemble result =====")
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
            model_dir, _ = model_type.set_save_model_dir(THIS_FILE_DIR, output_file_name, ARGS.corr_type, s_l, w_l)
            model_param_path = model_dir.parents[2].joinpath(ARGS.inference_model_paths[0])
            assert model_param_path.exists(), f"{model_param_path} not exists"
            model.load_state_dict(torch.load(model_param_path, map_location=device))
            model.eval()
            loss, edge_acc, preds, y_labels = model.test(inference_data, loss_fns=loss_fns_dict, test_data_split=ARGS.inference_data_split)

        assert preds.shape == y_labels.shape, f"preds.shape:{preds.shape} != y_labels.shape:{y_labels.shape}"
        loss = loss.item() if isinstance(loss, torch.Tensor) else loss
        edge_acc = edge_acc.item() if isinstance(edge_acc, torch.Tensor) else edge_acc
        if ARGS.custom_indices_metric_indices:
            preds = preds[:, ARGS.custom_indices_metric_indices].cpu().numpy()
            y_labels = y_labels[:, ARGS.custom_indices_metric_indices].cpu().numpy()
        else:
            preds, y_labels = preds.cpu().numpy(), y_labels.cpu().numpy()
        if ARGS.output_type == "class_probability":
            if len(ARGS.inference_models) == 1:
                conf_mat_save_fig_dir = THIS_FILE_DIR/f"models/exploration_model_result/model_result_figs/{ARGS.inference_models[0]}/{model_param_path.stem}"
            conf_mat_save_fig_name = f'confusion_matrix-{ARGS.inference_data_split}.png'
            conf_mat_save_fig_path = conf_mat_save_fig_dir/conf_mat_save_fig_name
            num_labels_classes = ARGS.target_mats_path.split("/")[-1].replace("bins_", "").count("_")
            conf_mat_save_fig_dir.mkdir(parents=True, exist_ok=True)
            plot_heatmap(preds, y_labels, num_classes=num_labels_classes, pic_title=ARGS.inference_data_split, can_show_conf_mat=True, save_fig_path=conf_mat_save_fig_path)
        LOGGER.info(f"loss_fns:{loss_fns_dict['fns']}")
        LOGGER.info(f"metric_fn:{basic_model_cfg['metric_fn'] if 'metric_fn' in basic_model_cfg.keys() else None}")
        LOGGER.info(f"Special args of loss_fns: {[(loss_fn, loss_args) for loss_fn, loss_args in loss_fns_dict['fn_args'].items() for arg in loss_args if arg not in ['input', 'target']]}")
        LOGGER.info(f"loss:{loss}, edge_acc:{edge_acc}")
