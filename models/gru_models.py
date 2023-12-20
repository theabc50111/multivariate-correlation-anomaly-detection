#!/usr/bin/env python
# coding: utf-8
import copy
import functools
import json
import re
from collections import OrderedDict
from datetime import datetime
from itertools import product
from pathlib import Path
from pprint import pformat
from typing import overload

import numpy as np
import torch
from torch.nn import GRU, Dropout, Linear, Sequential, Softmax
from tqdm import tqdm

from utils.log_utils import Log, TqdmToLogger

LOGGER = Log().init_logger(logger_name=__name__)


class GRUCorrClass(torch.nn.Module):
    """
    GRU model for predicting correlation class
    """
    def __init__(self, model_cfg: dict, **unused_kwargs):
        super().__init__()
        # set model config
        self.model_cfg = model_cfg
        self.num_labels_classes = self.model_cfg["num_labels_classes"] if self.model_cfg.get("num_labels_classes") else 0
        self.num_tr_batches = self.model_cfg["num_batches"]['train']
        self.num_val_batches = self.model_cfg["num_batches"]['val']
        self.num_test_batches = self.model_cfg["num_batches"]['test']
        self.best_model_info = {}
        self.epoch_metrics = {}
        self.init_epoch_metrics(loss_fns=self.model_cfg["loss_fns"])
        # set model components
        self.fc_dec_out_dim = self.model_cfg["gru_in_dim"]
        self.class_fc_out_dim = self.model_cfg["gru_in_dim"]
        self.gru = GRU(input_size=self.model_cfg['gru_in_dim'], hidden_size=self.model_cfg['gru_h'], num_layers=self.model_cfg['gru_l'], dropout=self.model_cfg["drop_p"] if "gru" in self.model_cfg["drop_pos"] else 0, batch_first=True)
        self.fc_decoder = Sequential(Linear(self.model_cfg['gru_h'], self.fc_dec_out_dim), Dropout(self.model_cfg["drop_p"] if "fc_decoder" in self.model_cfg["drop_pos"] else 0))
        for class_i in range(self.num_labels_classes):
            setattr(self, f"class_fc{class_i}", Sequential(Linear(self.fc_dec_out_dim, self.class_fc_out_dim), Dropout(self.model_cfg["drop_p"] if "class_fc" in self.model_cfg["drop_pos"] else 0)))
        self.softmax = Softmax(dim=1)
        if type(self) == GRUCorrClass:
            self.init_optimizer()

    # this forward() is based on reference to CNNOneDimGRUResMapCorrClass.forward()
    def forward(self, x, *unused_args, **unused_kwargs):
        batch_size = x.shape[0]
        batch_pred_probs = torch.empty(batch_size, self.num_labels_classes, self.class_fc_out_dim).fill_(np.nan)  # (batch_size, num_labels_classes, class_fc_out_dim)
        gru_output, _ = self.gru(x)  # (batch_size, seq_len, gru_h)
        fc_dec_output = self.fc_decoder(gru_output[:, -1, :]).unsqueeze(1)  # (batch_size, 1, fc_dec_out_dim), gru_output[-1] => only take last time-step
        for class_i in range(self.num_labels_classes):
            class_fc_output = getattr(self, f"class_fc{class_i}")(fc_dec_output)  # (batch_size, 1, class_fc_out_dim)
            logits = class_fc_output if class_i == 0 else torch.cat([logits, class_fc_output], dim=1)  # In the end of loop, logits.shape: (batch_size, num_labels_classes, class_fc_out_dim)
        batch_pred_probs = self.softmax(logits)  # (batch_size, num_labels_classes, class_fc_out_dim)
        assert not torch.isnan(batch_pred_probs).any(), f"batch_pred_probs contains NaN, batch_pred_probs: {batch_pred_probs}"
        assert torch.isclose(batch_pred_probs.sum(dim=1), torch.ones(batch_size, self.class_fc_out_dim)).all(), f"batch_pred_probs.sum(dim=1) must be close to 1, but batch_pred_probs.sum(dim=1)={batch_pred_probs.sum(dim=1)}"

        return batch_pred_probs

    def init_optimizer(self):
        """
        Initialize optimizer
        """
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        if self.model_cfg['can_use_optim_scheduler']:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.num_tr_batches*50, gamma=0.5)
        LOGGER.debug("!"*100)
        LOGGER.debug(f"{self.__class__} parameters:")
        for name, param in self.named_parameters():
            LOGGER.debug(name, param.shape)
        LOGGER.debug("!"*100)

    def init_best_model_info(self, train_data: dict, val_data: dict, loss_fns: dict, epochs: int):
        """
        Initialize best_model_info
        """
        self.best_model_info = {"num_train_data": train_data['model_input'].shape[1],
                                "num_val_data": val_data['model_input'].shape[1],
                                "seq_len": self.model_cfg['seq_len'],
                                "epochs": epochs,
                                "batch_size": self.model_cfg['batch_size'],
                                "fold_idx": self.model_cfg['fold_idx'],
                                "tr_batches_per_epoch": self.num_tr_batches,
                                "val_batches_per_epoch": self.num_val_batches,
                                "opt_lr": self.model_cfg['learning_rate'],
                                "opt_weight_decay": self.model_cfg['weight_decay'],
                                "optimizer": str(self.optimizer),
                                "gru_l": self.model_cfg['gru_l'],
                                "gru_h": self.model_cfg['gru_h'],
                                "loss_fns": [fn.__name__ if hasattr(fn, '__name__') else str(fn) for fn in loss_fns["fns"]],
                                "loss_weight": [{fn.__name__ if hasattr(fn, '__name__') else str(fn): str(getattr(fn, "weight", None))} for fn in loss_fns["fns"]],
                                "tol_edge_acc_loss_atol": self.model_cfg['tol_edge_acc_loss_atol'],
                                "drop_pos": self.model_cfg["drop_pos"],
                                "drop_p": self.model_cfg["drop_p"],
                                "max_val_edge_acc": float("-inf"),
                                "output_type": self.model_cfg['output_type'],
                                "model_input_cus_bins": self.model_cfg['model_input_cus_bins'],
                                "target_data_bins": self.model_cfg['target_data_bins']}
        if self.model_cfg.get("custom_indices_loss_indices"):
            self.best_model_info["custom_indices_loss_indices"] = self.model_cfg.get("custom_indices_loss_indices")
        if self.model_cfg.get("metric_fn"):
            self.best_model_info["metric_fn"] = str(self.model_cfg.get("metric_fn"))
        if self.model_cfg.get("custom_indices_metric_indices"):
            self.best_model_info["custom_indices_metric_indices"] = self.model_cfg.get("custom_indices_metric_indices")
        if hasattr(self, 'scheduler'):
            if hasattr(self.scheduler, '_milestones'):
                self.best_model_info["opt_scheduler"] = {"gamma": self.scheduler._schedulers[1].gamma, "milestoines": self.scheduler._milestones+list(self.scheduler._schedulers[1].milestones)}
            else:
                self.best_model_info["opt_scheduler"] = str(self.scheduler.__class__.__name__)
        else:
            self.best_model_info["opt_scheduler"] = None

        return self.best_model_info

    def init_epoch_metrics(self, loss_fns: dict):
        """
        Initialize epoch_metrics
        """
        self.epoch_metrics = {"tr_loss": torch.zeros(1),
                              "val_loss": torch.zeros(1),
                              "tr_edge_acc": torch.zeros(1),
                              "val_edge_acc": torch.zeros(1),
                              "gru_gradient": torch.zeros(1),
                              "fc_dec_gradient": torch.zeros(1),
                              "class_fc_gradient": torch.zeros(1)}
        data_split = ["train", "val"]
        self.epoch_metrics.update({(sp+"_"+str(loss_fn)): torch.zeros(1) for sp, loss_fn in product(data_split, loss_fns["fns"])})

        return self.epoch_metrics

    def infer_batch_data(self, batch_data: list):
        """
        Infer batch data and return batch_pred_probs, batch_preds and batch_y_labels
        """
        x, y = batch_data[0], batch_data[1]
        batch_pred_probs = self.forward(x, output_type=self.model_cfg['output_type'])
        batch_preds = torch.argmax(batch_pred_probs, dim=1)
        batch_y_labels = (y+1).to(torch.long)

        return batch_pred_probs, batch_preds, batch_y_labels

    def calc_batch_loss_edge_acc_metric(self, loss_fns: dict, loss_fn_input: torch.Tensor, loss_fn_target: torch.Tensor, num_batches: int,
                                        edge_acc_metric_input: torch.Tensor = None, edge_acc_metric_target: torch.Tensor = None):
        """
        Calculate loss function
        """
        has_calc_edge_acc = False
        batch_loss = torch.zeros(1)
        batch_loss_each_loss_fn = {}
        for fn in loss_fns["fns"]:
            fn_name = fn.__name__ if hasattr(fn, '__name__') else str(fn)
            loss_fns["fn_args"][fn_name].update({"input": loss_fn_input, "target": loss_fn_target})
            partial_fn = functools.partial(fn, **loss_fns["fn_args"][fn_name])
            loss = partial_fn()
            batch_loss += loss
            batch_loss_each_loss_fn[fn_name] = loss
            if has_calc_edge_acc:
                continue
            if self.model_cfg.get("metric_fn"):
                edge_acc = self.model_cfg.get("metric_fn")(loss_fn_input, loss_fn_target)
            elif edge_acc_metric_input is not None and edge_acc_metric_target is not None:
                edge_acc = (edge_acc_metric_input == edge_acc_metric_target).to(torch.float).mean()
            else:
                edge_acc = torch.zeros(1)
            has_calc_edge_acc = True
            batch_edge_acc = edge_acc

        return batch_loss, batch_edge_acc, batch_loss_each_loss_fn

    def update_weights(self, batch_loss: torch.Tensor):
        """
        Update weight of the model
        """
        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()
        if hasattr(self, "scheduler"):
            self.scheduler.step()

    def record_epoch_metrics_each_batch(self, batch_loss: torch.Tensor, batch_loss_each_loss_fn, batch_edge_acc: torch.Tensor, num_batches: int, rec_stage: str):
        """
        Record metrics at each batch and update to `epoch_metrics`.
        """
        assert rec_stage in ["train", "val", "test"], "rec_stage must be 'train' or 'val' or 'test'"
        epoch_metrics = self.epoch_metrics
        if rec_stage == "train":
            epoch_metrics["tr_edge_acc"] += batch_edge_acc/num_batches
            epoch_metrics["tr_loss"] += batch_loss/num_batches
            epoch_metrics["gru_gradient"] += sum(p.grad.abs().sum() for p in self.gru.parameters() if p.grad is not None)/num_batches
            epoch_metrics["fc_dec_gradient"] += sum(p.grad.abs().sum() for p in self.fc_decoder.parameters() if p.grad is not None)/num_batches
            epoch_metrics["class_fc_gradient"] += sum(p.grad.abs().sum() for attr_name in dir(self) if re.match(r"class_fc\d+", attr_name) for p in getattr(self, attr_name).parameters() if p.grad is not None)/num_batches
        elif rec_stage == "val":
            epoch_metrics['val_edge_acc'] += batch_edge_acc/num_batches
            epoch_metrics['val_loss'] += batch_loss/num_batches
        elif rec_stage == "test":
            pass
        for fn_name, loss in batch_loss_each_loss_fn.items():
            epoch_metrics[(rec_stage+"_"+fn_name)] += loss/num_batches

    def record_history(self, last_batch_output: dict, epoch_i: int):
        """
        Record training history
        """
        best_model_info = self.best_model_info
        epoch_metrics = self.epoch_metrics
        total_epochs = self.best_model_info["epochs"]
        if epoch_i in np.linspace(0, total_epochs-1, 5, dtype=int):
            epoch_metrics["tr_preds"] = last_batch_output['tr_preds']  # only record the last batch
            epoch_metrics["tr_labels"] = last_batch_output['tr_labels']
            epoch_metrics["val_preds"] = last_batch_output['val_preds']
            epoch_metrics["val_labels"] = last_batch_output['val_labels']
        for k, v in epoch_metrics.items():
            history_list = best_model_info.setdefault(k+"_history", [])
            if isinstance(v, torch.Tensor):
                if v.dim() == 0 or (v.dim() == 1 and v.shape[0] == 1):
                    history_list.append(v.item())
                elif v.dim() >= 2 or v.shape[0] > 1:
                    history_list.append(v.cpu().detach().numpy().tolist())
            else:
                history_list.append(v)
        if epoch_i == 0:
            best_model_info["model_structure"] = str(self)

    def ret_best_model(self, epoch_i: int):
        """
        Return best model
        """
        best_model_info = self.best_model_info
        epoch_metrics = self.epoch_metrics
        best_model = None
        if best_model_info.get("max_val_edge_acc") is not None:
            if epoch_metrics['val_edge_acc'] > best_model_info["max_val_edge_acc"]:
                best_model = copy.deepcopy(self.state_dict())
                best_model_info["best_val_epoch"] = epoch_i
                best_model_info["max_val_edge_acc_val_loss"] = epoch_metrics['val_loss'].item()
                best_model_info["max_val_edge_acc"] = epoch_metrics['val_edge_acc'].item()
        elif best_model_info.get("min_val_loss") is not None:
            if epoch_metrics['val_loss'] < best_model_info["min_val_loss"]:
                best_model = copy.deepcopy(self.state_dict())
                best_model_info["best_val_epoch"] = epoch_i
                best_model_info["min_val_loss"] = epoch_metrics['val_loss'].item()
                best_model_info["min_val_loss_edge_acc"] = epoch_metrics['val_edge_acc'].item()

        return best_model

    def show_model_config(self):
        """
        Show model information
        """
        observe_model_cfg = {item[0]: item[1] for item in self.model_cfg.items() if item[0] != 'dataset'}
        observe_model_cfg['optimizer'] = str(self.optimizer)
        if hasattr(self, 'scheduler'):
            if hasattr(self.scheduler, '_milestones'):
                observe_model_cfg['scheduler'] = {"scheduler_name": str(self.scheduler.__class__.__name__), "milestones": self.scheduler._milestones+list(self.scheduler._schedulers[1].milestones), "gamma": self.scheduler._schedulers[1].gamma}
            else:
                observe_model_cfg['scheduler'] = {"scheduler_name": str(self.scheduler.__class__.__name__)}
        LOGGER.info(f"\nModel Configuration of {self.__class__}: \n{pformat(observe_model_cfg, indent=1, width=200, compact=True)}")
        LOGGER.info("-"*30)
        LOGGER.info(f"\nInital best_model_info of {self.__class__}: \n{pformat(self.best_model_info, indent=1, width=200, compact=True)}")
        LOGGER.info("="*80)

    def show_training_process(self, epoch_i: int, batch_idx: int, last_batch_data: list, last_batch_output: dict):
        """
        Show training process
        """
        epoch_metrics = self.epoch_metrics
        if epoch_i == 0:
            LOGGER.info(f"\nModel Structure: \n{self}")
        if epoch_i % 10 == 0:  # show metrics every 10 epochs
            epoch_metric_log_msgs = " | ".join([f"{k}: {v.item():.8f}" for k, v in epoch_metrics.items() if v.dim() < 2])
            LOGGER.info(f"In Epoch {epoch_i:>3} | {epoch_metric_log_msgs} | lr: {self.optimizer.param_groups[0]['lr']:.9f}")
        if epoch_i % 100 == 0:  # show oredictive and real adjacency matrix every 500 epochs
            x, y = last_batch_data[0], last_batch_data[1]
            preds, y = last_batch_output['tr_preds'], last_batch_output['tr_labels']
            seq_len = self.model_cfg['seq_len']
            LOGGER.info("="*50)
            LOGGER.info(f"epoch_i: {epoch_i:>3}, batch_idx: {batch_idx:>3}, data_batch_idx:0")
            LOGGER.info(f"x.shape: {x.shape}, y.shape: {y.shape}, preds.shape: {preds.shape}")
            LOGGER.info(f"\nIn Epoch {epoch_i:>3}, batch_idx:{batch_idx}, data_batch_idx:0, input_corr_data_seq_len:{seq_len} \ninput_corr_data[0, {seq_len-1}, :5]:\n{x[0, seq_len-1, :5]}\npred_corr_data[0, :5]:\n{preds[0, :5]}\ny_labels[0, :5]:\n{y[0, :5]}")
            LOGGER.info("="*50)

    @overload
    def train(self, mode: bool = True) -> torch.nn.Module:
        ...

    @overload
    def train(self, train_data: np.ndarray = None, val_data: np.ndarray = None, loss_fns: dict = None, epochs: int = 1000) -> tuple:
        ...

    def train(self, mode: bool = True, train_data: np.ndarray = None, val_data: np.ndarray = None, loss_fns: dict = None, epochs: int = 1000, **unused_kwargs):
        """
        Train model
        """
        # In order to make original function of nn.Module.train() work, we need to override it
        super().train(mode=mode)
        if train_data is None:
            return self
        # Train on epochs
        assert self.model_cfg['output_type'] == "class_probability", "output_type must be class_probability"
        assert self.num_labels_classes == np.unique(train_data['target']).shape[0], f"num_labels_classes must be equal to the number of unique target in {self.__class__}, but self.num_labels_classes={self.num_labels_classes} and np.unique(train_data['target']).shape[0]={np.unique(train_data['target']).shape[0]}"
        self.init_best_model_info(train_data, val_data, loss_fns, epochs)
        self.show_model_config()
        best_model = []
        tqdm_out = TqdmToLogger(LOGGER)
        for epoch_i in tqdm(range(epochs), file=tqdm_out, miniters=10, desc="Training progress"):
            self.train()
            self.init_epoch_metrics(loss_fns)
            # Train on batches
            train_loader = self.yield_batch_data(model_input_data=train_data['model_input'], target_data=train_data['target'], batch_size=self.model_cfg['batch_size'], seq_len=self.model_cfg['seq_len'])
            for batch_idx, batch_data in enumerate(train_loader):
                batch_pred_probs, batch_preds, batch_y_labels = self.infer_batch_data(batch_data)
                calc_batch_loss_edge_acc_metric_kwargs = {"loss_fns": loss_fns, "loss_fn_input": batch_pred_probs, "loss_fn_target": batch_y_labels,
                                                          "edge_acc_metric_input": batch_preds, "edge_acc_metric_target": batch_y_labels, "num_batches": self.num_tr_batches}
                batch_loss, batch_edge_acc, batch_loss_each_loss_fn = self.calc_batch_loss_edge_acc_metric(**calc_batch_loss_edge_acc_metric_kwargs)
                self.update_weights(batch_loss)
                self.record_epoch_metrics_each_batch(batch_loss, batch_loss_each_loss_fn, batch_edge_acc, self.num_tr_batches, rec_stage="train")
            # Validation
            _, _, val_preds, val_labels = self.test(val_data, loss_fns=loss_fns, test_data_split="val")
            # record training history
            last_batch_output = {'tr_preds': batch_preds, 'tr_labels': batch_y_labels, 'val_preds': val_preds, 'val_labels': val_labels}
            self.record_history(last_batch_output, epoch_i)
            # get best model
            ret_model = self.ret_best_model(epoch_i)
            if ret_model is not None:
                best_model = ret_model
            # show training process
            self.show_training_process(epoch_i, batch_idx, batch_data, last_batch_output)

        # check if best_model is empty
        assert bool(best_model), "best_model is empty"

        return best_model, self.best_model_info

    def test(self, test_data: np.ndarray, loss_fns: dict, test_data_split: str):
        assert test_data_split in ["val", "test"], "test_data_split must be 'val' or 'test'"
        self.eval()
        test_loss = 0
        test_edge_acc = 0
        if test_data_split == "train":
            num_batches = self.num_tr_batches
        elif test_data_split == "val":
            num_batches = self.num_val_batches
        elif test_data_split == "test":
            num_batches = self.num_test_batches
        test_loader = self.yield_batch_data(model_input_data=test_data['model_input'], target_data=test_data['target'], batch_size=self.model_cfg['batch_size'], seq_len=self.model_cfg['seq_len'])
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(test_loader):
                batch_pred_probs, batch_preds, batch_y_labels = self.infer_batch_data(batch_data)
                calc_batch_loss_edge_acc_metric_kwargs = {"loss_fns": loss_fns, "loss_fn_input": batch_pred_probs, "loss_fn_target": batch_y_labels,
                                                          "edge_acc_metric_input": batch_preds, "edge_acc_metric_target": batch_y_labels, "num_batches": num_batches}
                batch_loss, batch_edge_acc, batch_loss_each_loss_fn = self.calc_batch_loss_edge_acc_metric(**calc_batch_loss_edge_acc_metric_kwargs)
                self.record_epoch_metrics_each_batch(batch_loss, batch_loss_each_loss_fn, batch_edge_acc, num_batches, rec_stage=test_data_split)
                # record test loss and edge_acc and preds and y_labels
                # it's designed to be used in inference stage
                test_edge_acc += batch_edge_acc/num_batches
                test_loss += batch_loss/num_batches
                test_preds = batch_preds if batch_idx == 0 else torch.cat((test_preds, batch_preds), dim=0)
                test_y_labels = batch_y_labels if batch_idx == 0 else torch.cat((test_y_labels, batch_y_labels), dim=0)

        return test_loss, test_edge_acc, test_preds, test_y_labels

    @staticmethod
    def save_model(unsaved_model: OrderedDict, model_info: dict, model_dir: Path, model_log_dir: Path):
        e_i = model_info.get("best_val_epoch")
        t_stamp = datetime.strftime(datetime.now(), "%Y%m%d%H%M%S")
        saved_model_name_prefix = f"epoch_{e_i}-{t_stamp}"
        torch.save(unsaved_model, model_dir/f"{saved_model_name_prefix}.pt")
        with open(model_log_dir/f"{saved_model_name_prefix}.json", "w") as f:
            json_str = json.dumps(model_info)
            f.write(json_str)
        LOGGER.info(f"model has been saved in:{model_dir}")

        return saved_model_name_prefix


    @staticmethod
    def yield_batch_data(model_input_data: np.ndarray, target_data: np.ndarray, seq_len: int = 10, batch_size: int = 5):
        """
        Yield batch data
        """
        num_pairs, all_timesetps = model_input_data.shape[0], model_input_data.shape[1]-1  # the graph of last "t" can't be used as train data
        input_arr = model_input_data.T
        target_arr = target_data.T
        for batch_start_t in range(0, all_timesetps, batch_size):
            cur_batch_size = batch_size if batch_start_t+batch_size <= all_timesetps-seq_len else all_timesetps-seq_len-batch_start_t
            if cur_batch_size <= 0:
                break
            batch_x = torch.empty((cur_batch_size, seq_len, num_pairs)).fill_(np.nan)
            batch_y = torch.empty((cur_batch_size, num_pairs)).fill_(np.nan)
            for data_batch_idx in range(cur_batch_size):
                begin_t, end_t = batch_start_t+data_batch_idx, batch_start_t+data_batch_idx+seq_len
                batch_x[data_batch_idx] = torch.tensor(input_arr[begin_t:end_t])
                batch_y[data_batch_idx] = torch.tensor(target_arr[end_t])

            assert not torch.isnan(batch_x).any() or not torch.isnan(batch_y).any(), "batch_x or batch_y contains nan"

            yield batch_x, batch_y


class GRUCorrClassCustomFeatures(GRUCorrClass):
    """
    GRU model for predicting correlation class with custom input features
    """
    def __init__(self, model_cfg: dict, **unused_kwargs):
        super(GRUCorrClassCustomFeatures, self).__init__(model_cfg, **unused_kwargs)
        if type(self) == GRUCorrClassCustomFeatures:
            self.init_optimizer()

    def init_best_model_info(self, train_data: dict, val_data: dict, loss_fns: dict, epochs: int):
        """
        Initialize best_model_info
        """
        super().init_best_model_info(train_data, val_data, loss_fns, epochs)
        self.best_model_info.update({"input_feature_idx": self.model_cfg["input_feature_idx"]})

        return self.best_model_info

    def yield_batch_data(self, model_input_data: np.ndarray, target_data: np.ndarray, seq_len: int = 10, batch_size: int = 5):
        """
        Yield batch data
        """
        assert (self.model_cfg["input_feature_idx"] is not None) and len(self.model_cfg["input_feature_idx"]) >= 1, "input_feature_idx must be a list of int"
        _, all_timesetps = model_input_data.shape[0], model_input_data.shape[1]-1  # the graph of last "t" can't be used as train data
        selected_feature_idx = self.model_cfg["input_feature_idx"]
        input_arr = model_input_data[selected_feature_idx, ::].T
        target_arr = target_data[selected_feature_idx, ::].T
        for batch_start_t in range(0, all_timesetps, batch_size):
            cur_batch_size = batch_size if batch_start_t+batch_size <= all_timesetps-seq_len else all_timesetps-seq_len-batch_start_t
            if cur_batch_size <= 0:
                break
            batch_x = torch.empty((cur_batch_size, seq_len, len(selected_feature_idx))).fill_(np.nan)
            batch_y = torch.empty((cur_batch_size, len(selected_feature_idx))).fill_(np.nan)
            for data_batch_idx in range(cur_batch_size):
                begin_t, end_t = batch_start_t+data_batch_idx, batch_start_t+data_batch_idx+seq_len
                batch_x[data_batch_idx] = torch.tensor(input_arr[begin_t:end_t])
                batch_y[data_batch_idx] = torch.tensor(target_arr[end_t])

            assert not torch.isnan(batch_x).any() or not torch.isnan(batch_y).any(), "batch_x or batch_y contains nan"

            yield batch_x, batch_y


class GRUCorrClassOneFeature(GRUCorrClass):
    """
    GRU model with one input feature for predicting correlation class
    """
    def __init__(self, model_cfg: dict, **unused_kwargs):
        super(GRUCorrClassOneFeature, self).__init__(model_cfg, **unused_kwargs)
        # set model config
        self.model_cfg = model_cfg
        del self.model_cfg["gru_in_dim"]
        self.num_gru = self.model_cfg["num_gru"]
        self.gru_in_dim = 1
        self.fc_dec_out_dim = self.gru_in_dim
        self.class_fc_out_dim = self.gru_in_dim
        # set model components
        del_attr_names = ["gru", "fc_decoder"] + [f"class_fc{class_i}" for class_i in range(self.num_labels_classes)]
        for attr_name in dir(self):
            if attr_name in del_attr_names:
                delattr(self, attr_name)
        for gru_i in range(self.num_gru):
            setattr(self, f"gru{gru_i}", GRU(input_size=self.gru_in_dim, hidden_size=self.model_cfg["gru_h"], num_layers=self.model_cfg["gru_l"], dropout=self.model_cfg["drop_p"] if "gru" in self.model_cfg["drop_pos"] else 0, batch_first=True))
            setattr(self, f"gru{gru_i}_fc_decoder", Sequential(Linear(self.model_cfg["gru_h"], self.fc_dec_out_dim), Dropout(self.model_cfg["drop_p"] if "fc_decoder" in self.model_cfg["drop_pos"] else 0)))
            for class_i in range(self.num_labels_classes):
                setattr(self, f"gru{gru_i}_class_fc{class_i}", Sequential(Linear(self.fc_dec_out_dim, self.class_fc_out_dim), Dropout(self.model_cfg["drop_p"] if "class_fc" in self.model_cfg["drop_pos"] else 0)))
        if type(self) == GRUCorrClassOneFeature:
            self.init_optimizer()

    # this forward() is based on reference to CNNOneDimGRUResMapCorrClass.forward()
    def forward(self, x, *unused_args, **unused_kwargs):
        batch_size = x.shape[0]
        split_x = torch.split(x, 1, dim=2)  # (batch_size, 1, seq_len)*num_pairs
        assert len(split_x) == self.model_cfg["num_gru"], f"len(split_x): {len(split_x)}, but it should be {self.model_cfg['num_gru']}"
        for gru_i, x_each_pair in enumerate(split_x):
            gru_input = x_each_pair  # (batch_size, seq_len, gru_in_dim)
            gru_output, _ = getattr(self, f"gru{gru_i}")(gru_input)  # (batch_size, seq_len, gru_h)
            fc_dec_output = getattr(self, f"gru{gru_i}_fc_decoder")(gru_output[:, -1, :]).unsqueeze(1)  # (batch_size, 1, fc_dec_out_dim), gru_output[-1] => only take last time-step
            for class_i in range(self.num_labels_classes):
                class_fc_output = getattr(self, f"gru{gru_i}_class_fc{class_i}")(fc_dec_output)  # (batch_size, 1, class_fc_out_dim)
                class_logits = class_fc_output if class_i == 0 else torch.cat((class_logits, class_fc_output), dim=1)  # (batch_size, num_labels_classes, class_fc_out_dim)
            logits = class_logits if gru_i == 0 else torch.cat((logits, class_logits), dim=2)  # (batch_size, num_labels_classes, num_out_channels*class_fc_out_dim)
        batch_pred_probs = self.softmax(logits)
        assert batch_pred_probs.shape == (batch_size, self.num_labels_classes, self.num_gru*self.class_fc_out_dim), f"batch_pred_probs.shape: {batch_pred_probs.shape}, but it should be (batch_size, self.num_labels_classes, self.num_gur*class_fc_out_dim)"
        assert torch.isclose(batch_pred_probs.sum(dim=1), torch.ones(batch_size, self.num_gru*self.class_fc_out_dim)).all(), f"batch_pred_probs.sum(dim=1): {batch_pred_probs.sum(dim=1)}, but it should all be 1"

        return batch_pred_probs

    def record_epoch_metrics_each_batch(self, batch_loss: torch.Tensor, batch_loss_each_loss_fn, batch_edge_acc: torch.Tensor, num_batches: int, rec_stage: str):
        """
        Record metrics at each batch and update to `epoch_metrics`.
        """
        assert rec_stage in ["train", "val", "test"], "rec_stage must be 'train' or 'val' or 'test'"
        epoch_metrics = self.epoch_metrics
        if rec_stage == "train":
            epoch_metrics["tr_edge_acc"] += batch_edge_acc/num_batches
            epoch_metrics["tr_loss"] += batch_loss/num_batches
            for attr in dir(self):
                if re.match(r"gru\d+", attr) and isinstance(getattr(self, attr), GRU):
                    epoch_metrics["gru_gradient"] += sum(p.grad.abs().sum() for p in getattr(self, attr).parameters() if p.grad is not None)/num_batches
                elif re.match(r"gru\d+_fc_decoder", attr) and isinstance(getattr(self, attr), Sequential):
                    epoch_metrics["fc_dec_gradient"] += sum(p.grad.abs().sum() for p in getattr(self, attr).parameters() if p.grad is not None)/num_batches
                elif re.match(r"gru\d+_class_fc\d+", attr) and isinstance(getattr(self, attr), Sequential):
                    epoch_metrics["class_fc_gradient"] += sum(p.grad.abs().sum() for p in getattr(self, attr).parameters() if p.grad is not None)/num_batches

        elif rec_stage == "val":
            epoch_metrics['val_edge_acc'] += batch_edge_acc/num_batches
            epoch_metrics['val_loss'] += batch_loss/num_batches
        elif rec_stage == "test":
            pass
        for fn_name, loss in batch_loss_each_loss_fn.items():
            epoch_metrics[(rec_stage+"_"+fn_name)] += loss/num_batches


class GRUCorrCoefPred(GRUCorrClass):
    """
    GRU model for predicting correlation coefficient
    """
    def __init__(self, model_cfg: dict, **unused_kwargs):
        super(GRUCorrCoefPred, self).__init__(model_cfg, **unused_kwargs)
        # set model config
        self.model_cfg = model_cfg
        # set model components
        if type(self) == GRUCorrCoefPred:
            self.init_optimizer()

    def forward(self, x, *unused_args, **unused_kwargs):
        gru_output, _ = self.gru(x)
        for data_batch_idx in range(x.shape[0]):
            pred = self.fc_decoder(gru_output[data_batch_idx, -1, :])  # gru_output[-1] => only take last time-step
            batch_preds = pred.reshape(1, -1) if data_batch_idx == 0 else torch.cat((batch_preds, pred.reshape(1, -1)), dim=0)

        return batch_preds

    def init_best_model_info(self, train_data: dict, val_data: dict, loss_fns: dict, epochs: int):
        """
        Initialize best_model_info
        """
        super().init_best_model_info(train_data, val_data, loss_fns, epochs)
        self.best_model_info.update({"min_val_loss": float('inf')})
        del self.best_model_info["max_val_edge_acc"]

        return self.best_model_info

    def init_epoch_metrics(self, loss_fns: dict):
        """
        Initialize epoch_metrics
        """
        super().init_epoch_metrics(loss_fns)
        del self.epoch_metrics["class_fc_gradient"]

        return self.epoch_metrics

    def infer_batch_data(self, batch_data: list):
        """
        Infer batch data and return predicted labels and real labels
        """
        batch_x, batch_y = batch_data[0], batch_data[1]
        batch_preds = self.forward(batch_x)

        return batch_preds, batch_y

    def record_epoch_metrics_each_batch(self, batch_loss: torch.Tensor, batch_loss_each_loss_fn: dict, batch_edge_acc: torch.Tensor, num_batches: int, rec_stage: str):
        """
        Record metrics at each batch and update to `epoch_metrics`.
        """
        assert rec_stage in ["train", "val", "test"], "rec_stage must be 'train' or 'val' or 'test'"
        epoch_metrics = self.epoch_metrics
        if rec_stage == "train":
            epoch_metrics["tr_edge_acc"] += batch_edge_acc/num_batches
            epoch_metrics["tr_loss"] += batch_loss/num_batches
            epoch_metrics["gru_gradient"] += sum(p.grad.abs().sum() for p in self.gru.parameters() if p.grad is not None)/num_batches
            epoch_metrics["fc_dec_gradient"] += sum(p.grad.abs().sum() for p in self.fc_decoder.parameters() if p.grad is not None)/num_batches
        elif rec_stage == "val":
            epoch_metrics['val_edge_acc'] += batch_edge_acc/num_batches
            epoch_metrics['val_loss'] += batch_loss/num_batches
        elif rec_stage == "test":
            pass
        for fn_name, loss in batch_loss_each_loss_fn.items():
            epoch_metrics[(rec_stage+"_"+fn_name)] += loss/num_batches

    def train(self, mode: bool = True, train_data: np.ndarray = None, val_data: np.ndarray = None, loss_fns: dict = None, epochs: int = 1000, **unused_kwargs):
        # In order to make original function of nn.Module.train() work, we need to override it
        super().train(mode=mode)
        if train_data is None:
            return self
        assert self.model_cfg['output_type'] == "corr_coef", "output_type must be corr_coef"
        assert self.num_labels_classes == 0, "num_labels_classes must be equal to 0 in {self.__class__}, but self.num_labels_classes={self.num_labels_classes}"
        self.init_best_model_info(train_data, val_data, loss_fns, epochs)
        self.show_model_config()
        best_model = []
        # Train on epochs
        for epoch_i in tqdm(range(epochs)):
            self.train()
            self.init_epoch_metrics(loss_fns)
            # Train on batches
            train_loader = self.yield_batch_data(model_input_data=train_data['model_input'], target_data=train_data['target'], batch_size=self.model_cfg['batch_size'], seq_len=self.model_cfg['seq_len'])
            for batch_idx, batch_data in enumerate(train_loader):
                batch_preds, batch_y_labels = self.infer_batch_data(batch_data)
                calc_batch_loss_edge_acc_metric_kwargs = {"loss_fns": loss_fns, "loss_fn_input": batch_preds, "loss_fn_target": batch_y_labels, "num_batches": self.num_tr_batches}
                batch_loss, batch_edge_acc, batch_loss_each_loss_fn = self.calc_batch_loss_edge_acc_metric(**calc_batch_loss_edge_acc_metric_kwargs)
                self.update_weights(batch_loss)
                self.record_epoch_metrics_each_batch(batch_loss, batch_loss_each_loss_fn, batch_edge_acc, self.num_tr_batches, rec_stage="train")
            # Validation
            _, _, val_preds, val_labels = self.test(val_data, loss_fns=loss_fns, test_data_split="val")
            # record training history and save best model
            last_batch_output = {'tr_preds': batch_preds, 'tr_labels': batch_y_labels, 'val_preds': val_preds, 'val_labels': val_labels}
            self.record_history(last_batch_output, epoch_i)
            # get best model
            ret_model = self.ret_best_model(epoch_i)
            if ret_model is not None:
                best_model = ret_model
            # show training process
            self.show_training_process(epoch_i, batch_idx, batch_data, last_batch_output)
        # check if best_model is empty
        assert bool(best_model), "best_model is empty"

        return best_model, self.best_model_info

    def test(self, test_data: np.ndarray, loss_fns: dict, test_data_split: str):
        assert test_data_split in ["val", "test"], "test_data_split must be 'val' or 'test'"
        self.eval()
        test_loss = 0
        test_edge_acc = 0
        if test_data_split == "train":
            num_batches = self.num_tr_batches
        elif test_data_split == "val":
            num_batches = self.num_val_batches
        elif test_data_split == "test":
            num_batches = self.num_test_batches
        test_loader = self.yield_batch_data(model_input_data=test_data['model_input'], target_data=test_data['target'], batch_size=self.model_cfg['batch_size'], seq_len=self.model_cfg['seq_len'])
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(test_loader):
                batch_preds, batch_y_labels = self.infer_batch_data(batch_data)
                calc_batch_loss_edge_acc_metric_kwargs = {"loss_fns": loss_fns, "loss_fn_input": batch_preds, "loss_fn_target": batch_y_labels, "num_batches": num_batches}
                batch_loss, batch_edge_acc, batch_loss_each_loss_fn = self.calc_batch_loss_edge_acc_metric(**calc_batch_loss_edge_acc_metric_kwargs)
                self.record_epoch_metrics_each_batch(batch_loss, batch_loss_each_loss_fn, batch_edge_acc, num_batches, rec_stage=test_data_split)
                # record test loss and edge_acc and preds and y_labels
                # it's designed to be used in inference stage
                test_edge_acc += batch_edge_acc/num_batches
                test_loss += batch_loss/num_batches
                test_preds = batch_preds if batch_idx == 0 else torch.cat((test_preds, batch_preds), dim=0)
                test_y_labels = batch_y_labels if batch_idx == 0 else torch.cat((test_y_labels, batch_y_labels), dim=0)

        return test_loss, test_edge_acc, test_preds, test_y_labels


