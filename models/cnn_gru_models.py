# Purpose: CNN-GRU models for correlation classification
import re
from math import floor

import torch
from torch.nn import GRU, Conv1d, Dropout, Linear, Sequential, Softmax

from .gru_models import GRUCorrClass


class CNNOneDimGRUCorrClass(GRUCorrClass):
    """
    CNN-GRU model for correlation classification
    """
    def __init__(self, model_cfg: dict, **unused_kwargs):
        super(CNNOneDimGRUCorrClass, self).__init__(model_cfg, **unused_kwargs)
        # set model config
        self.model_cfg = model_cfg
        del self.model_cfg["gru_in_dim"]
        self.seq_len = self.model_cfg["seq_len"]
        self.gru_in_dim = 1
        self.fc_dec_out_dim = self.gru_in_dim
        self.class_fc_out_dim = self.gru_in_dim
        # set model components
        del_attr_names = ["gru", "fc_decoder"] + [f"class_fc{class_i}" for class_i in range(self.num_labels_classes)]
        for attr_name in dir(self):
            if attr_name in del_attr_names:
                delattr(self, attr_name)
        conv1_kernel_size = 3  # kernel_size==2*padding+1 if stride==1
        conv1_padding = 1
        conv1_stride = 1
        conv1_out_seq_len = floor(((self.seq_len+2*conv1_padding-conv1_kernel_size)/conv1_stride)+1)
        assert conv1_out_seq_len == self.seq_len, f"conv1_out_seq_len: {conv1_out_seq_len}, self.seq_len: {self.seq_len} and they should be equal"
        self.conv1 = Conv1d(in_channels=self.model_cfg["cnn_in_channels"], out_channels=self.model_cfg["cnn_in_channels"], kernel_size=conv1_kernel_size, padding=conv1_padding, stride=conv1_stride)
        for channel_i in range(self.conv1.out_channels):
            setattr(self, f"channel{channel_i}_gru", GRU(input_size=self.gru_in_dim, hidden_size=self.model_cfg["gru_h"], num_layers=self.model_cfg["gru_l"], dropout=self.model_cfg["drop_p"] if "gru" in self.model_cfg["drop_pos"] else 0, batch_first=True))
            setattr(self, f"channel{channel_i}_fc_decoder", Sequential(Linear(self.model_cfg["gru_h"], self.fc_dec_out_dim), Dropout(self.model_cfg["drop_p"] if "fc_decoder" in self.model_cfg["drop_pos"] else 0)))
            for class_i in range(self.num_labels_classes):
                setattr(self, f"channel{channel_i}_class_fc{class_i}", Sequential(Linear(self.fc_dec_out_dim, self.class_fc_out_dim), Dropout(self.model_cfg["drop_p"] if "class_fc" in self.model_cfg["drop_pos"] else 0)))
        self.softmax = Softmax(dim=1)
        if type(self) == CNNOneDimGRUCorrClass:
            self.init_optimizer()

    def forward(self, x, *unused_args, **unused_kwargs):
        batch_size, seq_len, num_pairs = x.shape
        conv_input = x.permute(0, 2, 1)  # (batch_size, num_pairs, seq_len)
        conv_output = self.conv1(conv_input)  # (batch_size, num_out_channels, seq_len)
        split_conv_output = torch.split(conv_output, 1, dim=1)  # (batch_size, 1, seq_len)*conv1.out_channels
        for channel_i, conv_output_each_channel in enumerate(split_conv_output):
            trans_conv_output_each_channel = conv_output_each_channel.permute(0, 2, 1)
            gru_input = trans_conv_output_each_channel  # (batch_size, seq_len, gru_in_dim)
            gru_output, _ = getattr(self, f"channel{channel_i}_gru")(gru_input)  # (batch_size, seq_len, gru_h)
            fc_dec_output = getattr(self, f"channel{channel_i}_fc_decoder")(gru_output[:, -1, :]).unsqueeze(1)  # (batch_size, 1, fc_dec_out_dim), gru_output[-1] => only take last time-step
            for class_i in range(self.num_labels_classes):
                class_fc_output = getattr(self, f"channel{channel_i}_class_fc{class_i}")(fc_dec_output)  # (batch_size, 1, class_fc_out_dim)
                channel_logits = class_fc_output if class_i == 0 else torch.cat((channel_logits, class_fc_output), dim=1)  # (batch_size, num_labels_classes, class_fc_out_dim)
            logits = channel_logits if channel_i == 0 else torch.cat((logits, channel_logits), dim=2)  # (batch_size, num_labels_classes, num_out_channels*class_fc_out_dim)
        batch_pred_probs = self.softmax(logits)
        assert batch_pred_probs.shape == (batch_size, self.num_labels_classes, self.conv1.out_channels*self.class_fc_out_dim), f"batch_pred_probs.shape: {batch_pred_probs.shape}, but it should be (batch_size, self.num_labels_classes, self.conv1.out_channels*class_fc_out_dim)"
        assert torch.isclose(batch_pred_probs.sum(dim=1), torch.ones(batch_size, self.conv1.out_channels*self.class_fc_out_dim)).all(), f"batch_pred_probs.sum(dim=1): {batch_pred_probs.sum(dim=1)}, but it should all be 1"

        return batch_pred_probs

    def init_epoch_metrics(self, loss_fns: dict):
        """
        Initialize epoch_metrics
        """
        super().init_epoch_metrics(loss_fns)
        self.epoch_metrics["cnn_gradient"] = torch.zeros(1)

        return self.epoch_metrics

    def record_epoch_metrics_each_batch(self, batch_loss: torch.Tensor, batch_loss_each_loss_fn, batch_edge_acc: torch.Tensor, num_batches: int, rec_stage: str):
        """
        Record metrics at each batch and update to `epoch_metrics`.
        """
        assert rec_stage in ["train", "val", "test"], "rec_stage must be 'train' or 'val' or 'test'"
        epoch_metrics = self.epoch_metrics
        if rec_stage == "train":
            epoch_metrics["tr_edge_acc"] += batch_edge_acc/num_batches
            epoch_metrics["tr_loss"] += batch_loss/num_batches
            epoch_metrics["cnn_gradient"] += sum(p.grad.abs().sum() for p in self.conv1.parameters() if p.grad is not None)/num_batches
            for attr in dir(self):
                if re.match(r"channel\d+_gru", attr) and isinstance(getattr(self, attr), GRU):
                    epoch_metrics["gru_gradient"] += sum(p.grad.abs().sum() for p in getattr(self, attr).parameters() if p.grad is not None)/num_batches
                elif re.match(r"channel\d+_fc_decoder", attr) and isinstance(getattr(self, attr), Sequential):
                    epoch_metrics["fc_dec_gradient"] += sum(p.grad.abs().sum() for p in getattr(self, attr).parameters() if p.grad is not None)/num_batches
                elif re.match(r"channel\d+_class_fc\d+", attr) and isinstance(getattr(self, attr), Sequential):
                    epoch_metrics["class_fc_gradient"] += sum(p.grad.abs().sum() for p in getattr(self, attr).parameters() if p.grad is not None)/num_batches

        elif rec_stage == "val":
            epoch_metrics['val_edge_acc'] += batch_edge_acc/num_batches
            epoch_metrics['val_loss'] += batch_loss/num_batches
        elif rec_stage == "test":
            pass
        for fn_name, loss in batch_loss_each_loss_fn.items():
            epoch_metrics[(rec_stage+"_"+fn_name)] += loss/num_batches


class CNNOneDimGRUResMapCorrClass(CNNOneDimGRUCorrClass):
    """
    CNN-GRU model with residual mapping for correlation classification
    """
    def __init__(self, model_cfg: dict, **unused_kwargs):
        super(CNNOneDimGRUResMapCorrClass, self).__init__(model_cfg, **unused_kwargs)
        if type(self) == CNNOneDimGRUResMapCorrClass:
            self.init_optimizer()

    def forward(self, x, *unused_args, **unused_kwargs):
        batch_size, seq_len, num_pairs = x.shape
        conv_input = x.permute(0, 2, 1)  # (batch_size, num_pairs, seq_len)
        conv_output = self.conv1(conv_input)  # (batch_size, num_out_channels, seq_len)
        split_conv_output = torch.split(conv_output, 1, dim=1)  # (batch_size, 1, seq_len)*conv1.out_channels
        split_x = torch.split(x, 1, dim=2)  # (batch_size, 1, seq_len)*num_pairs
        assert len(split_conv_output) == len(split_x) == self.conv1.out_channels, f"len(split_conv_output): {len(split_conv_output)}, len(split_x): {len(split_x)}, self.conv1.out_channels: {self.conv1.out_channels} and they should be equal"
        for channel_i, (conv_output_each_channel, x_each_pair) in enumerate(zip(split_conv_output, split_x)):
            trans_conv_output_each_channel = conv_output_each_channel.permute(0, 2, 1)
            assert trans_conv_output_each_channel.shape == x_each_pair.shape, f"trans_conv_output_each_channel.shape: {trans_conv_output_each_channel.shape}, x_each_pair.shape: {x_each_pair.shape} and they should be equal"
            gru_input = trans_conv_output_each_channel+x_each_pair  # (batch_size, seq_len, gru_in_dim)
            gru_output, _ = getattr(self, f"channel{channel_i}_gru")(gru_input)  # (batch_size, seq_len, gru_h)
            fc_dec_output = getattr(self, f"channel{channel_i}_fc_decoder")(gru_output[:, -1, :]).unsqueeze(1)  # (batch_size, 1, fc_dec_out_dim), gru_output[-1] => only take last time-step
            for class_i in range(self.num_labels_classes):
                class_fc_output = getattr(self, f"channel{channel_i}_class_fc{class_i}")(fc_dec_output)  # (batch_size, 1, class_fc_out_dim)
                channel_logits = class_fc_output if class_i == 0 else torch.cat((channel_logits, class_fc_output), dim=1)  # (batch_size, num_labels_classes, class_fc_out_dim)
            logits = channel_logits if channel_i == 0 else torch.cat((logits, channel_logits), dim=2)  # (batch_size, num_labels_classes, num_out_channels*class_fc_out_dim)
        batch_pred_probs = self.softmax(logits)
        assert batch_pred_probs.shape == (batch_size, self.num_labels_classes, self.conv1.out_channels*self.class_fc_out_dim), f"batch_pred_probs.shape: {batch_pred_probs.shape}, but it should be (batch_size, self.num_labels_classes, self.conv1.out_channels*class_fc_out_dim)"
        assert torch.isclose(batch_pred_probs.sum(dim=1), torch.ones(batch_size, self.conv1.out_channels*self.class_fc_out_dim)).all(), f"batch_pred_probs.sum(dim=1): {batch_pred_probs.sum(dim=1)}, but it should all be 1"

        return batch_pred_probs
