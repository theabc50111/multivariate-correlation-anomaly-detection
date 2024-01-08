import re

import torch
from torch.nn import (GRU, Dropout, Linear, MultiheadAttention, Sequential,
                      Softmax)

from .gru_models import GRUCorrClass


class AttnOneDimGRUResMapCorrClass(GRUCorrClass):
    """
    GRU with attention and residual mapping for correlation classification.
    """
    def __init__(self, model_cfg: dict, **unused_kwargs):
        super(AttnOneDimGRUResMapCorrClass, self).__init__(model_cfg)
        # set model config
        self.model_cfg = model_cfg
        del self.model_cfg["gru_in_dim"]
        self.seq_len = self.model_cfg["seq_len"]
        self.gru_in_dim = 1
        self.fc_dec_out_dim = self.gru_in_dim
        self.class_fc_out_dim = self.gru_in_dim
        self.attn_num_heads = 1
        self.attn_embed_dim = self.seq_len
        self.attn_out_len = self.model_cfg["num_gru"]
        # set model components
        del_attr_names = ["gru", "fc_decoder"] + [f"class_fc{class_i}" for class_i in range(self.num_labels_classes)]
        for attr_name in dir(self):
            if attr_name in del_attr_names:
                delattr(self, attr_name)
        self.attn1 = MultiheadAttention(embed_dim=self.attn_embed_dim, num_heads=self.attn_num_heads, batch_first=True, dropout=self.model_cfg["drop_p"] if "attn" in self.model_cfg["drop_pos"] else 0)
        for attn_len_i in range(self.attn_out_len):
            setattr(self, f"attn_l_id{attn_len_i}_gru", GRU(input_size=self.gru_in_dim, hidden_size=self.model_cfg["gru_h"], num_layers=self.model_cfg["gru_l"], dropout=self.model_cfg["drop_p"] if "gru" in self.model_cfg["drop_pos"] else 0, batch_first=True))
            setattr(self, f"attn_l_id{attn_len_i}_fc_decoder", Sequential(Linear(self.model_cfg["gru_h"], self.fc_dec_out_dim), Dropout(self.model_cfg["drop_p"] if "fc_decoder" in self.model_cfg["drop_pos"] else 0)))
            for class_i in range(self.num_labels_classes):
                setattr(self, f"attn_l_id{attn_len_i}_class_fc{class_i}", Sequential(Linear(self.fc_dec_out_dim, self.class_fc_out_dim), Dropout(self.model_cfg["drop_p"] if "class_fc" in self.model_cfg["drop_pos"] else 0)))
        if type(self) == AttnOneDimGRUResMapCorrClass:
            self.init_optimizer()

    def forward(self, x: torch.Tensor, *unused_args, **unused_kwargs) -> torch.Tensor:
        """
        Forward pass of the model.
        """
        batch_size, seq_len, num_pairs = x.shape
        attn_input = x.permute(0, 2, 1)  # (batch_size, num_pairs, seq_len)
        attn1_out, attn1_weights = self.attn1(attn_input, attn_input, attn_input)  # (batch_size, num_pairs, attn_embed_dim), (batch_size, num_pairs, seq_len) ps. attn_embed_dim == seq_len
        split_attn1_out = torch.split(attn1_out, 1, dim=1)  # (batch_size, 1, attn_embed_dim) * num_pairs, ps. attn_embed_dim == seq_len
        split_x = torch.split(x, 1, dim=2)  # (batch_size, seq_len, 1) * num_pairs
        for attn_len_i, attn1_out_i, x_each_pair in enumerate(zip(split_attn1_out, split_x)):
            trans_attn1_out_i = attn1_out_i.permute(0, 2, 1)  # (batch_size, attn_embed_dim, 1), ps. attn_embed_dim == seq_len
            assert trans_attn1_out_i.shape == x_each_pair.shape, f"trans_attn1_out_i.shape:{trans_attn1_out_i.shape}, x_each_pair.shape:{x_each_pair.shape} are not equal and cannot be element-wise added."
            gru_input = trans_attn1_out_i + x_each_pair  # residual mapping
            gru_output, _ = getattr(self, f"attn_l_id{attn_len_i}_gru")(gru_input)  # (batch_size, seq_len, gru_h)
            fc_dec_output = getattr(self, f"attn_l_id{attn_len_i}_fc_decoder")(gru_output[:, -1, :]).unsqueeze(1)  # (batch_size, 1, fc_dec_out_dim), gru_output[:, -1, :] == (batch_size, gru_h)
            for class_i in range(self.num_labels_classes):
                class_fc_output = getattr(self, f"attn_l_id{attn_len_i}_class_fc{class_i}")(fc_dec_output)  # (batch_size, 1, class_fc_out_dim)
                attn_len_i_logits = class_fc_output if class_i == 0 else torch.cat([attn_len_i_logits, class_fc_output], dim=1)  # (batch_size, num_labels_classes, class_fc_out_dim), ps.  class_fc_out_dim == 1
            logits = attn_len_i_logits if attn_len_i == 0 else torch.cat([logits, attn_len_i_logits], dim=2)  # (batch_size, num_labels_classes, attn_out_len*class_fc_out_dim), ps.  class_fc_out_dim == 1
        batch_pred_probs = self.softmax(logits)
        assert batch_pred_probs.shape == (batch_size, self.num_labels_classes, self.attn_out_len*self.class_fc_out_dim), f"batch_pred_probs.shape:{batch_pred_probs.shape} is not equal to (batch_size, self.num_labels_classes, self.attn_out_len*self.class_fc_out_dim):{(batch_size, self.num_labels_classes, self.attn_out_len*self.class_fc_out_dim)}, ps. self.class_fc_out_dim=={self.class_fc_out_dim}"
        assert torch.allclose(torch.sum(batch_pred_probs, dim=1), torch.ones(batch_size, self.attn_out_len*self.class_fc_out_dim)), f"batch_pred_probs.sum(dim=1):{batch_pred_probs.sum(dim=1)} is not equal to 1."

        return batch_pred_probs

    def init_best_model_info(self, train_data: dict, val_data: dict, loss_fns: dict, epochs: int):
        """
        Initialize best_model_info
        """
        super().init_best_model_info(train_data, val_data, loss_fns, epochs)
        self.best_model_info.update({"attn_num_head": self.attn_num_heads})

        return self.best_model_info

    def init_epoch_metrics(self, loss_fns: dict):
        """
        Initialize epoch_metrics
        """
        super().init_epoch_metrics(loss_fns)
        self.epoch_metrics["attn_gradient"] = torch.zeros(1)

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
            epoch_metrics["attn_gradient"] += sum(p.grad.abs().sum() for p in self.attn1.parameters() if p.grad is not None)/num_batches
            for attr in dir(self):
                if re.match(r"attn_l_id\d+_gru", attr) and isinstance(getattr(self, attr), GRU):
                    epoch_metrics["gru_gradient"] += sum(p.grad.abs().sum() for p in getattr(self, attr).parameters() if p.grad is not None)/num_batches
                elif re.match(r"attn_l_id\d+_fc_decoder", attr) and isinstance(getattr(self, attr), Sequential):
                    epoch_metrics["fc_dec_gradient"] += sum(p.grad.abs().sum() for p in getattr(self, attr).parameters() if p.grad is not None)/num_batches
                elif re.match(r"attn_l_id\d+_class_fc\d+", attr) and isinstance(getattr(self, attr), Sequential):
                    epoch_metrics["class_fc_gradient"] += sum(p.grad.abs().sum() for p in getattr(self, attr).parameters() if p.grad is not None)/num_batches

        elif rec_stage == "val":
            epoch_metrics['val_edge_acc'] += batch_edge_acc/num_batches
            epoch_metrics['val_loss'] += batch_loss/num_batches
        elif rec_stage == "test":
            pass
        for fn_name, loss in batch_loss_each_loss_fn.items():
            epoch_metrics[(rec_stage+"_"+fn_name)] += loss/num_batches

