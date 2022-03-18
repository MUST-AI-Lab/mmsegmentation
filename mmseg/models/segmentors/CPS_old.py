from .. import build_segmentor, SEGMENTORS
from .multi_stream_segmentor import MultiStreamSegmentor
from mmseg.utils import dict_split, weighted_loss
import torch
import copy

@SEGMENTORS.register_module()
class CrossPesudoSupervision(MultiStreamSegmentor):
    def __init__(self, model, train_cfg=None, test_cfg=None):
        super(CrossPesudoSupervision, self).__init__(
            dict(branch1=build_segmentor(copy.deepcopy(model)), branch2=build_segmentor(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg
        )
        if train_cfg is not None:
            self.unsup_weight = self.train_cfg.unsup_weight

    def forward_train(self, img, img_metas, **kwargs):
        super().forward_train(img, img_metas, **kwargs)
        kwargs.update({"img": img})
        kwargs.update({"img_metas": img_metas})
        kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
        data_groups = dict_split(kwargs, "tag")
        for _, v in data_groups.items():
            v.pop("tag")

        loss = dict()
        """
        print("^^^^^^^^^^^^^^^^^^")
        for key, item in data_groups.items():
            print(key, len(item))
            for subkey, subitem in item.items():
                print(subkey, len(subitem))
        print("vvvvvvvvvvvvvvvvvv")
        """
        pred_sup_1 = self.branch1.encode_decode(
            data_groups['sup']['img'], data_groups['sup']['img_metas'])
        pred_unsup_1 = self.branch1.encode_decode(
            data_groups['unsup']['img'], data_groups['unsup']['img_metas'])
        pred_sup_2 = self.branch2.encode_decode(
            data_groups['sup']['img'], data_groups['sup']['img_metas'])
        pred_unsup_2 = self.branch2.encode_decode(
            data_groups['unsup']['img'], data_groups['unsup']['img_metas'])

        # 1. calculate CPS loss for all data
        pred_1 = torch.cat([pred_sup_1, pred_unsup_1], dim=0)
        pred_2 = torch.cat([pred_sup_2, pred_unsup_2], dim=0)
        loss_cps = self._forward_cps_train(pred_1, pred_2)
        loss.update(loss_cps)
        # 2. calculate supervised loss (cross-entropy) for labeled data
        loss['sup_loss1'] = self._loss(self.branch1, pred_sup_1, data_groups['sup']['gt_semantic_seg'].squeeze(1))
        loss['sup_loss2'] = self._loss(self.branch2, pred_sup_2, data_groups['sup']['gt_semantic_seg'].squeeze(1))
        return loss

    def _forward_cps_train(self, pred_1, pred_2):
        loss = {}
        label1 = pred_1.argmax(dim=1)
        label2 = pred_2.argmax(dim=1)
        loss['unsup_loss1'] = weighted_loss(self._loss(self.branch1, pred_1, label2)
                                            , self.unsup_weight)
        loss['unsup_loss2'] = weighted_loss(self._loss(self.branch2, pred_2, label1)
                                            , self.unsup_weight)
        return loss

    def _loss(self, branch, pred, label):
        """
        branch: EncoderDecoder
        """
        loss = branch.decode_head.loss_decode(pred, label, ignore_index=branch.decode_head.ignore_index)
        return loss

