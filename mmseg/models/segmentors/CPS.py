from .. import build_segmentor, SEGMENTORS, build_loss
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
        assert self.branch1.decode_head.ignore_index == self.branch2.decode_head.ignore_index, "'ignore_index' of two branches must be the same."
        self.ignore_index = self.branch1.decode_head.ignore_index
        # Register a new loss module for culculating CPS loss
        self.criterion = build_loss(dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))

    #def forward_train(self, img, img_metas, **kwargs):
    def train_step(self, data_batch, optimizer, **kwargs):
        """
        GAN-like updating logic
        
        Arg
        optimizer: dict of 2 optimizers
        """
        kwargs.update({"img": data_batch["img"]})
        kwargs.update({"img_metas": data_batch["img_metas"]})
        kwargs.update({"gt_semantic_seg": data_batch["gt_semantic_seg"]})
        kwargs.update({"tag": [meta["tag"] for meta in data_batch["img_metas"]]})
        data_groups = dict_split(kwargs, "tag")
        for _, v in data_groups.items():
            v.pop("tag")
        # TODO: optimizer->zero_grad
        log_vars = dict()
        
       
        
        optimizer["branch1"].zero_grad()
        #optimizer["branch2"].zero_grad()
        
        """
        print("^^^^^^^^^^^^^^^^^^")
        for key, item in data_groups.items():
            print(key, len(item))
            for subkey, subitem in item.items():
                print(subkey, len(subitem))
        print("vvvvvvvvvvvvvvvvvv")
        """
        pred_sup_1 = self.branch1.encode_decode(data_groups['sup']['img'], data_groups['sup']['img_metas'])
        pred_unsup_1 = self.branch1.encode_decode(data_groups['unsup']['img'], data_groups['unsup']['img_metas'])
        pred_1 = self.branch1.encode_decode(data_batch["img"], data_batch["img_metas"])

        #with torch.no_grad():
        pred_sup_2 = self.branch2.encode_decode(
                data_groups['sup']['img'], data_groups['sup']['img_metas'])
        pred_unsup_2 = self.branch2.encode_decode(data_groups['unsup']['img'], data_groups['unsup']['img_metas'])

        # 1. calculate CPS loss for all data
        pred_1 = torch.cat([pred_sup_1, pred_unsup_1], dim=0)
        pred_2 = torch.cat([pred_sup_2, pred_unsup_2], dim=0)
        log_vars_cps = self._forward_cps_train(pred_1, pred_2)
        # sum up the cps loss; distributed handling...and all_reduce
        # using loss_cps,log_var_cps=_parse_losses()
        loss_cps, log_vars_cps = self._parse_losses(log_vars_cps)
        # change log_vars_cps['loss'] key name: 
        log_vars_cps['cps_loss'] = log_vars_cps.pop("loss")
        log_vars.update(log_vars_cps)

        # 2. calculate supervised loss (cross-entropy) for labeled data
        log_vars_sup = dict()
        log_vars_sup['sup_loss1'] = self._loss(self.branch1, pred_sup_1, data_groups['sup']['gt_semantic_seg'].squeeze(1))
        log_vars_sup['sup_loss2'] = self._loss(self.branch2, pred_sup_2, data_groups['sup']['gt_semantic_seg'].squeeze(1))
        loss_sup, log_vars_sup = self._parse_losses(log_vars_sup)
        log_vars_sup['supervised_loss'] = log_vars_sup.pop("loss")
        log_vars.update(log_vars_sup)

        loss = loss_cps+loss_sup
        
        loss.backward()

        optimizer["branch1"].step()
        optimizer["branch2"].step()

        outputs = dict(
            log_vars=log_vars,
            num_samples=len(data_batch['img_metas']))

        return outputs

    def forward_train(self, imgs, img_metas, **kwargs):
        # This function should not be called
        raise NotImplementedError()

    def _forward_cps_train(self, pred_1, pred_2):
        loss = {}
        label1 = pred_1.argmax(dim=1)
        label2 = pred_2.argmax(dim=1)
        loss['unsup_loss1'] = weighted_loss(self._loss(self.branch1, pred_1, label2, True)
                                            , self.unsup_weight)
        loss['unsup_loss2'] = weighted_loss(self._loss(self.branch2, pred_2, label1, True), self.unsup_weight)
        return loss

    def _loss(self, branch, pred, label, is_CPS_loss = False):
        """
        branch: EncoderDecoder
        """
        if is_CPS_loss is False:
            loss = branch.decode_head.loss_decode(pred, label, ignore_index=self.ignore_index)
        else:
            loss = self.criterion(pred, label, ignore_index=self.ignore_index)
        return loss

